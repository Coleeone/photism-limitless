#!/usr/bin/env python3
"""Flag-gated, single-process Photism Limitless FX Drive queue worker; CLI only, no listeners.

Identity: baxter-photism-fx/1
Watches: I:\\My Drive\\BAXTER\\BAXTER_PHOTISM_JOBS
Version: 1.1.0
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import msvcrt
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(os.environ.get('PHOTISM_ROOT', Path(__file__).resolve().parent))
QUEUE = Path(os.environ.get('PHOTISM_QUEUE', r'D:\MetaMuse\MUSE_PHOTISM_JOBS' if Path(r'D:\MetaMuse').exists() else ROOT / 'queue'))
JOBS = ROOT / 'jobs'
FLAG = ROOT / 'photism-fx-worker-enabled.flag'
CLI = ROOT / 'photism-cli.exe'
FFPROBE = Path(shutil.which('ffprobe') or r'C:\Program Files\ffmpeg\bin\ffprobe.exe' or r'C:\Users\bmcol\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe')
WORKER_ID = 'baxter-photism-fx/1'
WORKER_VERSION = '1.1.0'
CLI_SHA256 = 'd43136a50f06d763bfc87cefb79f4531b5afa2f441c7ab2ed3fa370a0e465fc6'

JOB_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')

VALID_MODES = {
    'none', 'smear', 'droste', 'infinite', 'corridor', 'cube', 'planet',
    'windows', 'melt', 'shatter', 'prism', 'terrain', 'breathe', 'mosh',
    'sort', 'coral', 'scanline', 'thresh', 'solar', 'halftone', 'vhs'
}

# Param allowlist v2: name -> (type, min, max)
PARAM_SPEC_V2 = {
    'speed': (float, None, None),       # default 50.0, any finite positive/zero/negative speed rate
    'chaos': (float, 0.0, 100.0),
    'trails': (float, 0.0, 100.0),
    'decay': (float, 0.0, 100.0),
    'react': (float, 0.0, 200.0),
    'heat': (float, 0.0, 100.0),
    'symmetry': (float, 0.0, 12.0),
    'tiles': (float, 0.0, 12.0),
    'twist': (float, 0.0, 100.0),
    'slice': (float, 0.0, 100.0),
    'lens': (float, 0.0, 100.0),
    'lens-style': (int, 0, 6),
    'hue-drift': (float, 0.0, 100.0),
    'vignette': (float, 0.0, 100.0),
    'contrast': (float, 0.0, 100.0),
    'dry-wet': (float, 0.0, 100.0),
    'drive-low': (float, 0.0, 100.0),
    'drive-mid': (float, 0.0, 100.0),
    'drive-high': (float, 0.0, 100.0),
    'pulse': (float, 0.0, 100.0)
}

ALLOWED_TOP_KEYS = {'schema_version', 'job_id', 'ready', 'mode', 'params', 'input'}
ALLOWED_INPUT_KEYS = {'path', 'bytes', 'sha256'}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def safe_node(path: Path):
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
        raise ValueError(f'Link or junction forbidden: {path}')


def relative_path(value: str):
    if not isinstance(value, str) or not value or any(c in value for c in ':\x00<>"|?*'):
        raise ValueError(f'Invalid relative path: {value!r}')
    value = value.replace('\\', '/')
    parts = value.split('/')
    devices = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if PureWindowsPath(value).is_absolute() or any(
        p in ('', '.', '..') or p.rstrip(' .') != p or p.split('.')[0].upper() in devices
        or any(ord(c) < 32 for c in p) for p in parts
    ):
        raise ValueError(f'Unsafe relative path: {value!r}')
    return value


def file_digest(path: Path):
    safe_node(path)
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as f:
        while b := f.read(1024 * 1024):
            size += len(b)
            h.update(b)
    if path.stat().st_size != size:
        raise ValueError(f'File changed while hashing: {path}')
    return {'sha256': h.hexdigest().lower(), 'bytes': size}


def read_json(path: Path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError(f'Duplicate JSON key: {k}')
            out[k] = v
        return out
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def write_json(path: Path, value, exclusive=False):
    safe_node(path.parent)
    safe_node(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.writing')
    with tmp.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    if exclusive:
        os.rename(tmp, path)
    else:
        os.replace(tmp, path)


def probe_video(path: Path):
    safe_node(path)
    if not FFPROBE.exists():
        return {'error': f'ffprobe not found at {FFPROBE}'}
    cmd = [
        str(FFPROBE), '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,nb_frames,duration,r_frame_rate',
        '-of', 'json',
        str(path)
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
        if r.returncode != 0:
            return {'error': r.stderr.strip()}
        data = json.loads(r.stdout)
        streams = data.get('streams', [])
        if not streams:
            return {'error': 'No video streams found'}
        st = streams[0]
        nb_frames = st.get('nb_frames')
        if nb_frames is not None:
            try:
                nb_frames = int(nb_frames)
            except ValueError:
                pass
        duration = st.get('duration')
        if duration is not None:
            try:
                duration = float(duration)
            except ValueError:
                pass
        return {
            'codec': st.get('codec_name'),
            'width': st.get('width'),
            'height': st.get('height'),
            'frames': nb_frames,
            'duration': duration,
            'r_frame_rate': st.get('r_frame_rate')
        }
    except Exception as e:
        return {'error': str(e)}


def format_param_val(val) -> str:
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        # Format without scientific notation, e.g. 50.0 or 0.123
        s = f"{val:.6f}".rstrip('0').rstrip('.')
        return s if s else "0"
    return str(val)


def validate_manifest(manifest: dict, jid: str):
    if not isinstance(manifest, dict):
        raise ValueError('Manifest root must be an object')
    
    unknown_top = set(manifest.keys()) - ALLOWED_TOP_KEYS
    if unknown_top:
        raise ValueError(f'Unknown top-level fields: {sorted(unknown_top)}')
    
    if manifest.get('schema_version') != 1:
        raise ValueError('schema_version must be exactly 1')
    if manifest.get('ready') is not True:
        raise ValueError('ready must be boolean true')
    if manifest.get('job_id') != jid:
        raise ValueError(f"job_id in manifest '{manifest.get('job_id')}' does not match folder '{jid}'")
    
    mode = manifest.get('mode')
    if not isinstance(mode, str) or mode.lower() not in VALID_MODES:
        raise ValueError(f'Unknown or invalid mode: {mode!r}')
    mode = mode.lower()
    
    params = manifest.get('params')
    if not isinstance(params, dict):
        raise ValueError('params must be an object')
    
    unknown_params = set(params.keys()) - set(PARAM_SPEC_V2.keys())
    if unknown_params:
        raise ValueError(f'Unknown params in manifest: {sorted(unknown_params)}')
    
    validated_params = {}
    for p_name, p_val in params.items():
        expected_type, p_min, p_max = PARAM_SPEC_V2[p_name]
        if expected_type is int:
            if type(p_val) is not int:
                raise ValueError(f'Param {p_name} must be integer, got {type(p_val).__name__}: {p_val!r}')
        else: # float or int allowed
            if type(p_val) not in (int, float):
                raise ValueError(f'Param {p_name} must be numeric, got {type(p_val).__name__}: {p_val!r}')
            p_val = float(p_val)

        if p_min is not None and p_val < p_min:
            raise ValueError(f'Param {p_name}={p_val} below minimum {p_min}')
        if p_max is not None and p_val > p_max:
            raise ValueError(f'Param {p_name}={p_val} above maximum {p_max}')
        
        validated_params[p_name] = p_val

    inp = manifest.get('input')
    if not isinstance(inp, dict):
        raise ValueError('input must be an object')
    unknown_input_keys = set(inp.keys()) - ALLOWED_INPUT_KEYS
    if unknown_input_keys:
        raise ValueError(f'Unknown input fields: {sorted(unknown_input_keys)}')
    
    rel_p = relative_path(inp.get('path'))
    expected_bytes = inp.get('bytes')
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError(f'input.bytes must be positive integer: {expected_bytes!r}')
    expected_sha = inp.get('sha256')
    if not isinstance(expected_sha, str) or not re.fullmatch(r'[a-fA-F0-9]{64}', expected_sha):
        raise ValueError(f'input.sha256 must be 64-hex SHA-256 string: {expected_sha!r}')

    return {
        'mode': mode,
        'params': validated_params,
        'input_rel': rel_p,
        'expected_bytes': expected_bytes,
        'expected_sha256': expected_sha.lower()
    }


def process_job(drive_folder: Path, jid: str, logger: logging.Logger):
    t_start = time.perf_counter()
    started_at = utc_now()
    stage_dir = JOBS / jid
    stage_dir.mkdir(parents=True, exist_ok=False)

    logger.info(f"[{jid}] Claimed job, staged at {stage_dir}")

    manifest_drive_path = drive_folder / 'manifest.json'
    render_log_path = drive_folder / 'render-log.json'

    # Pre-verify CLI
    cli_dig = file_digest(CLI)
    if cli_dig['sha256'] != CLI_SHA256:
        raise RuntimeError(f"CLI binary hash mismatch: expected {CLI_SHA256}, got {cli_dig['sha256']}")

    t_stage_start = time.perf_counter()
    
    # Read manifest
    manifest_raw = read_json(manifest_drive_path)
    spec = validate_manifest(manifest_raw, jid)
    
    input_drive_path = drive_folder / spec['input_rel']
    if not input_drive_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_drive_path}")

    # Stage immutable snapshot
    staged_input = stage_dir / spec['input_rel']
    staged_input.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_drive_path, staged_input)

    staged_manifest = stage_dir / 'manifest.json'
    shutil.copyfile(manifest_drive_path, staged_manifest)

    # SHA-256 Preflight check on staged input
    actual_input_dig = file_digest(staged_input)
    hash_passed = (
        actual_input_dig['bytes'] == spec['expected_bytes'] and
        actual_input_dig['sha256'] == spec['expected_sha256']
    )

    hash_check_record = {
        'expected_bytes': spec['expected_bytes'],
        'actual_bytes': actual_input_dig['bytes'],
        'expected_sha256': spec['expected_sha256'],
        'actual_sha256': actual_input_dig['sha256'],
        'passed': hash_passed
    }

    if not hash_passed:
        raise ValueError(f"Input SHA-256 or byte size mismatch: {hash_check_record}")

    t_staging_sec = time.perf_counter() - t_stage_start

    # Build argv command array (no shell=True)
    staged_output = stage_dir / 'output.mp4'
    cmd = [
        str(CLI),
        '-i', str(staged_input),
        '-o', str(staged_output),
        '--mode', spec['mode']
    ]
    for p_name, p_val in sorted(spec['params'].items()):
        formatted_val = format_param_val(p_val)
        cmd.extend([f'--{p_name}', formatted_val])

    logger.info(f"[{jid}] Running CLI command: {cmd}")
    t_run_start = time.perf_counter()
    
    res = subprocess.run(cmd, cwd=stage_dir, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800)
    t_run_sec = time.perf_counter() - t_run_start
    
    if res.returncode != 0:
        raise RuntimeError(f"photism-cli exited {res.returncode}: stderr={res.stderr.strip()} stdout={res.stdout.strip()}")

    if not staged_output.exists() or staged_output.stat().st_size == 0:
        raise RuntimeError("photism-cli did not generate output.mp4 or generated a 0-byte file")

    # Probe output
    out_dig = file_digest(staged_output)
    probe_res = probe_video(staged_output)
    if 'error' in probe_res:
        raise RuntimeError(f"Output verification failed: {probe_res['error']}")

    # Zero-frame guard: fail if frames == 0 even if returncode was 0
    frames_count = probe_res.get('frames')
    if frames_count == 0:
        raise RuntimeError("CLI rendered 0 frames")

    # Publish outputs/output.mp4 back to Drive
    drive_outputs_dir = drive_folder / 'outputs'
    drive_outputs_dir.mkdir(parents=True, exist_ok=True)
    drive_output_file = drive_outputs_dir / 'output.mp4'
    shutil.copyfile(staged_output, drive_output_file)

    finished_at = utc_now()
    t_total_sec = time.perf_counter() - t_start

    render_log = {
        'worker': WORKER_ID,
        'worker_version': WORKER_VERSION,
        'job_id': jid,
        'final_status': 'completed',
        'started_at': started_at,
        'finished_at': finished_at,
        'cli_path': str(CLI),
        'cli_sha256': CLI_SHA256,
        'hash_checks': hash_check_record,
        'command': cmd,
        'outputs': {
            'path': 'outputs/output.mp4',
            'bytes': out_dig['bytes'],
            'sha256': out_dig['sha256']
        },
        'output_probes': probe_res,
        'timings': {
            'staging_seconds': round(t_staging_sec, 4),
            'run_seconds': round(t_run_sec, 4),
            'total_seconds': round(t_total_sec, 4)
        },
        'error': None
    }

    write_json(render_log_path, render_log)
    write_json(stage_dir / 'render-log.json', render_log)
    logger.info(f"[{jid}] Finished successfully in {round(t_total_sec, 2)}s")


def fail_job(drive_folder: Path, jid: str, error_msg: str, logger: logging.Logger):
    logger.error(f"[{jid}] Job failed: {error_msg}")
    finished_at = utc_now()
    render_log_path = drive_folder / 'render-log.json'
    stage_dir = JOBS / jid

    render_log = {
        'worker': WORKER_ID,
        'worker_version': WORKER_VERSION,
        'job_id': jid,
        'final_status': 'failed',
        'started_at': utc_now(),
        'finished_at': finished_at,
        'cli_path': str(CLI),
        'cli_sha256': CLI_SHA256,
        'hash_checks': None,
        'command': None,
        'outputs': None,
        'output_probes': None,
        'timings': {
            'staging_seconds': 0.0,
            'run_seconds': 0.0,
            'total_seconds': 0.0
        },
        'error': error_msg
    }
    try:
        write_json(render_log_path, render_log)
    except Exception as e:
        logger.error(f"Failed to write render-log.json to Drive: {e}")
    if stage_dir.exists():
        try:
            write_json(stage_dir / 'render-log.json', render_log)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Baxter Photism Limitless FX Drive Queue Render Worker")
    parser.add_argument('--session-minutes', type=int, default=480, help="Max run duration in minutes")
    parser.add_argument('--once', action='store_true', help="Process at most one job then exit")
    parser.add_argument('--job', type=str, default=None, help="Process only this specific job ID")
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    JOBS.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(parents=True, exist_ok=True)

    log_file = ROOT / 'logs' / 'photism_fx_worker.log'
    logger = logging.getLogger('PhotismWorker')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(str(log_file), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Exclusive single-process instance lock
    lock_file_path = ROOT / 'photism_worker.lock'
    lock_f = lock_file_path.open('w+')
    try:
        msvcrt.locking(lock_f.fileno(), msvcrt.LK_NBLCK, 1)
    except IOError:
        logger.error("Another instance of photism_fx_worker is already running. Exiting.")
        return 1

    logger.info(f"Starting {WORKER_ID} v{WORKER_VERSION}. Watching {QUEUE}. Session minutes: {args.session_minutes}")

    session_end = time.time() + (args.session_minutes * 60)

    try:
        while time.time() < session_end:
            # Check enabled flag gate
            if not FLAG.exists():
                logger.info(f"Flag {FLAG.name} absent. Worker paused. Sleeping 5s...")
                time.sleep(5)
                continue

            if not QUEUE.exists():
                logger.warning(f"Queue directory {QUEUE} not found. Sleeping 5s...")
                time.sleep(5)
                continue

            # Scan for candidate jobs
            entries = []
            try:
                entries = sorted([p for p in QUEUE.iterdir() if p.is_dir()])
            except Exception as e:
                logger.error(f"Error reading queue dir: {e}")
                time.sleep(3)
                continue

            found_job = False
            for drive_entry in entries:
                jid = drive_entry.name
                if not JOB_ID_RE.fullmatch(jid):
                    continue

                if args.job and jid != args.job:
                    continue

                manifest_file = drive_entry / 'manifest.json'
                render_log_file = drive_entry / 'render-log.json'
                staged_local = JOBS / jid

                # Skip if already rendered in drive, or claimed locally
                if render_log_file.exists() or staged_local.exists():
                    continue

                # Upload order: manifest.json is published LAST
                if not manifest_file.exists():
                    continue

                found_job = True
                logger.info(f"Found ready job: {jid}")
                try:
                    process_job(drive_entry, jid, logger)
                except Exception as ex:
                    fail_job(drive_entry, jid, str(ex), logger)

                if args.once:
                    logger.info("Ran with --once. Exiting.")
                    return 0
                break  # Process one job per cycle

            if not found_job:
                if args.once and args.job:
                    logger.info(f"Job {args.job} not found or not ready with --once. Exiting.")
                    return 0
                time.sleep(2)

    finally:
        try:
            msvcrt.locking(lock_f.fileno(), msvcrt.LK_UNLCK, 1)
            lock_f.close()
        except Exception:
            pass

    logger.info("Session expired. Worker stopped cleanly.")
    return 0


if __name__ == '__main__':
    exit(main())

