#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

#include "src/PhotismParams.h"

// Forward declaration of host CUDA launch function from PhotismLimitless.cu
extern "C" void LaunchSmearPipeline_CUDA(
    const float4* srcImg,
    float2* gradBuffer,
    const float4* historyIn,
    float4* historyOut,
    float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    PhotismParamsGPU params,
    cudaStream_t stream = 0);

// Helper to convert mode name to integer
int ParseMode(const std::string& name) {
    std::string s = name;
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    if (s == "none") return MODE_NONE;
    if (s == "smear") return MODE_SMEAR;
    if (s == "droste") return MODE_DROSTE;
    if (s == "infinite") return MODE_INFINITE;
    if (s == "corridor") return MODE_CORRIDOR;
    if (s == "cube") return MODE_CUBE;
    if (s == "planet") return MODE_PLANET;
    if (s == "windows") return MODE_WINDOWS;
    if (s == "melt") return MODE_MELT;
    if (s == "shatter") return MODE_SHATTER;
    if (s == "prism") return MODE_PRISM;
    if (s == "terrain") return MODE_TERRAIN;
    if (s == "breathe") return MODE_BREATHE;
    if (s == "mosh") return MODE_MOSH;
    if (s == "sort") return MODE_SORT;
    if (s == "coral") return MODE_CORAL;
    if (s == "scanline") return MODE_SCANLINE;
    if (s == "thresh") return MODE_THRESH;
    if (s == "solar") return MODE_SOLAR;
    if (s == "halftone") return MODE_HALFTONE;
    if (s == "vhs") return MODE_VHS;
    return atoi(name.c_str());
}

void PrintUsage() {
    std::cout << "=========================================================================\n"
              << " photism-cli: Headless GPU/CUDA Video & Pipe Processor for FilmCraft & Baxter\n"
              << "=========================================================================\n"
              << "Usage:\n"
              << "  photism-cli [options] -i <input.mp4> -o <output.mp4>\n"
              << "  ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgba - | photism-cli [options] --pipe-w 1920 --pipe-h 1080 | ffmpeg ...\n\n"
              << "Key Options:\n"
              << "  --mode <name/id>       Effect mode: smear, droste, infinite, corridor, cube, planet,\n"
              << "                         windows, melt, shatter, prism, terrain, breathe, mosh, sort,\n"
              << "                         coral, scanline, thresh, solar, halftone, vhs (default: droste)\n"
              << "  --speed <float>        Animation speed rate (default: 50.0)\n"
              << "  --chaos <float>        Turbulence / distortion (default: 23.0)\n"
              << "  --trails <float>       Motion blur / echo trail amount 0-100 (default: 82.0)\n"
              << "  --decay <float>        Trail half-life / persistence 0-100 (default: 55.0)\n"
              << "  --react <float>        Audio / drive responsiveness 0-200 (default: 36.0)\n"
              << "  --heat <float>         Luminescence / highlight glow 0-100 (default: 55.0)\n"
              << "  --symmetry <float>     Rotational kaleidoscope sectors 0-12 (default: 0.0)\n"
              << "  --tiles <float>        Grid tiling repetitions 0-12 (default: 0.0)\n"
              << "  --twist <float>        Vortex twist angle 0-100 (default: 0.0)\n"
              << "  --slice <float>        Glitch displacement slice 0-100 (default: 0.0)\n"
              << "  --lens <float>         Geometric lens distortion 0-100 (default: 0.0)\n"
              << "  --lens-style <int>     0:Ribs, 1:Fan, 2:Anamorphic, 3:Melt, 4:Glitch, 5:Pixels, 6:FlyEye\n"
              << "  --hue-drift <float>    Color spectrum rotation 0-100 (default: 0.0)\n"
              << "  --vignette <float>     Edge darkness falloff 0-100 (default: 0.0)\n"
              << "  --contrast <float>     Punch / S-curve contrast 0-100 (default: 98.0)\n"
              << "  --dry-wet <float>      Effect blend ratio 0-100 (default: 100.0)\n"
              << "  --drive-low <float>    Bass frequency drive injection 0-100 (default: 0.0)\n"
              << "  --drive-mid <float>    Mid frequency drive injection 0-100 (default: 0.0)\n"
              << "  --drive-high <float>   Treble frequency drive injection 0-100 (default: 0.0)\n"
              << "  --pulse <float>        Rhythmic bounce pulse 0-100 (default: 0.0)\n"
              << "  --fps <float>          Source frame rate (default: 30.0)\n"
              << "  --bitrate <str>        Output video bitrate for FFmpeg encode (default: 18M)\n"
              << "  --crf <int>            H.264 quality CRF (default: 17)\n"
              << "  --nvenc                Use hardware NVENC (h264_nvenc) on NVIDIA GPU (default: ON)\n"
              << "=========================================================================\n";
}

// Convert RGBA 8-bit to BGRA float4 on GPU
__global__ void kRgba8ToFloat4Bgra(const uchar4* __restrict__ src8, float4* __restrict__ dstF, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    uchar4 in = src8[idx];
    // Invert R and B to match Premiere/Photism BGRA format
    dstF[idx] = make_float4(
        static_cast<float>(in.z) / 255.0f, // Blue
        static_cast<float>(in.y) / 255.0f, // Green
        static_cast<float>(in.x) / 255.0f, // Red
        static_cast<float>(in.w) / 255.0f  // Alpha
    );
}

// Convert BGRA float4 to RGBA 8-bit on GPU
__global__ void kFloat4BgraToRgba8(const float4* __restrict__ srcF, uchar4* __restrict__ dst8, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    float4 in = srcF[idx];
    // Invert B and R to output standard RGBA
    dst8[idx] = make_uchar4(
        static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, in.z * 255.0f + 0.5f))), // Red
        static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, in.y * 255.0f + 0.5f))), // Green
        static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, in.x * 255.0f + 0.5f))), // Blue
        static_cast<unsigned char>(fminf(255.0f, fmaxf(0.0f, in.w * 255.0f + 0.5f)))  // Alpha
    );
}

int main(int argc, char* argv[]) {
    std::string inputFile = "";
    std::string outputFile = "";
    bool pipeMode = false;
    int pipeW = 1920;
    int pipeH = 1080;
    float fps = 30.0f;
    std::string bitrate = "18M";
    int crf = 17;
    bool useNvenc = true;

    PhotismParamsGPU params = {};
    params.mode        = MODE_DROSTE;
    params.react       = DFLT_REACT;
    params.chaos       = DFLT_CHAOS;
    params.trails      = DFLT_TRAILS;
    params.heat        = DFLT_HEAT;
    params.decay       = DFLT_DECAY;
    params.brightness  = DFLT_BRIGHTNESS;
    params.inputGain   = DFLT_INPUT_GAIN;
    params.symmetry    = DFLT_SYMMETRY;
    params.tiles       = DFLT_TILES;
    params.twist       = DFLT_TWIST;
    params.slice       = DFLT_SLICE;
    params.lens        = DFLT_LENS;
    params.lensStyle   = DFLT_LENS_STYLE;
    params.hueDrift    = DFLT_HUE_DRIFT;
    params.vignette    = DFLT_VIGNETTE;
    params.palette     = DFLT_PALETTE;
    params.speed       = DFLT_SPEED;
    params.shape       = DFLT_SHAPE;
    params.complexity  = DFLT_COMPLEXITY;
    params.fill        = DFLT_FILL;
    params.contrast    = DFLT_CONTRAST;
    params.dryWet      = DFLT_DRY_WET;
    params.driveLow    = DFLT_DRIVE_LOW;
    params.driveMid    = DFLT_DRIVE_MID;
    params.driveHigh   = DFLT_DRIVE_HIGH;
    params.pulse       = DFLT_PULSE;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) inputFile = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) outputFile = argv[++i];
        else if (arg == "--mode" && i + 1 < argc) params.mode = ParseMode(argv[++i]);
        else if (arg == "--speed" && i + 1 < argc) params.speed = atof(argv[++i]);
        else if (arg == "--chaos" && i + 1 < argc) params.chaos = atof(argv[++i]);
        else if (arg == "--trails" && i + 1 < argc) params.trails = atof(argv[++i]);
        else if (arg == "--heat" && i + 1 < argc) params.heat = atof(argv[++i]);
        else if (arg == "--decay" && i + 1 < argc) params.decay = atof(argv[++i]);
        else if (arg == "--brightness" && i + 1 < argc) params.brightness = atof(argv[++i]);
        else if (arg == "--input-gain" && i + 1 < argc) params.inputGain = atof(argv[++i]);
        else if (arg == "--symmetry" && i + 1 < argc) params.symmetry = atof(argv[++i]);
        else if (arg == "--tiles" && i + 1 < argc) params.tiles = atof(argv[++i]);
        else if (arg == "--twist" && i + 1 < argc) params.twist = atof(argv[++i]);
        else if (arg == "--slice" && i + 1 < argc) params.slice = atof(argv[++i]);
        else if (arg == "--lens" && i + 1 < argc) params.lens = atof(argv[++i]);
        else if (arg == "--lens-style" && i + 1 < argc) params.lensStyle = atoi(argv[++i]);
        else if (arg == "--hue-drift" && i + 1 < argc) params.hueDrift = atof(argv[++i]);
        else if (arg == "--vignette" && i + 1 < argc) params.vignette = atof(argv[++i]);
        else if (arg == "--palette" && i + 1 < argc) params.palette = atoi(argv[++i]);
        else if (arg == "--shape" && i + 1 < argc) params.shape = atof(argv[++i]);
        else if (arg == "--complexity" && i + 1 < argc) params.complexity = atof(argv[++i]);
        else if (arg == "--fill" && i + 1 < argc) params.fill = atof(argv[++i]);
        else if (arg == "--contrast" && i + 1 < argc) params.contrast = atof(argv[++i]);
        else if (arg == "--dry-wet" && i + 1 < argc) params.dryWet = atof(argv[++i]);
        else if (arg == "--react" && i + 1 < argc) params.react = atof(argv[++i]);
        else if (arg == "--drive-low" && i + 1 < argc) params.driveLow = atof(argv[++i]);
        else if (arg == "--drive-mid" && i + 1 < argc) params.driveMid = atof(argv[++i]);
        else if (arg == "--drive-high" && i + 1 < argc) params.driveHigh = atof(argv[++i]);
        else if (arg == "--pulse" && i + 1 < argc) params.pulse = atof(argv[++i]);
        else if (arg == "--fps" && i + 1 < argc) fps = atof(argv[++i]);
        else if (arg == "--pipe-w" && i + 1 < argc) { pipeW = atoi(argv[++i]); pipeMode = true; }
        else if (arg == "--pipe-h" && i + 1 < argc) { pipeH = atoi(argv[++i]); pipeMode = true; }
        else if (arg == "--bitrate" && i + 1 < argc) bitrate = argv[++i];
        else if (arg == "--crf" && i + 1 < argc) crf = atoi(argv[++i]);
        else if (arg == "--no-nvenc") useNvenc = false;
        else if (arg == "-h" || arg == "--help") { PrintUsage(); return 0; }
    }

    if (!pipeMode && (inputFile.empty() || outputFile.empty())) {
        std::cerr << "Error: Input and Output files must be specified (or use pipe mode with --pipe-w and --pipe-h).\n\n";
        PrintUsage();
        return 1;
    }

    // Initialize CUDA Device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA capable GPU found on this system.\n";
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cerr << "[photism-cli] Active GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n";

    FILE* inPipe = nullptr;
    FILE* outPipe = nullptr;
    int width = pipeW;
    int height = pipeH;

    if (!pipeMode) {
        // Probe input video resolution and frame rate using ffprobe
        std::string probeCmd = "ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 \"" + inputFile + "\"";
        FILE* pF = _popen(probeCmd.c_str(), "r");
        if (pF) {
            char buf[256];
            if (fgets(buf, sizeof(buf), pF)) {
                int pw = 0, ph = 0;
                int num = 0, den = 1;
                if (sscanf(buf, "%d,%d,%d/%d", &pw, &ph, &num, &den) >= 2) {
                    if (pw > 0 && ph > 0) {
                        width = pw;
                        height = ph;
                    }
                    if (den > 0 && num > 0) {
                        fps = static_cast<float>(num) / static_cast<float>(den);
                    }
                }
            }
            _pclose(pF);
        }
        std::cerr << "[photism-cli] Video Stream: " << width << "x" << height << " @ " << fps << " fps\n";

        // Spawn FFmpeg Decoder Process (pipes RGBA raw frames to stdin)
        std::string decodeCmd = "ffmpeg -v error -i \"" + inputFile + "\" -f rawvideo -pix_fmt rgba -";
        inPipe = _popen(decodeCmd.c_str(), "rb");
        if (!inPipe) {
            std::cerr << "Error: Failed to launch FFmpeg decode pipe.\n";
            return 1;
        }

        // Spawn FFmpeg Encoder Process (receives processed RGBA frames from stdout)
        std::string encoderStr = useNvenc ? "h264_nvenc -preset p7 -tune hq" : "libx264 -preset slow";
        std::string encodeCmd = "ffmpeg -v error -y -f rawvideo -pix_fmt rgba -s " + std::to_string(width) + "x" + std::to_string(height)
                              + " -r " + std::to_string(fps) + " -i - -i \"" + inputFile + "\" -map 0:v -map 1:a? -c:v " + encoderStr
                              + " -b:v " + bitrate + " -pix_fmt yuv420p -c:a copy \"" + outputFile + "\"";
        outPipe = _popen(encodeCmd.c_str(), "wb");
        if (!outPipe) {
            std::cerr << "Error: Failed to launch FFmpeg encode pipe.\n";
            _pclose(inPipe);
            return 1;
        }
    } else {
        inPipe = stdin;
        outPipe = stdout;
    }

    params.width = width;
    params.height = height;
    params.srcPitch = width;
    params.destPitch = width;
    params.is16f = 0;
    params.dt = 1.0f / fps;

    size_t numPixels = static_cast<size_t>(width) * height;
    size_t byteSize8 = numPixels * sizeof(uchar4);
    size_t byteSizeF4 = numPixels * sizeof(float4);

    // Allocate Host and Device Buffers
    std::vector<uchar4> hostIn(numPixels);
    std::vector<uchar4> hostOut(numPixels);

    uchar4* d_src8 = nullptr;
    uchar4* d_dst8 = nullptr;
    float4* d_srcF = nullptr;
    float4* d_destF = nullptr;
    float4* d_histA = nullptr;
    float4* d_histB = nullptr;
    float2* d_grad = nullptr;

    cudaMalloc(&d_src8, byteSize8);
    cudaMalloc(&d_dst8, byteSize8);
    cudaMalloc(&d_srcF, byteSizeF4);
    cudaMalloc(&d_destF, byteSizeF4);
    cudaMalloc(&d_histA, byteSizeF4);
    cudaMalloc(&d_histB, byteSizeF4);
    cudaMalloc(&d_grad, numPixels * sizeof(float2));

    cudaMemset(d_histA, 0, byteSizeF4);
    cudaMemset(d_histB, 0, byteSizeF4);

    dim3 threads(16, 16);
    dim3 grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

    int frameIndex = 0;
    float speedRate = (params.speed * 0.01f) * (1.0f + (params.react * 0.01f) * (params.driveMid * 0.01f));

    std::cerr << "[photism-cli] Processing frames with Mode: " << params.mode << "...\n";

    while (fread(hostIn.data(), 1, byteSize8, inPipe) == byteSize8) {
        params.frameIndex = frameIndex;
        params.phase = static_cast<float>(frameIndex) * params.dt * speedRate;

        // 1. Copy RGBA 8-bit frame to GPU
        cudaMemcpy(d_src8, hostIn.data(), byteSize8, cudaMemcpyHostToDevice);

        // 2. Unpack RGBA8 -> BGRA float4
        kRgba8ToFloat4Bgra<<<grid, threads>>>(d_src8, d_srcF, width, height);

        // 3. Run Photism GPU Limitless Pipeline
        LaunchSmearPipeline_CUDA(
            d_srcF,
            d_grad,
            d_histA,
            d_histB,
            d_destF,
            width,
            0,
            width,
            height,
            params,
            0
        );

        // 4. Pack BGRA float4 -> RGBA8
        kFloat4BgraToRgba8<<<grid, threads>>>(d_destF, d_dst8, width, height);

        // 5. Copy processed frame back to Host
        cudaMemcpy(hostOut.data(), d_dst8, byteSize8, cudaMemcpyDeviceToHost);

        // 6. Write to encoder pipe
        fwrite(hostOut.data(), 1, byteSize8, outPipe);

        // Ping-pong history buffers
        float4* tmp = d_histA;
        d_histA = d_histB;
        d_histB = tmp;

        frameIndex++;
        if (frameIndex % 30 == 0) {
            std::cerr << "\r[photism-cli] Rendered " << frameIndex << " frames ("
                      << static_cast<int>(frameIndex / fps) << "s)..." << std::flush;
        }
    }

    std::cerr << "\n[photism-cli] Finished! Total frames rendered: " << frameIndex << "\n";

    // Cleanup Pipes and CUDA Resources
    if (!pipeMode) {
        if (inPipe) _pclose(inPipe);
        if (outPipe) _pclose(outPipe);
    }

    cudaFree(d_src8);
    cudaFree(d_dst8);
    cudaFree(d_srcF);
    cudaFree(d_destF);
    cudaFree(d_histA);
    cudaFree(d_histB);
    cudaFree(d_grad);

    return 0;
}
