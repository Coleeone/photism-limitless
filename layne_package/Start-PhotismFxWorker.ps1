param([ValidateRange(1,480)][int]$SessionMinutes=480, [switch]$Once, [string]$Job)
$ErrorActionPreference='Stop'
$python = (Get-Command python -ErrorAction SilentlyContinue).Source; if (!$python) { $python = 'C:\Users\bmcol\AppData\Local\Programs\Python\Python313\python.exe' }
$worker=Join-Path $PSScriptRoot 'photism_fx_worker.py'
$workerArgs=@('-B',$worker,'--session-minutes',[string]$SessionMinutes)
if($Once){$workerArgs+='--once'}
if($Job){$workerArgs+=@('--job',$Job)}
& $python @workerArgs
exit $LASTEXITCODE

