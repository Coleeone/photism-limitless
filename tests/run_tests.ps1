$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\.."
$BuildDir = Join-Path $ProjectRoot "build"
$SrcDir = Join-Path $ProjectRoot "src"
$TestsDir = Join-Path $ProjectRoot "tests"

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$tempBat = Join-Path $BuildDir "get_env.bat"
"@echo off
call `"$vcvars`" >nul 2>&1
set
" | Out-File -FilePath $tempBat -Encoding ascii

$envLines = cmd.exe /c $tempBat
Remove-Item $tempBat -Force -ErrorAction SilentlyContinue

foreach ($line in $envLines) {
    if ($line -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
$testHarnessCu = Join-Path $TestsDir "test_harness.cu"
$limitlessCu = Join-Path $SrcDir "PhotismLimitless.cu"
$testExe = Join-Path $BuildDir "test_harness.exe"

Write-Host "Compiling test_harness.exe..."
& $nvcc -gencode arch=compute_120,code=sm_120 -use_fast_math -m64 -O2 -I$SrcDir $testHarnessCu $limitlessCu -o $testExe
if ($LASTEXITCODE -ne 0) { throw "Compilation of test_harness failed with code $LASTEXITCODE" }

Write-Host "Running test_harness.exe..."
& $testExe
