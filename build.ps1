param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"
$DistDir = Join-Path $BuildDir "dist"
$SrcDir = Join-Path $ProjectRoot "src"

$PremSdk = "E:\SDK\PremiereSDK\Premiere Pro 26.0 C++ SDK"
$AeSdk   = "E:\SDK\AfterEffectsSDK\AfterEffectsSDK_25.6_61_win\ae25.6_61.64bit.AfterEffectsSDK"
$CudaSdk = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"

Write-Host "============================================================"
Write-Host " Building Photism Limitless Effect Plugin"
Write-Host "============================================================"

if ($Clean -and (Test-Path $BuildDir)) {
    Remove-Item -Path $BuildDir -Recurse -Force
}

New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

# Setup Visual Studio x64 environment
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (!(Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

# Export environment variables from vcvars64
$envLines = cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
foreach ($line in $envLines) {
    if ($line -match "^([^=]+)=(.*)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

Write-Host "[1/5] Compiling PiPL Resource..."
$rFile   = Join-Path $SrcDir "PhotismLimitless.r"
$rrFile  = Join-Path $BuildDir "PhotismLimitless.rr"
$rcpFile = Join-Path $SrcDir "PhotismLimitless.rcp"
$rcFile  = Join-Path $SrcDir "PhotismLimitless.rc"
$resFile = Join-Path $BuildDir "PhotismLimitless.res"
$piplTool = Join-Path $AeSdk "Examples\Resources\PiPLtool.exe"

# Preprocess .r to .rr
& cl.exe /D "PRWIN_ENV" /I "$AeSdk\Examples\Headers" /I "$AeSdk\Examples\Resources" /EP $rFile | Out-File -FilePath $rrFile -Encoding ascii
if ($LASTEXITCODE -ne 0) { throw "PiPL preprocessor failed with code $LASTEXITCODE" }

# Run PiPLtool
& $piplTool $rrFile $rcpFile
if ($LASTEXITCODE -ne 0) { throw "PiPLtool failed with code $LASTEXITCODE" }

# Compile .rc to .res
& rc.exe /i "$AeSdk\Examples\Headers" /i "$AeSdk\Examples\Resources" /i $SrcDir /fo $resFile $rcFile
if ($LASTEXITCODE -ne 0) { throw "RC failed with code $LASTEXITCODE" }

Write-Host "[2/5] Compiling CUDA Kernels (CUDA 13.3 sm_75..sm_120)..."
$cuFile = Join-Path $SrcDir "PhotismLimitless.cu"
$cuObj  = Join-Path $BuildDir "PhotismLimitless_cu.obj"
$nvcc = Join-Path $CudaSdk "bin\nvcc.exe"

$nvccArgs = @(
    "-gencode", "arch=compute_75,code=sm_75",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_120,code=sm_120",
    "-use_fast_math",
    "-m64",
    "-c",
    "-Xcompiler", "/MD,/EHsc,/W3,/nologo,/O2",
    "-I$CudaSdk\include",
    "-I$SrcDir",
    "-I$PremSdk\Examples\Headers",
    "-I$PremSdk\Examples\Projects\GPUVideoFilter\Utils",
    "-I$AeSdk\Examples\Headers",
    "-I$AeSdk\Examples\Headers\SP",
    "-I$AeSdk\Examples\Util",
    $cuFile,
    "-o", $cuObj
)
& $nvcc $nvccArgs
if ($LASTEXITCODE -ne 0) { throw "NVCC compilation failed with code $LASTEXITCODE" }

Write-Host "[3/5] Compiling C++ Sources..."
$cxxFiles = @(
    (Join-Path $SrcDir "DeterministicState.cpp"),
    (Join-Path $SrcDir "PhotismLimitless_CPU.cpp"),
    (Join-Path $SrcDir "PhotismLimitless_GPU.cpp"),
    (Join-Path $AeSdk "Examples\Util\AEFX_SuiteHelper.c"),
    (Join-Path $AeSdk "Examples\Util\AEGP_SuiteHandler.cpp"),
    (Join-Path $AeSdk "Examples\Util\MissingSuiteError.cpp"),
    (Join-Path $AeSdk "Examples\Util\Smart_Utils.cpp")
)

$cxxIncludes = @(
    "/I$SrcDir",
    "/I$PremSdk\Examples\Headers",
    "/I$PremSdk\Examples\Headers\SP",
    "/I$PremSdk\Examples\Projects\GPUVideoFilter\Utils",
    "/I$AeSdk\Examples\Headers",
    "/I$AeSdk\Examples\Headers\SP",
    "/I$AeSdk\Examples\Resources",
    "/I$AeSdk\Examples\Util",
    "/I$CudaSdk\include"
)

$cxxFlags = @(
    "/c",
    "/MD",
    "/O2",
    "/EHsc",
    "/W3",
    "/nologo",
    "/DMSWindows",
    "/D_WINDOWS",
    "/D_USE_MATH_DEFINES",
    "/DNOMINMAX",
    "/DWIN32_LEAN_AND_MEAN"
)

$cxxObjs = @()
foreach ($file in $cxxFiles) {
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $objPath = Join-Path $BuildDir "$baseName.obj"
    $cxxObjs += $objPath
    Write-Host "  Compiling $([System.IO.Path]::GetFileName($file))..."
    & cl.exe $cxxFlags $cxxIncludes $file "/Fo$objPath"
    if ($LASTEXITCODE -ne 0) { throw "Compilation of $file failed with code $LASTEXITCODE" }
}

Write-Host "[4/5] Linking Plugin Binary (.aex)..."
$pluginArtifact = Join-Path $DistDir "PhotismLimitless.aex"
$linkLibs = @(
    "cudart_static.lib",
    "OpenCL.lib",
    "user32.lib",
    "gdi32.lib",
    "advapi32.lib"
)

$linkFlags = @(
    "/DLL",
    "/NOLOGO",
    "/MACHINE:X64",
    "/OPT:REF",
    "/OPT:ICF",
    "/LIBPATH:$CudaSdk\lib\x64",
    "/OUT:$pluginArtifact"
)

$linkInputs = $cxxObjs + @($cuObj, $resFile) + $linkLibs
& link.exe $linkFlags $linkInputs
if ($LASTEXITCODE -ne 0) { throw "Linking failed with code $LASTEXITCODE" }

Write-Host "[5/5] Plugin Built Successfully!"
Write-Host "Artifact: $pluginArtifact"

$fileSize = (Get-Item $pluginArtifact).Length
Write-Host "Size: $fileSize bytes"
