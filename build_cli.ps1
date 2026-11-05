param (
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$workspaceRoot = $PSScriptRoot
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (!(Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars"
}

# Export environment variables from vcvars64
$envLines = cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
foreach ($line in $envLines) {
    if ($line -match "^(.*?)=(.*)$") {
        Set-Item -Path "env:\$($matches[1])" -Value $matches[2]
    }
}

$nvcc = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
if (!(Test-Path $nvcc)) {
    throw "nvcc.exe not found at $nvcc"
}

$outDir = "$workspaceRoot\build\dist"
if (!(Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
}

$cliExe = "$outDir\photism-cli.exe"

Write-Host "============================================================"
Write-Host " Building photism-cli for FilmCraft & Baxter"
Write-Host "============================================================"

$nvccArgs = @(
    "-O3",
    "-std=c++17",
    "-arch=sm_120",
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_120,code=sm_120",
    "-I`"$workspaceRoot`"",
    "-I`"$workspaceRoot\src`"",
    "`"$workspaceRoot\src\photism_cli.cu`"",
    "`"$workspaceRoot\src\PhotismLimitless.cu`"",
    "-o", "`"$cliExe`""
)

& $nvcc @nvccArgs

if ($LASTEXITCODE -ne 0) {
    throw "Failed to compile photism-cli.exe"
}

Write-Host "[SUCCESS] Built photism-cli.exe!"
Write-Host "Artifact: $cliExe"
Get-Item $cliExe | Select-Object Name, Length, LastWriteTime
