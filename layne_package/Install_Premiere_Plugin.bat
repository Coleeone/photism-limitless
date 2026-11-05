@echo off
setlocal
echo ============================================================
echo  Installing Photism Limitless Plugin for Premiere Pro
echo ============================================================

set "DEST_DIR=C:\Program Files\Adobe\Common\Plug-ins\7.0\Mediacore"

if not exist "%DEST_DIR%" (
    echo Creating MediaCore directory: %DEST_DIR%
    mkdir "%DEST_DIR%" 2>nul
)

if not exist "%~dp0PhotismLimitless.aex" (
    echo [ERROR] PhotismLimitless.aex not found in current folder!
    pause
    exit /b 1
)

copy /Y "%~dp0PhotismLimitless.aex" "%DEST_DIR%\PhotismLimitless.aex"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to copy plugin. Please close Premiere Pro and run this script as Administrator.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] PhotismLimitless.aex installed to:
echo %DEST_DIR%\PhotismLimitless.aex
echo.

:: Clear Adobe Plugin Cache so it loads immediately on next boot
set "CACHE_DIR=%APPDATA%\Adobe\Common"
if exist "%CACHE_DIR%\PluginCache.opt" (
    del /F /Q "%CACHE_DIR%\PluginCache.opt" 2>nul
    echo Cleared plugin cache: %CACHE_DIR%\PluginCache.opt
)

echo ============================================================
echo  Done! Photism Limitless is ready in Premiere Pro Effects:
echo  Category: Video Effects / Photism / Photism Limitless
echo ============================================================
timeout /t 3 >nul
exit /b 0
