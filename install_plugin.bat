@echo off
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrative privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~fn0\"' -Verb RunAs"
    exit /b
)

echo Copying PhotismLimitless.aex to Premiere Pro MediaCore directory...
copy /y "%~dp0dist\PhotismLimitless.aex" "C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\PhotismLimitless.aex"

if %errorLevel% equ 0 (
    echo.
    echo ============================================================
    echo  [SUCCESS] Photism Limitless installed into Premiere Pro!
    echo ============================================================
    echo Location: C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\PhotismLimitless.aex
    echo You can now launch Premiere Pro 2026.
) else (
    echo.
    echo [ERROR] Failed to copy the plugin file.
)

pause
