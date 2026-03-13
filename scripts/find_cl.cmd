@echo off
REM ============================================================================
REM find_cl.cmd - Locate cl.exe (x64) via vswhere and output its SHORT path
REM              (8.3 format, forward slashes, no spaces - safe everywhere).
REM
REM Usage:  cmd /c scripts\find_cl.cmd
REM Output: C:/PROGRA~1/MIB055~1/.../cl.exe   (no spaces)
REM ============================================================================

setlocal EnableDelayedExpansion

REM Use short path for vswhere to avoid quoting issues with parentheses
for %%V in ("%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe") do set "VSWHERE=%%~sfV"
if not exist "!VSWHERE!" exit /b 1

REM Try full VS installations - loop keeps last match (= latest MSVC version)
set "CL_FOUND="
for /f "usebackq delims=" %%i in (`!VSWHERE! -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"`) do set "CL_FOUND=%%i"

REM Fall back to Build Tools SKU
if not defined CL_FOUND (
    for /f "usebackq delims=" %%i in (`!VSWHERE! -latest -products Microsoft.VisualStudio.Product.BuildTools -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"`) do set "CL_FOUND=%%i"
)

if not defined CL_FOUND exit /b 1

REM Output SHORT path with forward slashes (no spaces, safe for any shell)
for %%F in ("!CL_FOUND!") do set "SHORT=%%~sfF"
echo !SHORT:\=/!
exit /b 0
