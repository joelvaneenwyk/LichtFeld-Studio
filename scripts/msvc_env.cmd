@echo off
REM ============================================================================
REM msvc_env.cmd - Minimal MSVC environment setup
REM
REM Sets up the bare minimum for MSVC builds WITHOUT calling vcvarsall.bat
REM (which bloats PATH past the 8191-char batch limit on many dev machines).
REM
REM What it does:
REM   1. Adds cl.exe's directory + Windows SDK bin dir to PATH
REM   2. Sets LIB to MSVC + Windows SDK library directories
REM   3. Sets INCLUDE to MSVC + Windows SDK header directories
REM
REM Usage: call scripts\msvc_env.cmd && cmake ...
REM ============================================================================

REM If cl.exe is already on PATH, nothing to do
where cl.exe >nul 2>&1 && exit /b 0

setlocal EnableDelayedExpansion

REM --- Find cl.exe via vswhere ---
for %%V in ("%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe") do set "VSWHERE=%%~sfV"
if not exist "!VSWHERE!" (
    echo [msvc_env] ERROR: vswhere.exe not found. 1>&2
    exit /b 1
)

set "CL_FOUND="
for /f "usebackq delims=" %%i in (`!VSWHERE! -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"`) do set "CL_FOUND=%%i"
if not defined CL_FOUND (
    for /f "usebackq delims=" %%i in (`!VSWHERE! -latest -products Microsoft.VisualStudio.Product.BuildTools -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"`) do set "CL_FOUND=%%i"
)
if not defined CL_FOUND (
    echo [msvc_env] ERROR: cl.exe not found. 1>&2
    exit /b 1
)

REM --- Derive paths relative to cl.exe ---
REM cl.exe is at: <MSVC_ROOT>\bin\Hostx64\x64\cl.exe
REM Use ".." to navigate: x64 -> Hostx64 -> bin -> <MSVC_ROOT>
for %%F in ("!CL_FOUND!") do set "CL_DIR=%%~dpF"
set "MSVC_LIB=!CL_DIR!..\..\..\lib\x64"
set "MSVC_INCLUDE=!CL_DIR!..\..\..\include"

REM --- Find Windows SDK ---
set "SDK_BASE=%ProgramFiles(x86)%\Windows Kits\10"
set "SDK_VER="
if exist "!SDK_BASE!\Lib" (
    for /f "delims=" %%d in ('dir /b /ad /o-n "!SDK_BASE!\Lib" 2^>nul') do (
        if exist "!SDK_BASE!\Lib\%%d\um\x64\kernel32.lib" (
            set "SDK_VER=%%d"
            goto :sdk_found
        )
    )
)
:sdk_found

REM --- Build LIB and INCLUDE ---
set "NEW_LIB=!MSVC_LIB!"
set "NEW_INCLUDE=!MSVC_INCLUDE!"

if defined SDK_VER (
    set "NEW_LIB=!NEW_LIB!;!SDK_BASE!\Lib\!SDK_VER!\um\x64;!SDK_BASE!\Lib\!SDK_VER!\ucrt\x64"
    set "NEW_INCLUDE=!NEW_INCLUDE!;!SDK_BASE!\Include\!SDK_VER!\ucrt;!SDK_BASE!\Include\!SDK_VER!\um;!SDK_BASE!\Include\!SDK_VER!\shared"
)

REM --- Find Windows SDK bin directory (for mt.exe, rc.exe, etc.) ---
set "SDK_BIN="
if defined SDK_VER (
    set "SDK_BIN=!SDK_BASE!\bin\!SDK_VER!\x64"
)

REM --- Export to parent environment ---
endlocal & (
    set "PATH=%CL_DIR%;%SDK_BIN%;%PATH%"
    set "LIB=%NEW_LIB%"
    set "INCLUDE=%NEW_INCLUDE%"
)

echo [msvc_env] MSVC tools added to PATH 1>&2
exit /b 0
