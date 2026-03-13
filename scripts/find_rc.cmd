@echo off
REM ============================================================================
REM find_rc.cmd - Locate rc.exe (x64) from the Windows SDK
REM Output: SHORT path with forward slashes (no spaces)
REM ============================================================================

setlocal EnableDelayedExpansion

REM Search Windows Kits for rc.exe (latest version)
set "RC_FOUND="
for /f "delims=" %%d in ('dir /b /ad /o-n "%ProgramFiles(x86)%\Windows Kits\10\bin" 2^>nul') do (
    if exist "%ProgramFiles(x86)%\Windows Kits\10\bin\%%d\x64\rc.exe" (
        set "RC_FOUND=%ProgramFiles(x86)%\Windows Kits\10\bin\%%d\x64\rc.exe"
        goto :found
    )
)

:found
if not defined RC_FOUND exit /b 1

for %%F in ("!RC_FOUND!") do set "SHORT=%%~sfF"
echo !SHORT:\=/!
exit /b 0
