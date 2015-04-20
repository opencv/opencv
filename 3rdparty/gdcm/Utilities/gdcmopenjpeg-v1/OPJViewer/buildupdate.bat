::== buildupdate.bat
@echo off
setLocal EnableDelayedExpansion

for /f "tokens=2,* delims=^(^) " %%a in ('find /v "" ^< .\source\build.h') do (
rem echo %%a
set /A M = %%a + 1
echo Build %%a done^!
echo wxT^("!M!"^) > buildtemp283746825t347
)

if exist buildtemp283746825t347 move /Y buildtemp283746825t347 .\source\build.h
if exist buildtemp283746825t347 del /F /Q buildtemp283746825t347

::==