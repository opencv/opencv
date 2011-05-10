@ECHO OFF
SETLOCAL
PUSHD %~dp0
SET PROJECT_NAME=android-opencv
CALL ..\scripts\build.cmd %*
POPD
ENDLOCAL