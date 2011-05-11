@ECHO OFF
SETLOCAL
PUSHD %~dp0
SET PROJECT_NAME=HelloAndroid
CALL ..\..\scripts\build.cmd %*
POPD
ENDLOCAL