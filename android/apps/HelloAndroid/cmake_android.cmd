@ECHO OFF
SETLOCAL
PUSHD %~dp0
SET PROJECT_NAME=HelloAndroid
SET BUILD_DIR=build_armeabi
SET ARM_TARGET=armeabi
CALL ..\..\scripts\build.cmd %*
POPD
ENDLOCAL