@ECHO OFF
PUSHD %~dp0
SET PROJECT_NAME=android-opencv
CALL ..\scripts\cmake_android.cmd
POPD