@ECHO OFF

PUSHD %~dp0..
CALL .\scripts\build.cmd %*
POPD