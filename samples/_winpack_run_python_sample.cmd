@ECHO OFF
SETLOCAL

SET SCRIPT_DIR=%~dp0
SET "OPENCV_SETUPVARS_SCRIPT=setup_vars_opencv4.cmd"
SET "BUILD_DIR=%SCRIPT_DIR%\..\..\build"
IF NOT EXIST "%BUILD_DIR%\%OPENCV_SETUPVARS_SCRIPT%" (
  :: Winpack DLDT
  SET "BUILD_DIR=%SCRIPT_DIR%\..\..\..\build"
)
IF NOT EXIST "%BUILD_DIR%\%OPENCV_SETUPVARS_SCRIPT%" (
  ECHO ERROR: OpenCV Winpack installation is required
  pause
  exit
)
:: normalize path
for %%i in ("%PACKAGE_BUILD_DIR%") do SET "PACKAGE_BUILD_DIR=%%~fi"

:: Detect Python binary
python -V 2>nul
IF %ERRORLEVEL% EQU 0 (
  SET PYTHON=python
  GOTO :PYTHON_FOUND
)

CALL :QUERY_PYTHON 3.8
IF %ERRORLEVEL% EQU 0 GOTO :PYTHON_FOUND
CALL :QUERY_PYTHON 3.7
IF %ERRORLEVEL% EQU 0 GOTO :PYTHON_FOUND
CALL :QUERY_PYTHON 3.6
IF %ERRORLEVEL% EQU 0 GOTO :PYTHON_FOUND
CALL :QUERY_PYTHON 3.5
IF %ERRORLEVEL% EQU 0 GOTO :PYTHON_FOUND
CALL :QUERY_PYTHON 3.4
IF %ERRORLEVEL% EQU 0 GOTO :PYTHON_FOUND
CALL :QUERY_PYTHON 2.7
IF %ERRORLEVEL% EQU 0 GOTO :PYTHON_FOUND
GOTO :PYTHON_NOT_FOUND

:QUERY_PYTHON
SETLOCAL
SET PY_VERSION=%1
SET PYTHON_DIR=
CALL :regquery "HKCU\SOFTWARE\Python\PythonCore\%PY_VERSION%\InstallPath" PYTHON_DIR
IF EXIST "%PYTHON_DIR%\python.exe" (
  SET "PYTHON=%PYTHON_DIR%\python.exe"
  GOTO :QUERY_PYTHON_FOUND
)
CALL :regquery "HKLM\SOFTWARE\Python\PythonCore\%PY_VERSION%\InstallPath" PYTHON_DIR
IF EXIST "%PYTHON_DIR%\python.exe" (
  SET "PYTHON=%PYTHON_DIR%\python.exe"
  GOTO :QUERY_PYTHON_FOUND
)

::echo Python %PY_VERSION% is not detected
ENDLOCAL
EXIT /B 1

:QUERY_PYTHON_FOUND
ECHO Found Python %PY_VERSION% from Windows Registry: %PYTHON%
ENDLOCAL & SET PYTHON=%PYTHON%
EXIT /B 0

IF exist C:\Python27-x64\python.exe (
  SET PYTHON=C:\Python27-x64\python.exe
  GOTO :PYTHON_FOUND
)
IF exist C:\Python27\python.exe (
  SET PYTHON=C:\Python27\python.exe
  GOTO :PYTHON_FOUND
)

:PYTHON_NOT_FOUND
ECHO ERROR: Python not found
IF NOT DEFINED OPENCV_BATCH_MODE ( pause )
EXIT /B

:PYTHON_FOUND
ECHO Using Python: %PYTHON%

:: Don't generate unnecessary .pyc cache files
SET PYTHONDONTWRITEBYTECODE=1

IF [%1]==[] goto rundemo

set SRC_FILENAME=%~dpnx1
echo SRC_FILENAME=%SRC_FILENAME%
call :dirname "%SRC_FILENAME%" SRC_DIR
call :dirname "%PYTHON%" PYTHON_DIR
PUSHD %SRC_DIR%

CALL "%BUILD_DIR%\%OPENCV_SETUPVARS_SCRIPT%"
:: repair SCRIPT_DIR
SET "SCRIPT_DIR=%~dp0"

ECHO Run: %*
%PYTHON% %*
SET result=%errorlevel%
IF %result% NEQ 0 (
  IF NOT DEFINED OPENCV_BATCH_MODE (
    SET "PATH=%PYTHON_DIR%;%PATH%"
    echo ================================================================================
    echo **  Type 'python sample_name.py' to run sample
    echo **  Type 'exit' to exit from interactive shell and open the build directory
    echo ================================================================================
    cmd /k echo Current directory: %CD%
  )
)

POPD
EXIT /B %result%

:rundemo
PUSHD "%SCRIPT_DIR%\python"

CALL "%BUILD_DIR%\%OPENCV_SETUPVARS_SCRIPT%"
:: repair SCRIPT_DIR
SET "SCRIPT_DIR=%~dp0"

%PYTHON% demo.py
SET result=%errorlevel%
IF %result% NEQ 0 (
  IF NOT DEFINED OPENCV_BATCH_MODE ( pause )
)

POPD
EXIT /B %result%


:dirname file resultVar
  setlocal
  set _dir=%~dp1
  set _dir=%_dir:~0,-1%
  endlocal & set %2=%_dir%
  EXIT /B 0

:regquery name resultVar
  SETLOCAL
  FOR /F "tokens=*" %%A IN ('REG QUERY "%1" /reg:64 /ve 2^>NUL ^| FIND "REG_SZ"') DO SET _val=%%A
  IF "x%_val%x"=="xx" EXIT /B 1
  SET _val=%_val:*REG_SZ=%
  FOR /F "tokens=*" %%A IN ("%_val%") DO SET _val=%%A
  ENDLOCAL & SET %2=%_val%
  EXIT /B 0
