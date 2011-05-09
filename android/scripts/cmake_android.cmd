@ECHO OFF

:: don't modify the caller's environment
SETLOCAL

:: enable command extensions
VERIFY BADVALUE 2>NUL
SETLOCAL ENABLEEXTENSIONS
IF ERRORLEVEL 1 ECHO Unable to enable command extensions.

:: load configuration
PUSHD %~dp0
IF NOT EXIST .\wincfg.cmd GOTO nocfg
CALL .\wincfg.cmd
SET OPENCV_BUILD_DIR="%cd%"\..\%BUILD_DIR%
POPD

:: path to project root
SET SOURCE_DIR="%cd%"

:: create build dir
::rmdir /S /Q %BUILD_DIR% &:: uncomment this line to rebuild instead of build
MKDIR %BUILD_DIR% 2>NUL
PUSHD %BUILD_DIR%

:: run cmake
ECHO.
ECHO Runnning cmake...
ECHO ARM_TARGET=%ARM_TARGET%
ECHO.
IF NOT EXIST %SOURCE_DIR%\CMakeCache.android.initial.cmake GOTO other-cmake
:opencv-cmake
	%CMAKE_EXE% -G"MinGW Makefiles" -DARM_TARGET="%ARM_TARGET%" -C %SOURCE_DIR%\CMakeCache.android.initial.cmake -DCMAKE_TOOLCHAIN_FILE=%SOURCE_DIR%\android.toolchain.cmake -DCMAKE_MAKE_PROGRAM="%MAKE_EXE%" %SOURCE_DIR%\..
	IF ERRORLEVEL 1 GOTO cmakefails
	GOTO cmakefin
:other-cmake
	%CMAKE_EXE% -G"MinGW Makefiles" -DARM_TARGET="%ARM_TARGET%" -DOpenCV_DIR=%OPENCV_BUILD_DIR%  -DCMAKE_PROGRAM_PATH=%SWIG_DIR% -DCMAKE_TOOLCHAIN_FILE=%OPENCV_BUILD_DIR%\..\android.toolchain.cmake -DCMAKE_MAKE_PROGRAM="%MAKE_EXE%" %SOURCE_DIR%
	IF ERRORLEVEL 1 GOTO cmakefails
	GOTO cmakefin
:cmakefin

:: run make
ECHO.
ECHO Building native libs...
%MAKE_EXE% -j %NUMBER_OF_PROCESSORS% &:: VERBOSE=1
IF ERRORLEVEL 1 GOTO makefail

IF NOT EXIST ..\jni GOTO fin

:: configure java part
POPD
PUSHD .
ECHO.
ECHO Updating Android project...
CALL %ANDROID_SDK%\tools\android update project --name %PROJECT_NAME% --path .
IF ERRORLEVEL 1 GOTO androidfail

:: compile java part
ECHO.
ECHO Compiling Android project...
CALL %ANT_DIR%\bin\ant compile
IF ERRORLEVEL 1 GOTO antfail

GOTO fin

:nocfg
ECHO.
ECHO Could not find wincfg.cmd file.
ECHO.
ECHO   You should create opencv\android\scripts\wincfg.cmd
ECHO   from template opencv\android\scripts\wincfg.cmd.tmpl
GOTO fin

:antfail
ECHO.
ECHO failed to compile android project
GOTO fin

:androidfail
ECHO.
ECHO failed to update android project
GOTO fin

:makefail
ECHO.
ECHO make failed
GOTO fin

:cmakefail
ECHO. 
ECHO cmake failed
GOTO fin
  
:fin
POPD
ENDLOCAL