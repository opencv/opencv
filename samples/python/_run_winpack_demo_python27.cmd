@echo off
if NOT exist %CD%\..\..\..\build (
  echo ERROR: OpenCV Winpack installation is required
  pause
  exit
)

:: Path to FFMPEG binary files
set PATH=%PATH%;%CD%\..\..\..\build\bin\

:: Detect Python binary
python -V
if %ERRORLEVEL% EQU 0 (
  set PYTHON=python
) else (
  if exist C:\Python27-x64\python.exe (
    set PYTHON=C:\Python27-x64\python.exe
  ) else (
    if exist C:\Python27\python.exe (
      set PYTHON=C:\Python27\python.exe
    ) else (
      echo ERROR: Python not found
      pause
      exit
    )
  )
)
echo Using python: %PYTHON%

:: Detect python architecture
%PYTHON% -c "import platform; exit(64 if platform.architecture()[0] == '64bit' else 32)"
if %ERRORLEVEL% EQU 32 (
  echo Detected: Python 32-bit
  set PYTHONPATH=%CD%\..\..\..\build\python\2.7\x86
) else (
  if %ERRORLEVEL% EQU 64 (
    echo Detected: Python 64-bit
    set PYTHONPATH=%CD%\..\..\..\build\python\2.7\x64
  ) else (
    echo ERROR: Unknown python arch
    pause
    exit
  )
)

:: Launch demo
%PYTHON% demo.py
