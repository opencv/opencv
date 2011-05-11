:: this batch file copies compiled executable to the device,
:: runs it and gets resulting image back to the host
::
:: Here is sample output of successful run:
::
:: 204 KB/s (2887388 bytes in 13.790s)
:: Hello Android!
:: 304 KB/s (8723 bytes in 0.028s)

@ECHO OFF

:: enable command extensions
VERIFY BADVALUE 2>NUL
SETLOCAL ENABLEEXTENSIONS || (ECHO Unable to enable command extensions. & EXIT \B)

PUSHD %~dp0
SET PROJECT_NAME=HelloAndroid

:: try to load config file
SET CFG_PATH=..\..\scripts\wincfg.cmd
IF EXIST %CFG_PATH% CALL %CFG_PATH%

:: check if sdk path defined
IF NOT DEFINED ANDROID_SDK (ECHO. & ECHO You should set an environment variable ANDROID_SDK to the full path to your copy of Android SDK & GOTO end)
(PUSHD "%ANDROID_SDK%" 2>NUL && POPD) || (ECHO. & ECHO Directory "%ANDROID_SDK%" specified by ANDROID_SDK variable does not exist & GOTO end)
SET adb=%ANDROID_SDK%\platform-tools\adb.exe

:: copy file to device (usually takes 10 seconds or more)
%adb% push .\bin\armeabi\%PROJECT_NAME% /data/bin/sample/%PROJECT_NAME% || GOTO end

:: set execute permission
%adb% shell chmod 777 /data/bin/sample/%PROJECT_NAME% || GOTO end

:: execute our application
%adb% shell /data/bin/sample/%PROJECT_NAME% || GOTO end

:: get image result from device
%adb% pull /mnt/sdcard/HelloAndroid.png || GOTO end

GOTO end

:: cleanup (comment out GOTO above to enable cleanup)
%adb% shell rm /data/bin/sample/%PROJECT_NAME%
%adb% shell rm /mnt/sdcard/HelloAndroid.png

:end
POPD
ENDLOCAL