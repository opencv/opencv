#!/bin/sh
cd `dirname $0`
PROJECT_NAME=HelloAndroid
OUT_DIR=armeabi

# copy file to device (usually takes 10 seconds or more)
adb push ./bin/$OUT_DIR/$PROJECT_NAME /data/bin/sample/$PROJECT_NAME || return

# set execute permission
adb shell chmod 777 /data/bin/sample/$PROJECT_NAME || return

# execute our application
adb shell /data/bin/sample/$PROJECT_NAME || return

# get image result from device
adb pull /mnt/sdcard/HelloAndroid.png || return