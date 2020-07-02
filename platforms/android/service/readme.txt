How to select the proper version of OpenCV Manager
--------------------------------------------------

Since version 1.7 several packages of OpenCV Manager are built. Every package is targeted for some
specific hardware platform and includes corresponding OpenCV binaries. So, in all cases OpenCV
Manager uses built-in version of OpenCV. The new package selection logic in most cases simplifies
OpenCV installation on end user devices. In most cases OpenCV Manager may be installed automatically
from Google Play.

If Google Play is not available (i.e. on emulator, developer board, etc), you can install it
manually using adb tool:

    adb install <path-to-OpenCV-sdk>/apk/OpenCV_<version>_Manager_<app_version>_<platform>.apk

Example: OpenCV_3.4.5-dev_Manager_3.45_armeabi-v7a.apk

Use the list of platforms below to determine proper OpenCV Manager package for your device:

- armeabi (ARMv5, ARMv6)
- armeabi-v7a (ARMv7-A + NEON)
- arm64-v8a
- mips
- mips64
- x86
- x86_64
