How to select the proper version of OpenCV Manager
--------------------------------------------------

Since version 1.7 several packages of OpenCV Manager are built. Every package is targeted for some
specific hardware platform and includes corresponding OpenCV binaries. So, in all cases OpenCV
Manager uses built-in version of OpenCV. The new package selection logic in most cases simplifies
OpenCV installation on end user devices. In most cases OpenCV Manager may be installed automatically
from Google Play.

If Google Play is not available (i.e. on emulator, developer board, etc), you can install it
manually using adb tool:

    adb install <path-to-OpenCV-sdk>/apk/OpenCV_3.0.0_Manager_3.00_<platform>.apk

Use the list below to determine proper OpenCV Manager package for your device:

- OpenCV_3.0.0-dev_Manager_3.00_armeabi.apk - armeabi (ARMv5, ARMv6)
- OpenCV_3.0.0-dev_Manager_3.00_armeabi-v7a.apk - armeabi-v7a (ARMv7-A + NEON)
- OpenCV_3.0.0-dev_Manager_3.00_arm64-v8a.apk - arm64-v8a (ARM64-v8a)
- OpenCV_3.0.0-dev_Manager_3.00_mips.apk - mips (MIPS)
- OpenCV_3.0.0-dev_Manager_3.00_mips64.apk - mips64 (MIPS64)
- OpenCV_3.0.0-dev_Manager_3.00_x86.apk - x86
- OpenCV_3.0.0-dev_Manager_3.00_x86_64.apk - x86_64
