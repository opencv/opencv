OpenCV Manager selection
========================

Since version 1.7 several packages of OpenCV Manager is built. Every package includes OpenCV library
for package target platform. The internal library is used for most cases, except the rare one, when
arm-v7a without NEON instruction set processor is detected. In this case additional binary package
for arm-v7a is used. The new package selection logic in most cases simplifies OpenCV installation
on end user devices. In most cases OpenCV Manager may be installed automatically from Google Play.
For such case, when Google Play is not available, i.e. emulator, developer board, etc, you can
install it manually using adb tool:

    adb install OpenCV-2.4.3-android-sdk/apk/OpenCV_2.4.3.2_Manager_2.4_<platform_name>.apk

Use table to determine right OpenCV Manager package:

+----------------------------+-----------------+-----------------------------------------------------+
| Hardware Platform          | Android version | Package name                                        |
+============================+=================+=====================================================+
| Intel x86                  | >= 2.3          | OpenCV_2.4.3.2_Manager_2.4_x86.apk                  |
+----------------------------+-----------------+-----------------------------------------------------+
| MIPS                       | >= 2.3          | OpenCV_2.4.3.2_Manager_2.4_mips.apk                 |
+----------------------------+-----------------+-----------------------------------------------------+
| armeabi (arm-v5, arm-v6)   | >= 2.3          | OpenCV_2.4.3.2_Manager_2.4_armeabi.apk              |
+----------------------------+-----------------+-----------------------------------------------------+
| armeabi-v7a (arm-v7a-NEON) | >= 2.3          | OpenCV_2.4.3.2_Manager_2.4_armv7a-neon.apk          |
+----------------------------+-----------------+-----------------------------------------------------+
| armeabi-v7a (arm-v7a-NEON) | 2.2             | OpenCV_2.4.3.2_Manager_2.4_armv7a-neon-android8.apk |
+----------------------------+-----------------+-----------------------------------------------------+
