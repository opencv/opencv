How to select the proper version of OpenCV Manager
--------------------------------------------------

Since version 1.7 several packages of OpenCV Manager are built. Every package is targeted for some
specific hardware platform and includes corresponding OpenCV binaries. So, in most cases OpenCV
Manager uses built-in version of OpenCV. Separate package with OpenCV binaries is currently used in
a single rare case, when an ARMv7-A processor without NEON support is detected. In this case an
additional binary package is used. The new package selection logic in most cases simplifies OpenCV
installation on end user devices. In most cases OpenCV Manager may be installed automatically from
Google Play.

If Google Play is not available (i.e. on emulator, developer board, etc), you can install it
manually using adb tool:

.. code-block:: sh

    adb install OpenCV-2.4.7.1-android-sdk/apk/OpenCV_2.4.7.1_Manager_2.15_<platform>.apk

Use the table below to determine proper OpenCV Manager package for your device:

+------------------------------+--------------+------------------------------------------------------+
| Hardware Platform            | Android ver. | Package name                                         |
+==============================+==============+======================================================+
| armeabi-v7a (ARMv7-A + NEON) |    >= 2.3    | OpenCV_2.4.7.1_Manager_2.15_armv7a-neon.apk          |
+------------------------------+--------------+------------------------------------------------------+
| armeabi-v7a (ARMv7-A + NEON) |     = 2.2    | OpenCV_2.4.7.1_Manager_2.15_armv7a-neon-android8.apk |
+------------------------------+--------------+------------------------------------------------------+
| armeabi (ARMv5, ARMv6)       |    >= 2.3    | OpenCV_2.4.7.1_Manager_2.15_armeabi.apk              |
+------------------------------+--------------+------------------------------------------------------+
| Intel x86                    |    >= 2.3    | OpenCV_2.4.7.1_Manager_2.15_x86.apk                  |
+------------------------------+--------------+------------------------------------------------------+
| MIPS                         |    >= 2.3    | OpenCV_2.4.7.1_Manager_2.15_mips.apk                 |
+------------------------------+--------------+------------------------------------------------------+
