Most of the functionality is covered there:
http://code.opencv.org/projects/opencv/wiki/HowToUsePerfTests

===
Creating of usable tests backup with run.py
===

Direct copy would not work as expected because CMakeCache.txt contains link to the old binaries path and run.py script uses this path.

Script run.py with --move_tests option may be used to create runnable standalone backup of the tests.

Usage examples:

cd opecv/build
run.py --move_tests "new tests location" "current tests binaries location"

This copies tests binaries and CMakeCache.txt (run.py make use of this file) and also updates CMakeCache.txt with the new binaries path

NOTE: currently this functionality is not supported for OSX.