mkdir build_neon
cd build_neon
cmake -C ../CMakeCache.android.initial.cmake -DARM_TARGETS="armeabi-v7a with NEON" -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ../..

