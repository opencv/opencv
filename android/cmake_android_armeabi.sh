mkdir build_armeabi
cd build_armeabi
cmake -C ../CMakeCache.android.initial.cmake -DARM_TARGETS=armeabi -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ../..

