mkdir build_neon
cd build_neon
cmake -DOpenCV_DIR=../../../build_neon -DAndroidOpenCV_DIR=../../../android-opencv/build_neon -DARM_TARGETS="armeabi-v7a with NEON" -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ..

