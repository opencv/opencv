opencv_dir=`pwd`/../build_neon
mkdir build_neon
cd build_neon
cmake -DOpenCV_DIR=$opencv_dir -DARM_TARGETS="armeabi-v7a with NEON" -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ..

