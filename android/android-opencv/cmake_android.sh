opencv_dir=`pwd`/../build
mkdir build
cd build
cmake -DOpenCVDIR=$opencv_dir -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ..

