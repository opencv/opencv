=========================================
CMake Build
=========================================
#path to the android build of opencv
opencv_dir=`pwd`/../build
mkdir build
cd build
cmake -DOpenCV_DIR=$opencv_dir -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ..

=========================================
Android Build
=========================================
sh project_create.sh
ant compile

