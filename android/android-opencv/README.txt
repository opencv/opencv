=========================================
CMake Build
=========================================
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDTOOLCHAIN ..

=========================================
Android Build
=========================================
sh project_create.sh
ant compile
ant install
