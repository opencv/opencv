#!/bin/sh
cd `dirname $0`/..

ANDROID_DIR=`pwd`

rm -rf package
mkdir -p package
cd package

PRG_DIR=`pwd`
mkdir opencv


# neon-enabled build
cd $PRG_DIR
mkdir build-neon
cd build-neon

cmake -C "$ANDROID_DIR/CMakeCache.android.initial.cmake" -DARM_TARGET="armeabi-v7a with NEON" -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_ANDROID_EXAMPLES=OFF -DCMAKE_TOOLCHAIN_FILE="$ANDROID_DIR/android.toolchain.cmake" -DCMAKE_INSTALL_PREFIX="$PRG_DIR/opencv" "$ANDROID_DIR/.."  || exit 1
make -j8 install/strip || exit 1

cd "$PRG_DIR/opencv"
rm -rf doc include src .classpath .project AndroidManifest.xml default.properties share/OpenCV/haarcascades share/OpenCV/lbpcascades share/OpenCV/*.cmake share/OpenCV/OpenCV.mk
mv libs/armeabi-v7a libs/armeabi-v7a-neon
mv share/OpenCV/3rdparty/libs/armeabi-v7a share/OpenCV/3rdparty/libs/armeabi-v7a-neon


# armeabi-v7a build
cd "$PRG_DIR"
mkdir build
cd build

cmake -C "$ANDROID_DIR/CMakeCache.android.initial.cmake" -DARM_TARGET="armeabi-v7a" -DBUILD_DOCS=OFF -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF -DBUILD_ANDROID_EXAMPLES=ON -DCMAKE_TOOLCHAIN_FILE="$ANDROID_DIR/android.toolchain.cmake" -DCMAKE_INSTALL_PREFIX="$PRG_DIR/opencv" "$ANDROID_DIR/.."  || exit 1
make -j8 install/strip || exit 1

cd "$PRG_DIR/opencv"
rm -rf doc include src .classpath .project AndroidManifest.xml default.properties share/OpenCV/haarcascades share/OpenCV/lbpcascades share/OpenCV/*.cmake share/OpenCV/OpenCV.mk


# armeabi build
cd "$PRG_DIR/build"
rm -rf CMakeCache.txt

cmake -C "$ANDROID_DIR/CMakeCache.android.initial.cmake" -DARM_TARGET="armeabi" -DBUILD_DOCS=ON -DBUILD_TESTS=ON -DBUILD_EXAMPLES=OFF -DBUILD_ANDROID_EXAMPLES=ON -DINSTALL_ANDROID_EXAMPLES=ON -DCMAKE_TOOLCHAIN_FILE="$ANDROID_DIR/android.toolchain.cmake" -DCMAKE_INSTALL_PREFIX="$PRG_DIR/opencv" "$ANDROID_DIR/.."  || exit 1
make -j8 install/strip docs || exit 1

find doc -name "*.pdf" -exec cp {} $PRG_DIR/opencv/doc \;

cd $PRG_DIR
rm -rf opencv/doc/CMakeLists.txt
cp "$ANDROID_DIR/README.android" opencv/
cp "$ANDROID_DIR/../README" opencv/

CV_VERSION=`grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+" opencv/share/OpenCV/OpenCVConfig-version.cmake`
mv opencv opencv$CV_VERSION


#samples
cp -r "$ANDROID_DIR/../samples/android" "$PRG_DIR/samples"
cd "$PRG_DIR/samples"

#enable for loops over items with spaces in their name
IFS="
"
for dir in `ls -1 | grep -v hello-android`
do
  if [ -f "$dir/default.properties" ]
  then
    HAS_REFERENCE=`cat "$dir/default.properties" | grep -c android.library.reference.1`
    if [ $HAS_REFERENCE = 1 ]
    then
      echo -n > "$dir/default.properties"
      android update project --name "$dir" --target "android-8" --library "../../opencv$CV_VERSION" --path "$dir"
      echo 'android update project --name "$dir" --target "android-8" --library "../opencv$CV_VERSION" --path "$dir"'
    fi
  fi
done

echo "OPENCV_MK_PATH:=../../opencv$CV_VERSION/share/OpenCV/OpenCV.mk" > includeOpenCV.mk

cd "$PRG_DIR/samples"
#remove ignored files/folders
svn status --no-ignore | grep ^I | cut -c9- | xargs -d \\n rm -rf
#remove unversioned files/folders
svn status | grep ^\? | cut -c9- | xargs -d \\n rm -rf
#remove unneded CMakeLists.txt
rm CMakeLists.txt


# pack all files
cd $PRG_DIR
tar cjpf opencv$CV_VERSION.tar.bz2 --exclude-vcs opencv$CV_VERSION samples || exit -1
echo
echo "Package opencv$CV_VERSION.tar.bz2 is successfully created"
