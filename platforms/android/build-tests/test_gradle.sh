#!/bin/bash -e
SDK_DIR=$1

echo "OpenCV Android SDK path: ${SDK_DIR}"

ANDROID_HOME=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-${ANDROID_SDK?Required ANDROID_HOME/ANDROID_SDK/ANDROID_SDK_ROOT}}}
ANDROID_NDK=${ANDROID_NDK_HOME-${ANDROID_NDK:-${NDKROOT?Required ANDROID_NDK_HOME/ANDROID_NDK/NDKROOT}}}
OPENCV_GRADLE_VERBOSE_OPTIONS=${OPENCV_GRADLE_VERBOSE_OPTIONS:-'-i'}

echo "Android SDK: ${ANDROID_HOME}"
echo "Android NDK: ${ANDROID_NDK}"

if [ ! -d "${ANDROID_HOME}" ]; then
  echo "FATAL: Missing Android SDK directory"
  exit 1
fi
if [ ! -d "${ANDROID_NDK}" ]; then
  echo "FATAL: Missing Android NDK directory"
  exit 1
fi

export ANDROID_HOME=${ANDROID_HOME}
export ANDROID_SDK=${ANDROID_HOME}
export ANDROID_SDK_ROOT=${ANDROID_HOME}

export ANDROID_NDK=${ANDROID_NDK}
export ANDROID_NDK_HOME=${ANDROID_NDK}

echo "Cloning OpenCV Android SDK ..."
rm -rf "test-gradle"
cp -rp "${SDK_DIR}" "test-gradle"
echo "Cloning OpenCV Android SDK ... Done!"

# drop cmake bin name and "bin" folder from path
echo "ndk.dir=${ANDROID_NDK}" > "test-gradle/samples/local.properties"
echo "cmake.dir=$(dirname $(dirname $(which cmake)))" >> "test-gradle/samples/local.properties"

echo "Run gradle ..."
(cd "test-gradle/samples"; ./gradlew ${OPENCV_GRADLE_VERBOSE_OPTIONS} assemble)

echo "#"
echo "# Done!"
echo "#"
