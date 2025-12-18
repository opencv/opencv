#!/bin/bash -eu

# Build OpenCV
mkdir build
cd build

# MSan requires bundled libraries to prevent false positives from uninstrumented system libs
if [[ $SANITIZER = "memory" ]]; then
  EXTRA_CMAKE_FLAGS="-DBUILD_ZLIB=ON -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_TIFF=ON -DBUILD_WEBP=ON -DBUILD_OPENEXR=ON -DBUILD_PROTOBUF=ON"
else
  EXTRA_CMAKE_FLAGS=""
fi

cmake .. \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_C_FLAGS="$CFLAGS" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_DOCS=OFF \
  -DBUILD_JAVA=OFF \
  -DBUILD_PYTHON3=OFF \
  $EXTRA_CMAKE_FLAGS

make -j$(nproc)

# Locate libraries
# In static build, we need to link against all modules and 3rdparty libs.
# OpenCV creates a pkg-config file or we can find the libs in lib/ and 3rdparty/lib/

# Simple approach: link everything in lib/ and 3rdparty/lib/
# But order matters.
# Let's try to compile the fuzzer using the headers from source/build and linking.

cd ..

# Build the fuzzer
# We need include paths:
# -I. -I./include -I./modules/core/include -I./modules/imgcodecs/include ...
# Easier to use the installed/built headers if possible, or just add necessary -I

INCLUDES="-Iinclude -Imodules/core/include -Imodules/imgcodecs/include -Imodules/imgproc/include -Imodules/videoio/include -Imodules/highgui/include -Imodules/dnn/include -Ibuild"

# Static libs order is tricky.
# Using --start-group and --end-group handles circular deps/ordering issues for us (GNU ld specific, but standard in OSS-Fuzz env).

LIBS="-Wl,--start-group $(find build/lib build/3rdparty/lib -name '*.a') -Wl,--end-group -lpthread -ldl -lz"

$CXX $CXXFLAGS $LIB_FUZZING_ENGINE \
    fuzz/imdecode_fuzzer.cc -o $OUT/imdecode_fuzzer \
    $INCLUDES \
    $LIBS

# Build DNN fuzzer
$CXX $CXXFLAGS $LIB_FUZZING_ENGINE \
    fuzz/dnn_fuzzer.cc -o $OUT/dnn_fuzzer \
    $INCLUDES \
    $LIBS

# Build FileStorage fuzzer
$CXX $CXXFLAGS $LIB_FUZZING_ENGINE \
    fuzz/filestorage_fuzzer.cc -o $OUT/filestorage_fuzzer \
    $INCLUDES \
    $LIBS

# Zip corpus
zip -j $OUT/imdecode_fuzzer_seed_corpus.zip fuzz/corpus/imdecode/*
zip -j $OUT/dnn_fuzzer_seed_corpus.zip fuzz/corpus/dnn/*
zip -j $OUT/filestorage_fuzzer_seed_corpus.zip fuzz/corpus/filestorage/*


# Copy dictionary
cp fuzz/imdecode.dict $OUT/imdecode_fuzzer.dict

