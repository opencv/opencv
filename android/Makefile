# The path to the NDK, requires crystax version r-4 for now, due to support
#for the standard library
ifndef ANDROID_NDK_ROOT
$(error ANDROID_NDK_ROOT not defined)
endif

ANDROID_NDK_BASE = $(ANDROID_NDK_ROOT)

ifndef PROJECT_PATH
$(info PROJECT_PATH defaulting to this directory)
PROJECT_PATH=.
endif

#define OPENCV_ROOT when calling this makefile
ifndef OPENCV_ROOT
$(error Please define OPENCV_ROOT with something like the command \
make OPENCV_ROOT=<opencv>)
endif

$(info OPENCV_ROOT = $(OPENCV_ROOT))

# The name of the native library
LIBNAME = libandroid-opencv.so

# Find all the C++ sources in the native folder
SOURCES = $(wildcard jni/*.cpp)
HEADERS = $(wildcard jni/*.h)
SWIG_IS = $(wildcard jni/*.i)

ANDROID_MKS = $(wildcard jni/*.mk)

SWIG_MAIN = jni/android-cv.i

SWIG_JAVA_DIR = src/com/opencv/jni
SWIG_JAVA_OUT = $(wildcard $(SWIG_JAVA_DIR)/*.java)

SWIG_C_DIR = jni/gen
SWIG_C_OUT = $(SWIG_C_DIR)/android_cv_wrap.cpp

# The real native library stripped of symbols
LIB		= libs/armeabi-v7a/$(LIBNAME) libs/armeabi/$(LIBNAME)


all:	$(LIB)


#calls the ndk-build script, passing it OPENCV_ROOT and OPENCV_LIBS_DIR
$(LIB): $(SWIG_C_OUT) $(SOURCES) $(HEADERS) $(ANDROID_MKS)
	$(ANDROID_NDK_BASE)/ndk-build OPENCV_ROOT=$(OPENCV_ROOT) \
	PROJECT_PATH=$(PROJECT_PATH) V=$(V) $(NDK_FLAGS)


#this creates the swig wrappers
$(SWIG_C_OUT): $(SWIG_IS)
	make clean-swig &&\
	mkdir -p $(SWIG_C_DIR) &&\
	mkdir -p $(SWIG_JAVA_DIR) &&\
	swig -java -c++ -package "com.opencv.jni" \
	-outdir $(SWIG_JAVA_DIR) \
	-o $(SWIG_C_OUT) $(SWIG_MAIN)
	
	
#clean targets
.PHONY: clean  clean-swig cleanall nogdb

nogdb: $(LIB)
	rm -f libs/armeabi*/gdb*

#this deletes the generated swig java and the generated c wrapper
clean-swig:
	rm -f $(SWIG_JAVA_OUT) $(SWIG_C_OUT)
	
#does clean-swig and then uses the ndk-build clean
clean: clean-swig
	$(ANDROID_NDK_BASE)/ndk-build OPENCV_ROOT=$(OPENCV_ROOT) \
	PROJECT_PATH=$(PROJECT_PATH) clean V=$(V) $(NDK_FLAGS)
