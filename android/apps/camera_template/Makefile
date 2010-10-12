#Re-usable Makefile template for
#android-ndk + swig projects
#author: Ethan Rublee
#date: summer 2010

# The path to the NDK, requires crystax version r-4 for now, due to support
#for the standard library
ifndef ANDROID_NDK_BASE
ANDROID_NDK_BASE = $(HOME)/android-ndk-r4-crystax
$(info default ndk location set ANDROID_NDK_BASE = $(ANDROID_NDK_BASE))
endif

#define OPENCV_ROOT when calling this makefile
#OPENCV_ROOT = $(ANDROID_NDK_BASE)/apps/opencv
ifndef OPENCV_ROOT
$(error Please define OPENCV_ROOT with something like the command \
make OPENCV_ROOT=<opencv>)
endif

ifndef PROJECT_PATH
$(info PROJECT_PATH defaulting to this directory)
PROJECT_PATH=.
endif

$(info OPENCV_ROOT = $(OPENCV_ROOT))

# The name of the native library
LIBNAME = libfoobar.so

#define the main jni swig interface file
#the desired java package name
#and the desired java directory
#notice that the package and the java dir
#are related...
SWIG_MAIN = jni/foobar.i
SWIG_JAVA_PACKAGE = com.foo.bar.jni
SWIG_JAVA_DIR = src/com/foo/bar/jni

SWIG_BASE = foobar

#swig definitions - auto as long as all the
#swig interface files are in jni/*.i
SWIG_JAVA_OUT = $(wildcard $(SWIG_JAVA_DIR)/*.java)
SWIG_IS = $(wildcard jni/*.i)

#the directory where the jni sources are
C_DIR = jni

#directory where to put generated files
#relative to the C_DIR
GEN_DIR = gen

#the c swig is generated and put into the jni/gen folder
SWIG_C_DIR = $(C_DIR)/$(GEN_DIR)

#this file - jin/gen/foobar_swig.cpp must be included in the Android.mk
#for it to be built!
SWIG_C_OUT = $(SWIG_C_DIR)/$(SWIG_BASE)_swig.cpp

# The real native library stripped of symbols
LIB		= libs/armeabi-v7a/$(LIBNAME) libs/armeabi/$(LIBNAME)

# Find all the C++ sources in the native folder
SOURCES = $(wildcard jni/*.cpp)
HEADERS = $(wildcard jni/*.h)

ANDROID_MKS = $(wildcard jni/*.mk)

#this gets called by the make command
all:	$(LIB)

#calls the ndk-build script, passing it OPENCV_ROOT and OPENCV_LIBS_DIR
$(LIB): $(SWIG_C_OUT) $(SOURCES) $(HEADERS) $(ANDROID_MKS)
	$(ANDROID_NDK_BASE)/ndk-build OPENCV_ROOT=$(OPENCV_ROOT) \
	OPENCV_LIBS_DIR=$(OPENCV_LIBS_DIR) PROJECT_PATH=$(PROJECT_PATH) SWIG_C_OUT=$(GEN_DIR)/$(SWIG_BASE)_swig.cpp V=$(V) $(NDK_FLAGS)


#this creates the swig wrappers
#-I$(OPENCV_ROOT)/android/jni is an additional swig include path
$(SWIG_C_OUT): $(SWIG_IS)
	make clean-swig &&\
	mkdir -p $(SWIG_C_DIR) &&\
	mkdir -p $(SWIG_JAVA_DIR) &&\
	swig -java -c++ -I$(OPENCV_ROOT)/android/jni -package  "$(SWIG_JAVA_PACKAGE)" \
	-outdir $(SWIG_JAVA_DIR) \
	-o $(SWIG_C_OUT) $(SWIG_MAIN)
	
	
#clean targets
.PHONY: clean  clean-swig cleanall

#this deletes the generated swig java and the generated c wrapper
clean-swig:
	rm -f $(SWIG_JAVA_OUT) $(SWIG_C_OUT)
	
#does clean-swig and then uses the ndk-build clean
clean: clean-swig
	$(ANDROID_NDK_BASE)/ndk-build clean V=$(V) $(NDK_FLAGS)
