# The path to the NDK, requires crystax version r-4 for now, due to support
# for the standard library

# load environment from local make file
LOCAL_ENV_MK=local.env.mk
ifneq "$(wildcard $(LOCAL_ENV_MK))" ""
include $(LOCAL_ENV_MK)
else
$(shell cp sample.$(LOCAL_ENV_MK) $(LOCAL_ENV_MK))
$(info ERROR local environement not setup! try:)
$(info gedit $(LOCAL_ENV_MK))
$(error Please setup the $(LOCAL_ENV_MK) - the default was just created')
endif
ifndef ARM_TARGETS
ARM_TARGETS="armeabi armeabi-v7a"
endif
ANDROID_NDK_BASE = $(ANDROID_NDK_ROOT)

$(info OPENCV_CONFIG = $(OPENCV_CONFIG))

ifndef PROJECT_PATH
$(info PROJECT_PATH defaulting to this directory)
PROJECT_PATH=.
endif


# The name of the native library
LIBNAME = libcvcamera.so


# Find all the C++ sources in the native folder
SOURCES = $(wildcard jni/*.cpp)
HEADERS = $(wildcard jni/*.h)

ANDROID_MKS = $(wildcard jni/*.mk)

SWIG_IS = $(wildcard jni/*.i)

SWIG_MAIN = jni/cvcamera.i

SWIG_JAVA_DIR = src/com/theveganrobot/cvcamera/jni
SWIG_JAVA_OUT = $(wildcard $(SWIG_JAVA_DIR)/*.java)


SWIG_C_DIR = jni/gen
SWIG_C_OUT = $(SWIG_C_DIR)/cvcamera_swig.cpp

BUILD_DEFS=OPENCV_CONFIG=$(OPENCV_CONFIG) \
	PROJECT_PATH=$(PROJECT_PATH) \
	V=$(V) \
	$(NDK_FLAGS) \
	ARM_TARGETS=$(ARM_TARGETS)

# The real native library stripped of symbols
LIB		= libs/armeabi-v7a/$(LIBNAME) libs/armeabi/$(LIBNAME)


all:	$(LIB)


#calls the ndk-build script, passing it OPENCV_ROOT and OPENCV_LIBS_DIR
$(LIB): $(SWIG_C_OUT) $(SOURCES) $(HEADERS) $(ANDROID_MKS)
	$(ANDROID_NDK_BASE)/ndk-build $(BUILD_DEFS)


#this creates the swig wrappers
$(SWIG_C_OUT): $(SWIG_IS)
	make clean-swig &&\
	mkdir -p $(SWIG_C_DIR) &&\
	mkdir -p $(SWIG_JAVA_DIR) &&\
	swig -java -c++ -I../../android-jni/jni -package  "com.theveganrobot.cvcamera.jni" \
	-outdir $(SWIG_JAVA_DIR) \
	-o $(SWIG_C_OUT) $(SWIG_MAIN)
	
	
#clean targets
.PHONY: clean  clean-swig cleanall

#this deletes the generated swig java and the generated c wrapper
clean-swig:
	rm -f $(SWIG_JAVA_OUT) $(SWIG_C_OUT)
	
#does clean-swig and then uses the ndk-build clean
clean: clean-swig
	$(ANDROID_NDK_BASE)/ndk-build clean $(BUILD_DEFS)

