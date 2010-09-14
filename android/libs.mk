#ifndef OPENCV_LIBS_DIR
#$(error please define to something like: OPENCV_LIBS_DIR=$(OPENCV_ROOT)/bin/ndk/local/armeabi )
#endif

#$(info the opencv libs are here: $(OPENCV_LIBS_DIR) )

#newest ndk from crystax stores the libs in the obj folder????
OPENCV_LIB_DIRS := -L$(OPENCV_ROOT)/bin/ndk/local/armeabi-v7a -L$(OPENCV_ROOT)/bin/ndk/local/armeabi -L$(OPENCV_ROOT)/obj/local/armeabi-v7a -L$(OPENCV_ROOT)/obj/local/armeabi

#order of linking very important ---- may have stuff out of order here, but
#important that modules that are more dependent come first...

OPENCV_LIBS := $(OPENCV_LIB_DIRS) -lfeatures2d  -lcalib3d -limgproc -lobjdetect  \
     -lvideo  -lhighgui -lml -llegacy -lcore -lopencv_lapack -lflann \
    -lzlib -lpng -ljpeg -ljasper
    
ANDROID_OPENCV_LIBS := -L$(OPENCV_ROOT)/android/libs/armeabi -L$(OPENCV_ROOT)/android/libs/armeabi-v7a -landroid-opencv
