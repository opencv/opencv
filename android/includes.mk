ifndef OPENCV_ROOT
$(error Please define OPENCV_ROOT to point to the root folder of opencv)
endif
OPENCV_INCLUDES := $(OPENCV_ROOT)/3rdparty/include/ \
    $(OPENCV_ROOT)/modules/core/include/ $(OPENCV_ROOT)/modules/highgui/include/ \
    $(OPENCV_ROOT)/modules/imgproc/include $(OPENCV_ROOT)/modules/ml/include \
    $(OPENCV_ROOT)/modules/features2d/include \
    $(OPENCV_ROOT)/modules/legacy/include \
    $(OPENCV_ROOT)/modules/calib3d/include \
    $(OPENCV_ROOT)/modules/objdetect/include \
    $(OPENCV_ROOT)/modules/contrib/include \
    $(OPENCV_ROOT)/modules/video/include

ANDROID_OPENCV_INCLUDES := $(OPENCV_ROOT)/android/jni

#$(info the opencv includes are here: $(OPENCV_INCLUDES) )
