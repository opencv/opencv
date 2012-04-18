#include <jni.h>

extern "C" {

JNIEXPORT jint JNICALL
JNI_OnLoad(JavaVM* vm, void* reserved)
{
    JNIEnv* env;
    if (vm->GetEnv((void**) &env, JNI_VERSION_1_6) != JNI_OK)
        return -1;

    /* get class with (*env)->FindClass */
    /* register methods with (*env)->RegisterNatives */

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM *vm, void *reserved)
{
  //do nothing
}

} // extern "C"

#include "opencv2/opencv_modules.hpp"

#if HAVE_OPENCV_MODULES_NONFREE
#include "opencv2/nonfree/nonfree.hpp"
static bool makeUseOfNonfree = initModule_nonfree();
#endif

#if HAVE_OPENCV_MODULES_FEATURES2D
#include "opencv2/features2d/features2d.hpp"
static bool makeUseOfNonfree = initModule_features2d();
#endif