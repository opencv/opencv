/*
 * Copyright (C)2011-2023 D. R. Commander.  All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the libjpeg-turbo Project nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS",
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <limits.h>
#include "turbojpeg.h"
#include "jinclude.h"
#include <jni.h>
#include "java/org_libjpegturbo_turbojpeg_TJCompressor.h"
#include "java/org_libjpegturbo_turbojpeg_TJDecompressor.h"
#include "java/org_libjpegturbo_turbojpeg_TJTransformer.h"
#include "java/org_libjpegturbo_turbojpeg_TJ.h"

#define BAILIF0(f) { \
  if (!(f) || (*env)->ExceptionCheck(env)) { \
    goto bailout; \
  } \
}

#define BAILIF0NOEC(f) { \
  if (!(f)) { \
    goto bailout; \
  } \
}

#define THROW(msg, exceptionClass) { \
  jclass _exccls = (*env)->FindClass(env, exceptionClass); \
  \
  BAILIF0(_exccls); \
  (*env)->ThrowNew(env, _exccls, msg); \
  goto bailout; \
}

#define THROW_TJ() { \
  jclass _exccls; \
  jmethodID _excid; \
  jobject _excobj; \
  jstring _errstr; \
  \
  BAILIF0(_errstr = (*env)->NewStringUTF(env, tj3GetErrorStr(handle))); \
  BAILIF0(_exccls = (*env)->FindClass(env, \
    "org/libjpegturbo/turbojpeg/TJException")); \
  BAILIF0(_excid = (*env)->GetMethodID(env, _exccls, "<init>", \
                                       "(Ljava/lang/String;I)V")); \
  BAILIF0(_excobj = (*env)->NewObject(env, _exccls, _excid, _errstr, \
                                      tj3GetErrorCode(handle))); \
  (*env)->Throw(env, _excobj); \
  goto bailout; \
}

#define THROW_ARG(msg)  THROW(msg, "java/lang/IllegalArgumentException")

#define THROW_MEM() \
  THROW("Memory allocation failure", "java/lang/OutOfMemoryError");

#define GET_HANDLE() \
  jclass _cls = (*env)->GetObjectClass(env, obj); \
  jfieldID _fid; \
  \
  BAILIF0(_cls); \
  BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "handle", "J")); \
  handle = (tjhandle)(size_t)(*env)->GetLongField(env, obj, _fid);

#define SAFE_RELEASE(javaArray, cArray) { \
  if (javaArray && cArray) \
    (*env)->ReleasePrimitiveArrayCritical(env, javaArray, (void *)cArray, 0); \
  cArray = NULL; \
}

/* TurboJPEG 1.2.x: TJ::bufSize() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJ_bufSize
  (JNIEnv *env, jclass cls, jint width, jint height, jint jpegSubsamp)
{
  size_t retval = tj3JPEGBufSize(width, height, jpegSubsamp);

  if (retval == 0) THROW_ARG(tj3GetErrorStr(NULL));
  if (retval > (size_t)INT_MAX)
    THROW_ARG("Image is too large");

bailout:
  return (jint)retval;
}

/* TurboJPEG 1.4.x: TJ::bufSizeYUV() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJ_bufSizeYUV__IIII
  (JNIEnv *env, jclass cls, jint width, jint align, jint height, jint subsamp)
{
  size_t retval = tj3YUVBufSize(width, align, height, subsamp);

  if (retval == 0) THROW_ARG(tj3GetErrorStr(NULL));
  if (retval > (size_t)INT_MAX)
    THROW_ARG("Image is too large");

bailout:
  return (jint)retval;
}

/* TurboJPEG 1.4.x: TJ::planeSizeYUV() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJ_planeSizeYUV__IIIII
  (JNIEnv *env, jclass cls, jint componentID, jint width, jint stride,
   jint height, jint subsamp)
{
  size_t retval = tj3YUVPlaneSize(componentID, width, stride, height, subsamp);

  if (retval == 0) THROW_ARG(tj3GetErrorStr(NULL));
  if (retval > (size_t)INT_MAX)
    THROW_ARG("Image is too large");

bailout:
  return (jint)retval;
}

/* TurboJPEG 1.4.x: TJ::planeWidth() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJ_planeWidth__III
  (JNIEnv *env, jclass cls, jint componentID, jint width, jint subsamp)
{
  jint retval = (jint)tj3YUVPlaneWidth(componentID, width, subsamp);

  if (retval == 0) THROW_ARG(tj3GetErrorStr(NULL));

bailout:
  return retval;
}

/* TurboJPEG 1.4.x: TJ::planeHeight() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJ_planeHeight__III
  (JNIEnv *env, jclass cls, jint componentID, jint height, jint subsamp)
{
  jint retval = (jint)tj3YUVPlaneHeight(componentID, height, subsamp);

  if (retval == 0) THROW_ARG(tj3GetErrorStr(NULL));

bailout:
  return retval;
}

/* TurboJPEG 1.2.x: TJCompressor::init() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_init
  (JNIEnv *env, jobject obj)
{
  jclass cls;
  jfieldID fid;
  tjhandle handle;

  if ((handle = tj3Init(TJINIT_COMPRESS)) == NULL)
    THROW(tj3GetErrorStr(NULL), "org/libjpegturbo/turbojpeg/TJException");

  BAILIF0(cls = (*env)->GetObjectClass(env, obj));
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "handle", "J"));
  (*env)->SetLongField(env, obj, fid, (size_t)handle);

bailout:
  return;
}

/* TurboJPEG 3: TJCompressor::set() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_set
  (JNIEnv *env, jobject obj, jint param, jint value)
{
  tjhandle handle = 0;

  GET_HANDLE();

  if (tj3Set(handle, param, value) == -1)
    THROW_TJ();

bailout:
  return;
}

/* TurboJPEG 3: TJCompressor::get() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_get
  (JNIEnv *env, jobject obj, jint param)
{
  tjhandle handle = 0;

  GET_HANDLE();

  return tj3Get(handle, param);

bailout:
  return -1;
}

static jint TJCompressor_compress
  (JNIEnv *env, jobject obj, jarray src, jint srcElementSize, jint precision,
   jint x, jint y, jint width, jint pitch, jint height, jint pf,
   jbyteArray dst)
{
  tjhandle handle = 0;
  size_t jpegSize = 0;
  jsize arraySize = 0, actualPitch;
  void *srcBuf = NULL;
  unsigned char *jpegBuf = NULL;
  int jpegSubsamp;

  GET_HANDLE();

  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF || width < 1 ||
      height < 1 || pitch < 0)
    THROW_ARG("Invalid argument in compress*()");
  if (org_libjpegturbo_turbojpeg_TJ_NUMPF != TJ_NUMPF)
    THROW_ARG("Mismatch between Java and C API");

  actualPitch = (pitch == 0) ? width * tjPixelSize[pf] : pitch;
  arraySize = (y + height - 1) * actualPitch + (x + width) * tjPixelSize[pf];
  if ((*env)->GetArrayLength(env, src) * srcElementSize < arraySize)
    THROW_ARG("Source buffer is not large enough");
  jpegSubsamp = tj3Get(handle, TJPARAM_SUBSAMP);
  if (tj3Get(handle, TJPARAM_LOSSLESS) && jpegSubsamp != TJSAMP_GRAY)
    jpegSubsamp = TJSAMP_444;
  else if (jpegSubsamp == TJSAMP_UNKNOWN)
    THROW_ARG("TJPARAM_SUBSAMP must be specified");
  jpegSize = tj3JPEGBufSize(width, height, jpegSubsamp);
  if ((*env)->GetArrayLength(env, dst) < (jsize)jpegSize)
    THROW_ARG("Destination buffer is not large enough");

  if (tj3Set(handle, TJPARAM_NOREALLOC, 1) == -1)
    THROW_TJ();

  BAILIF0NOEC(srcBuf = (*env)->GetPrimitiveArrayCritical(env, src, 0));
  BAILIF0NOEC(jpegBuf = (*env)->GetPrimitiveArrayCritical(env, dst, 0));

  if (precision == 8) {
    if (tj3Compress8(handle, &((unsigned char *)srcBuf)[y * actualPitch +
                                                        x * tjPixelSize[pf]],
                     width, pitch, height, pf, &jpegBuf, &jpegSize) == -1) {
      SAFE_RELEASE(dst, jpegBuf);
      SAFE_RELEASE(src, srcBuf);
      THROW_TJ();
    }
  } else if (precision == 12) {
    if (tj3Compress12(handle, &((short *)srcBuf)[y * actualPitch +
                                                 x * tjPixelSize[pf]],
                      width, pitch, height, pf, &jpegBuf, &jpegSize) == -1) {
      SAFE_RELEASE(dst, jpegBuf);
      SAFE_RELEASE(src, srcBuf);
      THROW_TJ();
    }
  } else {
    if (tj3Compress16(handle, &((unsigned short *)srcBuf)[y * actualPitch +
                                                          x * tjPixelSize[pf]],
                      width, pitch, height, pf, &jpegBuf, &jpegSize) == -1) {
      SAFE_RELEASE(dst, jpegBuf);
      SAFE_RELEASE(src, srcBuf);
      THROW_TJ();
    }
  }

bailout:
  SAFE_RELEASE(dst, jpegBuf);
  SAFE_RELEASE(src, srcBuf);
  return (jint)jpegSize;
}

/* TurboJPEG 3: TJCompressor::compress8() byte source */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_compress8___3BIIIIII_3B
  (JNIEnv *env, jobject obj, jbyteArray src, jint x, jint y, jint width,
   jint pitch, jint height, jint pf, jbyteArray dst)
{
  return TJCompressor_compress(env, obj, src, 1, 8, x, y, width, pitch, height,
                               pf, dst);
}

/* TurboJPEG 3: TJCompressor::compress12() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_compress12
  (JNIEnv *env, jobject obj, jshortArray src, jint x, jint y, jint width,
   jint pitch, jint height, jint pf, jbyteArray dst)
{
  return TJCompressor_compress(env, obj, src, 1, 12, x, y, width, pitch,
                               height, pf, dst);
}

/* TurboJPEG 3: TJCompressor::compress16() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_compress16
  (JNIEnv *env, jobject obj, jshortArray src, jint x, jint y, jint width,
   jint pitch, jint height, jint pf, jbyteArray dst)
{
  return TJCompressor_compress(env, obj, src, 1, 16, x, y, width, pitch,
                               height, pf, dst);
}

/* TurboJPEG 3: TJCompressor::compress8() int source */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_compress8___3IIIIIII_3B
  (JNIEnv *env, jobject obj, jintArray src, jint x, jint y, jint width,
   jint stride, jint height, jint pf, jbyteArray dst)
{
  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF)
    THROW_ARG("Invalid argument in compress8()");
  if (tjPixelSize[pf] != sizeof(jint))
    THROW_ARG("Pixel format must be 32-bit when compressing from an integer buffer.");

  return TJCompressor_compress(env, obj, src, sizeof(jint), 8, x, y, width,
                               stride * sizeof(jint), height, pf, dst);

bailout:
  return 0;
}

/* TurboJPEG 3: TJCompressor::compressFromYUV8() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_compressFromYUV8
  (JNIEnv *env, jobject obj, jobjectArray srcobjs, jintArray jSrcOffsets,
   jint width, jintArray jSrcStrides, jint height, jbyteArray dst)
{
  tjhandle handle = 0;
  size_t jpegSize = 0;
  jbyteArray jSrcPlanes[3] = { NULL, NULL, NULL };
  const unsigned char *srcPlanesTmp[3] = { NULL, NULL, NULL };
  const unsigned char *srcPlanes[3] = { NULL, NULL, NULL };
  jint srcOffsetsTmp[3] = { 0, 0, 0 }, srcStridesTmp[3] = { 0, 0, 0 };
  int srcOffsets[3] = { 0, 0, 0 }, srcStrides[3] = { 0, 0, 0 };
  unsigned char *jpegBuf = NULL;
  int nc = 0, i, subsamp;

  GET_HANDLE();

  if (org_libjpegturbo_turbojpeg_TJ_NUMSAMP != TJ_NUMSAMP)
    THROW_ARG("Mismatch between Java and C API");

  if ((subsamp = tj3Get(handle, TJPARAM_SUBSAMP)) == TJSAMP_UNKNOWN)
    THROW_ARG("TJPARAM_SUBSAMP must be specified");
  nc = subsamp == TJSAMP_GRAY ? 1 : 3;
  if ((*env)->GetArrayLength(env, srcobjs) < nc)
    THROW_ARG("Planes array is too small for the subsampling type");
  if ((*env)->GetArrayLength(env, jSrcOffsets) < nc)
    THROW_ARG("Offsets array is too small for the subsampling type");
  if ((*env)->GetArrayLength(env, jSrcStrides) < nc)
    THROW_ARG("Strides array is too small for the subsampling type");

  jpegSize = tj3JPEGBufSize(width, height, subsamp);
  if ((*env)->GetArrayLength(env, dst) < (jsize)jpegSize)
    THROW_ARG("Destination buffer is not large enough");

  if (tj3Set(handle, TJPARAM_NOREALLOC, 1) == -1)
    THROW_TJ();

  (*env)->GetIntArrayRegion(env, jSrcOffsets, 0, nc, srcOffsetsTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    srcOffsets[i] = srcOffsetsTmp[i];

  (*env)->GetIntArrayRegion(env, jSrcStrides, 0, nc, srcStridesTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    srcStrides[i] = srcStridesTmp[i];

  for (i = 0; i < nc; i++) {
    size_t planeSize = tj3YUVPlaneSize(i, width, srcStrides[i], height,
                                       subsamp);
    int pw = tj3YUVPlaneWidth(i, width, subsamp);

    if (planeSize == 0 || pw == 0)
      THROW_ARG(tj3GetErrorStr(NULL));

    if (planeSize > (size_t)INT_MAX)
      THROW_ARG("Source plane is too large");
    if (srcOffsets[i] < 0)
      THROW_ARG("Invalid argument in compressFromYUV8()");
    if (srcStrides[i] < 0 && srcOffsets[i] - (int)planeSize + pw < 0)
      THROW_ARG("Negative plane stride would cause memory to be accessed below plane boundary");

    BAILIF0(jSrcPlanes[i] = (*env)->GetObjectArrayElement(env, srcobjs, i));
    if ((*env)->GetArrayLength(env, jSrcPlanes[i]) <
        srcOffsets[i] + (int)planeSize)
      THROW_ARG("Source plane is not large enough");
  }
  for (i = 0; i < nc; i++) {
    BAILIF0NOEC(srcPlanesTmp[i] =
                (*env)->GetPrimitiveArrayCritical(env, jSrcPlanes[i], 0));
    srcPlanes[i] = &srcPlanesTmp[i][srcOffsets[i]];
  }
  BAILIF0NOEC(jpegBuf = (*env)->GetPrimitiveArrayCritical(env, dst, 0));

  if (tj3CompressFromYUVPlanes8(handle, srcPlanes, width, srcStrides, height,
                                &jpegBuf, &jpegSize) == -1) {
    SAFE_RELEASE(dst, jpegBuf);
    for (i = 0; i < nc; i++)
      SAFE_RELEASE(jSrcPlanes[i], srcPlanesTmp[i]);
    THROW_TJ();
  }

bailout:
  SAFE_RELEASE(dst, jpegBuf);
  for (i = 0; i < nc; i++)
    SAFE_RELEASE(jSrcPlanes[i], srcPlanesTmp[i]);
  return (jint)jpegSize;
}

static void TJCompressor_encodeYUV8
  (JNIEnv *env, jobject obj, jarray src, jint srcElementSize, jint x, jint y,
   jint width, jint pitch, jint height, jint pf, jobjectArray dstobjs,
   jintArray jDstOffsets, jintArray jDstStrides)
{
  tjhandle handle = 0;
  jsize arraySize = 0, actualPitch;
  unsigned char *srcBuf = NULL;
  jbyteArray jDstPlanes[3] = { NULL, NULL, NULL };
  unsigned char *dstPlanesTmp[3] = { NULL, NULL, NULL };
  unsigned char *dstPlanes[3] = { NULL, NULL, NULL };
  jint dstOffsetsTmp[3] = { 0, 0, 0 }, dstStridesTmp[3] = { 0, 0, 0 };
  int dstOffsets[3] = { 0, 0, 0 }, dstStrides[3] = { 0, 0, 0 };
  int nc = 0, i, subsamp;

  GET_HANDLE();

  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF || width < 1 ||
      height < 1 || pitch < 0)
    THROW_ARG("Invalid argument in encodeYUV8()");
  if (org_libjpegturbo_turbojpeg_TJ_NUMPF != TJ_NUMPF ||
      org_libjpegturbo_turbojpeg_TJ_NUMSAMP != TJ_NUMSAMP)
    THROW_ARG("Mismatch between Java and C API");

  if ((subsamp = tj3Get(handle, TJPARAM_SUBSAMP)) == TJSAMP_UNKNOWN)
    THROW_ARG("TJPARAM_SUBSAMP must be specified");
  nc = subsamp == TJSAMP_GRAY ? 1 : 3;
  if ((*env)->GetArrayLength(env, dstobjs) < nc)
    THROW_ARG("Planes array is too small for the subsampling type");
  if ((*env)->GetArrayLength(env, jDstOffsets) < nc)
    THROW_ARG("Offsets array is too small for the subsampling type");
  if ((*env)->GetArrayLength(env, jDstStrides) < nc)
    THROW_ARG("Strides array is too small for the subsampling type");

  actualPitch = (pitch == 0) ? width * tjPixelSize[pf] : pitch;
  arraySize = (y + height - 1) * actualPitch + (x + width) * tjPixelSize[pf];
  if ((*env)->GetArrayLength(env, src) * srcElementSize < arraySize)
    THROW_ARG("Source buffer is not large enough");

  (*env)->GetIntArrayRegion(env, jDstOffsets, 0, nc, dstOffsetsTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    dstOffsets[i] = dstOffsetsTmp[i];

  (*env)->GetIntArrayRegion(env, jDstStrides, 0, nc, dstStridesTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    dstStrides[i] = dstStridesTmp[i];

  for (i = 0; i < nc; i++) {
    size_t planeSize = tj3YUVPlaneSize(i, width, dstStrides[i], height,
                                       subsamp);
    int pw = tj3YUVPlaneWidth(i, width, subsamp);

    if (planeSize == 0 || pw == 0)
      THROW_ARG(tj3GetErrorStr(NULL));

    if (planeSize > (size_t)INT_MAX)
      THROW_ARG("Destination plane is too large");
    if (dstOffsets[i] < 0)
      THROW_ARG("Invalid argument in encodeYUV8()");
    if (dstStrides[i] < 0 && dstOffsets[i] - (int)planeSize + pw < 0)
      THROW_ARG("Negative plane stride would cause memory to be accessed below plane boundary");

    BAILIF0(jDstPlanes[i] = (*env)->GetObjectArrayElement(env, dstobjs, i));
    if ((*env)->GetArrayLength(env, jDstPlanes[i]) <
        dstOffsets[i] + (int)planeSize)
      THROW_ARG("Destination plane is not large enough");
  }
  for (i = 0; i < nc; i++) {
    BAILIF0NOEC(dstPlanesTmp[i] =
                (*env)->GetPrimitiveArrayCritical(env, jDstPlanes[i], 0));
    dstPlanes[i] = &dstPlanesTmp[i][dstOffsets[i]];
  }
  BAILIF0NOEC(srcBuf = (*env)->GetPrimitiveArrayCritical(env, src, 0));

  if (tj3EncodeYUVPlanes8(handle,
                          &srcBuf[y * actualPitch + x * tjPixelSize[pf]],
                          width, pitch, height, pf, dstPlanes,
                          dstStrides) == -1) {
    SAFE_RELEASE(src, srcBuf);
    for (i = 0; i < nc; i++)
      SAFE_RELEASE(jDstPlanes[i], dstPlanesTmp[i]);
    THROW_TJ();
  }

bailout:
  SAFE_RELEASE(src, srcBuf);
  for (i = 0; i < nc; i++)
    SAFE_RELEASE(jDstPlanes[i], dstPlanesTmp[i]);
}

/* TurboJPEG 3: TJCompressor::encodeYUV8() byte source */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_encodeYUV8___3BIIIIII_3_3B_3I_3I
  (JNIEnv *env, jobject obj, jbyteArray src, jint x, jint y, jint width,
   jint pitch, jint height, jint pf, jobjectArray dstobjs,
   jintArray jDstOffsets, jintArray jDstStrides)
{
  TJCompressor_encodeYUV8(env, obj, src, 1, x, y, width, pitch, height, pf,
                          dstobjs, jDstOffsets, jDstStrides);
}

/* TurboJPEG 3: TJCompressor::encodeYUV8() int source */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_encodeYUV8___3IIIIIII_3_3B_3I_3I
  (JNIEnv *env, jobject obj, jintArray src, jint x, jint y, jint width,
   jint stride, jint height, jint pf, jobjectArray dstobjs,
   jintArray jDstOffsets, jintArray jDstStrides)
{
  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF)
    THROW_ARG("Invalid argument in encodeYUV8()");
  if (tjPixelSize[pf] != sizeof(jint))
    THROW_ARG("Pixel format must be 32-bit when encoding from an integer buffer.");

  TJCompressor_encodeYUV8(env, obj, src, sizeof(jint), x, y, width,
                          stride * sizeof(jint), height, pf, dstobjs,
                          jDstOffsets, jDstStrides);

bailout:
  return;
}

/* TurboJPEG 1.2.x: TJCompressor::destroy() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_destroy
  (JNIEnv *env, jobject obj)
{
  tjhandle handle = 0;

  GET_HANDLE();

  tj3Destroy(handle);
  (*env)->SetLongField(env, obj, _fid, 0);

bailout:
  return;
}

/* TurboJPEG 1.2.x: TJDecompressor::init() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_init
  (JNIEnv *env, jobject obj)
{
  jclass cls;
  jfieldID fid;
  tjhandle handle;

  if ((handle = tj3Init(TJINIT_DECOMPRESS)) == NULL)
    THROW(tj3GetErrorStr(NULL), "org/libjpegturbo/turbojpeg/TJException");

  BAILIF0(cls = (*env)->GetObjectClass(env, obj));
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "handle", "J"));
  (*env)->SetLongField(env, obj, fid, (size_t)handle);

bailout:
  return;
}

/* TurboJPEG 3: TJDecompressor::set() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_set
  (JNIEnv *env, jobject obj, jint param, jint value)
{
  Java_org_libjpegturbo_turbojpeg_TJCompressor_set(env, obj, param, value);
}

/* TurboJPEG 3: TJDecompressor::get() */
JNIEXPORT jint JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_get
  (JNIEnv *env, jobject obj, jint param)
{
  return Java_org_libjpegturbo_turbojpeg_TJCompressor_get(env, obj, param);
}

/* TurboJPEG 1.2.x: TJDecompressor::getScalingFactors() */
JNIEXPORT jobjectArray JNICALL Java_org_libjpegturbo_turbojpeg_TJ_getScalingFactors
  (JNIEnv *env, jclass cls)
{
  jclass sfcls = NULL;
  jfieldID fid = 0;
  tjscalingfactor *sf = NULL;
  int n = 0, i;
  jobject sfobj = NULL;
  jobjectArray sfjava = NULL;

  if ((sf = tj3GetScalingFactors(&n)) == NULL || n == 0)
    THROW_ARG(tj3GetErrorStr(NULL));

  BAILIF0(sfcls = (*env)->FindClass(env,
    "org/libjpegturbo/turbojpeg/TJScalingFactor"));
  BAILIF0(sfjava = (jobjectArray)(*env)->NewObjectArray(env, n, sfcls, 0));

  for (i = 0; i < n; i++) {
    BAILIF0(sfobj = (*env)->AllocObject(env, sfcls));
    BAILIF0(fid = (*env)->GetFieldID(env, sfcls, "num", "I"));
    (*env)->SetIntField(env, sfobj, fid, sf[i].num);
    BAILIF0(fid = (*env)->GetFieldID(env, sfcls, "denom", "I"));
    (*env)->SetIntField(env, sfobj, fid, sf[i].denom);
    (*env)->SetObjectArrayElement(env, sfjava, i, sfobj);
  }

bailout:
  return sfjava;
}

/* TurboJPEG 1.2.x: TJDecompressor::decompressHeader() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decompressHeader
  (JNIEnv *env, jobject obj, jbyteArray src, jint jpegSize)
{
  tjhandle handle = 0;
  unsigned char *jpegBuf = NULL;

  GET_HANDLE();

  if ((*env)->GetArrayLength(env, src) < jpegSize)
    THROW_ARG("Source buffer is not large enough");

  BAILIF0NOEC(jpegBuf = (*env)->GetPrimitiveArrayCritical(env, src, 0));

  if (tj3DecompressHeader(handle, jpegBuf, (size_t)jpegSize) == -1) {
    SAFE_RELEASE(src, jpegBuf);
    THROW_TJ();
  }

bailout:
  SAFE_RELEASE(src, jpegBuf);
}

/* TurboJPEG 3: TJDecompressor::setCroppingRegion() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_setCroppingRegion
  (JNIEnv *env, jobject obj)
{
  tjhandle handle = 0;
  jclass sfcls, crcls;
  jobject sfobj, crobj;
  tjregion croppingRegion;
  tjscalingfactor scalingFactor;

  GET_HANDLE();

  BAILIF0(sfcls = (*env)->FindClass(env,
    "org/libjpegturbo/turbojpeg/TJScalingFactor"));
  BAILIF0(_fid =
          (*env)->GetFieldID(env, _cls, "scalingFactor",
                             "Lorg/libjpegturbo/turbojpeg/TJScalingFactor;"));
  BAILIF0(sfobj = (*env)->GetObjectField(env, obj, _fid));
  BAILIF0(_fid = (*env)->GetFieldID(env, sfcls, "num", "I"));
  scalingFactor.num = (*env)->GetIntField(env, sfobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, sfcls, "denom", "I"));
  scalingFactor.denom = (*env)->GetIntField(env, sfobj, _fid);

  if (tj3SetScalingFactor(handle, scalingFactor) == -1)
    THROW_TJ();

  BAILIF0(crcls = (*env)->FindClass(env, "java/awt/Rectangle"));
  BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "croppingRegion",
                                    "Ljava/awt/Rectangle;"));
  BAILIF0(crobj = (*env)->GetObjectField(env, obj, _fid));
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "x", "I"));
  croppingRegion.x = (*env)->GetIntField(env, crobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "y", "I"));
  croppingRegion.y = (*env)->GetIntField(env, crobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "width", "I"));
  croppingRegion.w = (*env)->GetIntField(env, crobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "height", "I"));
  croppingRegion.h = (*env)->GetIntField(env, crobj, _fid);

  if (tj3SetCroppingRegion(handle, croppingRegion) == -1)
    THROW_TJ();

bailout:
  return;
}

static void TJDecompressor_decompress
  (JNIEnv *env, jobject obj, jbyteArray src, jint jpegSize, jarray dst,
   jint dstElementSize, int precision, jint x, jint y, jint pitch, jint pf)
{
  tjhandle handle = 0;
  jsize arraySize = 0, actualPitch;
  unsigned char *jpegBuf = NULL;
  void *dstBuf = NULL;
  jclass sfcls, crcls;
  jobject sfobj, crobj;
  tjscalingfactor scalingFactor;
  tjregion cr;
  int jpegWidth, jpegHeight, scaledWidth, scaledHeight;

  GET_HANDLE();

  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF)
    THROW_ARG("Invalid argument in decompress*()");
  if (org_libjpegturbo_turbojpeg_TJ_NUMPF != TJ_NUMPF)
    THROW_ARG("Mismatch between Java and C API");

  if ((*env)->GetArrayLength(env, src) < jpegSize)
    THROW_ARG("Source buffer is not large enough");
  if ((jpegWidth = tj3Get(handle, TJPARAM_JPEGWIDTH)) == -1)
    THROW_ARG("JPEG header has not yet been read");
  if ((jpegHeight = tj3Get(handle, TJPARAM_JPEGHEIGHT)) == -1)
    THROW_ARG("JPEG header has not yet been read");

  BAILIF0(sfcls = (*env)->FindClass(env,
    "org/libjpegturbo/turbojpeg/TJScalingFactor"));
  BAILIF0(_fid =
          (*env)->GetFieldID(env, _cls, "scalingFactor",
                             "Lorg/libjpegturbo/turbojpeg/TJScalingFactor;"));
  BAILIF0(sfobj = (*env)->GetObjectField(env, obj, _fid));
  BAILIF0(_fid = (*env)->GetFieldID(env, sfcls, "num", "I"));
  scalingFactor.num = (*env)->GetIntField(env, sfobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, sfcls, "denom", "I"));
  scalingFactor.denom = (*env)->GetIntField(env, sfobj, _fid);

  if (tj3SetScalingFactor(handle, scalingFactor) == -1)
    THROW_TJ();
  scaledWidth = TJSCALED(jpegWidth, scalingFactor);
  scaledHeight = TJSCALED(jpegHeight, scalingFactor);

  BAILIF0(crcls = (*env)->FindClass(env, "java/awt/Rectangle"));
  BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "croppingRegion",
                                    "Ljava/awt/Rectangle;"));
  BAILIF0(crobj = (*env)->GetObjectField(env, obj, _fid));
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "x", "I"));
  cr.x = (*env)->GetIntField(env, crobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "y", "I"));
  cr.y = (*env)->GetIntField(env, crobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "width", "I"));
  cr.w = (*env)->GetIntField(env, crobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, crcls, "height", "I"));
  cr.h = (*env)->GetIntField(env, crobj, _fid);
  if (cr.x != 0 || cr.y != 0 || cr.w != 0 || cr.h != 0) {
    scaledWidth = cr.w ? cr.w : scaledWidth - cr.x;
    scaledHeight = cr.h ? cr.h : scaledHeight - cr.y;
  }

  actualPitch = (pitch == 0) ? scaledWidth * tjPixelSize[pf] : pitch;
  arraySize = (y + scaledHeight - 1) * actualPitch +
              (x + scaledWidth) * tjPixelSize[pf];
  if ((*env)->GetArrayLength(env, dst) * dstElementSize < arraySize)
    THROW_ARG("Destination buffer is not large enough");

  BAILIF0NOEC(jpegBuf = (*env)->GetPrimitiveArrayCritical(env, src, 0));
  BAILIF0NOEC(dstBuf = (*env)->GetPrimitiveArrayCritical(env, dst, 0));

  if (precision == 8) {
    if (tj3Decompress8(handle, jpegBuf, (size_t)jpegSize,
                       &((unsigned char *)dstBuf)[y * actualPitch +
                                                  x * tjPixelSize[pf]],
                       pitch, pf) == -1) {
      SAFE_RELEASE(dst, dstBuf);
      SAFE_RELEASE(src, jpegBuf);
      THROW_TJ();
    }
  } else if (precision == 12) {
    if (tj3Decompress12(handle, jpegBuf, (size_t)jpegSize,
                        &((short *)dstBuf)[y * actualPitch +
                                           x * tjPixelSize[pf]],
                        pitch, pf) == -1) {
      SAFE_RELEASE(dst, dstBuf);
      SAFE_RELEASE(src, jpegBuf);
      THROW_TJ();
    }
  } else {
    if (tj3Decompress16(handle, jpegBuf, (size_t)jpegSize,
                        &((unsigned short *)dstBuf)[y * actualPitch +
                                                    x * tjPixelSize[pf]],
                        pitch, pf) == -1) {
      SAFE_RELEASE(dst, dstBuf);
      SAFE_RELEASE(src, jpegBuf);
      THROW_TJ();
    }
  }

bailout:
  SAFE_RELEASE(dst, dstBuf);
  SAFE_RELEASE(src, jpegBuf);
}

/* TurboJPEG 3: TJDecompressor::decompress8() byte destination */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decompress8___3BI_3BIIII
  (JNIEnv *env, jobject obj, jbyteArray src, jint jpegSize, jbyteArray dst,
   jint x, jint y, jint pitch, jint pf)
{
  TJDecompressor_decompress(env, obj, src, jpegSize, dst, 1, 8, x, y, pitch,
                            pf);
}

/* TurboJPEG 3: TJDecompressor::decompress12() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decompress12
  (JNIEnv *env, jobject obj, jbyteArray src, jint jpegSize, jshortArray dst,
   jint x, jint y, jint pitch, jint pf)
{
  TJDecompressor_decompress(env, obj, src, jpegSize, dst, 1, 12, x, y, pitch,
                            pf);
}

/* TurboJPEG 3: TJDecompressor::decompress16() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decompress16
  (JNIEnv *env, jobject obj, jbyteArray src, jint jpegSize, jshortArray dst,
   jint x, jint y, jint pitch, jint pf)
{
  TJDecompressor_decompress(env, obj, src, jpegSize, dst, 1, 16, x, y, pitch,
                            pf);
}

/* TurboJPEG 3: TJDecompressor::decompress8() int destination */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decompress8___3BI_3IIIII
  (JNIEnv *env, jobject obj, jbyteArray src, jint jpegSize, jintArray dst,
   jint x, jint y, jint stride, jint pf)
{
  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF)
    THROW_ARG("Invalid argument in decompress8()");
  if (tjPixelSize[pf] != sizeof(jint))
    THROW_ARG("Pixel format must be 32-bit when decompressing to an integer buffer.");

  TJDecompressor_decompress(env, obj, src, jpegSize, dst, sizeof(jint), 8, x,
                            y, stride * sizeof(jint), pf);

bailout:
  return;
}

/* TurboJPEG 3: TJDecompressor::decompressToYUV8() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decompressToYUV8
  (JNIEnv *env, jobject obj, jbyteArray src, jint jpegSize,
   jobjectArray dstobjs, jintArray jDstOffsets, jintArray jDstStrides)
{
  tjhandle handle = 0;
  unsigned char *jpegBuf = NULL;
  jbyteArray jDstPlanes[3] = { NULL, NULL, NULL };
  unsigned char *dstPlanesTmp[3] = { NULL, NULL, NULL };
  unsigned char *dstPlanes[3] = { NULL, NULL, NULL };
  jint dstOffsetsTmp[3] = { 0, 0, 0 }, dstStridesTmp[3] = { 0, 0, 0 };
  int dstOffsets[3] = { 0, 0, 0 }, dstStrides[3] = { 0, 0, 0 };
  jclass sfcls;
  jobject sfobj;
  int jpegSubsamp, jpegWidth = 0, jpegHeight = 0;
  int nc = 0, i, scaledWidth, scaledHeight;
  tjscalingfactor scalingFactor;

  GET_HANDLE();

  if ((*env)->GetArrayLength(env, src) < jpegSize)
    THROW_ARG("Source buffer is not large enough");
  if ((jpegWidth = tj3Get(handle, TJPARAM_JPEGWIDTH)) == -1)
    THROW_ARG("JPEG header has not yet been read");
  if ((jpegHeight = tj3Get(handle, TJPARAM_JPEGHEIGHT)) == -1)
    THROW_ARG("JPEG header has not yet been read");

  BAILIF0(sfcls = (*env)->FindClass(env,
    "org/libjpegturbo/turbojpeg/TJScalingFactor"));
  BAILIF0(_fid =
          (*env)->GetFieldID(env, _cls, "scalingFactor",
                             "Lorg/libjpegturbo/turbojpeg/TJScalingFactor;"));
  BAILIF0(sfobj = (*env)->GetObjectField(env, obj, _fid));
  BAILIF0(_fid = (*env)->GetFieldID(env, sfcls, "num", "I"));
  scalingFactor.num = (*env)->GetIntField(env, sfobj, _fid);
  BAILIF0(_fid = (*env)->GetFieldID(env, sfcls, "denom", "I"));
  scalingFactor.denom = (*env)->GetIntField(env, sfobj, _fid);

  if (tj3SetScalingFactor(handle, scalingFactor) == -1)
    THROW_TJ();
  scaledWidth = TJSCALED(jpegWidth, scalingFactor);
  scaledHeight = TJSCALED(jpegHeight, scalingFactor);

  if ((jpegSubsamp = tj3Get(handle, TJPARAM_SUBSAMP)) == TJSAMP_UNKNOWN)
    THROW_ARG("TJPARAM_SUBSAMP must be specified");
  nc = jpegSubsamp == TJSAMP_GRAY ? 1 : 3;

  (*env)->GetIntArrayRegion(env, jDstOffsets, 0, nc, dstOffsetsTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    dstOffsets[i] = dstOffsetsTmp[i];

  (*env)->GetIntArrayRegion(env, jDstStrides, 0, nc, dstStridesTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    dstStrides[i] = dstStridesTmp[i];

  for (i = 0; i < nc; i++) {
    size_t planeSize = tj3YUVPlaneSize(i, scaledWidth, dstStrides[i],
                                       scaledHeight, jpegSubsamp);
    int pw = tj3YUVPlaneWidth(i, scaledWidth, jpegSubsamp);

    if (planeSize == 0 || pw == 0)
      THROW_ARG(tj3GetErrorStr(NULL));

    if (planeSize > (size_t)INT_MAX)
      THROW_ARG("Destination plane is too large");
    if (dstOffsets[i] < 0)
      THROW_ARG("Invalid argument in decompressToYUV8()");
    if (dstStrides[i] < 0 && dstOffsets[i] - (int)planeSize + pw < 0)
      THROW_ARG("Negative plane stride would cause memory to be accessed below plane boundary");

    BAILIF0(jDstPlanes[i] = (*env)->GetObjectArrayElement(env, dstobjs, i));
    if ((*env)->GetArrayLength(env, jDstPlanes[i]) <
        dstOffsets[i] + (int)planeSize)
      THROW_ARG("Destination plane is not large enough");
  }
  for (i = 0; i < nc; i++) {
    BAILIF0NOEC(dstPlanesTmp[i] =
                (*env)->GetPrimitiveArrayCritical(env, jDstPlanes[i], 0));
    dstPlanes[i] = &dstPlanesTmp[i][dstOffsets[i]];
  }
  BAILIF0NOEC(jpegBuf = (*env)->GetPrimitiveArrayCritical(env, src, 0));

  if (tj3DecompressToYUVPlanes8(handle, jpegBuf, (size_t)jpegSize, dstPlanes,
                                dstStrides) == -1) {
    SAFE_RELEASE(src, jpegBuf);
    for (i = 0; i < nc; i++)
      SAFE_RELEASE(jDstPlanes[i], dstPlanesTmp[i]);
    THROW_TJ();
  }

bailout:
  SAFE_RELEASE(src, jpegBuf);
  for (i = 0; i < nc; i++)
    SAFE_RELEASE(jDstPlanes[i], dstPlanesTmp[i]);
}

static void TJDecompressor_decodeYUV8
  (JNIEnv *env, jobject obj, jobjectArray srcobjs, jintArray jSrcOffsets,
   jintArray jSrcStrides, jarray dst, jint dstElementSize, jint x, jint y,
   jint width, jint pitch, jint height, jint pf)
{
  tjhandle handle = 0;
  jsize arraySize = 0, actualPitch;
  jbyteArray jSrcPlanes[3] = { NULL, NULL, NULL };
  const unsigned char *srcPlanesTmp[3] = { NULL, NULL, NULL };
  const unsigned char *srcPlanes[3] = { NULL, NULL, NULL };
  jint srcOffsetsTmp[3] = { 0, 0, 0 }, srcStridesTmp[3] = { 0, 0, 0 };
  int srcOffsets[3] = { 0, 0, 0 }, srcStrides[3] = { 0, 0, 0 };
  unsigned char *dstBuf = NULL;
  int nc = 0, i, subsamp;

  GET_HANDLE();

  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF)
    THROW_ARG("Invalid argument in decodeYUV8()");
  if (org_libjpegturbo_turbojpeg_TJ_NUMPF != TJ_NUMPF ||
      org_libjpegturbo_turbojpeg_TJ_NUMSAMP != TJ_NUMSAMP)
    THROW_ARG("Mismatch between Java and C API");

  if ((subsamp = tj3Get(handle, TJPARAM_SUBSAMP)) == TJSAMP_UNKNOWN)
    THROW_ARG("TJPARAM_SUBSAMP must be specified");
  nc = subsamp == TJSAMP_GRAY ? 1 : 3;
  if ((*env)->GetArrayLength(env, srcobjs) < nc)
    THROW_ARG("Planes array is too small for the subsampling type");
  if ((*env)->GetArrayLength(env, jSrcOffsets) < nc)
    THROW_ARG("Offsets array is too small for the subsampling type");
  if ((*env)->GetArrayLength(env, jSrcStrides) < nc)
    THROW_ARG("Strides array is too small for the subsampling type");

  actualPitch = (pitch == 0) ? width * tjPixelSize[pf] : pitch;
  arraySize = (y + height - 1) * actualPitch + (x + width) * tjPixelSize[pf];
  if ((*env)->GetArrayLength(env, dst) * dstElementSize < arraySize)
    THROW_ARG("Destination buffer is not large enough");

  (*env)->GetIntArrayRegion(env, jSrcOffsets, 0, nc, srcOffsetsTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    srcOffsets[i] = srcOffsetsTmp[i];

  (*env)->GetIntArrayRegion(env, jSrcStrides, 0, nc, srcStridesTmp);
  if ((*env)->ExceptionCheck(env)) goto bailout;
  for (i = 0; i < 3; i++)
    srcStrides[i] = srcStridesTmp[i];

  for (i = 0; i < nc; i++) {
    size_t planeSize = tj3YUVPlaneSize(i, width, srcStrides[i], height,
                                       subsamp);
    int pw = tj3YUVPlaneWidth(i, width, subsamp);

    if (planeSize == 0 || pw == 0)
      THROW_ARG(tj3GetErrorStr(NULL));

    if (planeSize > (size_t)INT_MAX)
      THROW_ARG("Source plane is too large");
    if (srcOffsets[i] < 0)
      THROW_ARG("Invalid argument in decodeYUV8()");
    if (srcStrides[i] < 0 && srcOffsets[i] - (int)planeSize + pw < 0)
      THROW_ARG("Negative plane stride would cause memory to be accessed below plane boundary");

    BAILIF0(jSrcPlanes[i] = (*env)->GetObjectArrayElement(env, srcobjs, i));
    if ((*env)->GetArrayLength(env, jSrcPlanes[i]) <
        srcOffsets[i] + (int)planeSize)
      THROW_ARG("Source plane is not large enough");
  }
  for (i = 0; i < nc; i++) {
    BAILIF0NOEC(srcPlanesTmp[i] =
                (*env)->GetPrimitiveArrayCritical(env, jSrcPlanes[i], 0));
    srcPlanes[i] = &srcPlanesTmp[i][srcOffsets[i]];
  }
  BAILIF0NOEC(dstBuf = (*env)->GetPrimitiveArrayCritical(env, dst, 0));

  if (tj3DecodeYUVPlanes8(handle, srcPlanes, srcStrides,
                          &dstBuf[y * actualPitch + x * tjPixelSize[pf]],
                          width, pitch, height, pf) == -1) {
    SAFE_RELEASE(dst, dstBuf);
    for (i = 0; i < nc; i++)
      SAFE_RELEASE(jSrcPlanes[i], srcPlanesTmp[i]);
    THROW_TJ();
  }

bailout:
  SAFE_RELEASE(dst, dstBuf);
  for (i = 0; i < nc; i++)
    SAFE_RELEASE(jSrcPlanes[i], srcPlanesTmp[i]);
}

/* TurboJPEG 3: TJDecompressor::decodeYUV8() byte destination */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decodeYUV8___3_3B_3I_3I_3BIIIIII
  (JNIEnv *env, jobject obj, jobjectArray srcobjs, jintArray jSrcOffsets,
   jintArray jSrcStrides, jbyteArray dst, jint x, jint y, jint width,
   jint pitch, jint height, jint pf)
{
  TJDecompressor_decodeYUV8(env, obj, srcobjs, jSrcOffsets, jSrcStrides, dst,
                            1, x, y, width, pitch, height, pf);
}

/* TurboJPEG 3: TJDecompressor::decodeYUV8() int destination */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_decodeYUV8___3_3B_3I_3I_3IIIIIII
  (JNIEnv *env, jobject obj, jobjectArray srcobjs, jintArray jSrcOffsets,
   jintArray jSrcStrides, jintArray dst, jint x, jint y, jint width,
   jint stride, jint height, jint pf)
{
  if (pf < 0 || pf >= org_libjpegturbo_turbojpeg_TJ_NUMPF)
    THROW_ARG("Invalid argument in decodeYUV8()");
  if (tjPixelSize[pf] != sizeof(jint))
    THROW_ARG("Pixel format must be 32-bit when decoding to an integer buffer.");

  TJDecompressor_decodeYUV8(env, obj, srcobjs, jSrcOffsets, jSrcStrides, dst,
                            sizeof(jint), x, y, width, stride * sizeof(jint),
                            height, pf);

bailout:
  return;
}

/* TurboJPEG 1.2.x: TJTransformer::init() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJTransformer_init
  (JNIEnv *env, jobject obj)
{
  jclass cls;
  jfieldID fid;
  tjhandle handle;

  if ((handle = tj3Init(TJINIT_TRANSFORM)) == NULL)
    THROW(tj3GetErrorStr(NULL), "org/libjpegturbo/turbojpeg/TJException");

  BAILIF0(cls = (*env)->GetObjectClass(env, obj));
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "handle", "J"));
  (*env)->SetLongField(env, obj, fid, (size_t)handle);

bailout:
  return;
}

typedef struct _JNICustomFilterParams {
  JNIEnv *env;
  jobject tobj;
  jobject cfobj;
} JNICustomFilterParams;

static int JNICustomFilter(short *coeffs, tjregion arrayRegion,
                           tjregion planeRegion, int componentIndex,
                           int transformIndex, tjtransform *transform)
{
  JNICustomFilterParams *params = (JNICustomFilterParams *)transform->data;
  JNIEnv *env = params->env;
  jobject tobj = params->tobj, cfobj = params->cfobj;
  jobject arrayRegionObj, planeRegionObj, bufobj, borobj;
  jclass cls;
  jmethodID mid;
  jfieldID fid;

  BAILIF0(bufobj = (*env)->NewDirectByteBuffer(env, coeffs,
    sizeof(short) * arrayRegion.w * arrayRegion.h));
  BAILIF0(cls = (*env)->FindClass(env, "java/nio/ByteOrder"));
  BAILIF0(mid = (*env)->GetStaticMethodID(env, cls, "nativeOrder",
                                          "()Ljava/nio/ByteOrder;"));
  BAILIF0(borobj = (*env)->CallStaticObjectMethod(env, cls, mid));
  BAILIF0(cls = (*env)->GetObjectClass(env, bufobj));
  BAILIF0(mid = (*env)->GetMethodID(env, cls, "order",
    "(Ljava/nio/ByteOrder;)Ljava/nio/ByteBuffer;"));
  (*env)->CallObjectMethod(env, bufobj, mid, borobj);
  BAILIF0(mid = (*env)->GetMethodID(env, cls, "asShortBuffer",
                                    "()Ljava/nio/ShortBuffer;"));
  BAILIF0(bufobj = (*env)->CallObjectMethod(env, bufobj, mid));

  BAILIF0(cls = (*env)->FindClass(env, "java/awt/Rectangle"));
  BAILIF0(arrayRegionObj = (*env)->AllocObject(env, cls));
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "x", "I"));
  (*env)->SetIntField(env, arrayRegionObj, fid, arrayRegion.x);
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "y", "I"));
  (*env)->SetIntField(env, arrayRegionObj, fid, arrayRegion.y);
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "width", "I"));
  (*env)->SetIntField(env, arrayRegionObj, fid, arrayRegion.w);
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "height", "I"));
  (*env)->SetIntField(env, arrayRegionObj, fid, arrayRegion.h);

  BAILIF0(planeRegionObj = (*env)->AllocObject(env, cls));
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "x", "I"));
  (*env)->SetIntField(env, planeRegionObj, fid, planeRegion.x);
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "y", "I"));
  (*env)->SetIntField(env, planeRegionObj, fid, planeRegion.y);
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "width", "I"));
  (*env)->SetIntField(env, planeRegionObj, fid, planeRegion.w);
  BAILIF0(fid = (*env)->GetFieldID(env, cls, "height", "I"));
  (*env)->SetIntField(env, planeRegionObj, fid, planeRegion.h);

  BAILIF0(cls = (*env)->GetObjectClass(env, cfobj));
  BAILIF0(mid = (*env)->GetMethodID(env, cls, "customFilter",
    "(Ljava/nio/ShortBuffer;Ljava/awt/Rectangle;Ljava/awt/Rectangle;IILorg/libjpegturbo/turbojpeg/TJTransform;)V"));
  (*env)->CallVoidMethod(env, cfobj, mid, bufobj, arrayRegionObj,
                         planeRegionObj, componentIndex, transformIndex, tobj);

  return 0;

bailout:
  return -1;
}

/* TurboJPEG 1.2.x: TJTransformer::transform() */
JNIEXPORT jintArray JNICALL Java_org_libjpegturbo_turbojpeg_TJTransformer_transform
  (JNIEnv *env, jobject obj, jbyteArray jsrcBuf, jint jpegSize,
   jobjectArray dstobjs, jobjectArray tobjs)
{
  tjhandle handle = 0;
  unsigned char *jpegBuf = NULL, **dstBufs = NULL;
  jsize n = 0;
  size_t *dstSizes = NULL;
  tjtransform *t = NULL;
  jbyteArray *jdstBufs = NULL;
  int i, jpegWidth = 0, jpegHeight = 0, jpegSubsamp;
  jintArray jdstSizes = 0;
  jint *dstSizesi = NULL;
  JNICustomFilterParams *params = NULL;

  GET_HANDLE();

  if ((*env)->GetArrayLength(env, jsrcBuf) < jpegSize)
    THROW_ARG("Source buffer is not large enough");
  if ((jpegWidth = tj3Get(handle, TJPARAM_JPEGWIDTH)) == -1)
    THROW_ARG("JPEG header has not yet been read");
  if ((jpegHeight = tj3Get(handle, TJPARAM_JPEGHEIGHT)) == -1)
    THROW_ARG("JPEG header has not yet been read");
  if ((jpegSubsamp = tj3Get(handle, TJPARAM_SUBSAMP)) == TJSAMP_UNKNOWN)
    THROW_ARG("TJPARAM_SUBSAMP must be specified");

  n = (*env)->GetArrayLength(env, dstobjs);
  if (n != (*env)->GetArrayLength(env, tobjs))
    THROW_ARG("Mismatch between size of transforms array and destination buffers array");

  if ((dstBufs =
       (unsigned char **)malloc(sizeof(unsigned char *) * n)) == NULL)
    THROW_MEM();
  if ((jdstBufs = (jbyteArray *)malloc(sizeof(jbyteArray) * n)) == NULL)
    THROW_MEM();
  if ((dstSizes = (size_t *)malloc(sizeof(size_t) * n)) == NULL)
    THROW_MEM();
  if ((t = (tjtransform *)malloc(sizeof(tjtransform) * n)) == NULL)
    THROW_MEM();
  if ((params = (JNICustomFilterParams *)malloc(sizeof(JNICustomFilterParams) *
                                                n)) == NULL)
    THROW_MEM();
  for (i = 0; i < n; i++) {
    dstBufs[i] = NULL;  jdstBufs[i] = NULL;  dstSizes[i] = 0;
    memset(&t[i], 0, sizeof(tjtransform));
    memset(&params[i], 0, sizeof(JNICustomFilterParams));
  }

  for (i = 0; i < n; i++) {
    jobject tobj, cfobj;

    BAILIF0(tobj = (*env)->GetObjectArrayElement(env, tobjs, i));
    BAILIF0(_cls = (*env)->GetObjectClass(env, tobj));
    BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "op", "I"));
    t[i].op = (*env)->GetIntField(env, tobj, _fid);
    BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "options", "I"));
    t[i].options = (*env)->GetIntField(env, tobj, _fid);
    BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "x", "I"));
    t[i].r.x = (*env)->GetIntField(env, tobj, _fid);
    BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "y", "I"));
    t[i].r.y = (*env)->GetIntField(env, tobj, _fid);
    BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "width", "I"));
    t[i].r.w = (*env)->GetIntField(env, tobj, _fid);
    BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "height", "I"));
    t[i].r.h = (*env)->GetIntField(env, tobj, _fid);

    BAILIF0(_fid = (*env)->GetFieldID(env, _cls, "cf",
      "Lorg/libjpegturbo/turbojpeg/TJCustomFilter;"));
    cfobj = (*env)->GetObjectField(env, tobj, _fid);
    if (cfobj) {
      params[i].env = env;
      params[i].tobj = tobj;
      params[i].cfobj = cfobj;
      t[i].customFilter = JNICustomFilter;
      t[i].data = (void *)&params[i];
    }
  }

  if (tj3Set(handle, TJPARAM_NOREALLOC, 1) == -1)
    THROW_TJ();

  for (i = 0; i < n; i++) {
    int w = jpegWidth, h = jpegHeight;

    if (t[i].op == TJXOP_TRANSPOSE || t[i].op == TJXOP_TRANSVERSE ||
        t[i].op == TJXOP_ROT90 || t[i].op == TJXOP_ROT270) {
      w = jpegHeight;  h = jpegWidth;
    }
    if (t[i].r.w != 0) w = t[i].r.w;
    if (t[i].r.h != 0) h = t[i].r.h;
    BAILIF0(jdstBufs[i] = (*env)->GetObjectArrayElement(env, dstobjs, i));
    if ((size_t)(*env)->GetArrayLength(env, jdstBufs[i]) <
        tj3JPEGBufSize(w, h, jpegSubsamp))
      THROW_ARG("Destination buffer is not large enough");
  }
  BAILIF0NOEC(jpegBuf = (*env)->GetPrimitiveArrayCritical(env, jsrcBuf, 0));
  for (i = 0; i < n; i++)
    BAILIF0NOEC(dstBufs[i] =
                (*env)->GetPrimitiveArrayCritical(env, jdstBufs[i], 0));

  if (tj3Transform(handle, jpegBuf, jpegSize, n, dstBufs, dstSizes, t) == -1) {
    for (i = 0; i < n; i++)
      SAFE_RELEASE(jdstBufs[i], dstBufs[i]);
    SAFE_RELEASE(jsrcBuf, jpegBuf);
    THROW_TJ();
  }

  for (i = 0; i < n; i++)
    SAFE_RELEASE(jdstBufs[i], dstBufs[i]);
  SAFE_RELEASE(jsrcBuf, jpegBuf);

  jdstSizes = (*env)->NewIntArray(env, n);
  BAILIF0(dstSizesi = (*env)->GetIntArrayElements(env, jdstSizes, 0));
  for (i = 0; i < n; i++) dstSizesi[i] = (int)dstSizes[i];

bailout:
  if (dstSizesi) (*env)->ReleaseIntArrayElements(env, jdstSizes, dstSizesi, 0);
  if (dstBufs) {
    for (i = 0; i < n; i++) {
      if (dstBufs[i] && jdstBufs && jdstBufs[i])
        (*env)->ReleasePrimitiveArrayCritical(env, jdstBufs[i], dstBufs[i], 0);
    }
    free(dstBufs);
  }
  SAFE_RELEASE(jsrcBuf, jpegBuf);
  free(jdstBufs);
  free(dstSizes);
  free(t);
  return jdstSizes;
}

/* TurboJPEG 1.2.x: TJDecompressor::destroy() */
JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_destroy
  (JNIEnv *env, jobject obj)
{
  Java_org_libjpegturbo_turbojpeg_TJCompressor_destroy(env, obj);
}

/* Private image I/O routines (used only by TJBench) */
JNIEXPORT jobject JNICALL Java_org_libjpegturbo_turbojpeg_TJCompressor_loadImage
  (JNIEnv *env, jobject obj, jint precision, jstring jfilename,
   jintArray jwidth, jint align, jintArray jheight, jintArray jpixelFormat)
{
  tjhandle handle = NULL;
  void *dstBuf = NULL, *jdstPtr;
  int width, *warr, height, *harr, pixelFormat, *pfarr, n;
  const char *filename = NULL;
  jboolean isCopy;
  jobject jdstBuf = NULL;

  GET_HANDLE();

  if ((precision != 8 && precision != 12 && precision != 16) ||
      jfilename == NULL || jwidth == NULL ||
      (*env)->GetArrayLength(env, jwidth) < 1 || jheight == NULL ||
      (*env)->GetArrayLength(env, jheight) < 1 || jpixelFormat == NULL ||
      (*env)->GetArrayLength(env, jpixelFormat) < 1)
    THROW_ARG("Invalid argument in loadImage()");

  BAILIF0NOEC(warr = (*env)->GetPrimitiveArrayCritical(env, jwidth, 0));
  width = warr[0];
  (*env)->ReleasePrimitiveArrayCritical(env, jwidth, warr, 0);
  BAILIF0NOEC(harr = (*env)->GetPrimitiveArrayCritical(env, jheight, 0));
  height = harr[0];
  (*env)->ReleasePrimitiveArrayCritical(env, jheight, harr, 0);
  BAILIF0NOEC(pfarr = (*env)->GetPrimitiveArrayCritical(env, jpixelFormat, 0));
  pixelFormat = pfarr[0];
  (*env)->ReleasePrimitiveArrayCritical(env, jpixelFormat, pfarr, 0);
  BAILIF0(filename = (*env)->GetStringUTFChars(env, jfilename, &isCopy));

  if (precision == 8) {
    if ((dstBuf = tj3LoadImage8(handle, filename, &width, align, &height,
                                &pixelFormat)) == NULL)
      THROW_TJ();
  } else if (precision == 12) {
    if ((dstBuf = tj3LoadImage12(handle, filename, &width, align, &height,
                                 &pixelFormat)) == NULL)
      THROW_TJ();
  } else {
    if ((dstBuf = tj3LoadImage16(handle, filename, &width, align, &height,
                                 &pixelFormat)) == NULL)
      THROW_TJ();
  }

  (*env)->ReleaseStringUTFChars(env, jfilename, filename);
  filename = NULL;

  if ((unsigned long long)width * (unsigned long long)height *
      (unsigned long long)tjPixelSize[pixelFormat] >
      (unsigned long long)((unsigned int)-1))
    THROW_ARG("Image is too large");

  BAILIF0NOEC(warr = (*env)->GetPrimitiveArrayCritical(env, jwidth, 0));
  warr[0] = width;
  (*env)->ReleasePrimitiveArrayCritical(env, jwidth, warr, 0);
  BAILIF0NOEC(harr = (*env)->GetPrimitiveArrayCritical(env, jheight, 0));
  harr[0] = height;
  (*env)->ReleasePrimitiveArrayCritical(env, jheight, harr, 0);
  BAILIF0NOEC(pfarr = (*env)->GetPrimitiveArrayCritical(env, jpixelFormat, 0));
  pfarr[0] = pixelFormat;
  (*env)->ReleasePrimitiveArrayCritical(env, jpixelFormat, pfarr, 0);

  n = width * height * tjPixelSize[pixelFormat];
  if (precision == 8)
    jdstBuf = (*env)->NewByteArray(env, n);
  else
    jdstBuf = (*env)->NewShortArray(env, n);
  BAILIF0NOEC(jdstPtr = (*env)->GetPrimitiveArrayCritical(env, jdstBuf, 0));
  memcpy(jdstPtr, dstBuf, n * (precision > 8 ? 2 : 1));
  (*env)->ReleasePrimitiveArrayCritical(env, jdstBuf, jdstPtr, 0);

bailout:
  if (filename) (*env)->ReleaseStringUTFChars(env, jfilename, filename);
  tj3Free(dstBuf);
  return jdstBuf;
}


JNIEXPORT void JNICALL Java_org_libjpegturbo_turbojpeg_TJDecompressor_saveImage
  (JNIEnv *env, jobject obj, jint precision, jstring jfilename,
   jobject jsrcBuf, jint width, jint pitch, jint height, jint pixelFormat)
{
  tjhandle handle = NULL;
  void *srcBuf = NULL, *jsrcPtr;
  const char *filename = NULL;
  int n;
  jboolean isCopy;

  GET_HANDLE();

  if ((precision != 8 && precision != 12 && precision != 16) ||
      jfilename == NULL || jsrcBuf == NULL || width < 1 || height < 1 ||
      pixelFormat < 0 || pixelFormat >= TJ_NUMPF)
    THROW_ARG("Invalid argument in saveImage()");

  if ((unsigned long long)width * (unsigned long long)height *
      (unsigned long long)tjPixelSize[pixelFormat] >
      (unsigned long long)((unsigned int)-1))
    THROW_ARG("Image is too large");
  n = width * height * tjPixelSize[pixelFormat];
  if ((*env)->GetArrayLength(env, jsrcBuf) < n)
    THROW_ARG("Source buffer is not large enough");

  if ((srcBuf = malloc(n * (precision > 8 ? 2 : 1))) == NULL)
    THROW_MEM();

  BAILIF0NOEC(jsrcPtr = (*env)->GetPrimitiveArrayCritical(env, jsrcBuf, 0));
  memcpy(srcBuf, jsrcPtr, n * (precision > 8 ? 2 : 1));
  (*env)->ReleasePrimitiveArrayCritical(env, jsrcBuf, jsrcPtr, 0);
  BAILIF0(filename = (*env)->GetStringUTFChars(env, jfilename, &isCopy));

  if (precision == 8) {
    if (tj3SaveImage8(handle, filename, srcBuf, width, pitch, height,
                      pixelFormat) == -1)
      THROW_TJ();
  } else if (precision == 12) {
    if (tj3SaveImage12(handle, filename, srcBuf, width, pitch, height,
                       pixelFormat) == -1)
      THROW_TJ();
  } else {
    if (tj3SaveImage16(handle, filename, srcBuf, width, pitch, height,
                       pixelFormat) == -1)
      THROW_TJ();
  }

bailout:
  if (filename) (*env)->ReleaseStringUTFChars(env, jfilename, filename);
  free(srcBuf);
}
