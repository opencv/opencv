#include <jni.h>
/*
#include <android/log.h>
#define TEGRA_LOG_TAG "MAT_CPP"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, TEGRA_LOG_TAG, __VA_ARGS__))
*/


#include "opencv2/core/core.hpp"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nType
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return me->type(  );
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nRows
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return me->rows;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nCols
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return me->cols;
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nData
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return (jlong) me->data;
}

JNIEXPORT jboolean JNICALL Java_org_opencv_Mat_nIsEmpty
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return me->empty();
}

JNIEXPORT jboolean JNICALL Java_org_opencv_Mat_nIsCont
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return me->isContinuous();
}

JNIEXPORT jboolean JNICALL Java_org_opencv_Mat_nIsSubmat
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return me->isSubmatrix();
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nSubmat
	(JNIEnv* env, jclass cls, jlong self, jint r1, jint r2, jint c1, jint c2)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    return (jlong) new cv::Mat(*me, cv::Range(r1, r2>0 ? r2 : me->rows), cv::Range(c1, c2>0 ? c2 : me->cols));
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nClone
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    cv::Mat* it = new cv::Mat();
	me->copyTo(*it);
    return (jlong) it;
}

// unlike other nPut()-s this one (with double[]) should convert input values to correct type
#define PUT_ITEM(T, R, C) for(int ch=0; ch<me->channels() && count>0; ch++,count--) *((T*)me->ptr(R, C)+ch) = cv::saturate_cast<T>(*(src+ch))
JNIEXPORT jint JNICALL Java_org_opencv_Mat_nPutD
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jdoubleArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0;  // no native object behind
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

	int rest = ((me->rows - row) * me->cols - col) * me->channels();
	if(count>rest) count = rest;
	int res = count;
	double* values = (double*)env->GetPrimitiveArrayCritical(vals, 0);
	double* src = values;
	int r, c;
	for(c=col; c<me->cols && count>0; c++)
	{
		switch(me->depth()) {
			case CV_8U:  PUT_ITEM(uchar,  row, c); break;
			case CV_8S:  PUT_ITEM(schar,  row, c); break;
			case CV_16U: PUT_ITEM(ushort, row, c); break;
			case CV_16S: PUT_ITEM(short,  row, c); break;
			case CV_32S: PUT_ITEM(int,    row, c); break;
			case CV_32F: PUT_ITEM(float,  row, c); break;
			case CV_64F: PUT_ITEM(double, row, c); break;
		}
		src++;
	}

	for(r=row+1; r<me->rows && count>0; r++)
		for(c=0; c<me->cols && count>0; c++)
		{
			switch(me->depth()) {
				case CV_8U:  PUT_ITEM(uchar,  r, c); break;
				case CV_8S:  PUT_ITEM(schar,  r, c); break;
				case CV_16U: PUT_ITEM(ushort, r, c); break;
				case CV_16S: PUT_ITEM(short,  r, c); break;
				case CV_32S: PUT_ITEM(int,    r, c); break;
				case CV_32F: PUT_ITEM(float,  r, c); break;
				case CV_64F: PUT_ITEM(double, r, c); break;
			}
			src++;
		}

	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}


#ifdef __cplusplus
}
#endif

template<typename T> static int mat_put(cv::Mat* m, int row, int col, int count, char* buff)
{
	if(! m) return 0;
	if(! buff) return 0;

	int rest = ((m->rows - row) * m->cols - col) * m->channels() * sizeof(T);
	if(count>rest) count = rest;
	int res = count;

	if( m->isContinuous() )
	{
		memcpy(m->ptr(row, col), buff, count);
	} else {
		// row by row
		int num = (m->cols - col - 1) * m->channels() * sizeof(T); // 1st partial row
		if(count<num) num = count;
		uchar* data = m->ptr(row++, col);
		while(count>0){
			memcpy(data, buff, num);
			count -= num;
			buff += num;
			num = m->cols * m->channels() * sizeof(T);
			if(count<num) num = count;
			data = m->ptr(row++, 0);
		}
	}
	return res;
}


#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nPutB
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jbyteArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_8U && me->depth() != CV_8S) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_put<char>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nPutS
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jshortArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_8U && me->depth() != CV_8S) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_put<short>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nPutI
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jintArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_8U && me->depth() != CV_8S) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_put<int>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nPutF
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jfloatArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_8U && me->depth() != CV_8S) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_put<float>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}


#ifdef __cplusplus
}
#endif


template<typename T> int mat_get(cv::Mat* m, int row, int col, int count, char* buff)
{
	if(! m) return 0;
	if(! buff) return 0;

	int rest = ((m->rows - row) * m->cols - col) * m->channels() * sizeof(T);
	if(count>rest) count = rest;
	int res = count;

	if( m->isContinuous() )
	{
		memcpy(buff, m->ptr(row, col), count);
	} else {
		// row by row
		int num = (m->cols - col - 1) * m->channels() * sizeof(T); // 1st partial row
		if(count<num) num = count;
		uchar* data = m->ptr(row++, col);
		while(count>0){
			memcpy(buff, data, num);
			count -= num;
			buff += num;
			num = m->cols * m->channels() * sizeof(T);
			if(count<num) num = count;
			data = m->ptr(row++, 0);
		}
	}
	return res;
}

#ifdef __cplusplus
extern "C" {
#endif


JNIEXPORT jint JNICALL Java_org_opencv_Mat_nGetB
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jbyteArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_8U && me->depth() != CV_8S) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_get<char>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nGetS
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jshortArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_16U && me->depth() != CV_16S) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_get<short>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nGetI
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jintArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_32S) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_get<int>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nGetF
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jfloatArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_32F) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_get<float>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jint JNICALL Java_org_opencv_Mat_nGetD
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count, jdoubleArray vals)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->depth() != CV_64F) return 0; // incompatible type
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range
	
	char* values = (char*)env->GetPrimitiveArrayCritical(vals, 0);
	int res = mat_get<double>(me, row, col, count, values);
	env->ReleasePrimitiveArrayCritical(vals, values, 0);
	return res;
}

JNIEXPORT jdoubleArray JNICALL Java_org_opencv_Mat_nGet
	(JNIEnv* env, jclass cls, jlong self, jint row, jint col, jint count)
{
	cv::Mat* me = (cv::Mat*) self;
	if(! self) return 0; // no native object behind
	if(me->rows<=row || me->cols<=col) return 0; // indexes out of range

	jdoubleArray res = env->NewDoubleArray(me->channels());
	if(res){
		jdouble buff[me->channels()];
		int i;
		switch(me->depth()){
			case CV_8U:  for(i=0; i<me->channels(); i++) buff[i] = *((unsigned char*) me->ptr(row, col) + i); break;
			case CV_8S:  for(i=0; i<me->channels(); i++) buff[i] = *((signed char*)   me->ptr(row, col) + i); break;
			case CV_16U: for(i=0; i<me->channels(); i++) buff[i] = *((unsigned short*)me->ptr(row, col) + i); break;
			case CV_16S: for(i=0; i<me->channels(); i++) buff[i] = *((signed short*)  me->ptr(row, col) + i); break;
			case CV_32S: for(i=0; i<me->channels(); i++) buff[i] = *((int*)           me->ptr(row, col) + i); break;
			case CV_32F: for(i=0; i<me->channels(); i++) buff[i] = *((float*)         me->ptr(row, col) + i); break;
			case CV_64F: for(i=0; i<me->channels(); i++) buff[i] = *((double*)        me->ptr(row, col) + i); break;
		}
		env->SetDoubleArrayRegion(res, 0, me->channels(), buff);
	}
	return res;
}

JNIEXPORT void JNICALL Java_org_opencv_Mat_nSetTo
  (JNIEnv* env, jclass cls, jlong self, jdouble v0, jdouble v1, jdouble v2, jdouble v3)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    me->setTo( cv::Scalar(v0, v1, v2, v3) );
}

JNIEXPORT void JNICALL Java_org_opencv_Mat_nCopyTo
  (JNIEnv* env, jclass cls, jlong self, jlong m)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    cv::Mat* _m = (cv::Mat*) m; //TODO: check for NULL
    me->copyTo( *_m );
}

JNIEXPORT jdouble JNICALL Java_org_opencv_Mat_nDot
  (JNIEnv* env, jclass cls, jlong self, jlong m)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    cv::Mat* _m = (cv::Mat*) m; //TODO: check for NULL
    return me->dot( *_m );
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nCross
	(JNIEnv* env, jclass cls, jlong self, jlong m)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    cv::Mat* _m = (cv::Mat*) m; //TODO: check for NULL
    return (jlong) new cv::Mat(me->cross( *_m ));
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nInv
	(JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL    
	return (jlong) new cv::Mat(me->inv());
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nCreateMat__
  (JNIEnv* env, jclass cls)
{
    return (jlong) new cv::Mat();
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nEye
    (JNIEnv* env, jclass cls, jint _rows, jint _cols, jint _type)
{
    return (jlong) new cv::Mat(cv::Mat::eye( _rows, _cols, _type ));
}

JNIEXPORT jstring JNICALL Java_org_opencv_Mat_nDump
  (JNIEnv *env, jclass cls, jlong self)
{
	cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    std::stringstream s;
    s << *me;
	return env->NewStringUTF(s.str().c_str());
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nCreateMat__III
  (JNIEnv* env, jclass cls, jint _rows, jint _cols, jint _type)
{
    //LOGD("called with r=%d, c=%d", _rows, _cols);
    return (jlong) new cv::Mat( _rows, _cols, _type );;
}

JNIEXPORT jlong JNICALL Java_org_opencv_Mat_nCreateMat__IIIDDDD
  (JNIEnv* env, jclass cls, jint _rows, jint _cols, jint _type, jdouble v0, jdouble v1, jdouble v2, jdouble v3)
{
    return (jlong) new cv::Mat( _rows, _cols, _type, cv::Scalar(v0, v1, v2, v3) );
}

JNIEXPORT void JNICALL Java_org_opencv_Mat_nDelete
  (JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    delete me;
}

JNIEXPORT void JNICALL Java_org_opencv_Mat_nRelease
  (JNIEnv* env, jclass cls, jlong self)
{
    cv::Mat* me = (cv::Mat*) self; //TODO: check for NULL
    me->release();
}

#ifdef __cplusplus
}
#endif
