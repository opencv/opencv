#include "octhelpers.h"
#include <iostream>
#include <sstream>

int OctSwigObject_Check(octave_value op);

octave_value OctTuple_FromIntArray(int * arr, int len){
  octave_value obj = OctTuple_New(len);
  for(int i=0; i<len; i++){
    OctTuple_SetItem(obj, i, OctLong_FromLong( arr[i] ) );
  }
  return obj;
}

octave_value SWIG_SetResult(octave_value result, octave_value obj){
  result = OctTuple_New(1);
  OctTuple_SetItem(result, 0, obj);
  return result;
}

octave_value_list* SWIG_AppendResult(octave_value_list* result, octave_value* to_add, int num){
  for (int j=0;j<num;++j)
    result->append(to_add[j]);
  return result;
}

template <typename T>
std::ostream & cv_arr_write(std::ostream & out, T * data, int rows, int nch, int step){
  int i,j,k;
  char * cdata = (char *) data;
  std::string chdelim1="", chdelim2="";

  // only output channel parens if > 1
  if(nch>1){
    chdelim1="(";
    chdelim2=")";
  }

  out<<"[\n";
  for(i=0; i<rows; i++){
    out<<"[";

    // first element
    out<<chdelim1;
    out<<double(((T*)(cdata+i*step))[0]);
    for(k=1; k<nch; k++){
      out<<", "<<double(((T*)(cdata+i*step))[k]);
    }
    out<<chdelim2;

    // remaining elements
    for(j=nch*sizeof(T); j<step; j+=(nch*sizeof(T))){
      out<<", "<<chdelim1;
      out<<double(((T*)(cdata+i*step+j))[0]);
      for(k=1; k<nch; k++){
	out<<", "<<double(((T*)(cdata+i*step+j))[k]);
      }
      out<<chdelim2;
    }
    out<<"]\n";
  }
  out<<"]";
  return out;
}

void cvArrPrint(CvArr * arr){
  CV_FUNCNAME( "cvArrPrint" );
	    
  __BEGIN__;
  CvMat * mat;
  CvMat stub;

  mat = cvGetMat(arr, &stub);

  int cn = CV_MAT_CN(mat->type);
  int depth = CV_MAT_DEPTH(mat->type);
  int step = MAX(mat->step, cn*mat->cols*CV_ELEM_SIZE(depth));
  std::ostringstream str;

  switch(depth){
  case CV_8U:
    cv_arr_write(str, (uchar *)mat->data.ptr, mat->rows, cn, step);
    break;
  case CV_8S:
    cv_arr_write(str, (char *)mat->data.ptr, mat->rows, cn, step);
    break;
  case CV_16U:
    cv_arr_write(str, (ushort *)mat->data.ptr, mat->rows, cn, step);
    break;
  case CV_16S:
    cv_arr_write(str, (short *)mat->data.ptr, mat->rows, cn, step);
    break;
  case CV_32S:
    cv_arr_write(str, (int *)mat->data.ptr, mat->rows, cn, step);
    break;
  case CV_32F:
    cv_arr_write(str, (float *)mat->data.ptr, mat->rows, cn, step);
    break;
  case CV_64F:
    cv_arr_write(str, (double *)mat->data.ptr, mat->rows, cn, step);
    break;
  default:
    CV_ERROR( CV_StsError, "Unknown element type");
    break;
  }
  std::cout<<str.str()<<std::endl;

  __END__;
}

// deal with negative array indices
int OctLong_AsIndex( octave_value idx_object, int len ){
  int idx = OctLong_AsLong( idx_object );
  if(idx<0) return len+idx;
  return idx;
}

CvRect OctSlice_to_CvRect(CvArr * src, octave_value idx_object){
  CvSize sz = cvGetSize(src);
  //printf("Size %dx%d\n", sz.height, sz.width);
  int lower[2], upper[2];
  Oct_ssize_t len, start, stop, step, slicelength;

  if(OctInt_Check(idx_object) || OctLong_Check(idx_object)){
    // if array is a row vector, assume index into columns
    if(sz.height>1){
      lower[0] = OctLong_AsIndex( idx_object, sz.height );
      upper[0] = lower[0] + 1;
      lower[1] = 0;
      upper[1] = sz.width;
    }
    else{
      lower[0] = 0;
      upper[0] = sz.height;
      lower[1] = OctLong_AsIndex( idx_object, sz.width );
      upper[1] = lower[1]+1;
    }
  }

  // 1. Slice
  else if(OctSlice_Check(idx_object)){
    len = sz.height;
    if(OctSlice_GetIndicesEx( idx_object, len, &start, &stop, &step, &slicelength )!=0){
      error("Error in OctSlice_GetIndicesEx: returning NULL");
      return cvRect(0,0,0,0);
    }
    // if array is a row vector, assume index bounds are into columns
    if(sz.height>1){
      lower[0] = (int) start; // use c convention of start index = 0
      upper[0] = (int) stop;    // use c convention
      lower[1] = 0;
      upper[1] = sz.width;
    }
    else{
      lower[1] = (int) start; // use c convention of start index = 0
      upper[1] = (int) stop;    // use c convention
      lower[0] = 0;
      upper[0] = sz.height;
    }
  }

  // 2. Tuple
  else if(OctTuple_Check(idx_object)){
    //printf("OctTuple{\n");
    if(OctObject_Length(idx_object)!=2){
      error("Expected a sequence with 2 elements");
      return cvRect(0,0,0,0);
    }
    for(int i=0; i<2; i++){
      octave_value o = OctTuple_GetItem(idx_object, i);

      // 2a. Slice -- same as above
      if(OctSlice_Check(o)){
	//printf("OctSlice\n");
	len = (i==0 ? sz.height : sz.width);
	if(OctSlice_GetIndicesEx(o, len, &start, &stop, &step, &slicelength )!=0){
	  error("Error in OctSlice_GetIndicesEx: returning NULL");
	  return cvRect(0,0,0,0);
	}
	//printf("OctSlice_GetIndecesEx(%d, %d, %d, %d, %d)\n", len, start, stop, step, slicelength);
	lower[i] = start;
	upper[i] = stop;

      }

      // 2b. Integer
      else if(OctInt_Check(o) || OctLong_Check(o)){
	//printf("OctInt\n");
	lower[i] = OctLong_AsIndex(o, i==0 ? sz.height : sz.width);
	upper[i] = lower[i]+1;
      }

      else {
	error("Expected a slice or int as sequence item: returning NULL");
	return cvRect(0,0,0,0);
      }
    }
  }

  else {
    error("Expected a slice or sequence: returning NULL");
    return cvRect(0,0,0,0);
  }

  return cvRect(lower[1], lower[0], upper[1]-lower[1], upper[0]-lower[0]);
}

double OctObject_AsDouble(octave_value obj){
  if(OctNumber_Check(obj)){
    if(OctFloat_Check(obj)){
      return OctFloat_AsDouble(obj);
    }
    else if(OctInt_Check(obj) || OctLong_Check(obj)){
      return (double) OctLong_AsLong(obj);
    }
  }
  error("Could not convert octave object to Double");
  return -1;
}

long OctObject_AsLong(octave_value obj){
  if(OctNumber_Check(obj)){
    if(OctFloat_Check(obj)){
      return (long) OctFloat_AsDouble(obj);
    }
    else if(OctInt_Check(obj) || OctLong_Check(obj)){
      return OctLong_AsLong(obj);
    }
  }
  error("Could not convert octave object to Long");
  return -1;
}

// standard python container routines, adapted to octave

// * should matrix conversions happen here or at typemap layer? or both

bool OctNumber_Check(const octave_value& ov) {
  return ov.is_scalar_type();
}

octave_value OctBool_FromLong (long v) {
  return !!v;
}

bool OctInt_Check(const octave_value& ov) {
  return ov.is_integer_type();
}

long OctInt_AsLong (const octave_value& ov) {
  return ov.long_value();
}

octave_value OctInt_FromLong (long v) {
  return v;
}

bool OctLong_Check(const octave_value& ov) {
  return ov.is_scalar_type();
}

long OctLong_AsLong(const octave_value& ov) {
  return ov.long_value();
}

octave_value OctLong_FromLong(long v) {
  return v;
}

octave_value OctLong_FromUnsignedLong(unsigned long v) {
  return v;
}

bool OctFloat_Check(const octave_value& ov) {
  return ov.is_scalar_type();
}

octave_value OctFloat_FromDouble(double v) {
  return v;
}

double OctFloat_AsDouble (const octave_value& ov) {
  return ov.scalar_value();
}

octave_value OctSequence_New(int n) {
  return n ? Cell(1,n) : Cell(dim_vector(0,0));
}

bool OctSequence_Check(const octave_value& ov) {
  return ov.is_cell();
}

int OctSequence_Size(const octave_value& ov) {
  Cell c(ov.cell_value());
  return ov.cell_value().numel();
}

octave_value OctSequence_GetItem(const octave_value& ov,int i) {
  Cell c(ov.cell_value());
  if (i<0||i>=c.numel()) {
    error("index out of bounds");
    return octave_value();
  }
  return c(i);
}

void OctSequence_SetItem(octave_value& ov,int i,const octave_value& v) {
  Cell c(ov.cell_value());
  if (i<0||i>=c.numel())
    error("index out of bounds");
  else {
    c(i)=v;
    ov=c;
  }
}

octave_value OctTuple_New(int n) {
  return OctSequence_New(n);
}

bool OctTuple_Check(const octave_value& ov) {
  return OctSequence_Check(ov);
}

int OctTuple_Size(const octave_value& ov) {
  return OctSequence_Size(ov);
}

void OctTuple_SetItem(octave_value& ov,int i,const octave_value& v) {
  OctSequence_SetItem(ov,i,v);
}

octave_value OctTuple_GetItem(const octave_value& ov,int i) {
  return OctSequence_GetItem(ov,i);
}

octave_value OctList_New(int n) {
  return OctSequence_New(n);
}

bool OctList_Check(const octave_value& ov) {
  return OctSequence_Check(ov);
}

int OctList_Size(const octave_value& ov) {
  return OctSequence_Size(ov);
}

void OctList_SetItem(octave_value& ov,int i,const octave_value& v) {
  OctSequence_SetItem(ov,i,v);
}

octave_value OctList_GetItem(const octave_value& ov,int i) {
  return OctSequence_GetItem(ov,i);
}

bool OctSlice_Check(const octave_value& ov) {
  return false; // todo have these map to range and magic-colon types
}

int OctObject_Length(const octave_value& ov) {
  return 0;
}

bool OctSlice_GetIndicesEx(const octave_value& ov, Oct_ssize_t len, Oct_ssize_t* start, Oct_ssize_t* stop, Oct_ssize_t* step, Oct_ssize_t* slicelength ) {
  return false;
}
