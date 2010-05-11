#ifndef OCTHELPERS_H
#define OCTHELPERS_H

#include <octave/oct.h>
#include <octave/Cell.h>

// SWIG sets an include on 'tcl.h' without giving the path, which fails on Ubuntu
#if   defined HAVE_TCL_H
  #include <tcl.h>
#elif defined HAVE_TCL_TCL_H
  #include <tcl/tcl.h>
#endif

#include <cxcore.h>
#include <cv.h>

typedef unsigned int Oct_ssize_t;

// convert octave index object (tuple, integer, or slice) to CvRect for subsequent cvGetSubMat call
CvRect OctSlice_to_CvRect(CvArr * src, octave_value idx_object);

// prints array to stdout 
//  TODO: octave __str returns a string, so this should write to a string 
//
void cvArrPrint( CvArr * mat );

// Convert an integer array to octave tuple
octave_value OctTuple_FromIntArray(int * arr, int len);
	
// If result is not NULL or OctNone, release object and replace it with obj
octave_value SWIG_SetResult(octave_value result, octave_value obj);
	
// helper function to append one or more objects to the swig $result array
octave_value_list* SWIG_AppendResult(octave_value_list* result, octave_value* to_add, int num);

int OctObject_AsDoubleArray(octave_value obj, double * array, int len);
int OctObject_AsLongArray(octave_value obj, int * array, int len);
int OctObject_AsFloatArray(octave_value obj, float * array, int len);

// helper function to convert octave scalar or sequence to int, float or double arrays
double OctObject_AsDouble(octave_value obj);
long OctObject_AsLong(octave_value obj);

// standard python container routines, adapted to octave
bool OctNumber_Check(const octave_value& ov);
octave_value OctBool_FromLong (long v);
bool OctInt_Check(const octave_value& ov);
long OctInt_AsLong (const octave_value& ov);
octave_value OctInt_FromLong (long v);
bool OctLong_Check(const octave_value& ov);
long OctLong_AsLong(const octave_value& ov);
octave_value OctLong_FromLong(long v);
octave_value OctLong_FromUnsignedLong(unsigned long v);
bool OctFloat_Check(const octave_value& ov);
octave_value OctFloat_FromDouble(double v);
double OctFloat_AsDouble (const octave_value& ov);

octave_value OctSequence_New(int n);
bool OctSequence_Check(const octave_value& ov);
int OctSequence_Size(const octave_value& ov);
octave_value OctSequence_GetItem(const octave_value& ov,int i);
void OctSequence_SetItem(octave_value& ov,int i,const octave_value& v);

octave_value OctTuple_New(int n);
bool OctTuple_Check(const octave_value& ov);
int OctTuple_Size(const octave_value& ov);
void OctTuple_SetItem(octave_value& c,int i,const octave_value& ov);
octave_value OctTuple_GetItem(const octave_value& c,int i);

octave_value OctList_New(int n);
bool OctList_Check(const octave_value& ov);
int OctList_Size(const octave_value& ov);
void OctList_SetItem(octave_value& ov,int i,const octave_value& ov);
octave_value OctList_GetItem(const octave_value& ov,int i);

bool OctSlice_Check(const octave_value& ov);

int OctObject_Length(const octave_value& ov);
bool OctSlice_GetIndicesEx(const octave_value& ov, Oct_ssize_t len, Oct_ssize_t* start, Oct_ssize_t* stop, Oct_ssize_t* step, Oct_ssize_t* slicelength );

#endif //OCTHELPERS_H
