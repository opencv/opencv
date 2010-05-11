/* These functions need the SWIG_* functions defined in the wrapper */
%{


  static inline bool OctSwigObject_Check(const octave_value& ov) {
    return ov.type_id()==octave_swig_ref::static_type_id();
  }

  static CvArr * OctObject_to_CvArr(octave_value obj, bool * freearg);
  static CvArr * OctSequence_to_CvArr( octave_value obj );

  // convert a octave sequence/array/list object into a c-array
#define OctObject_AsArrayImpl(func, ctype, ptype)                              \
 int func(octave_value obj, ctype * array, int len){                         \
   void * mat_vptr=NULL;                                                     \
   void * im_vptr=NULL;                                                      \
   if(OctNumber_Check(obj)){                                                  \
     memset( array, 0, sizeof(ctype)*len );                                \
     array[0] = OctObject_As##ptype( obj );                                 \
   }                                                                         \
   else if(OctList_Check(obj) || OctTuple_Check(obj)){                         \
     int seqsize = OctSequence_Size(obj);                                   \
     for(int i=0; i<len && i<seqsize; i++){                                \
       if(i<seqsize){                                                    \
	 array[i] =  OctObject_As##ptype( OctSequence_GetItem(obj, i) ); \
       }                                                                 \
       else{                                                             \
	 array[i] = 0;                                                 \
       }                                                                 \
     }                                                                     \
   }                                                                         \
   else if( SWIG_ConvertPtr(obj, &mat_vptr, SWIGTYPE_p_CvMat, 0)!=-1 ||      \
	    SWIG_ConvertPtr(obj, &im_vptr, SWIGTYPE_p__IplImage, 0)!=-1)     \
     {                                                                         \
       CvMat * mat = (CvMat *) mat_vptr;                                     \
       CvMat stub;                                                           \
       if(im_vptr) mat = cvGetMat(im_vptr, &stub);                           \
       if( mat->rows!=1 && mat->cols!=1 ){                                   \
	 error("OctObject_As*Array: CvArr must be row or column vector" );   \
	 return -1;                                                        \
       }                                                                     \
       if( mat->rows==1 && mat->cols==1 ){                                   \
	 CvScalar val;                                                     \
	 if( len!=CV_MAT_CN(mat->type) ){                                  \
	   error("OctObject_As*Array: CvArr channels != length" );              \
	   return -1;                                                    \
	 }                                                                 \
	 val = cvGet1D(mat, 0);                                            \
	 for(int i=0; i<len; i++){                                         \
	   array[i] = (ctype) val.val[i];                                \
	 }                                                                 \
       }                                                                     \
       else{                                                                 \
	 mat = cvReshape(mat, &stub, -1, mat->rows*mat->cols);             \
	 if( mat->rows != len ){                                           \
	   error("OctObject_As*Array: CvArr rows or cols must equal length" ); \
	   return -1;                                                   \
	 }                                                                 \
	 for(int i=0; i<len; i++){                                         \
	   CvScalar val = cvGet1D(mat, i);                               \
	   array[i] = (ctype) val.val[0];                                \
	 }                                                                 \
       }                                                                     \
     }                                                                         \
   else{                                                                     \
     error("OctObject_As*Array: Expected a number, sequence or CvArr" );  \
     return -1;                                                            \
   }                                                                         \
   return 0;                                                                 \
 }

  OctObject_AsArrayImpl( OctObject_AsFloatArray, float, Double );
  OctObject_AsArrayImpl( OctObject_AsDoubleArray, double, Double );
  OctObject_AsArrayImpl( OctObject_AsLongArray, int, Long );

  static CvPoint OctObject_to_CvPoint(octave_value obj){
    CvPoint val;
    CvPoint *ptr;
    CvPoint2D32f * ptr2D32f;
    CvScalar * scalar;

    if( SWIG_ConvertPtr(obj, (void**)&ptr, SWIGTYPE_p_CvPoint, 0) != -1) {
      return *ptr;
    }
    if( SWIG_ConvertPtr(obj, (void**)&ptr2D32f, SWIGTYPE_p_CvPoint2D32f, 0) != -1) {
      return cvPointFrom32f( *ptr2D32f );
    }
    if( SWIG_ConvertPtr(obj, (void**)&scalar, SWIGTYPE_p_CvScalar, 0) != -1) {
      return cvPointFrom32f(cvPoint2D32f( scalar->val[0], scalar->val[1] ));
    }
    if(OctObject_AsLongArray(obj, (int *) &val, 2) != -1){
      return val;
    }

    error("could not convert to CvPoint");
    return cvPoint(0,0);
  }

  static CvPoint2D32f OctObject_to_CvPoint2D32f(octave_value obj){
    CvPoint2D32f val;
    CvPoint2D32f *ptr2D32f;
    CvPoint *ptr;
    CvScalar * scalar;
    if( SWIG_ConvertPtr(obj, (void**)&ptr2D32f, SWIGTYPE_p_CvPoint2D32f, 0) != -1) {
      return *ptr2D32f;
    }
    if( SWIG_ConvertPtr(obj, (void**)&ptr, SWIGTYPE_p_CvPoint, 0) != -1) {
      return cvPointTo32f(*ptr);
    }
    if( SWIG_ConvertPtr(obj, (void**)&scalar, SWIGTYPE_p_CvScalar, 0) != -1) {
      return cvPoint2D32f( scalar->val[0], scalar->val[1] );
    }
    if(OctObject_AsFloatArray(obj, (float *) &val, 2) != -1){
      return val;
    }
    error("could not convert to CvPoint2D32f");
    return cvPoint2D32f(0,0);
  }

  static CvScalar OctObject_to_CvScalar(octave_value obj){
    CvScalar val;
    CvScalar * ptr;
    CvPoint2D32f *ptr2D32f;
    CvPoint *pt_ptr;
    void * vptr;
    if( SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvScalar, 0 ) != -1)
      {
	ptr = (CvScalar *) vptr;
	return *ptr;
      }
    if( SWIG_ConvertPtr(obj, (void**)&ptr2D32f, SWIGTYPE_p_CvPoint2D32f, 0) != -1) {
      return cvScalar(ptr2D32f->x, ptr2D32f->y);
    }
    if( SWIG_ConvertPtr(obj, (void**)&pt_ptr, SWIGTYPE_p_CvPoint, 0) != -1) {
      return cvScalar(pt_ptr->x, pt_ptr->y);
    }
    if(OctObject_AsDoubleArray(obj, val.val, 4)!=-1){
      return val;
    }
    return cvScalar(-1,-1,-1,-1); 
  }

  // if octave sequence type, convert to CvMat or CvMatND
  static CvArr * OctObject_to_CvArr(octave_value obj, bool * freearg){
    CvArr * cvarr;
    *freearg = false;

    // check if OpenCV type
    if ( OctSwigObject_Check(obj) ){
      SWIG_ConvertPtr(obj, &cvarr, 0, SWIG_POINTER_EXCEPTION);
    }
    else if (OctList_Check(obj) || OctTuple_Check(obj)){
      cvarr = OctSequence_to_CvArr( obj );
      *freearg = (cvarr != NULL);
    }
    else if (OctLong_Check(obj) && OctLong_AsLong(obj)==0){
      return NULL;
    }
    else {
      SWIG_ConvertPtr(obj, (void**)&cvarr, 0, SWIG_POINTER_EXCEPTION);
    }
    return cvarr;
  }

  static int OctObject_GetElemType(octave_value obj){
    void *vptr;
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint, 0) != -1) return CV_32SC2;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvSize, 0) != -1) return CV_32SC2;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvRect, 0) != -1) return CV_32SC4;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvSize2D32f, 0) != -1) return CV_32FC2;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint2D32f, 0) != -1) return CV_32FC2;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint3D32f, 0) != -1) return CV_32FC3;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint2D64f, 0) != -1) return CV_64FC2;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvPoint3D64f, 0) != -1) return CV_64FC3;	
    if(SWIG_ConvertPtr(obj, &vptr, SWIGTYPE_p_CvScalar, 0) != -1) return CV_64FC4;	
    if(OctTuple_Check(obj) || OctList_Check(obj)) return CV_MAKE_TYPE(CV_32F, OctSequence_Size( obj ));
    if(OctLong_Check(obj)) return CV_32S;
    return CV_32F;
  }

  // Would like this to convert Octave lists to CvMat
  // Also lists of CvPoints, CvScalars, CvMats? etc
  static CvArr * OctSequence_to_CvArr( octave_value obj ){
    int dims[CV_MAX_DIM] = {1,1,1};
    int ndim=0;
    int cvtype;
    octave_value item;

    // figure out dimensions
    for(item = obj; 
	(OctTuple_Check(item) || OctList_Check(item));
	item = OctSequence_GetItem(item, 0))
      {
	dims[ndim] = OctSequence_Size( item ); 
	ndim++;
      }

    if(ndim==0){
      error("Cannot convert an empty octave object to a CvArr");
      return NULL;
    }

    cvtype = OctObject_GetElemType(item);
    // collapse last dim into NCH if we found a single channel, but the last dim is <=3
    if(CV_MAT_CN(cvtype)==1 && dims[ndim-1]>1 && dims[ndim-1]<4){
      cvtype=CV_MAKE_TYPE(cvtype, dims[ndim-1]);
      dims[ndim-1]=1;	
      ndim--;
    }

    if(cvtype==-1){
      error("Could not determine OpenCV element type of Octave sequence");
      return NULL;
    }

    // CvMat
    if(ndim<=2){
      CvMat *m = cvCreateMat(dims[0], dims[1], cvtype);
      for(int i=0; i<dims[0]; i++){
	octave_value rowobj = OctSequence_GetItem(obj, i);
	if( dims[1] > 1 ){
	  // double check size
	  assert((OctTuple_Check(rowobj) || OctList_Check(rowobj)) && 
		 OctSequence_Size(rowobj) == dims[1]);

	  for(int j=0; j<dims[1]; j++){
	    octave_value colobj = OctSequence_GetItem(rowobj, j);
	    cvSet2D( m, i, j, OctObject_to_CvScalar( colobj ) );
	  }
	}
	else{
	  cvSet1D(m, i, OctObject_to_CvScalar( rowobj ) );
	}
      }
      return (CvArr *) m;
    }

    // CvMatND
    error("Cannot convert Octave Object to CvArr -- ndim > 3");
    return NULL;
  }
%}
