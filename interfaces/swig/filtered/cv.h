# 1 "../../../include/opencv/cv.h"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "../../../include/opencv/cv.h"
# 58 "../../../include/opencv/cv.h"
# 1 "../../../include/opencv/cxcore.h" 1
# 70 "../../../include/opencv/cxcore.h"
# 1 "../../../include/opencv/cxtypes.h" 1
# 161 "../../../include/opencv/cxtypes.h"
typedef int64_t int64;
typedef uint64_t uint64;



typedef unsigned char uchar;
typedef unsigned short ushort;


typedef signed char schar;






typedef void CvArr;

typedef union Cv32suf
{
    int i;
    unsigned u;
    float f;
}
Cv32suf;

typedef union Cv64suf
{
    int64 i;
    uint64 u;
    double f;
}
Cv64suf;
# 226 "../../../include/opencv/cxtypes.h"
inline int cvRound( double value )
{
# 240 "../../../include/opencv/cxtypes.h"
    return (int)lrint(value);




}


inline int cvFloor( double value )
{

    int i = (int)value;
    return i - (i > value);
# 263 "../../../include/opencv/cxtypes.h"
}


inline int cvCeil( double value )
{

    int i = (int)value;
    return i + (i < value);
# 281 "../../../include/opencv/cxtypes.h"
}




inline int cvIsNaN( double value )
{





    Cv64suf ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) +
           ((unsigned)ieee754.u != 0) > 0x7ff00000;

}


inline int cvIsInf( double value )
{





    Cv64suf ieee754;
    ieee754.f = value;
    return ((unsigned)(ieee754.u >> 32) & 0x7fffffff) == 0x7ff00000 &&
           (unsigned)ieee754.u == 0;

}




typedef uint64 CvRNG;

inline CvRNG cvRNG( int64 seed = -1)
{
    CvRNG rng = seed ? (uint64)seed : (uint64)(int64)-1;
    return rng;
}


inline unsigned cvRandInt( CvRNG* rng )
{
    uint64 temp = *rng;
    temp = (uint64)(unsigned)temp*4164903690U + (temp >> 32);
    *rng = temp;
    return (unsigned)temp;
}


inline double cvRandReal( CvRNG* rng )
{
    return cvRandInt(rng)*2.3283064365386962890625e-10 ;
}
# 382 "../../../include/opencv/cxtypes.h"
typedef struct _IplImage
{
    int nSize;
    int ID;
    int nChannels;
    int alphaChannel;
    int depth;

    char colorModel[4];
    char channelSeq[4];
    int dataOrder;

    int origin;

    int align;

    int width;
    int height;
    struct _IplROI *roi;
    struct _IplImage *maskROI;
    void *imageId;
    struct _IplTileInfo *tileInfo;
    int imageSize;


    char *imageData;
    int widthStep;
    int BorderMode[4];
    int BorderConst[4];
    char *imageDataOrigin;


}
IplImage;

typedef struct _IplTileInfo IplTileInfo;

typedef struct _IplROI
{
    int coi;
    int xOffset;
    int yOffset;
    int width;
    int height;
}
IplROI;

typedef struct _IplConvKernel
{
    int nCols;
    int nRows;
    int anchorX;
    int anchorY;
    int *values;
    int nShiftR;
}
IplConvKernel;

typedef struct _IplConvKernelFP
{
    int nCols;
    int nRows;
    int anchorX;
    int anchorY;
    float *values;
}
IplConvKernelFP;
# 561 "../../../include/opencv/cxtypes.h"
typedef struct CvMat
{
    int type;
    int step;


    int* refcount;
    int hdr_refcount;

    union
    {
        uchar* ptr;
        short* s;
        int* i;
        float* fl;
        double* db;
    } data;


    union
    {
        int rows;
        int height;
    };

    union
    {
        int cols;
        int width;
    };





}
CvMat;
# 639 "../../../include/opencv/cxtypes.h"
inline CvMat cvMat( int rows, int cols, int type, void* data = NULL)
{
    CvMat m;

    assert( (unsigned)((type) & ((1 << 3) - 1)) <= 6 );
    type = ((type) & ((1 << 3)*64 - 1));
    m.type = 0x42420000 | (1 << 14) | type;
    m.cols = cols;
    m.rows = rows;
    m.step = m.cols*(((((type) & ((64 - 1) << 3)) >> 3) + 1) << ((((sizeof(size_t)/4+1)*16384|0x3a50) >> ((type) & ((1 << 3) - 1))*2) & 3));
    m.data.ptr = (uchar*)data;
    m.refcount = NULL;
    m.hdr_refcount = 0;

    return m;
}
# 669 "../../../include/opencv/cxtypes.h"
inline double cvmGet( const CvMat* mat, int row, int col )
{
    int type;

    type = ((mat->type) & ((1 << 3)*64 - 1));
    assert( (unsigned)row < (unsigned)mat->rows &&
            (unsigned)col < (unsigned)mat->cols );

    if( type == (((5) & ((1 << 3) - 1)) + (((1)-1) << 3)) )
        return ((float*)(mat->data.ptr + (size_t)mat->step*row))[col];
    else
    {
        assert( type == (((6) & ((1 << 3) - 1)) + (((1)-1) << 3)) );
        return ((double*)(mat->data.ptr + (size_t)mat->step*row))[col];
    }
}


inline void cvmSet( CvMat* mat, int row, int col, double value )
{
    int type;
    type = ((mat->type) & ((1 << 3)*64 - 1));
    assert( (unsigned)row < (unsigned)mat->rows &&
            (unsigned)col < (unsigned)mat->cols );

    if( type == (((5) & ((1 << 3) - 1)) + (((1)-1) << 3)) )
        ((float*)(mat->data.ptr + (size_t)mat->step*row))[col] = (float)value;
    else
    {
        assert( type == (((6) & ((1 << 3) - 1)) + (((1)-1) << 3)) );
        ((double*)(mat->data.ptr + (size_t)mat->step*row))[col] = (double)value;
    }
}


inline int cvIplDepth( int type )
{
    int depth = ((type) & ((1 << 3) - 1));
    return ((((sizeof(size_t)<<28)|0x8442211) >> ((depth) & ((1 << 3) - 1))*4) & 15)*8 | (depth == 1 || depth == 3 ||
           depth == 4 ? 0x80000000 : 0);
}
# 722 "../../../include/opencv/cxtypes.h"
typedef struct CvMatND
{
    int type;
    int dims;

    int* refcount;
    int hdr_refcount;

    union
    {
        uchar* ptr;
        float* fl;
        double* db;
        int* i;
        short* s;
    } data;

    struct
    {
        int size;
        int step;
    }
    dim[32];
}
CvMatND;
# 762 "../../../include/opencv/cxtypes.h"
struct CvSet;

typedef struct CvSparseMat
{
    int type;
    int dims;
    int* refcount;
    int hdr_refcount;

    struct CvSet* heap;
    void** hashtable;
    int hashsize;
    int valoffset;
    int idxoffset;
    int size[32];
}
CvSparseMat;
# 789 "../../../include/opencv/cxtypes.h"
typedef struct CvSparseNode
{
    unsigned hashval;
    struct CvSparseNode* next;
}
CvSparseNode;

typedef struct CvSparseMatIterator
{
    CvSparseMat* mat;
    CvSparseNode* node;
    int curidx;
}
CvSparseMatIterator;
# 811 "../../../include/opencv/cxtypes.h"
typedef int CvHistType;
# 827 "../../../include/opencv/cxtypes.h"
typedef struct CvHistogram
{
    int type;
    CvArr* bins;
    float thresh[32][2];
    float** thresh2;
    CvMatND mat;
}
CvHistogram;
# 857 "../../../include/opencv/cxtypes.h"
typedef struct CvRect
{
    int x;
    int y;
    int width;
    int height;
}
CvRect;

inline CvRect cvRect( int x, int y, int width, int height )
{
    CvRect r;

    r.x = x;
    r.y = y;
    r.width = width;
    r.height = height;

    return r;
}


inline IplROI cvRectToROI( CvRect rect, int coi )
{
    IplROI roi;
    roi.xOffset = rect.x;
    roi.yOffset = rect.y;
    roi.width = rect.width;
    roi.height = rect.height;
    roi.coi = coi;

    return roi;
}


inline CvRect cvROIToRect( IplROI roi )
{
    return cvRect( roi.xOffset, roi.yOffset, roi.width, roi.height );
}







typedef struct CvTermCriteria
{
    int type;


    int max_iter;
    double epsilon;
}
CvTermCriteria;

inline CvTermCriteria cvTermCriteria( int type, int max_iter, double epsilon )
{
    CvTermCriteria t;

    t.type = type;
    t.max_iter = max_iter;
    t.epsilon = (float)epsilon;

    return t;
}




typedef struct CvPoint
{
    int x;
    int y;
}
CvPoint;


inline CvPoint cvPoint( int x, int y )
{
    CvPoint p;

    p.x = x;
    p.y = y;

    return p;
}


typedef struct CvPoint2D32f
{
    float x;
    float y;
}
CvPoint2D32f;


inline CvPoint2D32f cvPoint2D32f( double x, double y )
{
    CvPoint2D32f p;

    p.x = (float)x;
    p.y = (float)y;

    return p;
}


inline CvPoint2D32f cvPointTo32f( CvPoint point )
{
    return cvPoint2D32f( (float)point.x, (float)point.y );
}


inline CvPoint cvPointFrom32f( CvPoint2D32f point )
{
    CvPoint ipt;
    ipt.x = cvRound(point.x);
    ipt.y = cvRound(point.y);

    return ipt;
}


typedef struct CvPoint3D32f
{
    float x;
    float y;
    float z;
}
CvPoint3D32f;


inline CvPoint3D32f cvPoint3D32f( double x, double y, double z )
{
    CvPoint3D32f p;

    p.x = (float)x;
    p.y = (float)y;
    p.z = (float)z;

    return p;
}


typedef struct CvPoint2D64f
{
    double x;
    double y;
}
CvPoint2D64f;


inline CvPoint2D64f cvPoint2D64f( double x, double y )
{
    CvPoint2D64f p;

    p.x = x;
    p.y = y;

    return p;
}


typedef struct CvPoint3D64f
{
    double x;
    double y;
    double z;
}
CvPoint3D64f;


inline CvPoint3D64f cvPoint3D64f( double x, double y, double z )
{
    CvPoint3D64f p;

    p.x = x;
    p.y = y;
    p.z = z;

    return p;
}




typedef struct
{
    int width;
    int height;
}
CvSize;

inline CvSize cvSize( int width, int height )
{
    CvSize s;

    s.width = width;
    s.height = height;

    return s;
}

typedef struct CvSize2D32f
{
    float width;
    float height;
}
CvSize2D32f;


inline CvSize2D32f cvSize2D32f( double width, double height )
{
    CvSize2D32f s;

    s.width = (float)width;
    s.height = (float)height;

    return s;
}

typedef struct CvBox2D
{
    CvPoint2D32f center;
    CvSize2D32f size;
    float angle;

}
CvBox2D;



typedef struct CvLineIterator
{

    uchar* ptr;


    int err;
    int plus_delta;
    int minus_delta;
    int plus_step;
    int minus_step;
}
CvLineIterator;





typedef struct CvSlice
{
    int start_index, end_index;
}
CvSlice;

inline CvSlice cvSlice( int start, int end )
{
    CvSlice slice;
    slice.start_index = start;
    slice.end_index = end;

    return slice;
}







typedef struct CvScalar
{
    double val[4];
}
CvScalar;

inline CvScalar cvScalar( double val0, double val1 = 0,
                               double val2 = 0, double val3 = 0)
{
    CvScalar scalar;
    scalar.val[0] = val0; scalar.val[1] = val1;
    scalar.val[2] = val2; scalar.val[3] = val3;
    return scalar;
}


inline CvScalar cvRealScalar( double val0 )
{
    CvScalar scalar;
    scalar.val[0] = val0;
    scalar.val[1] = scalar.val[2] = scalar.val[3] = 0;
    return scalar;
}

inline CvScalar cvScalarAll( double val0123 )
{
    CvScalar scalar;
    scalar.val[0] = val0123;
    scalar.val[1] = val0123;
    scalar.val[2] = val0123;
    scalar.val[3] = val0123;
    return scalar;
}







typedef struct CvMemBlock
{
    struct CvMemBlock* prev;
    struct CvMemBlock* next;
}
CvMemBlock;



typedef struct CvMemStorage
{
    int signature;
    CvMemBlock* bottom;
    CvMemBlock* top;
    struct CvMemStorage* parent;
    int block_size;
    int free_space;
}
CvMemStorage;






typedef struct CvMemStoragePos
{
    CvMemBlock* top;
    int free_space;
}
CvMemStoragePos;




typedef struct CvSeqBlock
{
    struct CvSeqBlock* prev;
    struct CvSeqBlock* next;
  int start_index;

    int count;
    schar* data;
}
CvSeqBlock;
# 1239 "../../../include/opencv/cxtypes.h"
typedef struct CvSeq
{
    int flags; int header_size; struct CvSeq* h_prev; struct CvSeq* h_next; struct CvSeq* v_prev; struct CvSeq* v_next; int total; int elem_size; schar* block_max; schar* ptr; int delta_elems; CvMemStorage* storage; CvSeqBlock* free_blocks; CvSeqBlock* first;
}
CvSeq;
# 1259 "../../../include/opencv/cxtypes.h"
typedef struct CvSetElem
{
    int flags; struct CvSetElem* next_free;
}
CvSetElem;






typedef struct CvSet
{
    int flags; int header_size; struct CvSeq* h_prev; struct CvSeq* h_next; struct CvSeq* v_prev; struct CvSeq* v_next; int total; int elem_size; schar* block_max; schar* ptr; int delta_elems; CvMemStorage* storage; CvSeqBlock* free_blocks; CvSeqBlock* first; CvSetElem* free_elems; int active_count;
}
CvSet;
# 1315 "../../../include/opencv/cxtypes.h"
typedef struct CvGraphEdge
{
    int flags; float weight; struct CvGraphEdge* next[2]; struct CvGraphVtx* vtx[2];
}
CvGraphEdge;

typedef struct CvGraphVtx
{
    int flags; struct CvGraphEdge* first;
}
CvGraphVtx;

typedef struct CvGraphVtx2D
{
    int flags; struct CvGraphEdge* first;
    CvPoint2D32f* ptr;
}
CvGraphVtx2D;
# 1342 "../../../include/opencv/cxtypes.h"
typedef struct CvGraph
{
    int flags; int header_size; struct CvSeq* h_prev; struct CvSeq* h_next; struct CvSeq* v_prev; struct CvSeq* v_next; int total; int elem_size; schar* block_max; schar* ptr; int delta_elems; CvMemStorage* storage; CvSeqBlock* free_blocks; CvSeqBlock* first; CvSetElem* free_elems; int active_count; CvSet* edges;
}
CvGraph;





typedef struct CvChain
{
    int flags; int header_size; struct CvSeq* h_prev; struct CvSeq* h_next; struct CvSeq* v_prev; struct CvSeq* v_next; int total; int elem_size; schar* block_max; schar* ptr; int delta_elems; CvMemStorage* storage; CvSeqBlock* free_blocks; CvSeqBlock* first;
    CvPoint origin;
}
CvChain;







typedef struct CvContour
{
    int flags; int header_size; struct CvSeq* h_prev; struct CvSeq* h_next; struct CvSeq* v_prev; struct CvSeq* v_next; int total; int elem_size; schar* block_max; schar* ptr; int delta_elems; CvMemStorage* storage; CvSeqBlock* free_blocks; CvSeqBlock* first; CvRect rect; int color; int reserved[3];
}
CvContour;

typedef CvContour CvPoint2DSeq;
# 1509 "../../../include/opencv/cxtypes.h"
typedef struct CvSeqWriter
{
    int header_size; CvSeq* seq; CvSeqBlock* block; schar* ptr; schar* block_min; schar* block_max;
}
CvSeqWriter;
# 1527 "../../../include/opencv/cxtypes.h"
typedef struct CvSeqReader
{
    int header_size; CvSeq* seq; CvSeqBlock* block; schar* ptr; schar* block_min; schar* block_max; int delta_index; schar* prev_elem;
}
CvSeqReader;
# 1647 "../../../include/opencv/cxtypes.h"
typedef struct CvFileStorage CvFileStorage;
# 1657 "../../../include/opencv/cxtypes.h"
typedef struct CvAttrList
{
    const char** attr;
    struct CvAttrList* next;
}
CvAttrList;

inline CvAttrList cvAttrList( const char** attr = NULL,
                                 CvAttrList* next = NULL )
{
    CvAttrList l;
    l.attr = attr;
    l.next = next;

    return l;
}

struct CvTypeInfo;
# 1710 "../../../include/opencv/cxtypes.h"
typedef struct CvString
{
    int len;
    char* ptr;
}
CvString;



typedef struct CvStringHashNode
{
    unsigned hashval;
    CvString str;
    struct CvStringHashNode* next;
}
CvStringHashNode;

typedef struct CvGenericHash CvFileNodeHash;


typedef struct CvFileNode
{
    int tag;
    struct CvTypeInfo* info;

    union
    {
        double f;
        int i;
        CvString str;
        CvSeq* seq;
        CvFileNodeHash* map;
    } data;
}
CvFileNode;


extern "C" {

typedef int ( *CvIsInstanceFunc)( const void* struct_ptr );
typedef void ( *CvReleaseFunc)( void** struct_dblptr );
typedef void* ( *CvReadFunc)( CvFileStorage* storage, CvFileNode* node );
typedef void ( *CvWriteFunc)( CvFileStorage* storage, const char* name,
                                      const void* struct_ptr, CvAttrList attributes );
typedef void* ( *CvCloneFunc)( const void* struct_ptr );

}


typedef struct CvTypeInfo
{
    int flags;
    int header_size;
    struct CvTypeInfo* prev;
    struct CvTypeInfo* next;
    const char* type_name;
    CvIsInstanceFunc is_instance;
    CvReleaseFunc release;
    CvReadFunc read;
    CvWriteFunc write;
    CvCloneFunc clone;
}
CvTypeInfo;




typedef struct CvPluginFuncInfo
{
    void** func_addr;
    void* default_func_addr;
    const char* func_names;
    int search_modules;
    int loaded_from;
}
CvPluginFuncInfo;

typedef struct CvModuleInfo
{
    struct CvModuleInfo* next;
    const char* name;
    const char* version;
    CvPluginFuncInfo* func_tab;
}
CvModuleInfo;
# 71 "../../../include/opencv/cxcore.h" 2
# 1 "../../../include/opencv/cxerror.h" 1
# 47 "../../../include/opencv/cxerror.h"
typedef int CVStatus;
# 72 "../../../include/opencv/cxcore.h" 2
# 1 "../../../include/opencv/cvver.h" 1
# 73 "../../../include/opencv/cxcore.h" 2


extern "C" {
# 86 "../../../include/opencv/cxcore.h"
extern "C" void* cvAlloc( size_t size );







extern "C" void cvFree_( void* ptr );



extern "C" IplImage* cvCreateImageHeader( CvSize size, int depth, int channels );


extern "C" IplImage* cvInitImageHeader( IplImage* image, CvSize size, int depth,
                                   int channels, int origin = 0,
                                   int align = 4);


extern "C" IplImage* cvCreateImage( CvSize size, int depth, int channels );


extern "C" void cvReleaseImageHeader( IplImage** image );


extern "C" void cvReleaseImage( IplImage** image );


extern "C" IplImage* cvCloneImage( const IplImage* image );



extern "C" void cvSetImageCOI( IplImage* image, int coi );


extern "C" int cvGetImageCOI( const IplImage* image );


extern "C" void cvSetImageROI( IplImage* image, CvRect rect );


extern "C" void cvResetImageROI( IplImage* image );


extern "C" CvRect cvGetImageROI( const IplImage* image );


extern "C" CvMat* cvCreateMatHeader( int rows, int cols, int type );




extern "C" CvMat* cvInitMatHeader( CvMat* mat, int rows, int cols,
                              int type, void* data = NULL,
                              int step = 0x7fffffff );


extern "C" CvMat* cvCreateMat( int rows, int cols, int type );



extern "C" void cvReleaseMat( CvMat** mat );



inline void cvDecRefData( CvArr* arr )
{
    if( (((arr) != NULL && (((const CvMat*)(arr))->type & 0xFFFF0000) == 0x42420000 && ((const CvMat*)(arr))->cols > 0 && ((const CvMat*)(arr))->rows > 0) && ((const CvMat*)(arr))->data.ptr != NULL))
    {
        CvMat* mat = (CvMat*)arr;
        mat->data.ptr = NULL;
        if( mat->refcount != NULL && --*mat->refcount == 0 )
            (cvFree_(*(&mat->refcount)), *(&mat->refcount)=0);
        mat->refcount = NULL;
    }
    else if( (((arr) != NULL && (((const CvMatND*)(arr))->type & 0xFFFF0000) == 0x42430000) && ((const CvMatND*)(arr))->data.ptr != NULL))
    {
        CvMatND* mat = (CvMatND*)arr;
        mat->data.ptr = NULL;
        if( mat->refcount != NULL && --*mat->refcount == 0 )
            (cvFree_(*(&mat->refcount)), *(&mat->refcount)=0);
        mat->refcount = NULL;
    }
}


inline int cvIncRefData( CvArr* arr )
{
    int refcount = 0;
    if( (((arr) != NULL && (((const CvMat*)(arr))->type & 0xFFFF0000) == 0x42420000 && ((const CvMat*)(arr))->cols > 0 && ((const CvMat*)(arr))->rows > 0) && ((const CvMat*)(arr))->data.ptr != NULL))
    {
        CvMat* mat = (CvMat*)arr;
        if( mat->refcount != NULL )
            refcount = ++*mat->refcount;
    }
    else if( (((arr) != NULL && (((const CvMatND*)(arr))->type & 0xFFFF0000) == 0x42430000) && ((const CvMatND*)(arr))->data.ptr != NULL))
    {
        CvMatND* mat = (CvMatND*)arr;
        if( mat->refcount != NULL )
            refcount = ++*mat->refcount;
    }
    return refcount;
}



extern "C" CvMat* cvCloneMat( const CvMat* mat );




extern "C" CvMat* cvGetSubRect( const CvArr* arr, CvMat* submat, CvRect rect );




extern "C" CvMat* cvGetRows( const CvArr* arr, CvMat* submat,
                        int start_row, int end_row,
                        int delta_row = 1);

inline CvMat* cvGetRow( const CvArr* arr, CvMat* submat, int row )
{
    return cvGetRows( arr, submat, row, row + 1, 1 );
}




extern "C" CvMat* cvGetCols( const CvArr* arr, CvMat* submat,
                        int start_col, int end_col );

inline CvMat* cvGetCol( const CvArr* arr, CvMat* submat, int col )
{
    return cvGetCols( arr, submat, col, col + 1 );
}





extern "C" CvMat* cvGetDiag( const CvArr* arr, CvMat* submat,
                            int diag = 0);


extern "C" void cvScalarToRawData( const CvScalar* scalar, void* data, int type,
                              int extend_to_12 = 0 );

extern "C" void cvRawDataToScalar( const void* data, int type, CvScalar* scalar );


extern "C" CvMatND* cvCreateMatNDHeader( int dims, const int* sizes, int type );


extern "C" CvMatND* cvCreateMatND( int dims, const int* sizes, int type );


extern "C" CvMatND* cvInitMatNDHeader( CvMatND* mat, int dims, const int* sizes,
                                    int type, void* data = NULL );


inline void cvReleaseMatND( CvMatND** mat )
{
    cvReleaseMat( (CvMat**)mat );
}


extern "C" CvMatND* cvCloneMatND( const CvMatND* mat );


extern "C" CvSparseMat* cvCreateSparseMat( int dims, const int* sizes, int type );


extern "C" void cvReleaseSparseMat( CvSparseMat** mat );


extern "C" CvSparseMat* cvCloneSparseMat( const CvSparseMat* mat );



extern "C" CvSparseNode* cvInitSparseMatIterator( const CvSparseMat* mat,
                                              CvSparseMatIterator* mat_iterator );


inline CvSparseNode* cvGetNextSparseNode( CvSparseMatIterator* mat_iterator )
{
    if( mat_iterator->node->next )
        return mat_iterator->node = mat_iterator->node->next;
    else
    {
        int idx;
        for( idx = ++mat_iterator->curidx; idx < mat_iterator->mat->hashsize; idx++ )
        {
            CvSparseNode* node = (CvSparseNode*)mat_iterator->mat->hashtable[idx];
            if( node )
            {
                mat_iterator->curidx = idx;
                return mat_iterator->node = node;
            }
        }
        return NULL;
    }
}





typedef struct CvNArrayIterator
{
    int count;
    int dims;
    CvSize size;
    uchar* ptr[10];
    int stack[32];
    CvMatND* hdr[10];

}
CvNArrayIterator;
# 313 "../../../include/opencv/cxcore.h"
extern "C" int cvInitNArrayIterator( int count, CvArr** arrs,
                                 const CvArr* mask, CvMatND* stubs,
                                 CvNArrayIterator* array_iterator,
                                 int flags = 0 );


extern "C" int cvNextNArraySlice( CvNArrayIterator* array_iterator );




extern "C" int cvGetElemType( const CvArr* arr );



extern "C" int cvGetDims( const CvArr* arr, int* sizes = NULL );





extern "C" int cvGetDimSize( const CvArr* arr, int index );




extern "C" uchar* cvPtr1D( const CvArr* arr, int idx0, int* type = NULL);
extern "C" uchar* cvPtr2D( const CvArr* arr, int idx0, int idx1, int* type = NULL );
extern "C" uchar* cvPtr3D( const CvArr* arr, int idx0, int idx1, int idx2,
                      int* type = NULL);





extern "C" uchar* cvPtrND( const CvArr* arr, const int* idx, int* type = NULL,
                      int create_node = 1,
                      unsigned* precalc_hashval = NULL);


extern "C" CvScalar cvGet1D( const CvArr* arr, int idx0 );
extern "C" CvScalar cvGet2D( const CvArr* arr, int idx0, int idx1 );
extern "C" CvScalar cvGet3D( const CvArr* arr, int idx0, int idx1, int idx2 );
extern "C" CvScalar cvGetND( const CvArr* arr, const int* idx );


extern "C" double cvGetReal1D( const CvArr* arr, int idx0 );
extern "C" double cvGetReal2D( const CvArr* arr, int idx0, int idx1 );
extern "C" double cvGetReal3D( const CvArr* arr, int idx0, int idx1, int idx2 );
extern "C" double cvGetRealND( const CvArr* arr, const int* idx );


extern "C" void cvSet1D( CvArr* arr, int idx0, CvScalar value );
extern "C" void cvSet2D( CvArr* arr, int idx0, int idx1, CvScalar value );
extern "C" void cvSet3D( CvArr* arr, int idx0, int idx1, int idx2, CvScalar value );
extern "C" void cvSetND( CvArr* arr, const int* idx, CvScalar value );


extern "C" void cvSetReal1D( CvArr* arr, int idx0, double value );
extern "C" void cvSetReal2D( CvArr* arr, int idx0, int idx1, double value );
extern "C" void cvSetReal3D( CvArr* arr, int idx0,
                        int idx1, int idx2, double value );
extern "C" void cvSetRealND( CvArr* arr, const int* idx, double value );



extern "C" void cvClearND( CvArr* arr, const int* idx );






extern "C" CvMat* cvGetMat( const CvArr* arr, CvMat* header,
                       int* coi = NULL,
                       int allowND = 0);


extern "C" IplImage* cvGetImage( const CvArr* arr, IplImage* image_header );
# 404 "../../../include/opencv/cxcore.h"
extern "C" CvArr* cvReshapeMatND( const CvArr* arr,
                             int sizeof_header, CvArr* header,
                             int new_cn, int new_dims, int* new_sizes );





extern "C" CvMat* cvReshape( const CvArr* arr, CvMat* header,
                        int new_cn, int new_rows = 0 );



extern "C" void cvRepeat( const CvArr* src, CvArr* dst );


extern "C" void cvCreateData( CvArr* arr );


extern "C" void cvReleaseData( CvArr* arr );




extern "C" void cvSetData( CvArr* arr, void* data, int step );




extern "C" void cvGetRawData( const CvArr* arr, uchar** data,
                         int* step = NULL,
                         CvSize* roi_size = NULL);


extern "C" CvSize cvGetSize( const CvArr* arr );


extern "C" void cvCopy( const CvArr* src, CvArr* dst,
                     const CvArr* mask = NULL );



extern "C" void cvSet( CvArr* arr, CvScalar value,
                    const CvArr* mask = NULL );


extern "C" void cvSetZero( CvArr* arr );





extern "C" void cvSplit( const CvArr* src, CvArr* dst0, CvArr* dst1,
                      CvArr* dst2, CvArr* dst3 );



extern "C" void cvMerge( const CvArr* src0, const CvArr* src1,
                      const CvArr* src2, const CvArr* src3,
                      CvArr* dst );



extern "C" void cvMixChannels( const CvArr** src, int src_count,
                            CvArr** dst, int dst_count,
                            const int* from_to, int pair_count );






extern "C" void cvConvertScale( const CvArr* src, CvArr* dst,
                             double scale = 1,
                             double shift = 0 );
# 489 "../../../include/opencv/cxcore.h"
extern "C" void cvConvertScaleAbs( const CvArr* src, CvArr* dst,
                                double scale = 1,
                                double shift = 0 );







extern "C" CvTermCriteria cvCheckTermCriteria( CvTermCriteria criteria,
                                           double default_eps,
                                           int default_max_iters );






extern "C" void cvAdd( const CvArr* src1, const CvArr* src2, CvArr* dst,
                    const CvArr* mask = NULL);


extern "C" void cvAddS( const CvArr* src, CvScalar value, CvArr* dst,
                     const CvArr* mask = NULL);


extern "C" void cvSub( const CvArr* src1, const CvArr* src2, CvArr* dst,
                    const CvArr* mask = NULL);


inline void cvSubS( const CvArr* src, CvScalar value, CvArr* dst,
                         const CvArr* mask = NULL)
{
    cvAddS( src, cvScalar( -value.val[0], -value.val[1], -value.val[2], -value.val[3]),
            dst, mask );
}


extern "C" void cvSubRS( const CvArr* src, CvScalar value, CvArr* dst,
                      const CvArr* mask = NULL);



extern "C" void cvMul( const CvArr* src1, const CvArr* src2,
                    CvArr* dst, double scale = 1 );




extern "C" void cvDiv( const CvArr* src1, const CvArr* src2,
                    CvArr* dst, double scale = 1);


extern "C" void cvScaleAdd( const CvArr* src1, CvScalar scale,
                         const CvArr* src2, CvArr* dst );



extern "C" void cvAddWeighted( const CvArr* src1, double alpha,
                            const CvArr* src2, double beta,
                            double gamma, CvArr* dst );


extern "C" double cvDotProduct( const CvArr* src1, const CvArr* src2 );


extern "C" void cvAnd( const CvArr* src1, const CvArr* src2,
                  CvArr* dst, const CvArr* mask = NULL);


extern "C" void cvAndS( const CvArr* src, CvScalar value,
                   CvArr* dst, const CvArr* mask = NULL);


extern "C" void cvOr( const CvArr* src1, const CvArr* src2,
                 CvArr* dst, const CvArr* mask = NULL);


extern "C" void cvOrS( const CvArr* src, CvScalar value,
                  CvArr* dst, const CvArr* mask = NULL);


extern "C" void cvXor( const CvArr* src1, const CvArr* src2,
                  CvArr* dst, const CvArr* mask = NULL);


extern "C" void cvXorS( const CvArr* src, CvScalar value,
                   CvArr* dst, const CvArr* mask = NULL);


extern "C" void cvNot( const CvArr* src, CvArr* dst );


extern "C" void cvInRange( const CvArr* src, const CvArr* lower,
                      const CvArr* upper, CvArr* dst );


extern "C" void cvInRangeS( const CvArr* src, CvScalar lower,
                       CvScalar upper, CvArr* dst );
# 601 "../../../include/opencv/cxcore.h"
extern "C" void cvCmp( const CvArr* src1, const CvArr* src2, CvArr* dst, int cmp_op );


extern "C" void cvCmpS( const CvArr* src, double value, CvArr* dst, int cmp_op );


extern "C" void cvMin( const CvArr* src1, const CvArr* src2, CvArr* dst );


extern "C" void cvMax( const CvArr* src1, const CvArr* src2, CvArr* dst );


extern "C" void cvMinS( const CvArr* src, double value, CvArr* dst );


extern "C" void cvMaxS( const CvArr* src, double value, CvArr* dst );


extern "C" void cvAbsDiff( const CvArr* src1, const CvArr* src2, CvArr* dst );


extern "C" void cvAbsDiffS( const CvArr* src, CvArr* dst, CvScalar value );
# 631 "../../../include/opencv/cxcore.h"
extern "C" void cvCartToPolar( const CvArr* x, const CvArr* y,
                            CvArr* magnitude, CvArr* angle = NULL,
                            int angle_in_degrees = 0);




extern "C" void cvPolarToCart( const CvArr* magnitude, const CvArr* angle,
                            CvArr* x, CvArr* y,
                            int angle_in_degrees = 0);


extern "C" void cvPow( const CvArr* src, CvArr* dst, double power );




extern "C" void cvExp( const CvArr* src, CvArr* dst );





extern "C" void cvLog( const CvArr* src, CvArr* dst );


extern "C" float cvFastArctan( float y, float x );


extern "C" float cvCbrt( float value );







extern "C" int cvCheckArr( const CvArr* arr, int flags = 0,
                        double min_val = 0, double max_val = 0);




extern "C" void cvRandArr( CvRNG* rng, CvArr* arr, int dist_type,
                      CvScalar param1, CvScalar param2 );

extern "C" void cvRandShuffle( CvArr* mat, CvRNG* rng,
                           double iter_factor = 1.);






extern "C" void cvSort( const CvArr* src, CvArr* dst = NULL,
                    CvArr* idxmat = NULL,
                    int flags = 0);


extern "C" int cvSolveCubic( const CvMat* coeffs, CvMat* roots );


extern "C" void cvSolvePoly(const CvMat* coeffs, CvMat *roots2,
   int maxiter = 20, int fig = 100);






extern "C" void cvCrossProduct( const CvArr* src1, const CvArr* src2, CvArr* dst );
# 712 "../../../include/opencv/cxcore.h"
extern "C" void cvGEMM( const CvArr* src1, const CvArr* src2, double alpha,
                     const CvArr* src3, double beta, CvArr* dst,
                     int tABC = 0);




extern "C" void cvTransform( const CvArr* src, CvArr* dst,
                          const CvMat* transmat,
                          const CvMat* shiftvec = NULL);



extern "C" void cvPerspectiveTransform( const CvArr* src, CvArr* dst,
                                     const CvMat* mat );


extern "C" void cvMulTransposed( const CvArr* src, CvArr* dst, int order,
                             const CvArr* delta = NULL,
                             double scale = 1. );


extern "C" void cvTranspose( const CvArr* src, CvArr* dst );



extern "C" void cvCompleteSymm( CvMat* matrix, int LtoR = 0 );




extern "C" void cvFlip( const CvArr* src, CvArr* dst = NULL,
                     int flip_mode = 0);
# 753 "../../../include/opencv/cxcore.h"
extern "C" void cvSVD( CvArr* A, CvArr* W, CvArr* U = NULL,
                     CvArr* V = NULL, int flags = 0);



extern "C" void cvSVBkSb( const CvArr* W, const CvArr* U,
                        const CvArr* V, const CvArr* B,
                        CvArr* X, int flags );
# 770 "../../../include/opencv/cxcore.h"
extern "C" double cvInvert( const CvArr* src, CvArr* dst,
                         int method = 0);




extern "C" int cvSolve( const CvArr* src1, const CvArr* src2, CvArr* dst,
                     int method = 0);


extern "C" double cvDet( const CvArr* mat );


extern "C" CvScalar cvTrace( const CvArr* mat );


extern "C" void cvEigenVV( CvArr* mat, CvArr* evects, CvArr* evals,
                        double eps = 0,
                        int lowindex = -1,
                        int highindex = -1);






extern "C" void cvSetIdentity( CvArr* mat, CvScalar value = cvRealScalar(1) );


extern "C" CvArr* cvRange( CvArr* mat, double start, double end );
# 821 "../../../include/opencv/cxcore.h"
extern "C" void cvCalcCovarMatrix( const CvArr** vects, int count,
                                CvArr* cov_mat, CvArr* avg, int flags );




extern "C" void cvCalcPCA( const CvArr* data, CvArr* mean,
                        CvArr* eigenvals, CvArr* eigenvects, int flags );

extern "C" void cvProjectPCA( const CvArr* data, const CvArr* mean,
                           const CvArr* eigenvects, CvArr* result );

extern "C" void cvBackProjectPCA( const CvArr* proj, const CvArr* mean,
                               const CvArr* eigenvects, CvArr* result );


extern "C" double cvMahalanobis( const CvArr* vec1, const CvArr* vec2, const CvArr* mat );







extern "C" CvScalar cvSum( const CvArr* arr );


extern "C" int cvCountNonZero( const CvArr* arr );


extern "C" CvScalar cvAvg( const CvArr* arr, const CvArr* mask = NULL );


extern "C" void cvAvgSdv( const CvArr* arr, CvScalar* mean, CvScalar* std_dev,
                       const CvArr* mask = NULL );


extern "C" void cvMinMaxLoc( const CvArr* arr, double* min_val, double* max_val,
                          CvPoint* min_loc = NULL,
                          CvPoint* max_loc = NULL,
                          const CvArr* mask = NULL );
# 880 "../../../include/opencv/cxcore.h"
extern "C" double cvNorm( const CvArr* arr1, const CvArr* arr2 = NULL,
                       int norm_type = 4,
                       const CvArr* mask = NULL );

extern "C" void cvNormalize( const CvArr* src, CvArr* dst,
                          double a = 1., double b = 0.,
                          int norm_type = 4,
                          const CvArr* mask = NULL );







extern "C" void cvReduce( const CvArr* src, CvArr* dst, int dim = -1,
                       int op = 0 );
# 914 "../../../include/opencv/cxcore.h"
extern "C" void cvDFT( const CvArr* src, CvArr* dst, int flags,
                    int nonzero_rows = 0 );



extern "C" void cvMulSpectrums( const CvArr* src1, const CvArr* src2,
                             CvArr* dst, int flags );


extern "C" int cvGetOptimalDFTSize( int size0 );


extern "C" void cvDCT( const CvArr* src, CvArr* dst, int flags );






extern "C" int cvSliceLength( CvSlice slice, const CvSeq* seq );





extern "C" CvMemStorage* cvCreateMemStorage( int block_size = 0);



extern "C" CvMemStorage* cvCreateChildMemStorage( CvMemStorage* parent );




extern "C" void cvReleaseMemStorage( CvMemStorage** storage );






extern "C" void cvClearMemStorage( CvMemStorage* storage );


extern "C" void cvSaveMemStoragePos( const CvMemStorage* storage, CvMemStoragePos* pos );


extern "C" void cvRestoreMemStoragePos( CvMemStorage* storage, CvMemStoragePos* pos );


extern "C" void* cvMemStorageAlloc( CvMemStorage* storage, size_t size );


extern "C" CvString cvMemStorageAllocString( CvMemStorage* storage, const char* ptr,
                                        int len = -1 );


extern "C" CvSeq* cvCreateSeq( int seq_flags, int header_size,
                            int elem_size, CvMemStorage* storage );



extern "C" void cvSetSeqBlockSize( CvSeq* seq, int delta_elems );



extern "C" schar* cvSeqPush( CvSeq* seq, const void* element = NULL);



extern "C" schar* cvSeqPushFront( CvSeq* seq, const void* element = NULL);



extern "C" void cvSeqPop( CvSeq* seq, void* element = NULL);



extern "C" void cvSeqPopFront( CvSeq* seq, void* element = NULL);





extern "C" void cvSeqPushMulti( CvSeq* seq, const void* elements,
                             int count, int in_front = 0 );


extern "C" void cvSeqPopMulti( CvSeq* seq, void* elements,
                            int count, int in_front = 0 );



extern "C" schar* cvSeqInsert( CvSeq* seq, int before_index,
                            const void* element = NULL);


extern "C" void cvSeqRemove( CvSeq* seq, int index );





extern "C" void cvClearSeq( CvSeq* seq );





extern "C" schar* cvGetSeqElem( const CvSeq* seq, int index );



extern "C" int cvSeqElemIdx( const CvSeq* seq, const void* element,
                         CvSeqBlock** block = NULL );


extern "C" void cvStartAppendToSeq( CvSeq* seq, CvSeqWriter* writer );



extern "C" void cvStartWriteSeq( int seq_flags, int header_size,
                              int elem_size, CvMemStorage* storage,
                              CvSeqWriter* writer );





extern "C" CvSeq* cvEndWriteSeq( CvSeqWriter* writer );




extern "C" void cvFlushSeqWriter( CvSeqWriter* writer );




extern "C" void cvStartReadSeq( const CvSeq* seq, CvSeqReader* reader,
                           int reverse = 0 );



extern "C" int cvGetSeqReaderPos( CvSeqReader* reader );




extern "C" void cvSetSeqReaderPos( CvSeqReader* reader, int index,
                                 int is_relative = 0);


extern "C" void* cvCvtSeqToArray( const CvSeq* seq, void* elements,
                               CvSlice slice = cvSlice(0, 0x3fffffff) );




extern "C" CvSeq* cvMakeSeqHeaderForArray( int seq_type, int header_size,
                                       int elem_size, void* elements, int total,
                                       CvSeq* seq, CvSeqBlock* block );


extern "C" CvSeq* cvSeqSlice( const CvSeq* seq, CvSlice slice,
                         CvMemStorage* storage = NULL,
                         int copy_data = 0);

inline CvSeq* cvCloneSeq( const CvSeq* seq, CvMemStorage* storage = NULL)
{
    return cvSeqSlice( seq, cvSlice(0, 0x3fffffff), storage, 1 );
}


extern "C" void cvSeqRemoveSlice( CvSeq* seq, CvSlice slice );


extern "C" void cvSeqInsertSlice( CvSeq* seq, int before_index, const CvArr* from_arr );


typedef int (* CvCmpFunc)(const void* a, const void* b, void* userdata );


extern "C" void cvSeqSort( CvSeq* seq, CvCmpFunc func, void* userdata = NULL );


extern "C" schar* cvSeqSearch( CvSeq* seq, const void* elem, CvCmpFunc func,
                           int is_sorted, int* elem_idx,
                           void* userdata = NULL );


extern "C" void cvSeqInvert( CvSeq* seq );


extern "C" int cvSeqPartition( const CvSeq* seq, CvMemStorage* storage,
                            CvSeq** labels, CvCmpFunc is_equal, void* userdata );


extern "C" void cvChangeSeqBlock( void* reader, int direction );
extern "C" void cvCreateSeqBlock( CvSeqWriter* writer );



extern "C" CvSet* cvCreateSet( int set_flags, int header_size,
                            int elem_size, CvMemStorage* storage );


extern "C" int cvSetAdd( CvSet* set_header, CvSetElem* elem = NULL,
                      CvSetElem** inserted_elem = NULL );


inline CvSetElem* cvSetNew( CvSet* set_header )
{
    CvSetElem* elem = set_header->free_elems;
    if( elem )
    {
        set_header->free_elems = elem->next_free;
        elem->flags = elem->flags & ((1 << 26) - 1);
        set_header->active_count++;
    }
    else
        cvSetAdd( set_header, NULL, (CvSetElem**)&elem );
    return elem;
}


inline void cvSetRemoveByPtr( CvSet* set_header, void* elem )
{
    CvSetElem* _elem = (CvSetElem*)elem;
    assert( _elem->flags >= 0 );
    _elem->next_free = set_header->free_elems;
    _elem->flags = (_elem->flags & ((1 << 26) - 1)) | (1 << (sizeof(int)*8-1));
    set_header->free_elems = _elem;
    set_header->active_count--;
}


extern "C" void cvSetRemove( CvSet* set_header, int index );



inline CvSetElem* cvGetSetElem( const CvSet* set_header, int index )
{
    CvSetElem* elem = (CvSetElem*)cvGetSeqElem( (CvSeq*)set_header, index );
    return elem && (((CvSetElem*)(elem))->flags >= 0) ? elem : 0;
}


extern "C" void cvClearSet( CvSet* set_header );


extern "C" CvGraph* cvCreateGraph( int graph_flags, int header_size,
                                int vtx_size, int edge_size,
                                CvMemStorage* storage );


extern "C" int cvGraphAddVtx( CvGraph* graph, const CvGraphVtx* vtx = NULL,
                           CvGraphVtx** inserted_vtx = NULL );



extern "C" int cvGraphRemoveVtx( CvGraph* graph, int index );
extern "C" int cvGraphRemoveVtxByPtr( CvGraph* graph, CvGraphVtx* vtx );






extern "C" int cvGraphAddEdge( CvGraph* graph,
                            int start_idx, int end_idx,
                            const CvGraphEdge* edge = NULL,
                            CvGraphEdge** inserted_edge = NULL );

extern "C" int cvGraphAddEdgeByPtr( CvGraph* graph,
                               CvGraphVtx* start_vtx, CvGraphVtx* end_vtx,
                               const CvGraphEdge* edge = NULL,
                               CvGraphEdge** inserted_edge = NULL );


extern "C" void cvGraphRemoveEdge( CvGraph* graph, int start_idx, int end_idx );
extern "C" void cvGraphRemoveEdgeByPtr( CvGraph* graph, CvGraphVtx* start_vtx,
                                     CvGraphVtx* end_vtx );


extern "C" CvGraphEdge* cvFindGraphEdge( const CvGraph* graph, int start_idx, int end_idx );
extern "C" CvGraphEdge* cvFindGraphEdgeByPtr( const CvGraph* graph,
                                           const CvGraphVtx* start_vtx,
                                           const CvGraphVtx* end_vtx );




extern "C" void cvClearGraph( CvGraph* graph );



extern "C" int cvGraphVtxDegree( const CvGraph* graph, int vtx_idx );
extern "C" int cvGraphVtxDegreeByPtr( const CvGraph* graph, const CvGraphVtx* vtx );
# 1248 "../../../include/opencv/cxcore.h"
typedef struct CvGraphScanner
{
    CvGraphVtx* vtx;
    CvGraphVtx* dst;
    CvGraphEdge* edge;

    CvGraph* graph;
    CvSeq* stack;
    int index;
    int mask;
}
CvGraphScanner;


extern "C" CvGraphScanner* cvCreateGraphScanner( CvGraph* graph,
                                             CvGraphVtx* vtx = NULL,
                                             int mask = -1);


extern "C" void cvReleaseGraphScanner( CvGraphScanner** scanner );


extern "C" int cvNextGraphItem( CvGraphScanner* scanner );


extern "C" CvGraph* cvCloneGraph( const CvGraph* graph, CvMemStorage* storage );
# 1295 "../../../include/opencv/cxcore.h"
extern "C" void cvLine( CvArr* img, CvPoint pt1, CvPoint pt2,
                     CvScalar color, int thickness = 1,
                     int line_type = 8, int shift = 0 );



extern "C" void cvRectangle( CvArr* img, CvPoint pt1, CvPoint pt2,
                          CvScalar color, int thickness = 1,
                          int line_type = 8,
                          int shift = 0);



extern "C" void cvCircle( CvArr* img, CvPoint center, int radius,
                       CvScalar color, int thickness = 1,
                       int line_type = 8, int shift = 0);




extern "C" void cvEllipse( CvArr* img, CvPoint center, CvSize axes,
                        double angle, double start_angle, double end_angle,
                        CvScalar color, int thickness = 1,
                        int line_type = 8, int shift = 0);

inline void cvEllipseBox( CvArr* img, CvBox2D box, CvScalar color,
                               int thickness = 1,
                               int line_type = 8, int shift = 0 )
{
    CvSize axes;
    axes.width = cvRound(box.size.height*0.5);
    axes.height = cvRound(box.size.width*0.5);

    cvEllipse( img, cvPointFrom32f( box.center ), axes, box.angle,
               0, 360, color, thickness, line_type, shift );
}


extern "C" void cvFillConvexPoly( CvArr* img, const CvPoint* pts, int npts, CvScalar color,
                               int line_type = 8, int shift = 0);


extern "C" void cvFillPoly( CvArr* img, CvPoint** pts, const int* npts,
                         int contours, CvScalar color,
                         int line_type = 8, int shift = 0 );


extern "C" void cvPolyLine( CvArr* img, CvPoint** pts, const int* npts, int contours,
                         int is_closed, CvScalar color, int thickness = 1,
                         int line_type = 8, int shift = 0 );
# 1355 "../../../include/opencv/cxcore.h"
extern "C" int cvClipLine( CvSize img_size, CvPoint* pt1, CvPoint* pt2 );




extern "C" int cvInitLineIterator( const CvArr* image, CvPoint pt1, CvPoint pt2,
                                CvLineIterator* line_iterator,
                                int connectivity = 8,
                                int left_to_right = 0);
# 1392 "../../../include/opencv/cxcore.h"
typedef struct CvFont
{
    int font_face;
    const int* ascii;
    const int* greek;
    const int* cyrillic;
    float hscale, vscale;
    float shear;
    int thickness;
    float dx;
    int line_type;
}
CvFont;


extern "C" void cvInitFont( CvFont* font, int font_face,
                         double hscale, double vscale,
                         double shear = 0,
                         int thickness = 1,
                         int line_type = 8);

inline CvFont cvFont( double scale, int thickness = 1 )
{
    CvFont font;
    cvInitFont( &font, 1, scale, scale, 0, thickness, 16 );
    return font;
}



extern "C" void cvPutText( CvArr* img, const char* text, CvPoint org,
                        const CvFont* font, CvScalar color );


extern "C" void cvGetTextSize( const char* text_string, const CvFont* font,
                            CvSize* text_size, int* baseline );




extern "C" CvScalar cvColorToScalar( double packed_color, int arrtype );







extern "C" int cvEllipse2Poly( CvPoint center, CvSize axes,
                 int angle, int arc_start, int arc_end, CvPoint * pts, int delta );


extern "C" void cvDrawContours( CvArr *img, CvSeq* contour,
                             CvScalar external_color, CvScalar hole_color,
                             int max_level, int thickness = 1,
                             int line_type = 8,
                             CvPoint offset = cvPoint(0,0));



extern "C" void cvLUT( const CvArr* src, CvArr* dst, const CvArr* lut );



typedef struct CvTreeNodeIterator
{
    const void* node;
    int level;
    int max_level;
}
CvTreeNodeIterator;

extern "C" void cvInitTreeNodeIterator( CvTreeNodeIterator* tree_iterator,
                                   const void* first, int max_level );
extern "C" void* cvNextTreeNode( CvTreeNodeIterator* tree_iterator );
extern "C" void* cvPrevTreeNode( CvTreeNodeIterator* tree_iterator );




extern "C" void cvInsertNodeIntoTree( void* node, void* parent, void* frame );


extern "C" void cvRemoveNodeFromTree( void* node, void* frame );



extern "C" CvSeq* cvTreeToNodeSeq( const void* first, int header_size,
                              CvMemStorage* storage );




extern "C" int cvKMeans2( const CvArr* samples, int cluster_count, CvArr* labels,
                      CvTermCriteria termcrit, int attempts = 1,
                      CvRNG* rng = 0, int flags = 0,
                      CvArr* _centers = 0, double* compactness = 0 );






extern "C" int cvRegisterModule( const CvModuleInfo* module_info );


extern "C" int cvUseOptimized( int on_off );


extern "C" void cvGetModuleInfo( const char* module_name,
                              const char** version,
                              const char** loaded_addon_plugins );


extern "C" int cvGetErrStatus( void );


extern "C" void cvSetErrStatus( int status );






extern "C" int cvGetErrMode( void );


extern "C" int cvSetErrMode( int mode );




extern "C" void cvError( int status, const char* func_name,
                    const char* err_msg, const char* file_name, int line );


extern "C" const char* cvErrorStr( int status );


extern "C" int cvGetErrInfo( const char** errcode_desc, const char** description,
                        const char** filename, int* line );


extern "C" int cvErrorFromIppStatus( int ipp_status );

typedef int ( *CvErrorCallback)( int status, const char* func_name,
                    const char* err_msg, const char* file_name, int line, void* userdata );


extern "C" CvErrorCallback cvRedirectError( CvErrorCallback error_handler,
                                       void* userdata = NULL,
                                       void** prev_userdata = NULL );







extern "C" int cvNulDevReport( int status, const char* func_name, const char* err_msg,
                          const char* file_name, int line, void* userdata );

extern "C" int cvStdErrReport( int status, const char* func_name, const char* err_msg,
                          const char* file_name, int line, void* userdata );

extern "C" int cvGuiBoxReport( int status, const char* func_name, const char* err_msg,
                          const char* file_name, int line, void* userdata );

typedef void* ( *CvAllocFunc)(size_t size, void* userdata);
typedef int ( *CvFreeFunc)(void* pptr, void* userdata);



extern "C" void cvSetMemoryManager( CvAllocFunc alloc_func = NULL,
                               CvFreeFunc free_func = NULL,
                               void* userdata = NULL);


typedef IplImage* (* Cv_iplCreateImageHeader)
                            (int,int,int,char*,char*,int,int,int,int,int,
                            IplROI*,IplImage*,void*,IplTileInfo*);
typedef void (* Cv_iplAllocateImageData)(IplImage*,int,int);
typedef void (* Cv_iplDeallocate)(IplImage*,int);
typedef IplROI* (* Cv_iplCreateROI)(int,int,int,int,int);
typedef IplImage* (* Cv_iplCloneImage)(const IplImage*);


extern "C" void cvSetIPLAllocators( Cv_iplCreateImageHeader create_header,
                               Cv_iplAllocateImageData allocate_data,
                               Cv_iplDeallocate deallocate,
                               Cv_iplCreateROI create_roi,
                               Cv_iplCloneImage clone_image );
# 1596 "../../../include/opencv/cxcore.h"
extern "C" CvFileStorage* cvOpenFileStorage( const char* filename,
                                          CvMemStorage* memstorage,
                                          int flags );


extern "C" void cvReleaseFileStorage( CvFileStorage** fs );


extern "C" const char* cvAttrValue( const CvAttrList* attr, const char* attr_name );


extern "C" void cvStartWriteStruct( CvFileStorage* fs, const char* name,
                                int struct_flags, const char* type_name = NULL,
                                CvAttrList attributes = cvAttrList());


extern "C" void cvEndWriteStruct( CvFileStorage* fs );


extern "C" void cvWriteInt( CvFileStorage* fs, const char* name, int value );


extern "C" void cvWriteReal( CvFileStorage* fs, const char* name, double value );


extern "C" void cvWriteString( CvFileStorage* fs, const char* name,
                           const char* str, int quote = 0 );


extern "C" void cvWriteComment( CvFileStorage* fs, const char* comment,
                            int eol_comment );



extern "C" void cvWrite( CvFileStorage* fs, const char* name, const void* ptr,
                         CvAttrList attributes = cvAttrList());


extern "C" void cvStartNextStream( CvFileStorage* fs );


extern "C" void cvWriteRawData( CvFileStorage* fs, const void* src,
                                int len, const char* dt );



extern "C" CvStringHashNode* cvGetHashedKey( CvFileStorage* fs, const char* name,
                                        int len = -1,
                                        int create_missing = 0);



extern "C" CvFileNode* cvGetRootFileNode( const CvFileStorage* fs,
                                     int stream_index = 0 );



extern "C" CvFileNode* cvGetFileNode( CvFileStorage* fs, CvFileNode* map,
                                 const CvStringHashNode* key,
                                 int create_missing = 0 );


extern "C" CvFileNode* cvGetFileNodeByName( const CvFileStorage* fs,
                                       const CvFileNode* map,
                                       const char* name );

inline int cvReadInt( const CvFileNode* node, int default_value = 0 )
{
    return !node ? default_value :
        (((node->tag) & 7) == 1) ? node->data.i :
        (((node->tag) & 7) == 2) ? cvRound(node->data.f) : 0x7fffffff;
}


inline int cvReadIntByName( const CvFileStorage* fs, const CvFileNode* map,
                         const char* name, int default_value = 0 )
{
    return cvReadInt( cvGetFileNodeByName( fs, map, name ), default_value );
}


inline double cvReadReal( const CvFileNode* node, double default_value = 0. )
{
    return !node ? default_value :
        (((node->tag) & 7) == 1) ? (double)node->data.i :
        (((node->tag) & 7) == 2) ? node->data.f : 1e300;
}


inline double cvReadRealByName( const CvFileStorage* fs, const CvFileNode* map,
                        const char* name, double default_value = 0. )
{
    return cvReadReal( cvGetFileNodeByName( fs, map, name ), default_value );
}


inline const char* cvReadString( const CvFileNode* node,
                        const char* default_value = NULL )
{
    return !node ? default_value : (((node->tag) & 7) == 3) ? node->data.str.ptr : 0;
}


inline const char* cvReadStringByName( const CvFileStorage* fs, const CvFileNode* map,
                        const char* name, const char* default_value = NULL )
{
    return cvReadString( cvGetFileNodeByName( fs, map, name ), default_value );
}



extern "C" void* cvRead( CvFileStorage* fs, CvFileNode* node,
                        CvAttrList* attributes = NULL);


inline void* cvReadByName( CvFileStorage* fs, const CvFileNode* map,
                              const char* name, CvAttrList* attributes = NULL )
{
    return cvRead( fs, cvGetFileNodeByName( fs, map, name ), attributes );
}



extern "C" void cvStartReadRawData( const CvFileStorage* fs, const CvFileNode* src,
                               CvSeqReader* reader );


extern "C" void cvReadRawDataSlice( const CvFileStorage* fs, CvSeqReader* reader,
                               int count, void* dst, const char* dt );


extern "C" void cvReadRawData( const CvFileStorage* fs, const CvFileNode* src,
                          void* dst, const char* dt );


extern "C" void cvWriteFileNode( CvFileStorage* fs, const char* new_node_name,
                            const CvFileNode* node, int embed );


extern "C" const char* cvGetFileNodeName( const CvFileNode* node );



extern "C" void cvRegisterType( const CvTypeInfo* info );
extern "C" void cvUnregisterType( const char* type_name );
extern "C" CvTypeInfo* cvFirstType(void);
extern "C" CvTypeInfo* cvFindType( const char* type_name );
extern "C" CvTypeInfo* cvTypeOf( const void* struct_ptr );


extern "C" void cvRelease( void** struct_ptr );
extern "C" void* cvClone( const void* struct_ptr );


extern "C" void cvSave( const char* filename, const void* struct_ptr,
                    const char* name = NULL,
                    const char* comment = NULL,
                    CvAttrList attributes = cvAttrList());
extern "C" void* cvLoad( const char* filename,
                     CvMemStorage* memstorage = NULL,
                     const char* name = NULL,
                     const char** real_name = NULL );





extern "C" int64 cvGetTickCount( void );
extern "C" double cvGetTickFrequency( void );




extern "C" int cvGetNumThreads( void );
extern "C" void cvSetNumThreads( int threads = 0 );

extern "C" int cvGetThreadNum( void );

}

struct CvModule
{
    CvModule( CvModuleInfo* _info );
    ~CvModule();
    CvModuleInfo* info;

    static CvModuleInfo* first;
    static CvModuleInfo* last;
};

struct CvType
{
    CvType( const char* type_name,
            CvIsInstanceFunc is_instance, CvReleaseFunc release=0,
            CvReadFunc read=0, CvWriteFunc write=0, CvCloneFunc clone=0 );
    ~CvType();
    CvTypeInfo* info;

    static CvTypeInfo* first;
    static CvTypeInfo* last;
};
# 59 "../../../include/opencv/cv.h" 2
# 1 "../../../include/opencv/cvtypes.h" 1
# 51 "../../../include/opencv/cvtypes.h"
typedef struct CvMoments
{
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double inv_sqrt_m00;
}
CvMoments;


typedef struct CvHuMoments
{
    double hu1, hu2, hu3, hu4, hu5, hu6, hu7;
}
CvHuMoments;



typedef struct CvConnectedComp
{
    double area;
    CvScalar value;
    CvRect rect;
    CvSeq* contour;

}
CvConnectedComp;





typedef struct _CvContourScanner* CvContourScanner;
# 99 "../../../include/opencv/cvtypes.h"
typedef struct CvChainPtReader
{
    int header_size; CvSeq* seq; CvSeqBlock* block; schar* ptr; schar* block_min; schar* block_max; int delta_index; schar* prev_elem;
    char code;
    CvPoint pt;
    schar deltas[8][2];
}
CvChainPtReader;
# 116 "../../../include/opencv/cvtypes.h"
typedef struct CvContourTree
{
    int flags; int header_size; struct CvSeq* h_prev; struct CvSeq* h_next; struct CvSeq* v_prev; struct CvSeq* v_next; int total; int elem_size; schar* block_max; schar* ptr; int delta_elems; CvMemStorage* storage; CvSeqBlock* free_blocks; CvSeqBlock* first;
    CvPoint p1;
    CvPoint p2;
}
CvContourTree;


typedef struct CvConvexityDefect
{
    CvPoint* start;
    CvPoint* end;
    CvPoint* depth_point;
    float depth;
}
CvConvexityDefect;



typedef size_t CvSubdiv2DEdge;
# 150 "../../../include/opencv/cvtypes.h"
typedef struct CvQuadEdge2D
{
    int flags; struct CvSubdiv2DPoint* pt[4]; CvSubdiv2DEdge next[4];
}
CvQuadEdge2D;

typedef struct CvSubdiv2DPoint
{
    int flags; CvSubdiv2DEdge first; CvPoint2D32f pt;
}
CvSubdiv2DPoint;
# 170 "../../../include/opencv/cvtypes.h"
typedef struct CvSubdiv2D
{
    int flags; int header_size; struct CvSeq* h_prev; struct CvSeq* h_next; struct CvSeq* v_prev; struct CvSeq* v_next; int total; int elem_size; schar* block_max; schar* ptr; int delta_elems; CvMemStorage* storage; CvSeqBlock* free_blocks; CvSeqBlock* first; CvSetElem* free_elems; int active_count; CvSet* edges; int quad_edges; int is_geometry_valid; CvSubdiv2DEdge recent_edge; CvPoint2D32f topleft; CvPoint2D32f bottomright;
}
CvSubdiv2D;


typedef enum CvSubdiv2DPointLocation
{
    CV_PTLOC_ERROR = -2,
    CV_PTLOC_OUTSIDE_RECT = -1,
    CV_PTLOC_INSIDE = 0,
    CV_PTLOC_VERTEX = 1,
    CV_PTLOC_ON_EDGE = 2
}
CvSubdiv2DPointLocation;

typedef enum CvNextEdgeType
{
    CV_NEXT_AROUND_ORG = 0x00,
    CV_NEXT_AROUND_DST = 0x22,
    CV_PREV_AROUND_ORG = 0x11,
    CV_PREV_AROUND_DST = 0x33,
    CV_NEXT_AROUND_LEFT = 0x13,
    CV_NEXT_AROUND_RIGHT = 0x31,
    CV_PREV_AROUND_LEFT = 0x20,
    CV_PREV_AROUND_RIGHT = 0x02
}
CvNextEdgeType;
# 216 "../../../include/opencv/cvtypes.h"
typedef enum CvFilter
{
    CV_GAUSSIAN_5x5 = 7
}
CvFilter;





typedef float* CvVect32f;
typedef float* CvMatr32f;
typedef double* CvVect64d;
typedef double* CvMatr64d;

typedef struct CvMatrix3
{
    float m[3][3];
}
CvMatrix3;



extern "C" {


typedef float ( * CvDistanceFunction)( const float* a, const float* b, void* user_param );


}


typedef struct CvKalman
{
    int MP;
    int DP;
    int CP;



    float* PosterState;
    float* PriorState;
    float* DynamMatr;
    float* MeasurementMatr;
    float* MNCovariance;
    float* PNCovariance;
    float* KalmGainMatr;
    float* PriorErrorCovariance;
    float* PosterErrorCovariance;
    float* Temp1;
    float* Temp2;


    CvMat* state_pre;

    CvMat* state_post;

    CvMat* transition_matrix;
    CvMat* control_matrix;

    CvMat* measurement_matrix;
    CvMat* process_noise_cov;
    CvMat* measurement_noise_cov;
    CvMat* error_cov_pre;

    CvMat* gain;

    CvMat* error_cov_post;

    CvMat* temp1;
    CvMat* temp2;
    CvMat* temp3;
    CvMat* temp4;
    CvMat* temp5;
}
CvKalman;
# 327 "../../../include/opencv/cvtypes.h"
typedef struct CvHaarFeature
{
    int tilted;
    struct
    {
        CvRect r;
        float weight;
    } rect[3];
}
CvHaarFeature;

typedef struct CvHaarClassifier
{
    int count;
    CvHaarFeature* haar_feature;
    float* threshold;
    int* left;
    int* right;
    float* alpha;
}
CvHaarClassifier;

typedef struct CvHaarStageClassifier
{
    int count;
    float threshold;
    CvHaarClassifier* classifier;

    int next;
    int child;
    int parent;
}
CvHaarStageClassifier;

typedef struct CvHidHaarClassifierCascade CvHidHaarClassifierCascade;

typedef struct CvHaarClassifierCascade
{
    int flags;
    int count;
    CvSize orig_window_size;
    CvSize real_window_size;
    double scale;
    CvHaarStageClassifier* stage_classifier;
    CvHidHaarClassifierCascade* hid_cascade;
}
CvHaarClassifierCascade;

typedef struct CvAvgComp
{
    CvRect rect;
    int neighbors;
}
CvAvgComp;

struct CvFeatureTree;
# 60 "../../../include/opencv/cv.h" 2


extern "C" {
# 71 "../../../include/opencv/cv.h"
extern "C" void cvCopyMakeBorder( const CvArr* src, CvArr* dst, CvPoint offset,
                              int bordertype, CvScalar value = cvScalarAll(0));
# 81 "../../../include/opencv/cv.h"
extern "C" void cvSmooth( const CvArr* src, CvArr* dst,
                      int smoothtype = 2,
                      int size1 = 3,
                      int size2 = 0,
                      double sigma1 = 0,
                      double sigma2 = 0);


extern "C" void cvFilter2D( const CvArr* src, CvArr* dst, const CvMat* kernel,
                        CvPoint anchor = cvPoint(-1,-1));


extern "C" void cvIntegral( const CvArr* image, CvArr* sum,
                       CvArr* sqsum = NULL,
                       CvArr* tilted_sum = NULL);






extern "C" void cvPyrDown( const CvArr* src, CvArr* dst,
                        int filter = CV_GAUSSIAN_5x5 );






extern "C" void cvPyrUp( const CvArr* src, CvArr* dst,
                      int filter = CV_GAUSSIAN_5x5 );


extern "C" CvMat** cvCreatePyramid( const CvArr* img, int extra_layers, double rate,
                                const CvSize* layer_sizes = 0,
                                CvArr* bufarr = 0,
                                int calc = 1,
                                int filter = CV_GAUSSIAN_5x5 );


extern "C" void cvReleasePyramid( CvMat*** pyramid, int extra_layers );






extern "C" void cvPyrSegmentation( IplImage* src, IplImage* dst,
                              CvMemStorage* storage, CvSeq** comp,
                              int level, double threshold1,
                              double threshold2 );


extern "C" void cvPyrMeanShiftFiltering( const CvArr* src, CvArr* dst,
    double sp, double sr, int max_level = 1,
    CvTermCriteria termcrit = cvTermCriteria(1 +2,5,1));


extern "C" void cvWatershed( const CvArr* image, CvArr* markers );





extern "C" void cvInpaint( const CvArr* src, const CvArr* inpaint_mask,
                       CvArr* dst, double inpaintRange, int flags );







extern "C" void cvSobel( const CvArr* src, CvArr* dst,
                    int xorder, int yorder,
                    int aperture_size = 3);


extern "C" void cvLaplace( const CvArr* src, CvArr* dst,
                      int aperture_size = 3 );
# 258 "../../../include/opencv/cv.h"
extern "C" void cvCvtColor( const CvArr* src, CvArr* dst, int code );
# 269 "../../../include/opencv/cv.h"
extern "C" void cvResize( const CvArr* src, CvArr* dst,
                       int interpolation = 1);


extern "C" void cvWarpAffine( const CvArr* src, CvArr* dst, const CvMat* map_matrix,
                           int flags = 1 +8,
                           CvScalar fillval = cvScalarAll(0) );


extern "C" CvMat* cvGetAffineTransform( const CvPoint2D32f * src,
                                    const CvPoint2D32f * dst,
                                    CvMat * map_matrix );


extern "C" CvMat* cv2DRotationMatrix( CvPoint2D32f center, double angle,
                                   double scale, CvMat* map_matrix );


extern "C" void cvWarpPerspective( const CvArr* src, CvArr* dst, const CvMat* map_matrix,
                                int flags = 1 +8,
                                CvScalar fillval = cvScalarAll(0) );


extern "C" CvMat* cvGetPerspectiveTransform( const CvPoint2D32f* src,
                                         const CvPoint2D32f* dst,
                                         CvMat* map_matrix );


extern "C" void cvRemap( const CvArr* src, CvArr* dst,
                      const CvArr* mapx, const CvArr* mapy,
                      int flags = 1 +8,
                      CvScalar fillval = cvScalarAll(0) );


extern "C" void cvConvertMaps( const CvArr* mapx, const CvArr* mapy,
                            CvArr* mapxy, CvArr* mapalpha );


extern "C" void cvLogPolar( const CvArr* src, CvArr* dst,
                         CvPoint2D32f center, double M,
                         int flags = 1 +8);


extern "C" void cvLinearPolar( const CvArr* src, CvArr* dst,
                         CvPoint2D32f center, double maxRadius,
                         int flags = 1 +8);







extern "C" IplConvKernel* cvCreateStructuringElementEx(
            int cols, int rows, int anchor_x, int anchor_y,
            int shape, int* values = NULL );


extern "C" void cvReleaseStructuringElement( IplConvKernel** element );



extern "C" void cvErode( const CvArr* src, CvArr* dst,
                      IplConvKernel* element = NULL,
                      int iterations = 1 );



extern "C" void cvDilate( const CvArr* src, CvArr* dst,
                       IplConvKernel* element = NULL,
                       int iterations = 1 );
# 348 "../../../include/opencv/cv.h"
extern "C" void cvMorphologyEx( const CvArr* src, CvArr* dst,
                             CvArr* temp, IplConvKernel* element,
                             int operation, int iterations = 1 );


extern "C" void cvMoments( const CvArr* arr, CvMoments* moments, int binary = 0);


extern "C" double cvGetSpatialMoment( CvMoments* moments, int x_order, int y_order );
extern "C" double cvGetCentralMoment( CvMoments* moments, int x_order, int y_order );
extern "C" double cvGetNormalizedCentralMoment( CvMoments* moments,
                                             int x_order, int y_order );


extern "C" void cvGetHuMoments( CvMoments* moments, CvHuMoments* hu_moments );





extern "C" int cvSampleLine( const CvArr* image, CvPoint pt1, CvPoint pt2, void* buffer,
                          int connectivity = 8);




extern "C" void cvGetRectSubPix( const CvArr* src, CvArr* dst, CvPoint2D32f center );







extern "C" void cvGetQuadrangleSubPix( const CvArr* src, CvArr* dst,
                                    const CvMat* map_matrix );
# 395 "../../../include/opencv/cv.h"
extern "C" void cvMatchTemplate( const CvArr* image, const CvArr* templ,
                              CvArr* result, int method );



extern "C" float cvCalcEMD2( const CvArr* signature1,
                          const CvArr* signature2,
                          int distance_type,
                          CvDistanceFunction distance_func = NULL,
                          const CvArr* cost_matrix = NULL,
                          CvArr* flow = NULL,
                          float* lower_bound = NULL,
                          void* userdata = NULL);







extern "C" int cvFindContours( CvArr* image, CvMemStorage* storage, CvSeq** first_contour,
                            int header_size = sizeof(CvContour),
                            int mode = 1,
                            int method = 2,
                            CvPoint offset = cvPoint(0,0));







extern "C" CvContourScanner cvStartFindContours( CvArr* image, CvMemStorage* storage,
                            int header_size = sizeof(CvContour),
                            int mode = 1,
                            int method = 2,
                            CvPoint offset = cvPoint(0,0));


extern "C" CvSeq* cvFindNextContour( CvContourScanner scanner );




extern "C" void cvSubstituteContour( CvContourScanner scanner, CvSeq* new_contour );



extern "C" CvSeq* cvEndFindContours( CvContourScanner* scanner );


extern "C" CvSeq* cvApproxChains( CvSeq* src_seq, CvMemStorage* storage,
                            int method = 2,
                            double parameter = 0,
                            int minimal_perimeter = 0,
                            int recursive = 0);





extern "C" void cvStartReadChainPoints( CvChain* chain, CvChainPtReader* reader );


extern "C" CvPoint cvReadChainPoint( CvChainPtReader* reader );
# 469 "../../../include/opencv/cv.h"
extern "C" void cvCalcOpticalFlowLK( const CvArr* prev, const CvArr* curr,
                                  CvSize win_size, CvArr* velx, CvArr* vely );


extern "C" void cvCalcOpticalFlowBM( const CvArr* prev, const CvArr* curr,
                                  CvSize block_size, CvSize shift_size,
                                  CvSize max_range, int use_previous,
                                  CvArr* velx, CvArr* vely );


extern "C" void cvCalcOpticalFlowHS( const CvArr* prev, const CvArr* curr,
                                  int use_previous, CvArr* velx, CvArr* vely,
                                  double lambda, CvTermCriteria criteria );
# 493 "../../../include/opencv/cv.h"
extern "C" void cvCalcOpticalFlowPyrLK( const CvArr* prev, const CvArr* curr,
                                     CvArr* prev_pyr, CvArr* curr_pyr,
                                     const CvPoint2D32f* prev_features,
                                     CvPoint2D32f* curr_features,
                                     int count,
                                     CvSize win_size,
                                     int level,
                                     char* status,
                                     float* track_error,
                                     CvTermCriteria criteria,
                                     int flags );




extern "C" void cvCalcAffineFlowPyrLK( const CvArr* prev, const CvArr* curr,
                                    CvArr* prev_pyr, CvArr* curr_pyr,
                                    const CvPoint2D32f* prev_features,
                                    CvPoint2D32f* curr_features,
                                    float* matrices, int count,
                                    CvSize win_size, int level,
                                    char* status, float* track_error,
                                    CvTermCriteria criteria, int flags );


extern "C" int cvEstimateRigidTransform( const CvArr* A, const CvArr* B,
                                      CvMat* M, int full_affine );
# 534 "../../../include/opencv/cv.h"
extern "C" void cvUpdateMotionHistory( const CvArr* silhouette, CvArr* mhi,
                                      double timestamp, double duration );



extern "C" void cvCalcMotionGradient( const CvArr* mhi, CvArr* mask, CvArr* orientation,
                                     double delta1, double delta2,
                                     int aperture_size = 3);




extern "C" double cvCalcGlobalOrientation( const CvArr* orientation, const CvArr* mask,
                                        const CvArr* mhi, double timestamp,
                                        double duration );



extern "C" CvSeq* cvSegmentMotion( const CvArr* mhi, CvArr* seg_mask,
                                CvMemStorage* storage,
                                double timestamp, double seg_thresh );




extern "C" void cvAcc( const CvArr* image, CvArr* sum,
                    const CvArr* mask = NULL );


extern "C" void cvSquareAcc( const CvArr* image, CvArr* sqsum,
                          const CvArr* mask = NULL );


extern "C" void cvMultiplyAcc( const CvArr* image1, const CvArr* image2, CvArr* acc,
                            const CvArr* mask = NULL );


extern "C" void cvRunningAvg( const CvArr* image, CvArr* acc, double alpha,
                           const CvArr* mask = NULL );
# 581 "../../../include/opencv/cv.h"
extern "C" int cvCamShift( const CvArr* prob_image, CvRect window,
                       CvTermCriteria criteria, CvConnectedComp* comp,
                       CvBox2D* box = NULL );



extern "C" int cvMeanShift( const CvArr* prob_image, CvRect window,
                        CvTermCriteria criteria, CvConnectedComp* comp );

extern "C" CvKalman* cvCreateKalman( int dynam_params, int measure_params,
                                int control_params = 0);


extern "C" void cvReleaseKalman( CvKalman** kalman);


extern "C" const CvMat* cvKalmanPredict( CvKalman* kalman,
                                     const CvMat* control = NULL);



extern "C" const CvMat* cvKalmanCorrect( CvKalman* kalman, const CvMat* measurement );






extern "C" void cvInitSubdivDelaunay2D( CvSubdiv2D* subdiv, CvRect rect );


extern "C" CvSubdiv2D* cvCreateSubdiv2D( int subdiv_type, int header_size,
                                      int vtx_size, int quadedge_size,
                                      CvMemStorage* storage );




inline CvSubdiv2D* cvCreateSubdivDelaunay2D( CvRect rect, CvMemStorage* storage )
{
    CvSubdiv2D* subdiv = cvCreateSubdiv2D( (4 << 9), sizeof(*subdiv),
                         sizeof(CvSubdiv2DPoint), sizeof(CvQuadEdge2D), storage );

    cvInitSubdivDelaunay2D( subdiv, rect );
    return subdiv;
}



extern "C" CvSubdiv2DPoint* cvSubdivDelaunay2DInsert( CvSubdiv2D* subdiv, CvPoint2D32f pt);




extern "C" CvSubdiv2DPointLocation cvSubdiv2DLocate(
                               CvSubdiv2D* subdiv, CvPoint2D32f pt,
                               CvSubdiv2DEdge* edge,
                               CvSubdiv2DPoint** vertex = NULL );


extern "C" void cvCalcSubdivVoronoi2D( CvSubdiv2D* subdiv );



extern "C" void cvClearSubdivVoronoi2D( CvSubdiv2D* subdiv );



extern "C" CvSubdiv2DPoint* cvFindNearestPoint2D( CvSubdiv2D* subdiv, CvPoint2D32f pt );




inline CvSubdiv2DEdge cvSubdiv2DNextEdge( CvSubdiv2DEdge edge )
{
    return (((CvQuadEdge2D*)((edge) & ~3))->next[(edge)&3]);
}


inline CvSubdiv2DEdge cvSubdiv2DRotateEdge( CvSubdiv2DEdge edge, int rotate )
{
    return (edge & ~3) + ((edge + rotate) & 3);
}

inline CvSubdiv2DEdge cvSubdiv2DSymEdge( CvSubdiv2DEdge edge )
{
    return edge ^ 2;
}

inline CvSubdiv2DEdge cvSubdiv2DGetEdge( CvSubdiv2DEdge edge, CvNextEdgeType type )
{
    CvQuadEdge2D* e = (CvQuadEdge2D*)(edge & ~3);
    edge = e->next[(edge + (int)type) & 3];
    return (edge & ~3) + ((edge + ((int)type >> 4)) & 3);
}


inline CvSubdiv2DPoint* cvSubdiv2DEdgeOrg( CvSubdiv2DEdge edge )
{
    CvQuadEdge2D* e = (CvQuadEdge2D*)(edge & ~3);
    return (CvSubdiv2DPoint*)e->pt[edge & 3];
}


inline CvSubdiv2DPoint* cvSubdiv2DEdgeDst( CvSubdiv2DEdge edge )
{
    CvQuadEdge2D* e = (CvQuadEdge2D*)(edge & ~3);
    return (CvSubdiv2DPoint*)e->pt[(edge + 2) & 3];
}


inline double cvTriangleArea( CvPoint2D32f a, CvPoint2D32f b, CvPoint2D32f c )
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
# 721 "../../../include/opencv/cv.h"
extern "C" CvSeq* cvApproxPoly( const void* src_seq,
                             int header_size, CvMemStorage* storage,
                             int method, double parameter,
                             int parameter2 = 0);



extern "C" double cvArcLength( const void* curve,
                            CvSlice slice = cvSlice(0, 0x3fffffff),
                            int is_closed = -1);


extern "C" CvRect cvBoundingRect( CvArr* points, int update = 0 );


extern "C" double cvContourArea( const CvArr* contour,
                              CvSlice slice = cvSlice(0, 0x3fffffff));


extern "C" CvBox2D cvMinAreaRect2( const CvArr* points,
                                CvMemStorage* storage = NULL);


extern "C" int cvMinEnclosingCircle( const CvArr* points,
                                  CvPoint2D32f* center, float* radius );






extern "C" double cvMatchShapes( const void* object1, const void* object2,
                              int method, double parameter = 0);


extern "C" CvContourTree* cvCreateContourTree( const CvSeq* contour,
                                            CvMemStorage* storage,
                                            double threshold );


extern "C" CvSeq* cvContourFromContourTree( const CvContourTree* tree,
                                         CvMemStorage* storage,
                                         CvTermCriteria criteria );




extern "C" double cvMatchContourTrees( const CvContourTree* tree1,
                                    const CvContourTree* tree2,
                                    int method, double threshold );


extern "C" CvSeq* cvConvexHull2( const CvArr* input,
                             void* hull_storage = NULL,
                             int orientation = 1,
                             int return_points = 0);


extern "C" int cvCheckContourConvexity( const CvArr* contour );


extern "C" CvSeq* cvConvexityDefects( const CvArr* contour, const CvArr* convexhull,
                                   CvMemStorage* storage = NULL);


extern "C" CvBox2D cvFitEllipse2( const CvArr* points );


extern "C" CvRect cvMaxRect( const CvRect* rect1, const CvRect* rect2 );


extern "C" void cvBoxPoints( CvBox2D box, CvPoint2D32f pt[4] );



extern "C" CvSeq* cvPointSeqFromMat( int seq_kind, const CvArr* mat,
                                 CvContour* contour_header,
                                 CvSeqBlock* block );





extern "C" double cvPointPolygonTest( const CvArr* contour,
                                  CvPoint2D32f pt, int measure_dist );






extern "C" CvHistogram* cvCreateHist( int dims, int* sizes, int type,
                                   float** ranges = NULL,
                                   int uniform = 1);


extern "C" void cvSetHistBinRanges( CvHistogram* hist, float** ranges,
                                int uniform = 1);


extern "C" CvHistogram* cvMakeHistHeaderForArray(
                            int dims, int* sizes, CvHistogram* hist,
                            float* data, float** ranges = NULL,
                            int uniform = 1);


extern "C" void cvReleaseHist( CvHistogram** hist );


extern "C" void cvClearHist( CvHistogram* hist );


extern "C" void cvGetMinMaxHistValue( const CvHistogram* hist,
                                   float* min_value, float* max_value,
                                   int* min_idx = NULL,
                                   int* max_idx = NULL);




extern "C" void cvNormalizeHist( CvHistogram* hist, double factor );



extern "C" void cvThreshHist( CvHistogram* hist, double threshold );







extern "C" double cvCompareHist( const CvHistogram* hist1,
                              const CvHistogram* hist2,
                              int method);



extern "C" void cvCopyHist( const CvHistogram* src, CvHistogram** dst );




extern "C" void cvCalcBayesianProb( CvHistogram** src, int number,
                                CvHistogram** dst);


extern "C" void cvCalcArrHist( CvArr** arr, CvHistogram* hist,
                            int accumulate = 0,
                            const CvArr* mask = NULL );

inline void cvCalcHist( IplImage** image, CvHistogram* hist,
                             int accumulate = 0,
                             const CvArr* mask = NULL )
{
    cvCalcArrHist( (CvArr**)image, hist, accumulate, mask );
}


extern "C" void cvCalcArrBackProject( CvArr** image, CvArr* dst,
                                   const CvHistogram* hist );





extern "C" void cvCalcArrBackProjectPatch( CvArr** image, CvArr* dst, CvSize range,
                                        CvHistogram* hist, int method,
                                        double factor );





extern "C" void cvCalcProbDensity( const CvHistogram* hist1, const CvHistogram* hist2,
                                CvHistogram* dst_hist, double scale = 255 );


extern "C" void cvEqualizeHist( const CvArr* src, CvArr* dst );






extern "C" void cvSnakeImage( const IplImage* image, CvPoint* points,
                           int length, float* alpha,
                           float* beta, float* gamma,
                           int coeff_usage, CvSize win,
                           CvTermCriteria criteria, int calc_gradient = 1);


extern "C" void cvDistTransform( const CvArr* src, CvArr* dst,
                              int distance_type = 2,
                              int mask_size = 3,
                              const float* mask = NULL,
                              CvArr* labels = NULL);
# 958 "../../../include/opencv/cv.h"
extern "C" double cvThreshold( const CvArr* src, CvArr* dst,
                            double threshold, double max_value,
                            int threshold_type );
# 970 "../../../include/opencv/cv.h"
extern "C" void cvAdaptiveThreshold( const CvArr* src, CvArr* dst, double max_value,
                                  int adaptive_method = 0,
                                  int threshold_type = 0,
                                  int block_size = 3,
                                  double param1 = 5);





extern "C" void cvFloodFill( CvArr* image, CvPoint seed_point,
                          CvScalar new_val, CvScalar lo_diff = cvScalarAll(0),
                          CvScalar up_diff = cvScalarAll(0),
                          CvConnectedComp* comp = NULL,
                          int flags = 4,
                          CvArr* mask = NULL);
# 994 "../../../include/opencv/cv.h"
extern "C" void cvCanny( const CvArr* image, CvArr* edges, double threshold1,
                      double threshold2, int aperture_size = 3 );




extern "C" void cvPreCornerDetect( const CvArr* image, CvArr* corners,
                              int aperture_size = 3 );



extern "C" void cvCornerEigenValsAndVecs( const CvArr* image, CvArr* eigenvv,
                                      int block_size, int aperture_size = 3 );



extern "C" void cvCornerMinEigenVal( const CvArr* image, CvArr* eigenval,
                                 int block_size, int aperture_size = 3 );



extern "C" void cvCornerHarris( const CvArr* image, CvArr* harris_responce,
                             int block_size, int aperture_size = 3,
                             double k = 0.04 );


extern "C" void cvFindCornerSubPix( const CvArr* image, CvPoint2D32f* corners,
                                 int count, CvSize win, CvSize zero_zone,
                                 CvTermCriteria criteria );



extern "C" void cvGoodFeaturesToTrack( const CvArr* image, CvArr* eig_image,
                                   CvArr* temp_image, CvPoint2D32f* corners,
                                   int* corner_count, double quality_level,
                                   double min_distance,
                                   const CvArr* mask = NULL,
                                   int block_size = 3,
                                   int use_harris = 0,
                                   double k = 0.04 );
# 1047 "../../../include/opencv/cv.h"
extern "C" CvSeq* cvHoughLines2( CvArr* image, void* line_storage, int method,
                              double rho, double theta, int threshold,
                              double param1 = 0, double param2 = 0);


extern "C" CvSeq* cvHoughCircles( CvArr* image, void* circle_storage,
                              int method, double dp, double min_dist,
                              double param1 = 100,
                              double param2 = 100,
                              int min_radius = 0,
                              int max_radius = 0);


extern "C" void cvFitLine( const CvArr* points, int dist_type, double param,
                        double reps, double aeps, float* line );



struct CvFeatureTree;


extern "C" struct CvFeatureTree* cvCreateKDTree(CvMat* desc);


extern "C" struct CvFeatureTree* cvCreateSpillTree( const CvMat* raw_data,
                                    const int naive = 50,
                                    const double rho = .7,
                                    const double tau = .1 );


extern "C" void cvReleaseFeatureTree(struct CvFeatureTree* tr);



extern "C" void cvFindFeatures(struct CvFeatureTree* tr, const CvMat* query_points,
                           CvMat* indices, CvMat* dist, int k, int emax = 20);



extern "C" int cvFindFeaturesBoxed(struct CvFeatureTree* tr,
                               CvMat* bounds_min, CvMat* bounds_max,
                               CvMat* out_indices);


struct CvLSH;
struct CvLSHOperations;



extern "C" struct CvLSH* cvCreateLSH(struct CvLSHOperations* ops, int d,
                                 int L = 10, int k = 10,
                                 int type = (((6) & ((1 << 3) - 1)) + (((1)-1) << 3)), double r = 4,
                                 int64 seed = -1);


extern "C" struct CvLSH* cvCreateMemoryLSH(int d, int n, int L = 10, int k = 10,
                                       int type = (((6) & ((1 << 3) - 1)) + (((1)-1) << 3)), double r = 4,
                                       int64 seed = -1);


extern "C" void cvReleaseLSH(struct CvLSH** lsh);


extern "C" unsigned int LSHSize(struct CvLSH* lsh);


extern "C" void cvLSHAdd(struct CvLSH* lsh, const CvMat* data, CvMat* indices = 0);


extern "C" void cvLSHRemove(struct CvLSH* lsh, const CvMat* indices);



extern "C" void cvLSHQuery(struct CvLSH* lsh, const CvMat* query_points,
                       CvMat* indices, CvMat* dist, int k, int emax);


typedef struct CvSURFPoint
{
    CvPoint2D32f pt;
    int laplacian;
    int size;
    float dir;
    float hessian;
} CvSURFPoint;

inline CvSURFPoint cvSURFPoint( CvPoint2D32f pt, int laplacian,
                                   int size, float dir = 0,
                                   float hessian = 0)
{
    CvSURFPoint kp;
    kp.pt = pt;
    kp.laplacian = laplacian;
    kp.size = size;
    kp.dir = dir;
    kp.hessian = hessian;
    return kp;
}

typedef struct CvSURFParams
{
    int extended;
    double hessianThreshold;

    int nOctaves;
    int nOctaveLayers;
}
CvSURFParams;

extern "C" CvSURFParams cvSURFParams( double hessianThreshold, int extended = 0 );



extern "C" void cvExtractSURF( const CvArr* img, const CvArr* mask,
                           CvSeq** keypoints, CvSeq** descriptors,
                           CvMemStorage* storage, CvSURFParams params, int useProvidedKeyPts = 0 );

typedef struct CvMSERParams
{

    int delta;

    int maxArea;
    int minArea;

    float maxVariation;

    float minDiversity;


    int maxEvolution;

    double areaThreshold;

    double minMargin;

    int edgeBlurSize;
}
CvMSERParams;

extern "C" CvMSERParams cvMSERParams( int delta = 5, int min_area = 60,
                           int max_area = 14400, float max_variation = .25f,
                           float min_diversity = .2f, int max_evolution = 200,
                           double area_threshold = 1.01,
                           double min_margin = .003,
                           int edge_blur_size = 5 );


extern "C" void cvExtractMSER( CvArr* _img, CvArr* _mask, CvSeq** contours, CvMemStorage* storage, CvMSERParams params );


typedef struct CvStarKeypoint
{
    CvPoint pt;
    int size;
    float response;
}
CvStarKeypoint;

inline CvStarKeypoint cvStarKeypoint(CvPoint pt, int size, float response)
{
    CvStarKeypoint kpt;
    kpt.pt = pt;
    kpt.size = size;
    kpt.response = response;
    return kpt;
}

typedef struct CvStarDetectorParams
{
    int maxSize;
    int responseThreshold;
    int lineThresholdProjected;
    int lineThresholdBinarized;
    int suppressNonmaxSize;
}
CvStarDetectorParams;

inline CvStarDetectorParams cvStarDetectorParams(
    int maxSize = 45,
    int responseThreshold = 30,
    int lineThresholdProjected = 10,
    int lineThresholdBinarized = 8,
    int suppressNonmaxSize = 5)
{
    CvStarDetectorParams params;
    params.maxSize = maxSize;
    params.responseThreshold = responseThreshold;
    params.lineThresholdProjected = lineThresholdProjected;
    params.lineThresholdBinarized = lineThresholdBinarized;
    params.suppressNonmaxSize = suppressNonmaxSize;

    return params;
}

extern "C" CvSeq* cvGetStarKeypoints( const CvArr* img, CvMemStorage* storage,
        CvStarDetectorParams params = cvStarDetectorParams());







extern "C" CvHaarClassifierCascade* cvLoadHaarClassifierCascade(
                    const char* directory, CvSize orig_window_size);

extern "C" void cvReleaseHaarClassifierCascade( CvHaarClassifierCascade** cascade );






extern "C" CvSeq* cvHaarDetectObjects( const CvArr* image,
                     CvHaarClassifierCascade* cascade,
                     CvMemStorage* storage, double scale_factor = 1.1,
                     int min_neighbors = 3, int flags = 0,
                     CvSize min_size = cvSize(0,0));


extern "C" void cvSetImagesForHaarClassifierCascade( CvHaarClassifierCascade* cascade,
                                                const CvArr* sum, const CvArr* sqsum,
                                                const CvArr* tilted_sum, double scale );


extern "C" int cvRunHaarClassifierCascade( const CvHaarClassifierCascade* cascade,
                                       CvPoint pt, int start_stage = 0);






extern "C" void cvUndistort2( const CvArr* src, CvArr* dst,
                          const CvMat* camera_matrix,
                          const CvMat* distortion_coeffs );



extern "C" void cvInitUndistortMap( const CvMat* camera_matrix,
                                const CvMat* distortion_coeffs,
                                CvArr* mapx, CvArr* mapy );


extern "C" void cvInitUndistortRectifyMap( const CvMat* camera_matrix,
                                       const CvMat* dist_coeffs,
                                       const CvMat *R, const CvMat* new_camera_matrix,
                                       CvArr* mapx, CvArr* mapy );



extern "C" void cvUndistortPoints( const CvMat* src, CvMat* dst,
                               const CvMat* camera_matrix,
                               const CvMat* dist_coeffs,
                               const CvMat* R = 0,
                               const CvMat* P = 0);


extern "C" int cvRodrigues2( const CvMat* src, CvMat* dst,
                         CvMat* jacobian = 0 );





extern "C" int cvFindHomography( const CvMat* src_points,
                             const CvMat* dst_points,
                             CvMat* homography,
                             int method = 0,
                             double ransacReprojThreshold = 0,
                             CvMat* mask = 0);


extern "C" void cvRQDecomp3x3( const CvMat *matrixM, CvMat *matrixR, CvMat *matrixQ,
                           CvMat *matrixQx = NULL,
                           CvMat *matrixQy = NULL,
                           CvMat *matrixQz = NULL,
                           CvPoint3D64f *eulerAngles = NULL);


extern "C" void cvDecomposeProjectionMatrix( const CvMat *projMatr, CvMat *calibMatr,
                                         CvMat *rotMatr, CvMat *posVect,
                                         CvMat *rotMatrX = NULL,
                                         CvMat *rotMatrY = NULL,
                                         CvMat *rotMatrZ = NULL,
                                         CvPoint3D64f *eulerAngles = NULL);


extern "C" void cvCalcMatMulDeriv( const CvMat* A, const CvMat* B, CvMat* dABdA, CvMat* dABdB );



extern "C" void cvComposeRT( const CvMat* _rvec1, const CvMat* _tvec1,
                         const CvMat* _rvec2, const CvMat* _tvec2,
                         CvMat* _rvec3, CvMat* _tvec3,
                         CvMat* dr3dr1 = 0, CvMat* dr3dt1 = 0,
                         CvMat* dr3dr2 = 0, CvMat* dr3dt2 = 0,
                         CvMat* dt3dr1 = 0, CvMat* dt3dt1 = 0,
                         CvMat* dt3dr2 = 0, CvMat* dt3dt2 = 0 );



extern "C" void cvProjectPoints2( const CvMat* object_points, const CvMat* rotation_vector,
                              const CvMat* translation_vector, const CvMat* camera_matrix,
                              const CvMat* distortion_coeffs, CvMat* image_points,
                              CvMat* dpdrot = NULL, CvMat* dpdt = NULL,
                              CvMat* dpdf = NULL, CvMat* dpdc = NULL,
                              CvMat* dpddist = NULL,
                              double aspect_ratio = 0);



extern "C" void cvFindExtrinsicCameraParams2( const CvMat* object_points,
                                          const CvMat* image_points,
                                          const CvMat* camera_matrix,
                                          const CvMat* distortion_coeffs,
                                          CvMat* rotation_vector,
                                          CvMat* translation_vector,
                                          int use_extrinsic_guess = 0 );



extern "C" void cvInitIntrinsicParams2D( const CvMat* object_points,
                                     const CvMat* image_points,
                                     const CvMat* npoints, CvSize image_size,
                                     CvMat* camera_matrix,
                                     double aspect_ratio = 1. );






extern "C" int cvFindChessboardCorners( const void* image, CvSize pattern_size,
                                    CvPoint2D32f* corners,
                                    int* corner_count = NULL,
                                    int flags = 1 + 2 );



extern "C" void cvDrawChessboardCorners( CvArr* image, CvSize pattern_size,
                                     CvPoint2D32f* corners,
                                     int count, int pattern_was_found );
# 1403 "../../../include/opencv/cv.h"
extern "C" void cvCalibrateCamera2( const CvMat* object_points,
                                const CvMat* image_points,
                                const CvMat* point_counts,
                                CvSize image_size,
                                CvMat* camera_matrix,
                                CvMat* distortion_coeffs,
                                CvMat* rotation_vectors = NULL,
                                CvMat* translation_vectors = NULL,
                                int flags = 0 );



extern "C" void cvCalibrationMatrixValues( const CvMat *camera_matrix,
                                CvSize image_size,
                                double aperture_width = 0,
                                double aperture_height = 0,
                                double *fovx = NULL,
                                double *fovy = NULL,
                                double *focal_length = NULL,
                                CvPoint2D64f *principal_point = NULL,
                                double *pixel_aspect_ratio = NULL);







extern "C" void cvStereoCalibrate( const CvMat* object_points, const CvMat* image_points1,
                               const CvMat* image_points2, const CvMat* npoints,
                               CvMat* camera_matrix1, CvMat* dist_coeffs1,
                               CvMat* camera_matrix2, CvMat* dist_coeffs2,
                               CvSize image_size, CvMat* R, CvMat* T,
                               CvMat* E = 0, CvMat* F = 0,
                               CvTermCriteria term_crit = cvTermCriteria( 1 +2,30,1e-6),

                               int flags = 256 );





extern "C" void cvStereoRectify( const CvMat* camera_matrix1, const CvMat* camera_matrix2,
                             const CvMat* dist_coeffs1, const CvMat* dist_coeffs2,
                             CvSize image_size, const CvMat* R, const CvMat* T,
                             CvMat* R1, CvMat* R2, CvMat* P1, CvMat* P2,
                             CvMat* Q = 0,
                             int flags = 1024 );



extern "C" int cvStereoRectifyUncalibrated( const CvMat* points1, const CvMat* points2,
                                        const CvMat* F, CvSize img_size,
                                        CvMat* H1, CvMat* H2,
                                        double threshold = 5);

typedef struct CvPOSITObject CvPOSITObject;


extern "C" CvPOSITObject* cvCreatePOSITObject( CvPoint3D32f* points, int point_count );




extern "C" void cvPOSIT( CvPOSITObject* posit_object, CvPoint2D32f* image_points,
                       double focal_length, CvTermCriteria criteria,
                       CvMatr32f rotation_matrix, CvVect32f translation_vector);


extern "C" void cvReleasePOSITObject( CvPOSITObject** posit_object );


extern "C" int cvRANSACUpdateNumIters( double p, double err_prob,
                                   int model_points, int max_iters );

extern "C" void cvConvertPointsHomogeneous( const CvMat* src, CvMat* dst );
# 1487 "../../../include/opencv/cv.h"
extern "C" int cvFindFundamentalMat( const CvMat* points1, const CvMat* points2,
                                 CvMat* fundamental_matrix,
                                 int method = 8,
                                 double param1 = 3., double param2 = 0.99,
                                 CvMat* status = NULL );




extern "C" void cvComputeCorrespondEpilines( const CvMat* points,
                                         int which_image,
                                         const CvMat* fundamental_matrix,
                                         CvMat* correspondent_lines );



extern "C" void cvTriangulatePoints(CvMat* projMatr1, CvMat* projMatr2,
                                CvMat* projPoints1, CvMat* projPoints2,
                                CvMat* points4D);

extern "C" void cvCorrectMatches(CvMat* F, CvMat* points1, CvMat* points2,
                             CvMat* new_points1, CvMat* new_points2);






typedef struct CvStereoBMState
{
    // pre-filtering (normalization of input images)
    int preFilterType; // =CV_STEREO_BM_NORMALIZED_RESPONSE now
    int preFilterSize; // averaging window size: ~5x5..21x21
    int preFilterCap; // the output of pre-filtering is clipped by [-preFilterCap,preFilterCap]

    // correspondence using Sum of Absolute Difference (SAD)
    int SADWindowSize; // ~5x5..21x21
    int minDisparity;  // minimum disparity (can be negative)
    int numberOfDisparities; // maximum disparity - minimum disparity (> 0)

    // post-filtering
    int textureThreshold;  // the disparity is only computed for pixels
                           // with textured enough neighborhood
    int uniquenessRatio;   // accept the computed disparity d* only if
                           // SAD(d) >= SAD(d*)*(1 + uniquenessRatio/100.)
                           // for any d != d*+/-1 within the search range.
    int speckleWindowSize; // disparity variation window
    int speckleRange; // acceptable range of variation in window

    int trySmallerWindows; // if 1, the results may be more accurate,
                           // at the expense of slower processing 
    CvRect roi1, roi2;
    int disp12MaxDiff;

    // temporary buffers
    CvMat* preFilteredImg0;
    CvMat* preFilteredImg1;
    CvMat* slidingSumBuf;
    CvMat* cost;
    CvMat* disp;
}
CvStereoBMState;





extern "C" CvStereoBMState* cvCreateStereoBMState(int preset = 0,
                                              int numberOfDisparities = 0);

extern "C" void cvReleaseStereoBMState( CvStereoBMState** state );

extern "C" void cvFindStereoCorrespondenceBM( const CvArr* left, const CvArr* right,
                                          CvArr* disparity, CvStereoBMState* state );




typedef struct CvStereoGCState
{
    int Ithreshold;
    int interactionRadius;
    float K, lambda, lambda1, lambda2;
    int occlusionCost;
    int minDisparity;
    int numberOfDisparities;
    int maxIters;

    CvMat* left;
    CvMat* right;
    CvMat* dispLeft;
    CvMat* dispRight;
    CvMat* ptrLeft;
    CvMat* ptrRight;
    CvMat* vtxBuf;
    CvMat* edgeBuf;
}
CvStereoGCState;

extern "C" CvStereoGCState* cvCreateStereoGCState( int numberOfDisparities, int maxIters );
extern "C" void cvReleaseStereoGCState( CvStereoGCState** state );

extern "C" void cvFindStereoCorrespondenceGC( const CvArr* left, const CvArr* right,
                                          CvArr* disparityLeft, CvArr* disparityRight,
                                          CvStereoGCState* state,
                                          int useDisparityGuess = 0 );


extern "C" void cvReprojectImageTo3D( const CvArr* disparityImage,
                                   CvArr* _3dImage, const CvMat* Q,
                                   int handleMissingValues = 0 );


}
