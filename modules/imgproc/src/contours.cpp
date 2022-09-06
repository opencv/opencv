/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

using namespace cv;

/* initializes 8-element array for fast access to 3x3 neighborhood of a pixel */
#define  CV_INIT_3X3_DELTAS( deltas, step, nch )            \
    ((deltas)[0] =  (nch),  (deltas)[1] = -(step) + (nch),  \
     (deltas)[2] = -(step), (deltas)[3] = -(step) - (nch),  \
     (deltas)[4] = -(nch),  (deltas)[5] =  (step) - (nch),  \
     (deltas)[6] =  (step), (deltas)[7] =  (step) + (nch))

static const CvPoint icvCodeDeltas[8] = {
    {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}
};


namespace cv
{

// Calculates bounding rectangle of a point set or retrieves already calculated
static Rect pointSetBoundingRect( const Mat& points )
{
    int npoints = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
    bool is_float = depth == CV_32F;

    if( npoints == 0 )
        return Rect();

#if CV_SIMD
    const int64_t* pts = points.ptr<int64_t>();

    if( !is_float )
    {
        v_int32 minval, maxval;
        minval = maxval = v_reinterpret_as_s32(vx_setall_s64(*pts)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - v_int32::nlanes/2; i+= v_int32::nlanes/2 )
        {
            v_int32 ptXY2 = v_reinterpret_as_s32(vx_load(pts + i));
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - v_int32::nlanes/4 )
        {
            v_int32 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += v_int64::nlanes/2;
        }
        for(int j = 16; j < CV_SIMD_WIDTH; j*=2)
        {
            minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = minval.get0();
        xmax = maxval.get0();
        ymin = v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))).get0();
        ymax = v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))).get0();
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_int32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
            for( i++; i < npoints; i++ )
            {
                v_int32x4 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, minval2.get0());
            xmax = max(xmax, maxval2.get0());
            ymin = min(ymin, v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval2))).get0());
            ymax = max(ymax, v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval2))).get0());
        }
#endif
    }
    else
    {
        v_float32 minval, maxval;
        minval = maxval = v_reinterpret_as_f32(vx_setall_s64(*pts)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - v_float32::nlanes/2; i+= v_float32::nlanes/2 )
        {
            v_float32 ptXY2 = v_reinterpret_as_f32(vx_load(pts + i));
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - v_float32::nlanes/4 )
        {
            v_float32 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += v_float32::nlanes/4;
        }
        for(int j = 16; j < CV_SIMD_WIDTH; j*=2)
        {
            minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = cvFloor(minval.get0());
        xmax = cvFloor(maxval.get0());
        ymin = cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))).get0());
        ymax = cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))).get0());
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_float32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
            for( i++; i < npoints; i++ )
            {
                v_float32x4 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, cvFloor(minval2.get0()));
            xmax = max(xmax, cvFloor(maxval2.get0()));
            ymin = min(ymin, cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval2))).get0()));
            ymax = max(ymax, cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval2))).get0()));
        }
#endif
    }
#else
    const Point* pts = points.ptr<Point>();
    Point pt = pts[0];

    if( !is_float )
    {
        xmin = xmax = pt.x;
        ymin = ymax = pt.y;

        for( i = 1; i < npoints; i++ )
        {
            pt = pts[i];

            if( xmin > pt.x )
                xmin = pt.x;

            if( xmax < pt.x )
                xmax = pt.x;

            if( ymin > pt.y )
                ymin = pt.y;

            if( ymax < pt.y )
                ymax = pt.y;
        }
    }
    else
    {
        Cv32suf v;
        // init values
        xmin = xmax = CV_TOGGLE_FLT(pt.x);
        ymin = ymax = CV_TOGGLE_FLT(pt.y);

        for( i = 1; i < npoints; i++ )
        {
            pt = pts[i];
            pt.x = CV_TOGGLE_FLT(pt.x);
            pt.y = CV_TOGGLE_FLT(pt.y);

            if( xmin > pt.x )
                xmin = pt.x;

            if( xmax < pt.x )
                xmax = pt.x;

            if( ymin > pt.y )
                ymin = pt.y;

            if( ymax < pt.y )
                ymax = pt.y;
        }

        v.i = CV_TOGGLE_FLT(xmin); xmin = cvFloor(v.f);
        v.i = CV_TOGGLE_FLT(ymin); ymin = cvFloor(v.f);
        // because right and bottom sides of the bounding rectangle are not inclusive
        // (note +1 in width and height calculation below), cvFloor is used here instead of cvCeil
        v.i = CV_TOGGLE_FLT(xmax); xmax = cvFloor(v.f);
        v.i = CV_TOGGLE_FLT(ymax); ymax = cvFloor(v.f);
    }
#endif

    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}


static Rect maskBoundingRect( const Mat& img )
{
    CV_Assert( img.depth() <= CV_8S && img.channels() == 1 );

    Size size = img.size();
    int xmin = size.width, ymin = -1, xmax = -1, ymax = -1, i, j, k;

    for( i = 0; i < size.height; i++ )
    {
        const uchar* _ptr = img.ptr(i);
        const uchar* ptr = (const uchar*)alignPtr(_ptr, 4);
        int have_nz = 0, k_min, offset = (int)(ptr - _ptr);
        j = 0;
        offset = MIN(offset, size.width);
        for( ; j < offset; j++ )
            if( _ptr[j] )
            {
                have_nz = 1;
                break;
            }
        if( j < offset )
        {
            if( j < xmin )
                xmin = j;
            if( j > xmax )
                xmax = j;
        }
        if( offset < size.width )
        {
            xmin -= offset;
            xmax -= offset;
            size.width -= offset;
            j = 0;
            for( ; j <= xmin - 4; j += 4 )
                if( *((int*)(ptr+j)) )
                    break;
            for( ; j < xmin; j++ )
                if( ptr[j] )
                {
                    xmin = j;
                    if( j > xmax )
                        xmax = j;
                    have_nz = 1;
                    break;
                }
            k_min = MAX(j-1, xmax);
            k = size.width - 1;
            for( ; k > k_min && (k&3) != 3; k-- )
                if( ptr[k] )
                    break;
            if( k > k_min && (k&3) == 3 )
            {
                for( ; k > k_min+3; k -= 4 )
                    if( *((int*)(ptr+k-3)) )
                        break;
            }
            for( ; k > k_min; k-- )
                if( ptr[k] )
                {
                    xmax = k;
                    have_nz = 1;
                    break;
                }
            if( !have_nz )
            {
                j &= ~3;
                for( ; j <= k - 3; j += 4 )
                    if( *((int*)(ptr+j)) )
                        break;
                for( ; j <= k; j++ )
                    if( ptr[j] )
                    {
                        have_nz = 1;
                        break;
                    }
            }
            xmin += offset;
            xmax += offset;
            size.width += offset;
        }
        if( have_nz )
        {
            if( ymin < 0 )
                ymin = i;
            ymax = i;
        }
    }

    if( xmin >= size.width )
        xmin = ymin = 0;
    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

} // namespace cv::

cv::Rect cv::boundingRect(InputArray array)
{
    CV_INSTRUMENT_REGION();

    Mat m = array.getMat();
    return m.depth() <= CV_8U ? maskBoundingRect(m) : pointSetBoundingRect(m);
}

// utility function for cvBoundingRect
static CvSeq* cvPointSeqFromMat( int seq_kind, const CvArr* arr,
                                  CvContour* contour_header, CvSeqBlock* block )
{
    CV_Assert( arr != 0 && contour_header != 0 && block != 0 );

    int eltype;
    CvMat hdr;
    CvMat* mat = (CvMat*)arr;

    if( !CV_IS_MAT( mat ))
        CV_Error( CV_StsBadArg, "Input array is not a valid matrix" );

    if( CV_MAT_CN(mat->type) == 1 && mat->width == 2 )
        mat = cvReshape(mat, &hdr, 2);

    eltype = CV_MAT_TYPE( mat->type );
    if( eltype != CV_32SC2 && eltype != CV_32FC2 )
        CV_Error( CV_StsUnsupportedFormat,
        "The matrix can not be converted to point sequence because of "
        "inappropriate element type" );

    if( (mat->width != 1 && mat->height != 1) || !CV_IS_MAT_CONT(mat->type))
        CV_Error( CV_StsBadArg,
        "The matrix converted to point sequence must be "
        "1-dimensional and continuous" );

    cvMakeSeqHeaderForArray(
            (seq_kind & (CV_SEQ_KIND_MASK|CV_SEQ_FLAG_CLOSED)) | eltype,
            sizeof(CvContour), CV_ELEM_SIZE(eltype), mat->data.ptr,
            mat->width*mat->height, (CvSeq*)contour_header, block );

    return (CvSeq*)contour_header;
}

/* Calculates bounding rectangle of a point set or retrieves already calculated */
static CvRect cvBoundingRect( CvArr* array, int update )
{
    cv::Rect rect;
    CvContour contour_header;
    CvSeq* ptseq = 0;
    CvSeqBlock block;

    CvMat stub, *mat = 0;
    int calculate = update;

    if( CV_IS_SEQ( array ))
    {
        ptseq = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( ptseq ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );

        if( ptseq->header_size < (int)sizeof(CvContour))
        {
            update = 0;
            calculate = 1;
        }
    }
    else
    {
        mat = cvGetMat( array, &stub );
        if( CV_MAT_TYPE(mat->type) == CV_32SC2 ||
            CV_MAT_TYPE(mat->type) == CV_32FC2 )
        {
            ptseq = cvPointSeqFromMat(CV_SEQ_KIND_GENERIC, mat, &contour_header, &block);
            mat = 0;
        }
        else if( CV_MAT_TYPE(mat->type) != CV_8UC1 &&
                CV_MAT_TYPE(mat->type) != CV_8SC1 )
            CV_Error( CV_StsUnsupportedFormat,
                "The image/matrix format is not supported by the function" );
        update = 0;
        calculate = 1;
    }

    if( !calculate )
        return ((CvContour*)ptseq)->rect;

    if( mat )
    {
        rect = cvRect(cv::maskBoundingRect(cv::cvarrToMat(mat)));
    }
    else if( ptseq->total )
    {
        cv::AutoBuffer<double> abuf;
        rect = cvRect(cv::pointSetBoundingRect(cv::cvarrToMat(ptseq, false, false, 0, &abuf)));
    }
    if( update )
        ((CvContour*)ptseq)->rect = cvRect(rect);
    return cvRect(rect);
}

static double cvThreshold( const void* srcarr, void* dstarr, double thresh, double maxval, int type )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), dst0 = dst;

    CV_Assert( src.size == dst.size && src.channels() == dst.channels() &&
        (src.depth() == dst.depth() || dst.depth() == CV_8U));

    thresh = cv::threshold( src, dst, thresh, maxval, type );
    if( dst0.data != dst.data )
        dst.convertTo( dst0, dst0.depth() );
    return thresh;
}

/****************************************************************************************\
*                         Raster->Chain Tree (Suzuki algorithms)                         *
\****************************************************************************************/

typedef struct _CvContourInfo
{
    int flags;
    struct _CvContourInfo *next;        /* next contour with the same mark value */
    struct _CvContourInfo *parent;      /* information about parent contour */
    CvSeq *contour;             /* corresponding contour (may be 0, if rejected) */
    CvRect rect;                /* bounding rectangle */
    CvPoint origin;             /* origin point (where the contour was traced from) */
    int is_hole;                /* hole flag */
}
_CvContourInfo;


/*
  Structure that is used for sequential retrieving contours from the image.
  It supports both hierarchical and plane variants of Suzuki algorithm.
*/
typedef struct _CvContourScanner
{
    CvMemStorage *storage1;     /* contains fetched contours */
    CvMemStorage *storage2;     /* contains approximated contours
                                   (!=storage1 if approx_method2 != approx_method1) */
    CvMemStorage *cinfo_storage;        /* contains _CvContourInfo nodes */
    CvSet *cinfo_set;           /* set of _CvContourInfo nodes */
    CvMemStoragePos initial_pos;        /* starting storage pos */
    CvMemStoragePos backup_pos; /* beginning of the latest approx. contour */
    CvMemStoragePos backup_pos2;        /* ending of the latest approx. contour */
    schar *img0;                /* image origin */
    schar *img;                 /* current image row */
    int img_step;               /* image step */
    CvSize img_size;            /* ROI size */
    CvPoint offset;             /* ROI offset: coordinates, added to each contour point */
    CvPoint pt;                 /* current scanner position */
    CvPoint lnbd;               /* position of the last met contour */
    int nbd;                    /* current mark val */
    _CvContourInfo *l_cinfo;    /* information about latest approx. contour */
    _CvContourInfo cinfo_temp;  /* temporary var which is used in simple modes */
    _CvContourInfo frame_info;  /* information about frame */
    CvSeq frame;                /* frame itself */
    int approx_method1;         /* approx method when tracing */
    int approx_method2;         /* final approx method */
    int mode;                   /* contour scanning mode:
                                   0 - external only
                                   1 - all the contours w/o any hierarchy
                                   2 - connected components (i.e. two-level structure -
                                   external contours and holes),
                                   3 - full hierarchy;
                                   4 - connected components of a multi-level image
                                */
    int subst_flag;
    int seq_type1;              /* type of fetched contours */
    int header_size1;           /* hdr size of fetched contours */
    int elem_size1;             /* elem size of fetched contours */
    int seq_type2;              /*                                       */
    int header_size2;           /*        the same for approx. contours  */
    int elem_size2;             /*                                       */
    _CvContourInfo *cinfo_table[128];
}
_CvContourScanner;

/*
Internal structure that is used for sequential retrieving contours from the image.
It supports both hierarchical and plane variants of Suzuki algorithm.
*/
typedef struct _CvContourScanner* CvContourScanner;


#define _CV_FIND_CONTOURS_FLAGS_EXTERNAL_ONLY    1
#define _CV_FIND_CONTOURS_FLAGS_HIERARCHIC       2

/*
   Initializes scanner structure.
   Prepare image for scanning ( clear borders and convert all pixels to 0-1.
*/
static CvContourScanner
cvStartFindContours_Impl( void* _img, CvMemStorage* storage,
                     int  header_size, int mode,
                     int  method, CvPoint offset, int needFillBorder )
{
    if( !storage )
        CV_Error( CV_StsNullPtr, "" );

    CvMat stub, *mat = cvGetMat( _img, &stub );

    if( CV_MAT_TYPE(mat->type) == CV_32SC1 && mode == cv::RETR_CCOMP )
        mode = cv::RETR_FLOODFILL;

    if( !((CV_IS_MASK_ARR( mat ) && mode < cv::RETR_FLOODFILL) ||
          (CV_MAT_TYPE(mat->type) == CV_32SC1 && mode == cv::RETR_FLOODFILL)) )
        CV_Error( CV_StsUnsupportedFormat,
                  "[Start]FindContours supports only CV_8UC1 images when mode != cv::RETR_FLOODFILL "
                  "otherwise supports CV_32SC1 images only" );

    CvSize size = cvSize( mat->width, mat->height );
    int step = mat->step;
    uchar* img = (uchar*)(mat->data.ptr);

    if( method < 0 || method > cv::CHAIN_APPROX_TC89_KCOS )
        CV_Error( CV_StsOutOfRange, "" );

    if( header_size < (int) (method == cv::CHAIN_CODE ? sizeof( CvChain ) : sizeof( CvContour )))
        CV_Error( CV_StsBadSize, "" );

    CvContourScanner scanner = (CvContourScanner)cvAlloc( sizeof( *scanner ));
    memset( scanner, 0, sizeof(*scanner) );

    scanner->storage1 = scanner->storage2 = storage;
    scanner->img0 = (schar *) img;
    scanner->img = (schar *) (img + step);
    scanner->img_step = step;
    scanner->img_size.width = size.width - 1;   /* exclude rightest column */
    scanner->img_size.height = size.height - 1; /* exclude bottomost row */
    scanner->mode = mode;
    scanner->offset = offset;
    scanner->pt.x = scanner->pt.y = 1;
    scanner->lnbd.x = 0;
    scanner->lnbd.y = 1;
    scanner->nbd = 2;
    scanner->frame_info.contour = &(scanner->frame);
    scanner->frame_info.is_hole = 1;
    scanner->frame_info.next = 0;
    scanner->frame_info.parent = 0;
    scanner->frame_info.rect = cvRect( 0, 0, size.width, size.height );
    scanner->l_cinfo = 0;
    scanner->subst_flag = 0;

    scanner->frame.flags = CV_SEQ_FLAG_HOLE;

    scanner->approx_method2 = scanner->approx_method1 = method;

    if( method == cv::CHAIN_APPROX_TC89_L1 || method == cv::CHAIN_APPROX_TC89_KCOS )
        scanner->approx_method1 = cv::CHAIN_CODE;

    if( scanner->approx_method1 == cv::CHAIN_CODE )
    {
        scanner->seq_type1 = CV_SEQ_CHAIN_CONTOUR;
        scanner->header_size1 = scanner->approx_method1 == scanner->approx_method2 ?
            header_size : sizeof( CvChain );
        scanner->elem_size1 = sizeof( char );
    }
    else
    {
        scanner->seq_type1 = CV_SEQ_POLYGON;
        scanner->header_size1 = scanner->approx_method1 == scanner->approx_method2 ?
            header_size : sizeof( CvContour );
        scanner->elem_size1 = sizeof( CvPoint );
    }

    scanner->header_size2 = header_size;

    if( scanner->approx_method2 == cv::CHAIN_CODE )
    {
        scanner->seq_type2 = scanner->seq_type1;
        scanner->elem_size2 = scanner->elem_size1;
    }
    else
    {
        scanner->seq_type2 = CV_SEQ_POLYGON;
        scanner->elem_size2 = sizeof( CvPoint );
    }

    scanner->seq_type1 = scanner->approx_method1 == cv::CHAIN_CODE ?
        CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;

    scanner->seq_type2 = scanner->approx_method2 == cv::CHAIN_CODE ?
        CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;

    cvSaveMemStoragePos( storage, &(scanner->initial_pos) );

    if( method > cv::CHAIN_APPROX_SIMPLE )
    {
        scanner->storage1 = cvCreateChildMemStorage( scanner->storage2 );
    }

    if( mode > cv::RETR_LIST )
    {
        scanner->cinfo_storage = cvCreateChildMemStorage( scanner->storage2 );
        scanner->cinfo_set = cvCreateSet( 0, sizeof( CvSet ), sizeof( _CvContourInfo ),
                                          scanner->cinfo_storage );
    }

    CV_Assert(step >= 0);
    CV_Assert(size.height >= 1);

    /* make zero borders */
    if(needFillBorder)
    {
        int esz = CV_ELEM_SIZE(mat->type);
        memset( img, 0, size.width*esz );
        memset( img + static_cast<size_t>(step) * (size.height - 1), 0, size.width*esz );

        img += step;
        for( int y = 1; y < size.height - 1; y++, img += step )
        {
            for( int k = 0; k < esz; k++ )
                img[k] = img[(size.width - 1)*esz + k] = (schar)0;
        }
    }

    /* converts all pixels to 0 or 1 */
    if( CV_MAT_TYPE(mat->type) != CV_32S )
        cvThreshold( mat, mat, 0, 1, cv::THRESH_BINARY );

    return scanner;
}

static void
icvEndProcessContour( CvContourScanner scanner )
{
    _CvContourInfo *l_cinfo = scanner->l_cinfo;

    if( l_cinfo )
    {
        if( scanner->subst_flag )
        {
            CvMemStoragePos temp;

            cvSaveMemStoragePos( scanner->storage2, &temp );

            if( temp.top == scanner->backup_pos2.top &&
                temp.free_space == scanner->backup_pos2.free_space )
            {
                cvRestoreMemStoragePos( scanner->storage2, &scanner->backup_pos );
            }
            scanner->subst_flag = 0;
        }

        if( l_cinfo->contour )
        {
            cvInsertNodeIntoTree( l_cinfo->contour, l_cinfo->parent->contour,
                                  &(scanner->frame) );
        }
        scanner->l_cinfo = 0;
    }
}

static const int MAX_SIZE = 16;

/*
    marks domain border with +/-<constant> and stores the contour into CvSeq.
        method:
            <0  - chain
            ==0 - direct
            >0  - simple approximation
*/
static void
icvFetchContour( schar                  *ptr,
                 int                    step,
                 CvPoint                pt,
                 CvSeq*                 contour,
                 int    _method )
{
    const schar     nbd = 2;
    int             deltas[MAX_SIZE];
    CvSeqWriter     writer;
    schar           *i0 = ptr, *i1, *i3, *i4 = 0;
    int             prev_s = -1, s, s_end;
    int             method = _method - 1;

    CV_DbgAssert( (unsigned) _method <= cv::CHAIN_APPROX_SIMPLE );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *) contour)->origin = pt;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = (schar) (nbd | -128);
        if( method >= 0 )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            CV_Assert(i3 != NULL);
            s_end = s;
            s = std::min(s, MAX_SIZE - 1);

            while( s < MAX_SIZE - 1 )
            {
                i4 = i3 + deltas[++s];
                CV_Assert(i4 != NULL);
                if( *i4 != 0 )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = (schar) (nbd | -128);
            }
            else if( *i3 == 1 )
            {
                *i3 = nbd;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;

                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else
            {
                if( s != prev_s || method == 0 )
                {
                    CV_WRITE_SEQ_ELEM( pt, writer );
                    prev_s = s;
                }

                pt.x += icvCodeDeltas[s].x;
                pt.y += icvCodeDeltas[s].y;

            }

            if( i4 == i0 && i3 == i1 )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    cvEndWriteSeq( &writer );

    if( _method != cv::CHAIN_CODE )
        cvBoundingRect( contour, 1 );

    CV_DbgAssert( (writer.seq->total == 0 && writer.seq->first == 0) ||
            writer.seq->total > writer.seq->first->count ||
            (writer.seq->first->prev == writer.seq->first &&
             writer.seq->first->next == writer.seq->first) );
}



/*
   trace contour until certain point is met.
   returns 1 if met and this is the last contour
   encountered by a raster scan reaching the point, 0 else.
*/
static int
icvTraceContour( schar *ptr, int step, schar *stop_ptr, int is_hole )
{
    int deltas[MAX_SIZE];
    schar *i0 = ptr, *i1, *i3, *i4 = NULL;
    int s, s_end;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    CV_DbgAssert( (*i0 & -2) != 0 );

    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    i3 = i0;

    /* check single pixel domain */
    if( s != s_end )
    {
        /* follow border */
        for( ;; )
        {
            CV_Assert(i3 != NULL);

            s = std::min(s, MAX_SIZE - 1);
            while( s < MAX_SIZE - 1 )
            {
                i4 = i3 + deltas[++s];
                CV_Assert(i4 != NULL);
                if( *i4 != 0 )
                    break;
            }

            if (i3 == stop_ptr) {
                if (!(*i3 & 0x80)) {
                    /* it's the only contour */
                    return 1;
                }

                /* check if this is the last contour */
                /* encountered during a raster scan  */
                schar *i5;
                int t = s;
                while (true)
                {
                    t = (t - 1) & 7;
                    i5 = i3 + deltas[t];
                    if (*i5 != 0)
                        break;
                    if (t == 0)
                        return 1;
                }
            }

            if( (i4 == i0 && i3 == i1) )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
    else {
        return i3 == stop_ptr;
    }

    return 0;
}


static void
icvFetchContourEx( schar*               ptr,
                   int                  step,
                   CvPoint              pt,
                   CvSeq*               contour,
                   int  _method,
                   int                  nbd,
                   CvRect*              _rect )
{
    int         deltas[MAX_SIZE];
    CvSeqWriter writer;
    schar        *i0 = ptr, *i1, *i3, *i4 = NULL;
    cv::Rect    rect;
    int         prev_s = -1, s, s_end;
    int         method = _method - 1;

    CV_DbgAssert( (unsigned) _method <= cv::CHAIN_APPROX_SIMPLE );
    CV_DbgAssert( 1 < nbd && nbd < 128 );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *)contour)->origin = pt;

    rect.x = rect.width = pt.x;
    rect.y = rect.height = pt.y;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( *i1 == 0 && s != s_end );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = (schar) (nbd | 0x80);
        if( method >= 0 )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
    }
    else
    {
        i3 = i0;

        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            CV_Assert(i3 != NULL);
            s_end = s;
            s = std::min(s, MAX_SIZE - 1);

            while( s < MAX_SIZE - 1 )
            {
                i4 = i3 + deltas[++s];
                CV_Assert(i4 != NULL);
                if( *i4 != 0 )
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = (schar) (nbd | 0x80);
            }
            else if( *i3 == 1 )
            {
                *i3 = (schar) nbd;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;
                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else if( s != prev_s || method == 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
            }

            if( s != prev_s )
            {
                /* update bounds */
                if( pt.x < rect.x )
                    rect.x = pt.x;
                else if( pt.x > rect.width )
                    rect.width = pt.x;

                if( pt.y < rect.y )
                    rect.y = pt.y;
                else if( pt.y > rect.height )
                    rect.height = pt.y;
            }

            prev_s = s;
            pt.x += icvCodeDeltas[s].x;
            pt.y += icvCodeDeltas[s].y;

            if( i4 == i0 && i3 == i1 )  break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;

    cvEndWriteSeq( &writer );

    if( _method != cv::CHAIN_CODE )
        ((CvContour*)contour)->rect = cvRect(rect);

    CV_DbgAssert( (writer.seq->total == 0 && writer.seq->first == 0) ||
            writer.seq->total > writer.seq->first->count ||
            (writer.seq->first->prev == writer.seq->first &&
             writer.seq->first->next == writer.seq->first) );

    if( _rect )  *_rect = cvRect(rect);
}


static int
icvTraceContour_32s( int *ptr, int step, int *stop_ptr, int is_hole )
{
    CV_Assert(ptr != NULL);
    int deltas[MAX_SIZE];
    int *i0 = ptr, *i1, *i3, *i4 = NULL;
    int s, s_end;
    const int   right_flag = INT_MIN;
    const int   new_flag = (int)((unsigned)INT_MIN >> 1);
    const int   value_mask = ~(right_flag | new_flag);
    const int   ccomp_val = *i0 & value_mask;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( (*i1 & value_mask) != ccomp_val && s != s_end );

    i3 = i0;

    /* check single pixel domain */
    if( s != s_end )
    {
        /* follow border */
        for( ;; )
        {
            CV_Assert(i3 != NULL);
            s = std::min(s, MAX_SIZE - 1);

            while( s < MAX_SIZE - 1 )
            {
                i4 = i3 + deltas[++s];
                CV_Assert(i4 != NULL);
                if( (*i4 & value_mask) == ccomp_val )
                    break;
            }

            if( i3 == stop_ptr || (i4 == i0 && i3 == i1) )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
    return i3 == stop_ptr;
}


static void
icvFetchContourEx_32s( int*                 ptr,
                       int                  step,
                       CvPoint              pt,
                       CvSeq*               contour,
                       int                  _method,
                       CvRect*              _rect )
{
    CV_Assert(ptr != NULL);
    int         deltas[MAX_SIZE];
    CvSeqWriter writer;
    int        *i0 = ptr, *i1, *i3, *i4;
    cv::Rect    rect;
    int         prev_s = -1, s, s_end;
    int         method = _method - 1;
    const int   right_flag = INT_MIN;
    const int   new_flag = (int)((unsigned)INT_MIN >> 1);
    const int   value_mask = ~(right_flag | new_flag);
    const int   ccomp_val = *i0 & value_mask;
    const int   nbd0 = ccomp_val | new_flag;
    const int   nbd1 = nbd0 | right_flag;

    CV_DbgAssert( (unsigned) _method <= cv::CHAIN_APPROX_SIMPLE );

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    /* initialize writer */
    cvStartAppendToSeq( contour, &writer );

    if( method < 0 )
        ((CvChain *)contour)->origin = pt;

    rect.x = rect.width = pt.x;
    rect.y = rect.height = pt.y;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while( (*i1 & value_mask) != ccomp_val && s != s_end && ( s < MAX_SIZE - 1 ) );

    if( s == s_end )            /* single pixel domain */
    {
        *i0 = nbd1;
        if( method >= 0 )
        {
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for( ;; )
        {
            CV_Assert(i3 != NULL);
            s_end = s;

            do
            {
                i4 = i3 + deltas[++s];
                CV_Assert(i4 != NULL);
            }
            while( (*i4 & value_mask) != ccomp_val && ( s < MAX_SIZE - 1 ) );
            s &= 7;

            /* check "right" bound */
            if( (unsigned) (s - 1) < (unsigned) s_end )
            {
                *i3 = nbd1;
            }
            else if( *i3 == ccomp_val )
            {
                *i3 = nbd0;
            }

            if( method < 0 )
            {
                schar _s = (schar) s;
                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else if( s != prev_s || method == 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
            }

            if( s != prev_s )
            {
                /* update bounds */
                if( pt.x < rect.x )
                    rect.x = pt.x;
                else if( pt.x > rect.width )
                    rect.width = pt.x;

                if( pt.y < rect.y )
                    rect.y = pt.y;
                else if( pt.y > rect.height )
                    rect.height = pt.y;
            }

            prev_s = s;
            pt.x += icvCodeDeltas[s].x;
            pt.y += icvCodeDeltas[s].y;

            if( i4 == i0 && i3 == i1 )  break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;

    cvEndWriteSeq( &writer );

    if( _method != cv::CHAIN_CODE )
        ((CvContour*)contour)->rect = cvRect(rect);

    CV_DbgAssert( (writer.seq->total == 0 && writer.seq->first == 0) ||
           writer.seq->total > writer.seq->first->count ||
           (writer.seq->first->prev == writer.seq->first &&
            writer.seq->first->next == writer.seq->first) );

    if (_rect) *_rect = cvRect(rect);
}


static CvSeq * cvFindNextContour( CvContourScanner scanner )
{
    if( !scanner )
        CV_Error( CV_StsNullPtr, "" );

    CV_Assert(scanner->img_step >= 0);

    icvEndProcessContour( scanner );

    /* initialize local state */
    schar* img0 = scanner->img0;
    schar* img = scanner->img;
    int step = scanner->img_step;
    int step_i = step / sizeof(int);
    int x = scanner->pt.x;
    int y = scanner->pt.y;
    int width = scanner->img_size.width;
    int height = scanner->img_size.height;
    int mode = scanner->mode;
    cv::Point2i lnbd = scanner->lnbd;
    int nbd = scanner->nbd;
    int prev = img[x - 1];
    int new_mask = -2;

    if( mode == cv::RETR_FLOODFILL )
    {
        prev = ((int*)img)[x - 1];
        new_mask = INT_MIN / 2;
    }

    for( ; y < height; y++, img += step )
    {
        int* img0_i = 0;
        int* img_i = 0;
        int p = 0;

        if( mode == cv::RETR_FLOODFILL )
        {
            img0_i = (int*)img0;
            img_i = (int*)img;
        }

        for( ; x < width; x++ )
        {
            if( img_i )
            {
                for( ; x < width && ((p = img_i[x]) == prev || (p & ~new_mask) == (prev & ~new_mask)); x++ )
                    prev = p;
            }
            else
            {
#if CV_SIMD
                if ((p = img[x]) != prev)
                {
                    goto _next_contour;
                }
                else
                {
                    v_uint8 v_prev = vx_setall_u8((uchar)prev);
                    for (; x <= width - v_uint8::nlanes; x += v_uint8::nlanes)
                    {
                        v_uint8 vmask = (vx_load((uchar*)(img + x)) != v_prev);
                        if (v_check_any(vmask))
                        {
                            p = img[(x += v_scan_forward(vmask))];
                            goto _next_contour;
                        }
                    }
                }
#endif
                for( ; x < width && (p = img[x]) == prev; x++ )
                    ;
            }

            if( x >= width )
                break;
#if CV_SIMD
        _next_contour:
#endif
            {
                _CvContourInfo *par_info = 0;
                CvSeq *seq = 0;
                int is_hole = 0;
                cv::Point2i origin;

                /* if not external contour */
                if( (!img_i && !(prev == 0 && p == 1)) ||
                    (img_i && !(((prev & new_mask) != 0 || prev == 0) && (p & new_mask) == 0)) )
                {
                    /* check hole */
                    if( (!img_i && (p != 0 || prev < 1)) ||
                        (img_i && ((prev & new_mask) != 0 || (p & new_mask) != 0)))
                        goto resume_scan;

                    if( prev & new_mask )
                    {
                        lnbd.x = x - 1;
                    }
                    is_hole = 1;
                }

                if( mode == 0 && (is_hole || img0[lnbd.y * static_cast<size_t>(step) + lnbd.x] > 0) )
                    goto resume_scan;

                origin.y = y;
                origin.x = x - is_hole;

                /* find contour parent */
                if( mode <= 1 || (!is_hole && (mode == cv::RETR_CCOMP || mode == cv::RETR_FLOODFILL)) || lnbd.x <= 0 )
                {
                    par_info = &(scanner->frame_info);
                }
                else
                {
                    int lval = (img0_i ?
                        img0_i[lnbd.y * static_cast<size_t>(step_i) + lnbd.x] :
                        (int)img0[lnbd.y * static_cast<size_t>(step) + lnbd.x]) & 0x7f;
                    _CvContourInfo *cur = scanner->cinfo_table[lval];

                    /* find the first bounding contour */
                    while( cur )
                    {
                        if( (unsigned) (lnbd.x - cur->rect.x) < (unsigned) cur->rect.width &&
                            (unsigned) (lnbd.y - cur->rect.y) < (unsigned) cur->rect.height )
                        {
                            if( par_info )
                            {
                                if( (img0_i &&
                                     icvTraceContour_32s( img0_i + par_info->origin.y * static_cast<size_t>(step_i) +
                                                          par_info->origin.x, step_i, img_i + lnbd.x,
                                                          par_info->is_hole ) > 0) ||
                                    (!img0_i &&
                                     icvTraceContour( img0 + par_info->origin.y * static_cast<size_t>(step) +
                                                      par_info->origin.x, step, img + lnbd.x,
                                                      par_info->is_hole ) > 0) )
                                    break;
                            }
                            par_info = cur;
                        }
                        cur = cur->next;
                    }

                    CV_Assert( par_info != 0 );

                    /* if current contour is a hole and previous contour is a hole or
                       current contour is external and previous contour is external then
                       the parent of the contour is the parent of the previous contour else
                       the parent is the previous contour itself. */
                    if( par_info->is_hole == is_hole )
                    {
                        par_info = par_info->parent;
                        /* every contour must have a parent
                           (at least, the frame of the image) */
                        if( !par_info )
                            par_info = &(scanner->frame_info);
                    }

                    /* hole flag of the parent must differ from the flag of the contour */
                    CV_Assert( par_info->is_hole != is_hole );
                    if( par_info->contour == 0 )        /* removed contour */
                        goto resume_scan;
                }

                lnbd.x = x - is_hole;

                cvSaveMemStoragePos( scanner->storage2, &(scanner->backup_pos) );

                seq = cvCreateSeq( scanner->seq_type1, scanner->header_size1,
                                   scanner->elem_size1, scanner->storage1 );
                seq->flags |= is_hole ? CV_SEQ_FLAG_HOLE : 0;

                /* initialize header */
                _CvContourInfo *l_cinfo = 0;
                if( mode <= 1 )
                {
                    l_cinfo = &(scanner->cinfo_temp);
                    icvFetchContour( img + x - is_hole, step,
                                     cvPoint( origin.x + scanner->offset.x,
                                              origin.y + scanner->offset.y),
                                     seq, scanner->approx_method1 );
                }
                else
                {
                    cvSetAdd(scanner->cinfo_set, 0, (CvSetElem**)&l_cinfo);
                    CV_Assert(l_cinfo);
                    int lval;

                    if( img_i )
                    {
                        lval = img_i[x - is_hole] & 127;
                        icvFetchContourEx_32s(img_i + x - is_hole, step_i,
                                              cvPoint( origin.x + scanner->offset.x,
                                                       origin.y + scanner->offset.y),
                                              seq, scanner->approx_method1,
                                              &(l_cinfo->rect) );
                    }
                    else
                    {
                        lval = nbd;
                        // change nbd
                        nbd = (nbd + 1) & 127;
                        nbd += nbd == 0 ? 3 : 0;
                        icvFetchContourEx( img + x - is_hole, step,
                                           cvPoint( origin.x + scanner->offset.x,
                                                    origin.y + scanner->offset.y),
                                           seq, scanner->approx_method1,
                                           lval, &(l_cinfo->rect) );
                    }
                    l_cinfo->rect.x -= scanner->offset.x;
                    l_cinfo->rect.y -= scanner->offset.y;

                    l_cinfo->next = scanner->cinfo_table[lval];
                    scanner->cinfo_table[lval] = l_cinfo;
                }

                l_cinfo->is_hole = is_hole;
                l_cinfo->contour = seq;
                l_cinfo->origin = cvPoint(origin);
                l_cinfo->parent = par_info;

                if( scanner->approx_method1 != scanner->approx_method2 )
                {
                    l_cinfo->contour = icvApproximateChainTC89( (CvChain *) seq,
                                                      scanner->header_size2,
                                                      scanner->storage2,
                                                      scanner->approx_method2 );
                    cvClearMemStorage( scanner->storage1 );
                }

                l_cinfo->contour->v_prev = l_cinfo->parent->contour;

                if( par_info->contour == 0 )
                {
                    l_cinfo->contour = 0;
                    if( scanner->storage1 == scanner->storage2 )
                    {
                        cvRestoreMemStoragePos( scanner->storage1, &(scanner->backup_pos) );
                    }
                    else
                    {
                        cvClearMemStorage( scanner->storage1 );
                    }
                    p = img[x];
                    goto resume_scan;
                }

                cvSaveMemStoragePos( scanner->storage2, &(scanner->backup_pos2) );
                scanner->l_cinfo = l_cinfo;
                scanner->pt.x = !img_i ? x + 1 : x + 1 - is_hole;
                scanner->pt.y = y;
                scanner->lnbd = cvPoint(lnbd);
                scanner->img = (schar *) img;
                scanner->nbd = nbd;
                return l_cinfo->contour;
            }
        resume_scan:
            {
                prev = p;
                /* update lnbd */
                if( prev & -2 )
                {
                    lnbd.x = x;
                }
            }
        }                       /* end of loop on x */

        lnbd.x = 0;
        lnbd.y = y + 1;
        x = 1;
        prev = 0;
    }                           /* end of loop on y */

    return 0;
}


/*
   The function add to tree the last retrieved/substituted contour,
   releases temp_storage, restores state of dst_storage (if needed), and
   returns pointer to root of the contour tree */
static  CvSeq * cvEndFindContours( CvContourScanner * _scanner )
{
    CvContourScanner scanner;
    CvSeq *first = 0;

    if( !_scanner )
        CV_Error( CV_StsNullPtr, "" );
    scanner = *_scanner;

    if( scanner )
    {
        icvEndProcessContour( scanner );

        if( scanner->storage1 != scanner->storage2 )
            cvReleaseMemStorage( &(scanner->storage1) );

        if( scanner->cinfo_storage )
            cvReleaseMemStorage( &(scanner->cinfo_storage) );

        first = scanner->frame.v_next;
        cvFree( _scanner );
    }

    return first;
}


#define ICV_SINGLE                  0
#define ICV_CONNECTING_ABOVE        1
#define ICV_CONNECTING_BELOW        -1

#define CV_GET_WRITTEN_ELEM( writer ) ((writer).ptr - (writer).seq->elem_size)

typedef  struct CvLinkedRunPoint
{
    struct CvLinkedRunPoint* link;
    struct CvLinkedRunPoint* next;
    CvPoint pt;
}
CvLinkedRunPoint;

inline int findStartContourPoint(uchar *src_data, CvSize img_size, int j)
{
#if CV_SIMD
    v_uint8 v_zero = vx_setzero_u8();
    for (; j <= img_size.width - v_uint8::nlanes; j += v_uint8::nlanes)
    {
        v_uint8 vmask = (vx_load((uchar*)(src_data + j)) != v_zero);
        if (v_check_any(vmask))
        {
            j += v_scan_forward(vmask);
            return j;
        }
    }
#endif
    for (; j < img_size.width && !src_data[j]; ++j)
        ;
    return j;
}

inline int findEndContourPoint(uchar *src_data, CvSize img_size, int j)
{
#if CV_SIMD
    if (j < img_size.width && !src_data[j])
    {
        return j;
    }
    else
    {
        v_uint8 v_zero = vx_setzero_u8();
        for (; j <= img_size.width - v_uint8::nlanes; j += v_uint8::nlanes)
        {
            v_uint8 vmask = (vx_load((uchar*)(src_data + j)) == v_zero);
            if (v_check_any(vmask))
            {
                j += v_scan_forward(vmask);
                return j;
            }
        }
    }
#endif
    for (; j < img_size.width && src_data[j]; ++j)
        ;

    return j;
}

static int
icvFindContoursInInterval( const CvArr* src,
                           /*int minValue, int maxValue,*/
                           CvMemStorage* storage,
                           CvSeq** result,
                           int contourHeaderSize )
{
    int count = 0;
    cv::Ptr<CvMemStorage> storage00;
    cv::Ptr<CvMemStorage> storage01;
    CvSeq* first = 0;

    int j, k, n;

    uchar*  src_data = 0;
    int  img_step = 0;
    cv::Size img_size;

    int  connect_flag;
    int  lower_total;
    int  upper_total;
    int  all_total;

    CvSeq*  runs;
    CvLinkedRunPoint  tmp;
    CvLinkedRunPoint*  tmp_prev;
    CvLinkedRunPoint*  upper_line = 0;
    CvLinkedRunPoint*  lower_line = 0;
    CvLinkedRunPoint*  last_elem;

    CvLinkedRunPoint*  upper_run = 0;
    CvLinkedRunPoint*  lower_run = 0;
    CvLinkedRunPoint*  prev_point = 0;

    CvSeqWriter  writer_ext;
    CvSeqWriter  writer_int;
    CvSeqWriter  writer;
    CvSeqReader  reader;

    CvSeq* external_contours;
    CvSeq* internal_contours;
    CvSeq* prev = 0;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    if( !result )
        CV_Error( CV_StsNullPtr, "NULL double CvSeq pointer" );

    if( contourHeaderSize < (int)sizeof(CvContour))
        CV_Error( CV_StsBadSize, "Contour header size must be >= sizeof(CvContour)" );

    storage00.reset(cvCreateChildMemStorage(storage));
    storage01.reset(cvCreateChildMemStorage(storage));

    CvMat stub, *mat;

    mat = cvGetMat( src, &stub );
    if( !CV_IS_MASK_ARR(mat))
        CV_Error( CV_StsBadArg, "Input array must be 8uC1 or 8sC1" );
    src_data = mat->data.ptr;
    img_step = mat->step;
    img_size = cvGetMatSize(mat);

    // Create temporary sequences
    runs = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvLinkedRunPoint), storage00 );
    cvStartAppendToSeq( runs, &writer );

    cvStartWriteSeq( 0, sizeof(CvSeq), sizeof(CvLinkedRunPoint*), storage01, &writer_ext );
    cvStartWriteSeq( 0, sizeof(CvSeq), sizeof(CvLinkedRunPoint*), storage01, &writer_int );

    tmp_prev = &(tmp);
    tmp_prev->next = 0;
    tmp_prev->link = 0;

    // First line. None of runs is binded
    tmp.pt.x = 0;
    tmp.pt.y = 0;
    CV_WRITE_SEQ_ELEM( tmp, writer );
    upper_line = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );

    tmp_prev = upper_line;
    for( j = 0; j < img_size.width; )
    {
        j = findStartContourPoint(src_data, cvSize(img_size), j);

        if( j == img_size.width )
            break;

        tmp.pt.x = j;
        CV_WRITE_SEQ_ELEM( tmp, writer );
        tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
        tmp_prev = tmp_prev->next;

        j = findEndContourPoint(src_data, cvSize(img_size), j + 1);

        tmp.pt.x = j - 1;
        CV_WRITE_SEQ_ELEM( tmp, writer );
        tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
        tmp_prev->link = tmp_prev->next;
        // First point of contour
        CV_WRITE_SEQ_ELEM( tmp_prev, writer_ext );
        tmp_prev = tmp_prev->next;
    }
    cvFlushSeqWriter( &writer );
    upper_line = upper_line->next;
    upper_total = runs->total - 1;
    last_elem = tmp_prev;
    tmp_prev->next = 0;

    for( int i = 1; i < img_size.height; i++ )
    {
//------// Find runs in next line
        src_data += img_step;
        tmp.pt.y = i;
        all_total = runs->total;
        for( j = 0; j < img_size.width; )
        {
            j = findStartContourPoint(src_data, cvSize(img_size), j);

            if( j == img_size.width ) break;

            tmp.pt.x = j;
            CV_WRITE_SEQ_ELEM( tmp, writer );
            tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
            tmp_prev = tmp_prev->next;

            j = findEndContourPoint(src_data, cvSize(img_size), j + 1);

            tmp.pt.x = j - 1;
            CV_WRITE_SEQ_ELEM( tmp, writer );
            tmp_prev = tmp_prev->next = (CvLinkedRunPoint*)CV_GET_WRITTEN_ELEM( writer );
        }//j
        cvFlushSeqWriter( &writer );
        lower_line = last_elem->next;
        lower_total = runs->total - all_total;
        last_elem = tmp_prev;
        tmp_prev->next = 0;
//------//
//------// Find links between runs of lower_line and upper_line
        upper_run = upper_line;
        lower_run = lower_line;
        connect_flag = ICV_SINGLE;

        for( k = 0, n = 0; k < upper_total/2 && n < lower_total/2; )
        {
            switch( connect_flag )
            {
            case ICV_SINGLE:
                if( upper_run->next->pt.x < lower_run->next->pt.x )
                {
                    if( upper_run->next->pt.x >= lower_run->pt.x  -1 )
                    {
                        lower_run->link = upper_run;
                        connect_flag = ICV_CONNECTING_ABOVE;
                        prev_point = upper_run->next;
                    }
                    else
                        upper_run->next->link = upper_run;
                    k++;
                    upper_run = upper_run->next->next;
                }
                else
                {
                    if( upper_run->pt.x <= lower_run->next->pt.x  +1 )
                    {
                        lower_run->link = upper_run;
                        connect_flag = ICV_CONNECTING_BELOW;
                        prev_point = lower_run->next;
                    }
                    else
                    {
                        lower_run->link = lower_run->next;
                        // First point of contour
                        CV_WRITE_SEQ_ELEM( lower_run, writer_ext );
                    }
                    n++;
                    lower_run = lower_run->next->next;
                }
                break;
            case ICV_CONNECTING_ABOVE:
                if( upper_run->pt.x > lower_run->next->pt.x +1 )
                {
                    prev_point->link = lower_run->next;
                    connect_flag = ICV_SINGLE;
                    n++;
                    lower_run = lower_run->next->next;
                }
                else
                {
                    prev_point->link = upper_run;
                    if( upper_run->next->pt.x < lower_run->next->pt.x )
                    {
                        k++;
                        prev_point = upper_run->next;
                        upper_run = upper_run->next->next;
                    }
                    else
                    {
                        connect_flag = ICV_CONNECTING_BELOW;
                        prev_point = lower_run->next;
                        n++;
                        lower_run = lower_run->next->next;
                    }
                }
                break;
            case ICV_CONNECTING_BELOW:
                if( lower_run->pt.x > upper_run->next->pt.x +1 )
                {
                    upper_run->next->link = prev_point;
                    connect_flag = ICV_SINGLE;
                    k++;
                    upper_run = upper_run->next->next;
                }
                else
                {
                    // First point of contour
                    CV_WRITE_SEQ_ELEM( lower_run, writer_int );

                    lower_run->link = prev_point;
                    if( lower_run->next->pt.x < upper_run->next->pt.x )
                    {
                        n++;
                        prev_point = lower_run->next;
                        lower_run = lower_run->next->next;
                    }
                    else
                    {
                        connect_flag = ICV_CONNECTING_ABOVE;
                        k++;
                        prev_point = upper_run->next;
                        upper_run = upper_run->next->next;
                    }
                }
                break;
            }
        }// k, n

        for( ; n < lower_total/2; n++ )
        {
            if( connect_flag != ICV_SINGLE )
            {
                prev_point->link = lower_run->next;
                connect_flag = ICV_SINGLE;
                lower_run = lower_run->next->next;
                continue;
            }
            lower_run->link = lower_run->next;

            //First point of contour
            CV_WRITE_SEQ_ELEM( lower_run, writer_ext );

            lower_run = lower_run->next->next;
        }

        for( ; k < upper_total/2; k++ )
        {
            if( connect_flag != ICV_SINGLE )
            {
                upper_run->next->link = prev_point;
                connect_flag = ICV_SINGLE;
                upper_run = upper_run->next->next;
                continue;
            }
            upper_run->next->link = upper_run;
            upper_run = upper_run->next->next;
        }
        upper_line = lower_line;
        upper_total = lower_total;
    }//i

    upper_run = upper_line;

    //the last line of image
    for( k = 0; k < upper_total/2; k++ )
    {
        upper_run->next->link = upper_run;
        upper_run = upper_run->next->next;
    }

//------//
//------//Find end read contours
    external_contours = cvEndWriteSeq( &writer_ext );
    internal_contours = cvEndWriteSeq( &writer_int );

    for( k = 0; k < 2; k++ )
    {
        CvSeq* contours = k == 0 ? external_contours : internal_contours;

        cvStartReadSeq( contours, &reader );

        for( j = 0; j < contours->total; j++, count++ )
        {
            CvLinkedRunPoint* p_temp;
            CvLinkedRunPoint* p00;
            CvLinkedRunPoint* p01;
            CvSeq* contour;

            CV_READ_SEQ_ELEM( p00, reader );
            p01 = p00;

            if( !p00->link )
                continue;

            cvStartWriteSeq( CV_SEQ_ELTYPE_POINT | CV_SEQ_POLYLINE | CV_SEQ_FLAG_CLOSED,
                             contourHeaderSize, sizeof(CvPoint), storage, &writer );
            do
            {
                CV_WRITE_SEQ_ELEM( p00->pt, writer );
                p_temp = p00;
                p00 = p00->link;
                p_temp->link = 0;
            }
            while( p00 != p01 );

            contour = cvEndWriteSeq( &writer );
            cvBoundingRect( contour, 1 );

            if( k != 0 )
                contour->flags |= CV_SEQ_FLAG_HOLE;

            if( !first )
                prev = first = contour;
            else
            {
                contour->h_prev = prev;
                prev = prev->h_next = contour;
            }
        }
    }

    if( !first )
        count = -1;

    if( result )
        *result = first;

    return count;
}

static int
cvFindContours_Impl( void*  img,  CvMemStorage*  storage,
                CvSeq**  firstContour, int  cntHeaderSize,
                int  mode,
                int  method, CvPoint offset, int needFillBorder )
{
    CvContourScanner scanner = 0;
    CvSeq *contour = 0;
    int count = -1;

    if( !firstContour )
        CV_Error( CV_StsNullPtr, "NULL double CvSeq pointer" );

    *firstContour = 0;

    if( method == cv::LINK_RUNS )
    {
        if( offset.x != 0 || offset.y != 0 )
            CV_Error( CV_StsOutOfRange,
            "Nonzero offset is not supported in cv::LINK_RUNS yet" );

        count = icvFindContoursInInterval( img, storage, firstContour, cntHeaderSize );
    }
    else
    {
        try
        {
            scanner = cvStartFindContours_Impl( img, storage, cntHeaderSize, mode, method, offset,
                                            needFillBorder);

            do
            {
                count++;
                contour = cvFindNextContour( scanner );
            }
            while( contour != 0 );
        }
        catch(...)
        {
            if( scanner )
                cvEndFindContours(&scanner);
            throw;
        }

        *firstContour = cvEndFindContours( &scanner );
    }

    return count;
}

void cv::findContours( InputArray _image, OutputArrayOfArrays _contours,
                   OutputArray _hierarchy, int mode, int method, Point offset )
{
    CV_INSTRUMENT_REGION();

    // Sanity check: output must be of type vector<vector<Point>>
    CV_Assert((_contours.kind() == _InputArray::STD_VECTOR_VECTOR || _contours.kind() == _InputArray::STD_VECTOR_MAT ||
                _contours.kind() == _InputArray::STD_VECTOR_UMAT));

    CV_Assert(_contours.empty() || (_contours.channels() == 2 && _contours.depth() == CV_32S));

    Mat image0 = _image.getMat(), image;
    Point offset0(0, 0);
    if(method != cv::LINK_RUNS)
    {
        offset0 = Point(-1, -1);
        copyMakeBorder(image0, image, 1, 1, 1, 1, BORDER_CONSTANT | BORDER_ISOLATED, Scalar(0));
    }
    else
    {
        image = image0;
    }
    MemStorage storage(cvCreateMemStorage());
    CvMat _cimage = cvMat(image);
    CvSeq* _ccontours = 0;
    if( _hierarchy.needed() )
        _hierarchy.clear();
    cvFindContours_Impl(&_cimage, storage, &_ccontours, sizeof(CvContour), mode, method, cvPoint(offset0 + offset), 0);
    if( !_ccontours )
    {
        _contours.clear();
        return;
    }
    Seq<CvSeq*> all_contours(cvTreeToNodeSeq( _ccontours, sizeof(CvSeq), storage ));
    int i, total = (int)all_contours.size();
    _contours.create(total, 1, 0, -1, true);
    SeqIterator<CvSeq*> it = all_contours.begin();
    for( i = 0; i < total; i++, ++it )
    {
        CvSeq* c = *it;
        ((CvContour*)c)->color = (int)i;
        _contours.create((int)c->total, 1, CV_32SC2, i, true);
        Mat ci = _contours.getMat(i);
        CV_Assert( ci.isContinuous() );
        cvCvtSeqToArray(c, ci.ptr());
    }

    if( _hierarchy.needed() )
    {
        _hierarchy.create(1, total, CV_32SC4, -1, true);
        Vec4i* hierarchy = _hierarchy.getMat().ptr<Vec4i>();

        it = all_contours.begin();
        for( i = 0; i < total; i++, ++it )
        {
            CvSeq* c = *it;
            int h_next = c->h_next ? ((CvContour*)c->h_next)->color : -1;
            int h_prev = c->h_prev ? ((CvContour*)c->h_prev)->color : -1;
            int v_next = c->v_next ? ((CvContour*)c->v_next)->color : -1;
            int v_prev = c->v_prev ? ((CvContour*)c->v_prev)->color : -1;
            hierarchy[i] = Vec4i(h_next, h_prev, v_next, v_prev);
        }
    }
}

void cv::findContours( InputArray _image, OutputArrayOfArrays _contours,
                       int mode, int method, Point offset)
{
    CV_INSTRUMENT_REGION();

    findContours(_image, _contours, noArray(), mode, method, offset);
}

/* End of file. */
