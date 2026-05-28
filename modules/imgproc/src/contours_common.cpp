// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "contours_common.hpp"
#include <map>
#include <limits>
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/core/check.hpp"

using namespace std;
using namespace cv;

// calculates length of a curve (e.g. contour perimeter)
double cv::arcLength( InputArray _curve, bool is_closed )
{
    CV_INSTRUMENT_REGION();

    Mat curve = _curve.getMat();
    int count = curve.checkVector(2);
    int depth = curve.depth();
    CV_Assert( count >= 0 && (depth == CV_32F || depth == CV_32S));
    double perimeter = 0;

    int i;

    if( count <= 1 )
        return 0.;

    bool is_float = depth == CV_32F;
    int last = is_closed ? count-1 : 0;
    const Point* pti = curve.ptr<Point>();
    const Point2f* ptf = curve.ptr<Point2f>();

    Point2f prev = is_float ? ptf[last] : Point2f((float)pti[last].x,(float)pti[last].y);

    for( i = 0; i < count; i++ )
    {
        Point2f p = is_float ? ptf[i] : Point2f((float)pti[i].x,(float)pti[i].y);
        float dx = p.x - prev.x, dy = p.y - prev.y;
        perimeter += std::sqrt(dx*dx + dy*dy);

        prev = p;
    }

    return perimeter;
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
                if( j < xmin )
                    xmin = j;
                if( j > xmax )
                    xmax = j;
                have_nz = 1;
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

// Calculates bounding rectangle of a point set or retrieves already calculated
static Rect pointSetBoundingRect( const Mat& points )
{
    int npoints = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i = 0;
    bool is_float = depth == CV_32F;

    if( npoints == 0 )
        return Rect();

    if( !is_float )
    {
        const int32_t* pts = points.ptr<int32_t>();
        int64_t firstval = 0;
        std::memcpy(&firstval, pts, sizeof(pts[0]) * 2);
        xmin = xmax = pts[0];
        ymin = ymax = pts[1];
        #if CV_SIMD || CV_SIMD_SCALABLE
        v_int32 minval, maxval;
        minval = maxval = v_reinterpret_as_s32(vx_setall_s64(firstval)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        const int nlanes = VTraits<v_int32>::vlanes()/2;
        for (; i < npoints; i += nlanes)
        {
            if (i > npoints - nlanes)
            {
                if (i == 0)
                    break;
                i = npoints - nlanes;
            }
            v_int32 ptXY2 = vx_load(pts + 2 * i);
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        constexpr int max_nlanes = VTraits<v_int32>::max_nlanes;
        int arr_minval[max_nlanes], arr_maxval[max_nlanes];
        vx_store(arr_minval, minval);
        vx_store(arr_maxval, maxval);
        for (int j = 0; j < nlanes; j++)
        {
            xmin = std::min(xmin, arr_minval[2*j]);
            ymin = std::min(ymin, arr_minval[2*j+1]);
            xmax = std::max(xmax, arr_maxval[2*j]);
            ymax = std::max(ymax, arr_maxval[2*j+1]);
        }
        #endif
        for( ; i < npoints; i++ )
        {
            int pt_x = pts[2*i];
            int pt_y = pts[2*i+1];

            xmin = std::min(xmin, pt_x);
            xmax = std::max(xmax, pt_x);
            ymin = std::min(ymin, pt_y);
            ymax = std::max(ymax, pt_y);
        }
    }
    else
    {
        const float* pts = points.ptr<float>();
        int64_t firstval = 0;
        std::memcpy(&firstval, pts, sizeof(pts[0]) * 2);
        xmin = xmax = cvFloor(pts[0]);
        ymin = ymax = cvFloor(pts[1]);
        #if CV_SIMD || CV_SIMD_SCALABLE
        v_float32 minval, maxval;
        minval = maxval = v_reinterpret_as_f32(vx_setall_s64(firstval)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        const int nlanes = VTraits<v_float32>::vlanes()/2;
        for (; i < npoints; i += nlanes)
        {
            if (i > npoints - nlanes)
            {
                if (i == 0)
                    break;
                i = npoints - nlanes;
            }
            v_float32 ptXY2 = vx_load(pts + 2 * i);
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        constexpr int max_nlanes = VTraits<v_int32>::max_nlanes;
        float arr_minval[max_nlanes], arr_maxval[max_nlanes];
        vx_store(arr_minval, minval);
        vx_store(arr_maxval, maxval);
        for (int j = 0; j < nlanes; j++)
        {
            int _xmin = cvFloor(arr_minval[2*j]), _ymin = cvFloor(arr_minval[2*j+1]);
            int _xmax = cvFloor(arr_maxval[2*j]), _ymax = cvFloor(arr_maxval[2*j+1]);
            xmin = std::min(xmin, _xmin);
            ymin = std::min(ymin, _ymin);
            xmax = std::max(xmax, _xmax);
            ymax = std::max(ymax, _ymax);
        }
        #endif
        for( ; i < npoints; i++ )
        {
            // because right and bottom sides of the bounding rectangle are not inclusive
            // (note +1 in width and height calculation below), cvFloor is used here instead of cvCeil
            int pt_x = cvFloor(pts[2*i]);
            int pt_y = cvFloor(pts[2*i+1]);

            xmin = std::min(xmin, pt_x);
            xmax = std::max(xmax, pt_x);
            ymin = std::min(ymin, pt_y);
            ymax = std::max(ymax, pt_y);
        }
    }

    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

cv::Rect cv::boundingRect(InputArray array)
{
    CV_INSTRUMENT_REGION();

    Mat m = array.getMat();
    return m.depth() <= CV_8U ? maskBoundingRect(m) : pointSetBoundingRect(m);
}

// area of a whole sequence
double cv::contourArea( InputArray _contour, bool oriented )
{
    CV_INSTRUMENT_REGION();

    Mat contour = _contour.getMat();
    int npoints = contour.checkVector(2);
    int depth = contour.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    if( npoints == 0 )
        return 0.;

    double a00 = 0;
    bool is_float = depth == CV_32F;
    const Point* ptsi = contour.ptr<Point>();
    const Point2f* ptsf = contour.ptr<Point2f>();
    Point2f prev = is_float ? ptsf[npoints-1] : Point2f((float)ptsi[npoints-1].x, (float)ptsi[npoints-1].y);

    for( int i = 0; i < npoints; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        a00 += (double)prev.x * p.y - (double)prev.y * p.x;
        prev = p;
    }

    a00 *= 0.5;
    if( !oriented )
        a00 = fabs(a00);

    return a00;
}

void cv::contourTreeToResults(CTree& tree,
                              int res_type,
                              OutputArrayOfArrays& _contours,
                              OutputArray& _hierarchy)
{
    // check if there are no results
    if (tree.isEmpty() || (tree.elem(0).body.isEmpty() && (tree.elem(0).first_child == -1)))
    {
        _contours.clear();
        return;
    }

    CV_Assert(tree.size() < (size_t)numeric_limits<int>::max());
    // mapping for indexes (original -> resulting)
    // -1 - based indexing
    vector<int> index_mapping(tree.size() + 1, -1);

    const int total = (int)tree.size() - 1;
    _contours.create(total, 1, 0, -1, true);
    {
        int i = 0;
        CIterator it(tree);
        while (!it.isDone())
        {
            const CNode& elem = it.getNext_s();
            CV_Assert(elem.self() != -1);
            if (elem.self() == 0)
                continue;
            index_mapping.at(elem.self() + 1) = i;
            CV_Assert(elem.body.size() < (size_t)numeric_limits<int>::max());
            const int sz = (int)elem.body.size();
            _contours.create(sz, 1, res_type, i, true);
            if (sz > 0)
            {
                Mat cmat = _contours.getMat(i);
                CV_Assert(cmat.isContinuous());
                elem.body.copyTo(cmat.data);
            }
            ++i;
        }
    }

    if (_hierarchy.needed())
    {
        _hierarchy.create(1, total, CV_32SC4, -1, true);
        Mat h_mat = _hierarchy.getMat();
        int i = 0;
        CIterator it(tree);
        while (!it.isDone())
        {
            const CNode& elem = it.getNext_s();
            if (elem.self() == 0)
                continue;
            Vec4i& h_vec = h_mat.at<Vec4i>(i);
            h_vec = Vec4i(index_mapping.at(elem.next + 1),
                          index_mapping.at(elem.prev + 1),
                          index_mapping.at(elem.first_child + 1),
                          index_mapping.at(elem.parent + 1));
            ++i;
        }
    }
}
