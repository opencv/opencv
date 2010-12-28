#include "precomp.hpp"

using namespace cv;

namespace cvtest
{

Size randomSize(RNG& rng, double maxSizeLog)
{
    double width_log = rng.uniform(0., maxSizeLog);
    double height_log = rng.uniform(0., maxSizeLog - width_log);
    if( (unsigned)rng % 2 != 0 )
        std::swap(width_log, height_log);
    Size sz;
    sz.width = cvRound(exp(width_log));
    sz.height = cvRound(exp(height_log));
    return sz;
}

void randomSize(RNG& rng, int minDims, int maxDims, double maxSizeLog, vector<int>& sz)
{
    int i, dims = rng.uniform(minDims, maxDims+1);
    sz.resize(dims);
    for( i = 0; i < dims; i++ )
    {
        double v = rng.uniform(0., maxSizeLog);
        maxSizeLog -= v;
        sz[i] = cvRound(exp(v));
    }
    for( i = 0; i < dims; i++ )
    {
        int j = rng.uniform(0, dims);
        int k = rng.uniform(0, dims);
        std::swap(sz[j], sz[k]);
    }
}

int randomType(RNG& rng, int typeMask, int minChannels, int maxChannels)
{
    int channels = rng.uniform(minChannels, maxChannels+1);
    int depth = 0;
    CV_Assert((typeMask & TYPE_MASK_ALL) != 0);
    for(;;)
    {
        depth = rng.uniform(CV_8U, CV_64F+1);
        if( ((1 << depth) & typeMask) != 0 )
            break;
    }
    return CV_MAKETYPE(depth, channels);
}

Mat randomMat(RNG& rng, Size size, int type, bool useRoi)
{
    
}

Mat randomMat(RNG& rng, const vector<int>& size, int type, bool useRoi)
{
}
    
Mat add(const Mat& _a, double alpha, const Mat& _b, double beta,
        Scalar gamma, Mat& c, int ctype, bool calcAbs)
{
    Mat a = _a, b = _b;
    if( a.empty() || alpha == 0 )
    {
        // both alpha and beta can be 0, but at least one of a and b must be non-empty array,
        // otherwise we do not know the size of the output (and may be type of the output, when ctype<0)
        CV_Assert( !a.empty() || !b.empty() );
        if( !b.empty() )
        {
            a = b;
            alpha = beta;
            b = Mat();
            beta = 0;
        }
    }
    if( b.empty() || beta == 0 )
    {
        b = Mat();
        beta = 0;
    }
    else
        CV_Assert(a.size == b.size);
    
    if( ctype < 0 )
        ctype = a.depth();
    ctype = CV_MAKETYPE(CV_MAT_DEPTH(ctype), a.channels());
    c.create(a.dims, &a.size[0], ctype);
    const Mat *arrays[3];
    Mat planes[3], buf[3];
    arrays[0] = &a;
    arrays[1] = b.empty() ? 0 : &b;
    arrays[2] = &c;
    
    NAryMatIterator it(arrays, planes, 3);
    int i, nplanes = it.nplanes, cn=a.channels();
    size_t total = planes[0].total(), maxsize = min(12*12*max(12/cn, 1), total);
    
    CV_Assert(planes[0].rows == 1);
    buf[0].create(1, (int)maxsize, CV_64FC(cn));
    if(!b.empty())
        buf[1].create(1, maxsize, CV_64FC(cn));
    buf[2].create(1, maxsize, CV_64FC(cn));
    scalarToRawData(gamma, buf[2].data, CV_64FC(cn), (int)(maxsize*cn));
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        for( size_t j = 0; j < total; j += maxsize )
        {
            size_t j2 = min(j + maxsize, total);
            Mat apart0 = planes[0].colRange((int)j, (int)j2);
            Mat cpart0 = planes[2].colRange((int)j, (int)j2);
            Mat apart = buf[0].colRange(0, (int)(j2 - j));
            
            apart0.convertTo(apart, apart.type(), alpha);
            size_t k, n = (j2 - j)*cn;
            double* aptr = (double*)apart.data;
            const double* gptr = (const double*)buf[2].data;
            
            if( b.empty() )
            {
                for( k = 0; k < n; k++ )
                    aptr[k] += gptr[k];
            }
            else
            {
                Mat bpart0 = planes[1].colRange((int)j, (int)j2);
                Mat bpart = buf[1].colRange(0, (int)(j2 - j));
                bpart0.convertTo(bpart, bpart.type(), beta);
                const double* bptr = (const double*)bpart.data;
                
                for( k = 0; k < n; k++ )
                    aptr[k] += bptr[k] + gptr[k];
            }
            if( calcAbs )
                for( k = 0; k < n; k++ )
                    aptr[k] = fabs(aptr[k]);
            apart.convertTo(cpart0, cpart0.type(), 1, 0);
        }
    }
}


static template<typename _Tp1, typename _Tp2> inline void
convert(const _Tp1* src, _Tp2* dst, size_t total, double alpha, double beta)
{
    size_t i;
    if( alpha == 1 && beta == 0 )
        for( i = 0; i < total; i++ )
            dst[i] = saturate_cast<_Tp2>(src[i]);
    else if( beta == 0 )
        for( i = 0; i < total; i++ )
            dst[i] = saturate_cast<_Tp2>(src[i]*alpha);
    else
        for( i = 0; i < total; i++ )
            dst[i] = saturate_cast<_Tp2>(src[i]*alpha + beta);
}
    
void convert(const Mat& src, Mat& dst, int dtype, double alpha, double beta)
{
    dtype = CV_MAKETYPE(CV_MAT_DEPTH(dtype), src.channels());
    dst.create(src.dims, &src.size[0], dtype);
    if( alpha == 0 )
    {
        set( dst, Scalar::all(beta) );
        return;
    }
    if( dtype == src.type() && alpha == 1 && beta == 0 )
    {
        copy( src, dst );
        return;
    }
    
    const Mat *arrays[]={&src, &dst};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t j, total = total = planes[0].total()*planes[0].channels();
    int i, nplanes = it.nplanes;
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        
        switch( src.depth() )
        {
        case 
        
        }
        
        for( j = 0; j < total; j++, sptr += elemSize, dptr += elemSize )
        {
            if( mptr[j] )
                for( k = 0; k < elemSize; k++ )
                    dptr[k] = sptr[k];
        }
    }
}
    
     
void copy(const Mat& src, Mat& dst, const Mat& mask)
{
    dst.create(src.dims, &src.size[0], src.type());
    
    if(mask.empty())
    {
        const Mat* arrays[] = {&src, &dst};
        Mat planes[2];
        NAryMatIterator it(arrays, planes, 2);
        int i, nplanes = it.nplanes;
        size_t planeSize = planes[0].total()*src.elemSize();
        
        for( i = 0; i < nplanes; i++, ++it )
            memcpy(planes[1].data, planes[0].data, planeSize);
        
        return;
    }
    
    CV_Assert( src.size == mask.size && mask.type() == CV_8U );
    
    const Mat *arrays[3]={&src, &dst, &mask};
    Mat planes[3];
    
    NAryMatIterator it(arrays, planes, 3);
    size_t j, k, elemSize = src.elemSize(), total = planes[0].total();
    int i, nplanes = it.nplanes;
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        const uchar* sptr = planes[0].data;
        uchar* dptr = planes[1].data;
        const uchar* mptr = planes[2].data;
        
        for( j = 0; j < total; j++, sptr += elemSize, dptr += elemSize )
        {
            if( mptr[j] )
                for( k = 0; k < elemSize; k++ )
                    dptr[k] = sptr[k];
        }
    }
}

    
void set(Mat& dst, const Scalar& gamma, const Mat& mask)
{
    double buf[12];
    scalarToRawData(gama, &buf, dst.type(), dst.channels());
    const uchar* gptr = (const uchar*)&buf[0];
    
    if(mask.empty())
    {
        const Mat* arrays[] = {&dst};
        Mat plane;
        NAryMatIterator it(arrays, &plane, 1);
        int i, nplanes = it.nplanes;
        size_t j, k, elemSize = dst.elemSize(), planeSize = planes[0].total()*elemSize;
        
        for( k = 1; k < elemSize; k++ )
            if( gptr[k] != gptr[0] )
                break;
        bool uniform = k >= elemSize;
        
        for( i = 0; i < nplanes; i++, ++it )
        {
            uchar* dptr = plane.data;
            if( uniform )
                memset( dptr, gptr[0], planeSize );
            else if( i == 0 )
            {
                for( j = 0; j < planeSize; j += elemSize, dptr += elemSize )
                    for( k = 0; k < elemSize; k++ )
                        dptr[k] = gptr[k];
            }
            else
                memcpy(dtr, dst.data, planeSize);
        }
        return;
    }
    
    CV_Assert( dst.size == mask.size && mask.type() == CV_8U );
    
    const Mat *arrays[2]={&dst, &mask};
    Mat planes[2];
    
    NAryMatIterator it(arrays, planes, 2);
    size_t j, k, elemSize = src.elemSize(), total = planes[0].total();
    int i, nplanes = it.nplanes;
    
    for( i = 0; i < nplanes; i++, ++it)
    {
        uchar* dptr = planes[0].data;
        const uchar* mptr = planes[1].data;
        
        for( j = 0; j < total; j++, dptr += elemSize )
        {
            if( mptr[j] )
                for( k = 0; k < elemSize; k++ )
                    dptr[k] = gptr[k];
        }
    }
}
    
    
void minMaxFilter(const Mat& a, Mat& maxresult, const Mat& minresult, const Mat& kernel, Point anchor);
void filter2D(const Mat& src, Mat& dst, int ddepth, const Mat& kernel, Point anchor, double delta, int borderType);
void copyMakeBorder(const Mat& src, Mat& dst, int top, int bottom, int left, int right, int borderType, Scalar borderValue);
void minMaxLoc(const Mat& src, double* maxval, double* minval,
               vector<int>* maxloc, vector<int>* minloc, const Mat& mask=Mat());
double norm(const Mat& src, int normType, const Mat& mask=Mat());
double norm(const Mat& src1, const Mat& src2, int normType, const Mat& mask=Mat());
bool cmpEps(const Mat& src1, const Mat& src2, int int_maxdiff, int flt_maxulp, vector<int>* loc);
void logicOp(const Mat& src1, const Mat& src2, Mat& dst, char c);
void logicOp(const Mat& src, const Scalar& s, Mat& dst, char c);
void compare(const Mat& src1, const Mat& src2, Mat& dst, int cmpop);
void compare(const Mat& src, const Scalar& s, Mat& dst, int cmpop);    

}
