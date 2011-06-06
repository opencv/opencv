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
#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
static IppStatus sts = ippInit();
#endif
/****************************************************************************************/

/* lightweight convolution with 3x3 kernel */
void icvSepConvSmall3_32f( float* src, int src_step, float* dst, int dst_step,
            CvSize src_size, const float* kx, const float* ky, float* buffer )
{
    int  dst_width, buffer_step = 0;
    int  x, y;

    assert( src && dst && src_size.width > 2 && src_size.height > 2 &&
            (src_step & 3) == 0 && (dst_step & 3) == 0 &&
            (kx || ky) && (buffer || !kx || !ky));

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);

    dst_width = src_size.width - 2;

    if( !kx )
    {
        /* set vars, so that vertical convolution
           will write results into destination ROI and
           horizontal convolution won't run */
        src_size.width = dst_width;
        buffer_step = dst_step;
        buffer = dst;
        dst_width = 0;
    }

    assert( src_step >= src_size.width && dst_step >= dst_width );

    src_size.height -= 3;
    if( !ky )
    {
        /* set vars, so that vertical convolution won't run and
           horizontal convolution will write results into destination ROI */
        src_size.height += 3;
        buffer_step = src_step;
        buffer = src;
        src_size.width = 0;
    }

    for( y = 0; y <= src_size.height; y++, src += src_step,
                                           dst += dst_step,
                                           buffer += buffer_step )
    {
        float* src2 = src + src_step;
        float* src3 = src + src_step*2;
        for( x = 0; x < src_size.width; x++ )
        {
            buffer[x] = (float)(ky[0]*src[x] + ky[1]*src2[x] + ky[2]*src3[x]);
        }

        for( x = 0; x < dst_width; x++ )
        {
            dst[x] = (float)(kx[0]*buffer[x] + kx[1]*buffer[x+1] + kx[2]*buffer[x+2]);
        }
    }
}


/****************************************************************************************\
                             Sobel & Scharr Derivative Filters
\****************************************************************************************/

namespace cv
{

static void getScharrKernels( OutputArray _kx, OutputArray _ky,
                              int dx, int dy, bool normalize, int ktype )
{
    const int ksize = 3;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    _kx.create(ksize, 1, ktype, -1, true);
    _ky.create(ksize, 1, ktype, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();

    CV_Assert( dx >= 0 && dy >= 0 && dx+dy == 1 );

    for( int k = 0; k < 2; k++ )
    {
        Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        int kerI[3];

        if( order == 0 )
            kerI[0] = 3, kerI[1] = 10, kerI[2] = 3;
        else if( order == 1 )
            kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;

        Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
        double scale = !normalize || order == 1 ? 1. : 1./32;
        temp.convertTo(*kernel, ktype, scale);
    }
}


static void getSobelKernels( OutputArray _kx, OutputArray _ky,
                             int dx, int dy, int _ksize, bool normalize, int ktype )
{
    int i, j, ksizeX = _ksize, ksizeY = _ksize;
    if( ksizeX == 1 && dx > 0 )
        ksizeX = 3;
    if( ksizeY == 1 && dy > 0 )
        ksizeY = 3;

    CV_Assert( ktype == CV_32F || ktype == CV_64F );

    _kx.create(ksizeX, 1, ktype, -1, true);
    _ky.create(ksizeY, 1, ktype, -1, true);
    Mat kx = _kx.getMat();
    Mat ky = _ky.getMat();    

    if( _ksize % 2 == 0 || _ksize > 31 )
        CV_Error( CV_StsOutOfRange, "The kernel size must be odd and not larger than 31" );
    vector<int> kerI(std::max(ksizeX, ksizeY) + 1);

    CV_Assert( dx >= 0 && dy >= 0 && dx+dy > 0 );

    for( int k = 0; k < 2; k++ )
    {
        Mat* kernel = k == 0 ? &kx : &ky;
        int order = k == 0 ? dx : dy;
        int ksize = k == 0 ? ksizeX : ksizeY;

        CV_Assert( ksize > order );

        if( ksize == 1 )
            kerI[0] = 1;
        else if( ksize == 3 )
        {
            if( order == 0 )
                kerI[0] = 1, kerI[1] = 2, kerI[2] = 1;
            else if( order == 1 )
                kerI[0] = -1, kerI[1] = 0, kerI[2] = 1;
            else
                kerI[0] = 1, kerI[1] = -2, kerI[2] = 1;
        }
        else
        {
            int oldval, newval;
            kerI[0] = 1;
            for( i = 0; i < ksize; i++ )
                kerI[i+1] = 0;

            for( i = 0; i < ksize - order - 1; i++ )
            {
                oldval = kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j]+kerI[j-1];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }

            for( i = 0; i < order; i++ )
            {
                oldval = -kerI[0];
                for( j = 1; j <= ksize; j++ )
                {
                    newval = kerI[j-1] - kerI[j];
                    kerI[j-1] = oldval;
                    oldval = newval;
                }
            }
        }

        Mat temp(kernel->rows, kernel->cols, CV_32S, &kerI[0]);
        double scale = !normalize ? 1. : 1./(1 << (ksize-order-1));
        temp.convertTo(*kernel, ktype, scale);
    }
}

}

void cv::getDerivKernels( OutputArray kx, OutputArray ky, int dx, int dy,
                          int ksize, bool normalize, int ktype )
{
    if( ksize <= 0 )
        getScharrKernels( kx, ky, dx, dy, normalize, ktype );
    else
        getSobelKernels( kx, ky, dx, dy, ksize, normalize, ktype );
}


cv::Ptr<cv::FilterEngine> cv::createDerivFilter(int srcType, int dstType,
                                                int dx, int dy, int ksize, int borderType )
{
    Mat kx, ky;
    getDerivKernels( kx, ky, dx, dy, ksize, false, CV_32F );
    return createSeparableLinearFilter(srcType, dstType,
        kx, ky, Point(-1,-1), 0, borderType );
}

#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)

namespace cv
{

static bool IPPDerivScharr(const Mat& src, Mat& dst, int ddepth, int dx, int dy, double scale)
{
   int bufSize = 0;
   cv::AutoBuffer<char> buffer;
   IppiSize roi = ippiSize(src.cols, src.rows);

   if( ddepth < 0 )
     ddepth = src.depth();

   dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );

   switch(src.type())
   {
      case CV_8U:
         {
            if(scale != 1)
                return false;

            switch(dst.type())
            {
               case CV_16S:
               {
                  if((dx == 1) && (dy == 0))
                  {
                     ippiFilterScharrVertGetBufferSize_8u16s_C1R(roi,&bufSize);
                     buffer.allocate(bufSize);

                     ippiFilterScharrVertBorder_8u16s_C1R((const Ipp8u*)src.data, src.step,
                        (Ipp16s*)dst.data, dst.step, roi, ippBorderRepl, 0, (Ipp8u*)(char*)buffer);

                     return true;
                  }

                  if((dx == 0) && (dy == 1))
                  {
                     ippiFilterScharrHorizGetBufferSize_8u16s_C1R(roi,&bufSize);
                     buffer.allocate(bufSize);

                     ippiFilterScharrHorizBorder_8u16s_C1R((const Ipp8u*)src.data, src.step,
                        (Ipp16s*)dst.data, dst.step, roi, ippBorderRepl, 0, (Ipp8u*)(char*)buffer);

                     return true;
                  }
               }

               default:
                  return false;
            }
         }

      case CV_32F:
         {
            switch(dst.type())
            {
               case CV_32F:
               if((dx == 1) && (dy == 0))
               {
                  ippiFilterScharrVertGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows),&bufSize);
                  buffer.allocate(bufSize);

                  ippiFilterScharrVertBorder_32f_C1R((const Ipp32f*)src.data, src.step,
                     (Ipp32f*)dst.data, dst.step, ippiSize(src.cols, src.rows),
                                            ippBorderRepl, 0, (Ipp8u*)(char*)buffer);
                  if(scale != 1)
                     /* IPP is fast, so MulC produce very little perf degradation */
                     ippiMulC_32f_C1IR((Ipp32f)scale,(Ipp32f*)dst.data,dst.step,ippiSize(dst.cols*dst.channels(),dst.rows));

                  return true;
               }

               if((dx == 0) && (dy == 1))
               {
                  ippiFilterScharrHorizGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows),&bufSize);
                  buffer.allocate(bufSize);

                  ippiFilterScharrHorizBorder_32f_C1R((const Ipp32f*)src.data, src.step,
                     (Ipp32f*)dst.data, dst.step, ippiSize(src.cols, src.rows),
                                            ippBorderRepl, 0, (Ipp8u*)(char*)buffer);
                  if(scale != 1)
                     ippiMulC_32f_C1IR((Ipp32f)scale,(Ipp32f *)dst.data,dst.step,ippiSize(dst.cols*dst.channels(),dst.rows));

                  return true;
               }

               default:
                  return false;
            }
         }

      default:
         return false;
   }
}


static bool IPPDeriv(const Mat& src, Mat& dst, int ddepth, int dx, int dy, int ksize, double scale)
{
   int bufSize = 0;
   cv::AutoBuffer<char> buffer;

   if(ksize == 3 || ksize == 5)
   {
      if( ddepth < 0 )
          ddepth = src.depth();

      if(src.type() == CV_8U && dst.type() == CV_16S && scale == 1)
      {
         if((dx == 1) && (dy == 0))
         {
            ippiFilterSobelNegVertGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelNegVertBorder_8u16s_C1R((const Ipp8u*)src.data, src.step,
               (Ipp16s*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);
            return true;
         }

         if((dx == 0) && (dy == 1))
         {
            ippiFilterSobelHorizGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelHorizBorder_8u16s_C1R((const Ipp8u*)src.data, src.step,
               (Ipp16s*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);

            return true;
         }

         if((dx == 2) && (dy == 0))
         {
            ippiFilterSobelVertSecondGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelVertSecondBorder_8u16s_C1R((const Ipp8u*)src.data, src.step,
               (Ipp16s*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);

            return true;
         }

         if((dx == 0) && (dy == 2))
         {
            ippiFilterSobelHorizSecondGetBufferSize_8u16s_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelHorizSecondBorder_8u16s_C1R((const Ipp8u*)src.data, src.step,
               (Ipp16s*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);

            return true;
         }
      }

      if(src.type() == CV_32F && dst.type() == CV_32F)
      {
         if((dx == 1) && (dy == 0))
         {
            ippiFilterSobelNegVertGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelNegVertBorder_32f_C1R((const Ipp32f*)src.data, src.step,
               (Ipp32f*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);
            if(scale != 1)
               ippiMulC_32f_C1IR((Ipp32f)scale,(Ipp32f *)dst.data,dst.step,ippiSize(dst.cols*dst.channels(),dst.rows));

            return true;
         }

         if((dx == 0) && (dy == 1))
         {
            ippiFilterSobelHorizGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelHorizBorder_32f_C1R((const Ipp32f*)src.data, src.step,
               (Ipp32f*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);
            if(scale != 1)
               ippiMulC_32f_C1IR((Ipp32f)scale,(Ipp32f *)dst.data,dst.step,ippiSize(dst.cols*dst.channels(),dst.rows));

            return true;
         }

         if((dx == 2) && (dy == 0))
         {
            ippiFilterSobelVertSecondGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelVertSecondBorder_32f_C1R((const Ipp32f*)src.data, src.step,
               (Ipp32f*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);
            if(scale != 1)
               ippiMulC_32f_C1IR((Ipp32f)scale,(Ipp32f *)dst.data,dst.step,ippiSize(dst.cols*dst.channels(),dst.rows));

            return true;
         }

         if((dx == 0) && (dy == 2))
         {
            ippiFilterSobelHorizSecondGetBufferSize_32f_C1R(ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),&bufSize);
            buffer.allocate(bufSize);

            ippiFilterSobelHorizSecondBorder_32f_C1R((const Ipp32f*)src.data, src.step,
               (Ipp32f*)dst.data, dst.step, ippiSize(src.cols, src.rows), (IppiMaskSize)(ksize*10+ksize),
                                      ippBorderRepl, 0, (Ipp8u*)(char*)buffer);
            if(scale != 1)
               ippiMulC_32f_C1IR((Ipp32f)scale,(Ipp32f *)dst.data,dst.step,ippiSize(dst.cols*dst.channels(),dst.rows));

            return true;
         }
      }
   }

   if(ksize <= 0)
      return IPPDerivScharr(src, dst, ddepth, dx, dy, scale);

   return false;
}

}
    
#endif

void cv::Sobel( InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
                int ksize, double scale, double delta, int borderType )
{
    Mat src = _src.getMat();
    if (ddepth < 0)
        ddepth = src.depth();
    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();
    
#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    if(dx < 3 && dy < 3 && src.channels() == 1 && borderType == 1)
    {
        if(IPPDeriv(src, dst, ddepth, dx, dy, ksize,scale))
            return;
    }
#endif
    int ktype = std::max(CV_32F, std::max(ddepth, src.depth()));

    Mat kx, ky;
    getDerivKernels( kx, ky, dx, dy, ksize, false, ktype );
    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if( dx == 0 )
            kx *= scale;
        else
            ky *= scale;
    }
    sepFilter2D( src, dst, ddepth, kx, ky, Point(-1,-1), delta, borderType );
}


void cv::Scharr( InputArray _src, OutputArray _dst, int ddepth, int dx, int dy,
                 double scale, double delta, int borderType )
{
    Mat src = _src.getMat();
    if (ddepth < 0)
        ddepth = src.depth();
    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();
    
#if defined (HAVE_IPP) && (IPP_VERSION_MAJOR >= 7)
    if(dx < 2 && dy < 2 && src.channels() == 1 && borderType == 1)
    {
        if(IPPDerivScharr(src, dst, ddepth, dx, dy, scale))
            return;
    }
#endif
    int ktype = std::max(CV_32F, std::max(ddepth, src.depth()));

    Mat kx, ky;
    getScharrKernels( kx, ky, dx, dy, false, ktype );
    if( scale != 1 )
    {
        // usually the smoothing part is the slowest to compute,
        // so try to scale it instead of the faster differenciating part
        if( dx == 0 )
            kx *= scale;
        else
            ky *= scale;
    }
    sepFilter2D( src, dst, ddepth, kx, ky, Point(-1,-1), delta, borderType );
}


void cv::Laplacian( InputArray _src, OutputArray _dst, int ddepth, int ksize,
                    double scale, double delta, int borderType )
{
    Mat src = _src.getMat();    
    if (ddepth < 0)
        ddepth = src.depth();
    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();
    
    if( ksize == 1 || ksize == 3 )
    {
        float K[2][9] =
        {{0, 1, 0, 1, -4, 1, 0, 1, 0},
         {2, 0, 2, 0, -8, 0, 2, 0, 2}};
        Mat kernel(3, 3, CV_32F, K[ksize == 3]);
        if( scale != 1 )
            kernel *= scale;
        filter2D( src, dst, ddepth, kernel, Point(-1,-1), delta, borderType );
    }
    else
    {
        const size_t STRIPE_SIZE = 1 << 14;

        int depth = src.depth();
        int ktype = std::max(CV_32F, std::max(ddepth, depth));
        int wdepth = depth == CV_8U && ksize <= 5 ? CV_16S : depth <= CV_32F ? CV_32F : CV_64F;
        int wtype = CV_MAKETYPE(wdepth, src.channels());
        Mat kd, ks;
        getSobelKernels( kd, ks, 2, 0, ksize, false, ktype );
        if( ddepth < 0 )
            ddepth = src.depth();
        int dtype = CV_MAKETYPE(ddepth, src.channels());

        int dy0 = std::min(std::max((int)(STRIPE_SIZE/(getElemSize(src.type())*src.cols)), 1), src.rows);
        Ptr<FilterEngine> fx = createSeparableLinearFilter(src.type(),
            wtype, kd, ks, Point(-1,-1), 0, borderType, borderType, Scalar() ); 
        Ptr<FilterEngine> fy = createSeparableLinearFilter(src.type(),
            wtype, ks, kd, Point(-1,-1), 0, borderType, borderType, Scalar() );

        int y = fx->start(src), dsty = 0, dy = 0;
        fy->start(src);
        const uchar* sptr = src.data + y*src.step;

        Mat d2x( dy0 + kd.rows - 1, src.cols, wtype );
        Mat d2y( dy0 + kd.rows - 1, src.cols, wtype );

        for( ; dsty < src.rows; sptr += dy0*src.step, dsty += dy )
        {
            fx->proceed( sptr, (int)src.step, dy0, d2x.data, (int)d2x.step );
            dy = fy->proceed( sptr, (int)src.step, dy0, d2y.data, (int)d2y.step );
            if( dy > 0 )
            {
                Mat dstripe = dst.rowRange(dsty, dsty + dy);
                d2x.rows = d2y.rows = dy; // modify the headers, which should work
                d2x += d2y;
                d2x.convertTo( dstripe, dtype, scale, delta );
            }
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////

CV_IMPL void
cvSobel( const void* srcarr, void* dstarr, int dx, int dy, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() &&
        ((src.depth() == CV_8U && (dst.depth() == CV_16S || dst.depth() == CV_32F)) ||
        (src.depth() == CV_32F && dst.depth() == CV_32F)));

    cv::Sobel( src, dst, dst.depth(), dx, dy, aperture_size, 1, 0, cv::BORDER_REPLICATE );
    if( CV_IS_IMAGE(srcarr) && ((IplImage*)srcarr)->origin && dy % 2 != 0 )
        dst *= -1;
}


CV_IMPL void
cvLaplace( const void* srcarr, void* dstarr, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() &&
        ((src.depth() == CV_8U && (dst.depth() == CV_16S || dst.depth() == CV_32F)) ||
        (src.depth() == CV_32F && dst.depth() == CV_32F)));

    cv::Laplacian( src, dst, dst.depth(), aperture_size, 1, 0, cv::BORDER_REPLICATE );
}

/* End of file. */
