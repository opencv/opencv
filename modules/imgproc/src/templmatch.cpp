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

void
icvCrossCorr( const CvArr* _img, const CvArr* _templ, CvArr* _corr,
              CvPoint anchor, double delta, int borderType )
{
    const int CV_MAX_THREADS = 1;
    const double block_scale = 4.5;
    const int min_block_size = 256;
    cv::Ptr<CvMat> dft_img[CV_MAX_THREADS];
    cv::Ptr<CvMat> dft_templ;
    std::vector<uchar> buf[CV_MAX_THREADS];
    int k, num_threads = 0;
    
    CvMat istub, *img = (CvMat*)_img;
    CvMat tstub, *templ = (CvMat*)_templ;
    CvMat cstub, *corr = (CvMat*)_corr;
    CvSize dftsize, blocksize;
    int depth, templ_depth, corr_depth, max_depth = CV_32F,
        cn, templ_cn, corr_cn, buf_size = 0,
        tile_count_x, tile_count_y, tile_count;

    img = cvGetMat( img, &istub );
    templ = cvGetMat( templ, &tstub );
    corr = cvGetMat( corr, &cstub );

    if( CV_MAT_DEPTH( img->type ) != CV_8U &&
        CV_MAT_DEPTH( img->type ) != CV_16U &&
        CV_MAT_DEPTH( img->type ) != CV_32F &&
        CV_MAT_DEPTH( img->type ) != CV_64F )
        CV_Error( CV_StsUnsupportedFormat,
        "The function supports only 8u, 16u and 32f data types" );

    if( !CV_ARE_DEPTHS_EQ( img, templ ) && CV_MAT_DEPTH( templ->type ) != CV_32F )
        CV_Error( CV_StsUnsupportedFormat,
        "Template (kernel) must be of the same depth as the input image, or be 32f" );
    
    if( !CV_ARE_DEPTHS_EQ( img, corr ) && CV_MAT_DEPTH( corr->type ) != CV_32F &&
        CV_MAT_DEPTH( corr->type ) != CV_64F )
        CV_Error( CV_StsUnsupportedFormat,
        "The output image must have the same depth as the input image, or be 32f/64f" );

    if( (!CV_ARE_CNS_EQ( img, corr ) || CV_MAT_CN(templ->type) > 1) &&
        (CV_MAT_CN( corr->type ) > 1 || !CV_ARE_CNS_EQ( img, templ)) )
        CV_Error( CV_StsUnsupportedFormat,
        "The output must have the same number of channels as the input (when the template has 1 channel), "
        "or the output must have 1 channel when the input and the template have the same number of channels" );

    depth = CV_MAT_DEPTH(img->type);
    cn = CV_MAT_CN(img->type);
    templ_depth = CV_MAT_DEPTH(templ->type);
    templ_cn = CV_MAT_CN(templ->type);
    corr_depth = CV_MAT_DEPTH(corr->type);
    corr_cn = CV_MAT_CN(corr->type);

    CV_Assert( corr_cn == 1 || delta == 0 );

    max_depth = MAX( max_depth, templ_depth );
    max_depth = MAX( max_depth, depth );
    max_depth = MAX( max_depth, corr_depth );
    if( depth > CV_8U )
        max_depth = CV_64F;

    /*if( img->cols < templ->cols || img->rows < templ->rows )
        CV_Error( CV_StsUnmatchedSizes,
        "Such a combination of image and template/filter size is not supported" );*/

    if( corr->rows > img->rows + templ->rows - 1 ||
        corr->cols > img->cols + templ->cols - 1 )
        CV_Error( CV_StsUnmatchedSizes,
        "output image should not be greater than (W + w - 1)x(H + h - 1)" );

    blocksize.width = cvRound(templ->cols*block_scale);
    blocksize.width = MAX( blocksize.width, min_block_size - templ->cols + 1 );
    blocksize.width = MIN( blocksize.width, corr->cols );
    blocksize.height = cvRound(templ->rows*block_scale);
    blocksize.height = MAX( blocksize.height, min_block_size - templ->rows + 1 );
    blocksize.height = MIN( blocksize.height, corr->rows );

    dftsize.width = cvGetOptimalDFTSize(blocksize.width + templ->cols - 1);
    if( dftsize.width == 1 )
        dftsize.width = 2;
    dftsize.height = cvGetOptimalDFTSize(blocksize.height + templ->rows - 1);
    if( dftsize.width <= 0 || dftsize.height <= 0 )
        CV_Error( CV_StsOutOfRange, "the input arrays are too big" );

    // recompute block size
    blocksize.width = dftsize.width - templ->cols + 1;
    blocksize.width = MIN( blocksize.width, corr->cols );
    blocksize.height = dftsize.height - templ->rows + 1;
    blocksize.height = MIN( blocksize.height, corr->rows );

    dft_templ = cvCreateMat( dftsize.height*templ_cn, dftsize.width, max_depth );

#ifdef USE_OPENMP
    num_threads = cvGetNumThreads();
#else
    num_threads = 1;
#endif

    for( k = 0; k < num_threads; k++ )
        dft_img[k] = cvCreateMat( dftsize.height, dftsize.width, max_depth );

    if( templ_cn > 1 && templ_depth != max_depth )
        buf_size = templ->cols*templ->rows*CV_ELEM_SIZE(templ_depth);

    if( cn > 1 && depth != max_depth )
        buf_size = MAX( buf_size, (blocksize.width + templ->cols - 1)*
            (blocksize.height + templ->rows - 1)*CV_ELEM_SIZE(depth));

    if( (corr_cn > 1 || cn > 1) && corr_depth != max_depth )
        buf_size = MAX( buf_size, blocksize.width*blocksize.height*CV_ELEM_SIZE(corr_depth));

    if( buf_size > 0 )
    {
        for( k = 0; k < num_threads; k++ )
            buf[k].resize(buf_size);
    }

    // compute DFT of each template plane
    for( k = 0; k < templ_cn; k++ )
    {
        CvMat dstub, *src, *dst, temp;
        CvMat* planes[] = { 0, 0, 0, 0 };
        int yofs = k*dftsize.height;

        src = templ;
        dst = cvGetSubRect( dft_templ, &dstub, cvRect(0,yofs,templ->cols,templ->rows));
    
        if( templ_cn > 1 )
        {
            planes[k] = templ_depth == max_depth ? dst :
                cvInitMatHeader( &temp, templ->rows, templ->cols, templ_depth, &buf[0][0] );
            cvSplit( templ, planes[0], planes[1], planes[2], planes[3] );
            src = planes[k];
            planes[k] = 0;
        }

        if( dst != src )
            cvConvert( src, dst );

        if( dft_templ->cols > templ->cols )
        {
            cvGetSubRect( dft_templ, dst, cvRect(templ->cols, yofs,
                          dft_templ->cols - templ->cols, templ->rows) );
            cvZero( dst );
        }
        cvGetSubRect( dft_templ, dst, cvRect(0,yofs,dftsize.width,dftsize.height) );
        cvDFT( dst, dst, CV_DXT_FORWARD + CV_DXT_SCALE, templ->rows );
    }

    tile_count_x = (corr->cols + blocksize.width - 1)/blocksize.width;
    tile_count_y = (corr->rows + blocksize.height - 1)/blocksize.height;
    tile_count = tile_count_x*tile_count_y;

#if defined _OPENMP && defined USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
#endif
    // calculate correlation by blocks
    for( k = 0; k < tile_count; k++ )
    {
#ifdef USE_OPENMP
        int thread_idx = cvGetThreadNum();
#else
        int thread_idx = 0;
#endif
        int x = (k%tile_count_x)*blocksize.width;
        int y = (k/tile_count_x)*blocksize.height;
        int i, yofs;
        CvMat sstub, dstub, *src, *dst, temp;
        CvMat* planes[] = { 0, 0, 0, 0 };
        CvMat* _dft_img = dft_img[thread_idx];
        uchar* _buf = buf_size > 0 ? &buf[thread_idx][0] : 0;
        CvSize csz = { blocksize.width, blocksize.height }, isz;
        int x0 = x - anchor.x, y0 = y - anchor.y;
        int x1 = MAX( 0, x0 ), y1 = MAX( 0, y0 ), x2, y2;
        csz.width = MIN( csz.width, corr->cols - x );
        csz.height = MIN( csz.height, corr->rows - y );
        isz.width = csz.width + templ->cols - 1;
        isz.height = csz.height + templ->rows - 1;
        x2 = MIN( img->cols, x0 + isz.width );
        y2 = MIN( img->rows, y0 + isz.height );
        
        for( i = 0; i < cn; i++ )
        {
            CvMat dstub1, *dst1;
            yofs = i*dftsize.height;

            src = cvGetSubRect( img, &sstub, cvRect(x1,y1,x2-x1,y2-y1) );
            dst = cvGetSubRect( _dft_img, &dstub,
                cvRect(0,0,isz.width,isz.height) );
            dst1 = dst;
            
            if( x2 - x1 < isz.width || y2 - y1 < isz.height )
                dst1 = cvGetSubRect( _dft_img, &dstub1,
                    cvRect( x1 - x0, y1 - y0, x2 - x1, y2 - y1 ));

            if( cn > 1 )
            {
                planes[i] = dst1;
                if( depth != max_depth )
                    planes[i] = cvInitMatHeader( &temp, y2 - y1, x2 - x1, depth, _buf );
                cvSplit( src, planes[0], planes[1], planes[2], planes[3] );
                src = planes[i];
                planes[i] = 0;
            }

            if( dst1 != src )
                cvConvert( src, dst1 );

            if( dst != dst1 )
                cvCopyMakeBorder( dst1, dst, cvPoint(x1 - x0, y1 - y0), borderType );

            if( dftsize.width > isz.width )
            {
                cvGetSubRect( _dft_img, dst, cvRect(isz.width, 0,
                      dftsize.width - isz.width,dftsize.height) );
                cvZero( dst );
            }

            cvDFT( _dft_img, _dft_img, CV_DXT_FORWARD, isz.height );
            cvGetSubRect( dft_templ, dst,
                cvRect(0,(templ_cn>1?yofs:0),dftsize.width,dftsize.height) );

            cvMulSpectrums( _dft_img, dst, _dft_img, CV_DXT_MUL_CONJ );
            cvDFT( _dft_img, _dft_img, CV_DXT_INVERSE, csz.height );

            src = cvGetSubRect( _dft_img, &sstub, cvRect(0,0,csz.width,csz.height) );
            dst = cvGetSubRect( corr, &dstub, cvRect(x,y,csz.width,csz.height) );

            if( corr_cn > 1 )
            {
                planes[i] = src;
                if( corr_depth != max_depth )
                {
                    planes[i] = cvInitMatHeader( &temp, csz.height, csz.width,
                                                 corr_depth, _buf );
                    cvConvertScale( src, planes[i], 1, delta );
                }
                cvMerge( planes[0], planes[1], planes[2], planes[3], dst );
                planes[i] = 0;                    
            }
            else
            {
                if( i == 0 )
                    cvConvertScale( src, dst, 1, delta );
                else
                {
                    if( max_depth > corr_depth )
                    {
                        cvInitMatHeader( &temp, csz.height, csz.width,
                                         corr_depth, _buf );
                        cvConvert( src, &temp );
                        src = &temp;
                    }
                    cvAcc( src, dst );
                }
            }
        }
    }
}

void
cv::crossCorr( const Mat& img, const Mat& templ, Mat& corr,
           Point anchor, double delta, int borderType )
{
    CvMat _img = img, _templ = templ, _corr = corr;
    icvCrossCorr( &_img, &_templ, &_corr, anchor, delta, borderType );
}


/*****************************************************************************************/

CV_IMPL void
cvMatchTemplate( const CvArr* _img, const CvArr* _templ, CvArr* _result, int method )
{
    cv::Ptr<CvMat> sum, sqsum;
    
    int coi1 = 0, coi2 = 0;
    int depth, cn;
    int i, j, k;
    CvMat stub, *img = (CvMat*)_img;
    CvMat tstub, *templ = (CvMat*)_templ;
    CvMat rstub, *result = (CvMat*)_result;
    CvScalar templ_mean = cvScalarAll(0);
    double templ_norm = 0, templ_sum2 = 0;
    
    int idx = 0, idx2 = 0;
    double *p0, *p1, *p2, *p3;
    double *q0, *q1, *q2, *q3;
    double inv_area;
    int sum_step, sqsum_step;
    int num_type = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
                   method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
    int is_normed = method == CV_TM_CCORR_NORMED ||
                    method == CV_TM_SQDIFF_NORMED ||
                    method == CV_TM_CCOEFF_NORMED;

    img = cvGetMat( img, &stub, &coi1 );
    templ = cvGetMat( templ, &tstub, &coi2 );
    result = cvGetMat( result, &rstub );

    if( CV_MAT_DEPTH( img->type ) != CV_8U &&
        CV_MAT_DEPTH( img->type ) != CV_32F )
        CV_Error( CV_StsUnsupportedFormat,
        "The function supports only 8u and 32f data types" );

    if( !CV_ARE_TYPES_EQ( img, templ ))
        CV_Error( CV_StsUnmatchedSizes, "image and template should have the same type" );

    if( CV_MAT_TYPE( result->type ) != CV_32FC1 )
        CV_Error( CV_StsUnsupportedFormat, "output image should have 32f type" );

    if( img->rows < templ->rows || img->cols < templ->cols )
    {
        CvMat* t;
        CV_SWAP( img, templ, t );
    }

    if( result->rows != img->rows - templ->rows + 1 ||
        result->cols != img->cols - templ->cols + 1 )
        CV_Error( CV_StsUnmatchedSizes, "output image should be (W - w + 1)x(H - h + 1)" );

    if( method < CV_TM_SQDIFF || method > CV_TM_CCOEFF_NORMED )
        CV_Error( CV_StsBadArg, "unknown comparison method" );

    depth = CV_MAT_DEPTH(img->type);
    cn = CV_MAT_CN(img->type);

    icvCrossCorr( img, templ, result );

    if( method == CV_TM_CCORR )
        return;

    inv_area = 1./((double)templ->rows * templ->cols);

    sum = cvCreateMat( img->rows + 1, img->cols + 1, CV_MAKETYPE( CV_64F, cn ));
    if( method == CV_TM_CCOEFF )
    {
        cvIntegral( img, sum, 0, 0 );
        templ_mean = cvAvg( templ );
        q0 = q1 = q2 = q3 = 0;
    }
    else
    {
        CvScalar _templ_sdv = cvScalarAll(0);
        sqsum = cvCreateMat( img->rows + 1, img->cols + 1, CV_MAKETYPE( CV_64F, cn ));
        cvIntegral( img, sum, sqsum, 0 );
        cvAvgSdv( templ, &templ_mean, &_templ_sdv );

        templ_norm = CV_SQR(_templ_sdv.val[0]) + CV_SQR(_templ_sdv.val[1]) +
                    CV_SQR(_templ_sdv.val[2]) + CV_SQR(_templ_sdv.val[3]);

        if( templ_norm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED )
        {
            cvSet( result, cvScalarAll(1.) );
            return;
        }
        
        templ_sum2 = templ_norm +
                     CV_SQR(templ_mean.val[0]) + CV_SQR(templ_mean.val[1]) +
                     CV_SQR(templ_mean.val[2]) + CV_SQR(templ_mean.val[3]);

        if( num_type != 1 )
        {
            templ_mean = cvScalarAll(0);
            templ_norm = templ_sum2;
        }
        
        templ_sum2 /= inv_area;
        templ_norm = sqrt(templ_norm);
        templ_norm /= sqrt(inv_area); // care of accuracy here

        q0 = (double*)sqsum->data.ptr;
        q1 = q0 + templ->cols*cn;
        q2 = (double*)(sqsum->data.ptr + templ->rows*sqsum->step);
        q3 = q2 + templ->cols*cn;
    }

    p0 = (double*)sum->data.ptr;
    p1 = p0 + templ->cols*cn;
    p2 = (double*)(sum->data.ptr + templ->rows*sum->step);
    p3 = p2 + templ->cols*cn;

    sum_step = sum ? sum->step / sizeof(double) : 0;
    sqsum_step = sqsum ? sqsum->step / sizeof(double) : 0;

    for( i = 0; i < result->rows; i++ )
    {
        float* rrow = (float*)(result->data.ptr + i*result->step);
        idx = i * sum_step;
        idx2 = i * sqsum_step;

        for( j = 0; j < result->cols; j++, idx += cn, idx2 += cn )
        {
            double num = rrow[j], t;
            double wnd_mean2 = 0, wnd_sum2 = 0;
            
            if( num_type == 1 )
            {
                for( k = 0; k < cn; k++ )
                {
                    t = p0[idx+k] - p1[idx+k] - p2[idx+k] + p3[idx+k];
                    wnd_mean2 += CV_SQR(t);
                    num -= t*templ_mean.val[k];
                }

                wnd_mean2 *= inv_area;
            }

            if( is_normed || num_type == 2 )
            {
                for( k = 0; k < cn; k++ )
                {
                    t = q0[idx2+k] - q1[idx2+k] - q2[idx2+k] + q3[idx2+k];
                    wnd_sum2 += t;
                }

                if( num_type == 2 )
                    num = wnd_sum2 - 2*num + templ_sum2;
            }

            if( is_normed )
            {
                t = sqrt(MAX(wnd_sum2 - wnd_mean2,0))*templ_norm;
                if( fabs(num) < t )
                    num /= t;
                else if( fabs(num) < t*1.125 )
                    num = num > 0 ? 1 : -1;
                else
                    num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
            }

            rrow[j] = (float)num;
        }
    }
}

void cv::matchTemplate( const Mat& image, const Mat& templ, Mat& result, int method )
{
    result.create( std::abs(image.rows - templ.rows) + 1,
                   std::abs(image.cols - templ.cols) + 1, CV_32F );
    CvMat _image = image, _templ = templ, _result = result;
    cvMatchTemplate( &_image, &_templ, &_result, method );    
}

/* End of file. */
