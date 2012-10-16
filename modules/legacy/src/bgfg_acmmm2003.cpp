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


// This file implements the foreground/background pixel
// discrimination algorithm described in
//
//     Foreground Object Detection from Videos Containing Complex Background
//     Li, Huan, Gu, Tian 2003 9p
//     http://muq.org/~cynbe/bib/foreground-object-detection-from-videos-containing-complex-background.pdf


#include "precomp.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <algorithm>

static double* _cv_max_element( double* start, double* end )
{
    double* p = start++;

    for( ; start != end;  ++start) {

        if (*p < *start)   p = start;
    }

    return p;
}

static void CV_CDECL icvReleaseFGDStatModel( CvFGDStatModel** model );
static int CV_CDECL icvUpdateFGDStatModel( IplImage* curr_frame,
                                           CvFGDStatModel*  model,
                                           double );

// Function cvCreateFGDStatModel initializes foreground detection process
// parameters:
//      first_frame - frame from video sequence
//      parameters  - (optional) if NULL default parameters of the algorithm will be used
//      p_model     - pointer to CvFGDStatModel structure
CV_IMPL CvBGStatModel*
cvCreateFGDStatModel( IplImage* first_frame, CvFGDStatModelParams* parameters )
{
    CvFGDStatModel* p_model = 0;

    CV_FUNCNAME( "cvCreateFGDStatModel" );

    __BEGIN__;

    int i, j, k, pixel_count, buf_size;
    CvFGDStatModelParams params;

    if( !CV_IS_IMAGE(first_frame) )
        CV_ERROR( CV_StsBadArg, "Invalid or NULL first_frame parameter" );

    if (first_frame->nChannels != 3)
        CV_ERROR( CV_StsBadArg, "first_frame must have 3 color channels" );

    // Initialize parameters:
    if( parameters == NULL )
    {
        params.Lc      = CV_BGFG_FGD_LC;
        params.N1c     = CV_BGFG_FGD_N1C;
        params.N2c     = CV_BGFG_FGD_N2C;

        params.Lcc     = CV_BGFG_FGD_LCC;
        params.N1cc    = CV_BGFG_FGD_N1CC;
        params.N2cc    = CV_BGFG_FGD_N2CC;

        params.delta   = CV_BGFG_FGD_DELTA;

        params.alpha1  = CV_BGFG_FGD_ALPHA_1;
        params.alpha2  = CV_BGFG_FGD_ALPHA_2;
        params.alpha3  = CV_BGFG_FGD_ALPHA_3;

        params.T       = CV_BGFG_FGD_T;
        params.minArea = CV_BGFG_FGD_MINAREA;

        params.is_obj_without_holes = 1;
        params.perform_morphing     = 1;
    }
    else
    {
        params = *parameters;
    }

    CV_CALL( p_model = (CvFGDStatModel*)cvAlloc( sizeof(*p_model) ));
    memset( p_model, 0, sizeof(*p_model) );
    p_model->type = CV_BG_MODEL_FGD;
    p_model->release = (CvReleaseBGStatModel)icvReleaseFGDStatModel;
    p_model->update = (CvUpdateBGStatModel)icvUpdateFGDStatModel;;
    p_model->params = params;

    // Initialize storage pools:
    pixel_count = first_frame->width * first_frame->height;

    buf_size = pixel_count*sizeof(p_model->pixel_stat[0]);
    CV_CALL( p_model->pixel_stat = (CvBGPixelStat*)cvAlloc(buf_size) );
    memset( p_model->pixel_stat, 0, buf_size );

    buf_size = pixel_count*params.N2c*sizeof(p_model->pixel_stat[0].ctable[0]);
    CV_CALL( p_model->pixel_stat[0].ctable = (CvBGPixelCStatTable*)cvAlloc(buf_size) );
    memset( p_model->pixel_stat[0].ctable, 0, buf_size );

    buf_size = pixel_count*params.N2cc*sizeof(p_model->pixel_stat[0].cctable[0]);
    CV_CALL( p_model->pixel_stat[0].cctable = (CvBGPixelCCStatTable*)cvAlloc(buf_size) );
    memset( p_model->pixel_stat[0].cctable, 0, buf_size );

    for(     i = 0, k = 0; i < first_frame->height; i++ ) {
        for( j = 0;        j < first_frame->width;  j++, k++ )
        {
            p_model->pixel_stat[k].ctable = p_model->pixel_stat[0].ctable + k*params.N2c;
            p_model->pixel_stat[k].cctable = p_model->pixel_stat[0].cctable + k*params.N2cc;
        }
    }

    // Init temporary images:
    CV_CALL( p_model->Ftd = cvCreateImage(cvSize(first_frame->width, first_frame->height), IPL_DEPTH_8U, 1));
    CV_CALL( p_model->Fbd = cvCreateImage(cvSize(first_frame->width, first_frame->height), IPL_DEPTH_8U, 1));
    CV_CALL( p_model->foreground = cvCreateImage(cvSize(first_frame->width, first_frame->height), IPL_DEPTH_8U, 1));

    CV_CALL( p_model->background = cvCloneImage(first_frame));
    CV_CALL( p_model->prev_frame = cvCloneImage(first_frame));
    CV_CALL( p_model->storage = cvCreateMemStorage());

    __END__;

    if( cvGetErrStatus() < 0 )
    {
        CvBGStatModel* base_ptr = (CvBGStatModel*)p_model;

        if( p_model && p_model->release )
            p_model->release( &base_ptr );
        else
            cvFree( &p_model );
        p_model = 0;
    }

    return (CvBGStatModel*)p_model;
}


static void CV_CDECL
icvReleaseFGDStatModel( CvFGDStatModel** _model )
{
    CV_FUNCNAME( "icvReleaseFGDStatModel" );

    __BEGIN__;

    if( !_model )
        CV_ERROR( CV_StsNullPtr, "" );

    if( *_model )
    {
        CvFGDStatModel* model = *_model;
        if( model->pixel_stat )
        {
            cvFree( &model->pixel_stat[0].ctable );
            cvFree( &model->pixel_stat[0].cctable );
            cvFree( &model->pixel_stat );
        }

        cvReleaseImage( &model->Ftd );
        cvReleaseImage( &model->Fbd );
        cvReleaseImage( &model->foreground );
        cvReleaseImage( &model->background );
        cvReleaseImage( &model->prev_frame );
        cvReleaseMemStorage(&model->storage);

        cvFree( _model );
    }

    __END__;
}

//  Function cvChangeDetection performs change detection for Foreground detection algorithm
// parameters:
//      prev_frame -
//      curr_frame -
//      change_mask -
CV_IMPL int
cvChangeDetection( IplImage*  prev_frame,
                   IplImage*  curr_frame,
                   IplImage*  change_mask )
{
    int i, j, b, x, y, thres;
    const int PIXELRANGE=256;

    if( !prev_frame
    ||  !curr_frame
    ||  !change_mask
    ||   prev_frame->nChannels  != 3
    ||   curr_frame->nChannels  != 3
    ||   change_mask->nChannels != 1
    ||   prev_frame->depth  != IPL_DEPTH_8U
    ||   curr_frame->depth  != IPL_DEPTH_8U
    ||   change_mask->depth != IPL_DEPTH_8U
    ||   prev_frame->width  != curr_frame->width
    ||   prev_frame->height != curr_frame->height
    ||   prev_frame->width  != change_mask->width
    ||   prev_frame->height != change_mask->height
    ){
        return 0;
    }

    cvZero ( change_mask );

    // All operations per colour
    for (b=0 ; b<prev_frame->nChannels ; b++) {

        // Create histogram:

        long HISTOGRAM[PIXELRANGE];
        for (i=0 ; i<PIXELRANGE; i++) HISTOGRAM[i]=0;

        for (y=0 ; y<curr_frame->height ; y++)
        {
            uchar* rowStart1 = (uchar*)curr_frame->imageData + y * curr_frame->widthStep + b;
            uchar* rowStart2 = (uchar*)prev_frame->imageData + y * prev_frame->widthStep + b;
            for (x=0 ; x<curr_frame->width ; x++, rowStart1+=curr_frame->nChannels, rowStart2+=prev_frame->nChannels) {
                int diff = abs( int(*rowStart1) - int(*rowStart2) );
                HISTOGRAM[diff]++;
            }
        }

        double relativeVariance[PIXELRANGE];
        for (i=0 ; i<PIXELRANGE; i++) relativeVariance[i]=0;

        for (thres=PIXELRANGE-2; thres>=0 ; thres--)
        {
            //            fprintf(stderr, "Iter %d\n", thres);
            double sum=0;
            double sqsum=0;
            int count=0;
            //            fprintf(stderr, "Iter %d entering loop\n", thres);
            for (j=thres ; j<PIXELRANGE ; j++) {
                sum   += double(j)*double(HISTOGRAM[j]);
                sqsum += double(j*j)*double(HISTOGRAM[j]);
                count += HISTOGRAM[j];
            }
            count = count == 0 ? 1 : count;
            //            fprintf(stderr, "Iter %d finishing loop\n", thres);
            double my = sum / count;
            double sigma = sqrt( sqsum/count - my*my);
            //            fprintf(stderr, "Iter %d sum=%g sqsum=%g count=%d sigma = %g\n", thres, sum, sqsum, count, sigma);
            //            fprintf(stderr, "Writing to %x\n", &(relativeVariance[thres]));
            relativeVariance[thres] = sigma;
            //            fprintf(stderr, "Iter %d finished\n", thres);
        }

        // Find maximum:
        uchar bestThres = 0;

        double* pBestThres = _cv_max_element(relativeVariance, relativeVariance+PIXELRANGE);
        bestThres = (uchar)(*pBestThres); if (bestThres <10) bestThres=10;

        for (y=0 ; y<prev_frame->height ; y++)
        {
            uchar* rowStart1 = (uchar*)(curr_frame->imageData) + y * curr_frame->widthStep + b;
            uchar* rowStart2 = (uchar*)(prev_frame->imageData) + y * prev_frame->widthStep + b;
            uchar* rowStart3 = (uchar*)(change_mask->imageData) + y * change_mask->widthStep;
            for (x = 0; x < curr_frame->width; x++, rowStart1+=curr_frame->nChannels,
                rowStart2+=prev_frame->nChannels, rowStart3+=change_mask->nChannels) {
                // OR between different color channels
                int diff = abs( int(*rowStart1) - int(*rowStart2) );
                if ( diff > bestThres)
                    *rowStart3 |=255;
            }
        }
    }

    return 1;
}


#define MIN_PV 1E-10


#define V_C(k,l) ctable[k].v[l]
#define PV_C(k) ctable[k].Pv
#define PVB_C(k) ctable[k].Pvb
#define V_CC(k,l) cctable[k].v[l]
#define PV_CC(k) cctable[k].Pv
#define PVB_CC(k) cctable[k].Pvb


// Function cvUpdateFGDStatModel updates statistical model and returns number of foreground regions
// parameters:
//      curr_frame  - current frame from video sequence
//      p_model     - pointer to CvFGDStatModel structure
static int CV_CDECL
icvUpdateFGDStatModel( IplImage* curr_frame, CvFGDStatModel*  model, double )
{
    int            mask_step = model->Ftd->widthStep;
    CvSeq         *first_seq = NULL, *prev_seq = NULL, *seq = NULL;
    IplImage*      prev_frame = model->prev_frame;
    int            region_count = 0;
    int            FG_pixels_count = 0;
    int            deltaC  = cvRound(model->params.delta * 256 / model->params.Lc);
    int            deltaCC = cvRound(model->params.delta * 256 / model->params.Lcc);
    int            i, j, k, l;

    //clear storages
    cvClearMemStorage(model->storage);
    cvZero(model->foreground);

    // From foreground pixel candidates using image differencing
    // with adaptive thresholding.  The algorithm is from:
    //
    //    Thresholding for Change Detection
    //    Paul L. Rosin 1998 6p
    //    http://www.cis.temple.edu/~latecki/Courses/CIS750-03/Papers/thresh-iccv.pdf
    //
    cvChangeDetection( prev_frame, curr_frame, model->Ftd );
    cvChangeDetection( model->background, curr_frame, model->Fbd );

    for( i = 0; i < model->Ftd->height; i++ )
    {
        for( j = 0; j < model->Ftd->width; j++ )
        {
            if( ((uchar*)model->Fbd->imageData)[i*mask_step+j] || ((uchar*)model->Ftd->imageData)[i*mask_step+j] )
            {
            float Pb  = 0;
                float Pv  = 0;
                float Pvb = 0;

                CvBGPixelStat* stat = model->pixel_stat + i * model->Ftd->width + j;

                CvBGPixelCStatTable*   ctable = stat->ctable;
                CvBGPixelCCStatTable* cctable = stat->cctable;

                uchar* curr_data = (uchar*)(curr_frame->imageData) + i*curr_frame->widthStep + j*3;
                uchar* prev_data = (uchar*)(prev_frame->imageData) + i*prev_frame->widthStep + j*3;

                int val = 0;

                // Is it a motion pixel?
                if( ((uchar*)model->Ftd->imageData)[i*mask_step+j] )
                {
            if( !stat->is_trained_dyn_model ) {

                        val = 1;

            } else {

                        // Compare with stored CCt vectors:
                        for( k = 0;  PV_CC(k) > model->params.alpha2 && k < model->params.N1cc;  k++ )
                        {
                            if ( abs( V_CC(k,0) - prev_data[0]) <=  deltaCC &&
                                 abs( V_CC(k,1) - prev_data[1]) <=  deltaCC &&
                                 abs( V_CC(k,2) - prev_data[2]) <=  deltaCC &&
                                 abs( V_CC(k,3) - curr_data[0]) <=  deltaCC &&
                                 abs( V_CC(k,4) - curr_data[1]) <=  deltaCC &&
                                 abs( V_CC(k,5) - curr_data[2]) <=  deltaCC)
                            {
                                Pv += PV_CC(k);
                                Pvb += PVB_CC(k);
                            }
                        }
                        Pb = stat->Pbcc;
                        if( 2 * Pvb * Pb <= Pv ) val = 1;
                    }
                }
                else if( stat->is_trained_st_model )
                {
                    // Compare with stored Ct vectors:
                    for( k = 0;  PV_C(k) > model->params.alpha2 && k < model->params.N1c;  k++ )
                    {
                        if ( abs( V_C(k,0) - curr_data[0]) <=  deltaC &&
                             abs( V_C(k,1) - curr_data[1]) <=  deltaC &&
                             abs( V_C(k,2) - curr_data[2]) <=  deltaC )
                        {
                            Pv += PV_C(k);
                            Pvb += PVB_C(k);
                        }
                    }
                    Pb = stat->Pbc;
                    if( 2 * Pvb * Pb <= Pv ) val = 1;
                }

                // Update foreground:
                ((uchar*)model->foreground->imageData)[i*mask_step+j] = (uchar)(val*255);
                FG_pixels_count += val;

            }		// end if( change detection...
        }		// for j...
    }			// for i...
    //end BG/FG classification

    // Foreground segmentation.
    // Smooth foreground map:
    if( model->params.perform_morphing ){
        cvMorphologyEx( model->foreground, model->foreground, 0, 0, CV_MOP_OPEN,  model->params.perform_morphing );
        cvMorphologyEx( model->foreground, model->foreground, 0, 0, CV_MOP_CLOSE, model->params.perform_morphing );
    }


    if( model->params.minArea > 0 || model->params.is_obj_without_holes ){

        // Discard under-size foreground regions:
    //
        cvFindContours( model->foreground, model->storage, &first_seq, sizeof(CvContour), CV_RETR_LIST );
        for( seq = first_seq; seq; seq = seq->h_next )
        {
            CvContour* cnt = (CvContour*)seq;
            if( cnt->rect.width * cnt->rect.height < model->params.minArea ||
                (model->params.is_obj_without_holes && CV_IS_SEQ_HOLE(seq)) )
            {
                // Delete under-size contour:
                prev_seq = seq->h_prev;
                if( prev_seq )
                {
                    prev_seq->h_next = seq->h_next;
                    if( seq->h_next ) seq->h_next->h_prev = prev_seq;
                }
                else
                {
                    first_seq = seq->h_next;
                    if( seq->h_next ) seq->h_next->h_prev = NULL;
                }
            }
            else
            {
                region_count++;
            }
        }
        model->foreground_regions = first_seq;
        cvZero(model->foreground);
        cvDrawContours(model->foreground, first_seq, CV_RGB(0, 0, 255), CV_RGB(0, 0, 255), 10, -1);

    } else {

        model->foreground_regions = NULL;
    }

    // Check ALL BG update condition:
    if( ((float)FG_pixels_count/(model->Ftd->width*model->Ftd->height)) > CV_BGFG_FGD_BG_UPDATE_TRESH )
    {
         for( i = 0; i < model->Ftd->height; i++ )
             for( j = 0; j < model->Ftd->width; j++ )
             {
                 CvBGPixelStat* stat = model->pixel_stat + i * model->Ftd->width + j;
                 stat->is_trained_st_model = stat->is_trained_dyn_model = 1;
             }
    }


    // Update background model:
    for( i = 0; i < model->Ftd->height; i++ )
    {
        for( j = 0; j < model->Ftd->width; j++ )
        {
            CvBGPixelStat* stat = model->pixel_stat + i * model->Ftd->width + j;
            CvBGPixelCStatTable* ctable = stat->ctable;
            CvBGPixelCCStatTable* cctable = stat->cctable;

            uchar *curr_data = (uchar*)(curr_frame->imageData)+i*curr_frame->widthStep+j*3;
            uchar *prev_data = (uchar*)(prev_frame->imageData)+i*prev_frame->widthStep+j*3;

            if( ((uchar*)model->Ftd->imageData)[i*mask_step+j] || !stat->is_trained_dyn_model )
            {
                float alpha = stat->is_trained_dyn_model ? model->params.alpha2 : model->params.alpha3;
                float diff = 0;
                int dist, min_dist = 2147483647, indx = -1;

                //update Pb
                stat->Pbcc *= (1.f-alpha);
                if( !((uchar*)model->foreground->imageData)[i*mask_step+j] )
                {
                    stat->Pbcc += alpha;
                }

                // Find best Vi match:
                for(k = 0; PV_CC(k) && k < model->params.N2cc; k++ )
                {
                    // Exponential decay of memory
                    PV_CC(k)  *= (1-alpha);
                    PVB_CC(k) *= (1-alpha);
                    if( PV_CC(k) < MIN_PV )
                    {
                        PV_CC(k) = 0;
                        PVB_CC(k) = 0;
                        continue;
                    }

                    dist = 0;
                    for( l = 0; l < 3; l++ )
                    {
                        int val = abs( V_CC(k,l) - prev_data[l] );
                        if( val > deltaCC ) break;
                        dist += val;
                        val = abs( V_CC(k,l+3) - curr_data[l] );
                        if( val > deltaCC) break;
                        dist += val;
                    }
                    if( l == 3 && dist < min_dist )
                    {
                        min_dist = dist;
                        indx = k;
                    }
                }


                if( indx < 0 )
                {   // Replace N2th elem in the table by new feature:
                    indx = model->params.N2cc - 1;
                    PV_CC(indx) = alpha;
                    PVB_CC(indx) = alpha;
                    //udate Vt
                    for( l = 0; l < 3; l++ )
                    {
                        V_CC(indx,l) = prev_data[l];
                        V_CC(indx,l+3) = curr_data[l];
                    }
                }
                else
                {   // Update:
                    PV_CC(indx) += alpha;
                    if( !((uchar*)model->foreground->imageData)[i*mask_step+j] )
                    {
                        PVB_CC(indx) += alpha;
                    }
                }

                //re-sort CCt table by Pv
                for( k = 0; k < indx; k++ )
                {
                    if( PV_CC(k) <= PV_CC(indx) )
                    {
                        //shift elements
                        CvBGPixelCCStatTable tmp1, tmp2 = cctable[indx];
                        for( l = k; l <= indx; l++ )
                        {
                            tmp1 = cctable[l];
                            cctable[l] = tmp2;
                            tmp2 = tmp1;
                        }
                        break;
                    }
                }


                float sum1=0, sum2=0;
                //check "once-off" changes
                for(k = 0; PV_CC(k) && k < model->params.N1cc; k++ )
                {
                    sum1 += PV_CC(k);
                    sum2 += PVB_CC(k);
                }
                if( sum1 > model->params.T ) stat->is_trained_dyn_model = 1;

                diff = sum1 - stat->Pbcc * sum2;
                // Update stat table:
                if( diff >  model->params.T )
                {
                    //printf("once off change at motion mode\n");
                    //new BG features are discovered
                    for( k = 0; PV_CC(k) && k < model->params.N1cc; k++ )
                    {
                        PVB_CC(k) =
                            (PV_CC(k)-stat->Pbcc*PVB_CC(k))/(1-stat->Pbcc);
                    }
                    assert(stat->Pbcc<=1 && stat->Pbcc>=0);
                }
            }

            // Handle "stationary" pixel:
            if( !((uchar*)model->Ftd->imageData)[i*mask_step+j] )
            {
                float alpha = stat->is_trained_st_model ? model->params.alpha2 : model->params.alpha3;
                float diff = 0;
                int dist, min_dist = 2147483647, indx = -1;

                //update Pb
                stat->Pbc *= (1.f-alpha);
                if( !((uchar*)model->foreground->imageData)[i*mask_step+j] )
                {
                    stat->Pbc += alpha;
                }

                //find best Vi match
                for( k = 0; k < model->params.N2c; k++ )
                {
                    // Exponential decay of memory
                    PV_C(k) *= (1-alpha);
                    PVB_C(k) *= (1-alpha);
                    if( PV_C(k) < MIN_PV )
                    {
                        PV_C(k) = 0;
                        PVB_C(k) = 0;
                        continue;
                    }

                    dist = 0;
                    for( l = 0; l < 3; l++ )
                    {
                        int val = abs( V_C(k,l) - curr_data[l] );
                        if( val > deltaC ) break;
                        dist += val;
                    }
                    if( l == 3 && dist < min_dist )
                    {
                        min_dist = dist;
                        indx = k;
                    }
                }

                if( indx < 0 )
                {//N2th elem in the table is replaced by a new features
                    indx = model->params.N2c - 1;
                    PV_C(indx) = alpha;
                    PVB_C(indx) = alpha;
                    //udate Vt
                    for( l = 0; l < 3; l++ )
                    {
                        V_C(indx,l) = curr_data[l];
                    }
                } else
                {//update
                    PV_C(indx) += alpha;
                    if( !((uchar*)model->foreground->imageData)[i*mask_step+j] )
                    {
                        PVB_C(indx) += alpha;
                    }
                }

                //re-sort Ct table by Pv
                for( k = 0; k < indx; k++ )
                {
                    if( PV_C(k) <= PV_C(indx) )
                    {
                        //shift elements
                        CvBGPixelCStatTable tmp1, tmp2 = ctable[indx];
                        for( l = k; l <= indx; l++ )
                        {
                            tmp1 = ctable[l];
                            ctable[l] = tmp2;
                            tmp2 = tmp1;
                        }
                        break;
                    }
                }

                // Check "once-off" changes:
                float sum1=0, sum2=0;
                for( k = 0; PV_C(k) && k < model->params.N1c; k++ )
                {
                    sum1 += PV_C(k);
                    sum2 += PVB_C(k);
                }
                diff = sum1 - stat->Pbc * sum2;
                if( sum1 > model->params.T ) stat->is_trained_st_model = 1;

                // Update stat table:
                if( diff >  model->params.T )
                {
                    //printf("once off change at stat mode\n");
                    //new BG features are discovered
                    for( k = 0; PV_C(k) && k < model->params.N1c; k++ )
                    {
                        PVB_C(k) = (PV_C(k)-stat->Pbc*PVB_C(k))/(1-stat->Pbc);
                    }
                    stat->Pbc = 1 - stat->Pbc;
                }
            }		// if !(change detection) at pixel (i,j)

            // Update the reference BG image:
            if( !((uchar*)model->foreground->imageData)[i*mask_step+j])
            {
                uchar* ptr = ((uchar*)model->background->imageData) + i*model->background->widthStep+j*3;

                if( !((uchar*)model->Ftd->imageData)[i*mask_step+j] &&
                    !((uchar*)model->Fbd->imageData)[i*mask_step+j] )
                {
                    // Apply IIR filter:
                    for( l = 0; l < 3; l++ )
                    {
                        int a = cvRound(ptr[l]*(1 - model->params.alpha1) + model->params.alpha1*curr_data[l]);
                        ptr[l] = (uchar)a;
                        //((uchar*)model->background->imageData)[i*model->background->widthStep+j*3+l]*=(1 - model->params.alpha1);
                        //((uchar*)model->background->imageData)[i*model->background->widthStep+j*3+l] += model->params.alpha1*curr_data[l];
                    }
                }
                else
                {
                    // Background change detected:
                    for( l = 0; l < 3; l++ )
                    {
                        //((uchar*)model->background->imageData)[i*model->background->widthStep+j*3+l] = curr_data[l];
                        ptr[l] = curr_data[l];
                    }
                }
            }
        }		// j
    }			// i

    // Keep previous frame:
    cvCopy( curr_frame, model->prev_frame );

    return region_count;
}

/* End of file. */
