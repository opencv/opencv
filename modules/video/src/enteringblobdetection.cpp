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

/*
This file implements the virtual interface defined as "CvBlobDetector".
This implementation based on a simple algorithm:
A new blob is detected when several successive frames contains connected components
which have uniform motion not at an unreasonably high speed.
Separation from border and already tracked blobs are also considered.

For an entrypoint into the literature see:

     Appearance Models for Occlusion Handling
     Andrew Senior &t al, 8p 2001
     http://www.research.ibm.com/peoplevision/PETS2001.pdf

*/

//#define USE_OBJECT_DETECTOR

#include "precomp.hpp"

static int CompareContour(const void* a, const void* b, void* )
{
    float           dx,dy;
    float           h,w,ht,wt;
    CvPoint2D32f    pa,pb;
    CvRect          ra,rb;
    CvSeq*          pCA = *(CvSeq**)a;
    CvSeq*          pCB = *(CvSeq**)b;
    ra = ((CvContour*)pCA)->rect;
    rb = ((CvContour*)pCB)->rect;
    pa.x = ra.x + ra.width*0.5f;
    pa.y = ra.y + ra.height*0.5f;
    pb.x = rb.x + rb.width*0.5f;
    pb.y = rb.y + rb.height*0.5f;
    w = (ra.width+rb.width)*0.5f;
    h = (ra.height+rb.height)*0.5f;

    dx = (float)(fabs(pa.x - pb.x)-w);
    dy = (float)(fabs(pa.y - pb.y)-h);

    //wt = MAX(ra.width,rb.width)*0.1f;
    wt = 0;
    ht = MAX(ra.height,rb.height)*0.3f;
    return (dx < wt && dy < ht);
}

void cvFindBlobsByCCClasters(IplImage* pFG, CvBlobSeq* pBlobs, CvMemStorage* storage)
{   /* Create contours: */
    IplImage*       pIB = NULL;
    CvSeq*          cnt = NULL;
    CvSeq*          cnt_list = cvCreateSeq(0,sizeof(CvSeq),sizeof(CvSeq*), storage );
    CvSeq*          clasters = NULL;
    int             claster_cur, claster_num;

    pIB = cvCloneImage(pFG);
    cvThreshold(pIB,pIB,128,255,CV_THRESH_BINARY);
    cvFindContours(pIB,storage, &cnt, sizeof(CvContour), CV_RETR_EXTERNAL);
    cvReleaseImage(&pIB);

    /* Create cnt_list.      */
    /* Process each contour: */
    for(; cnt; cnt=cnt->h_next)
    {
        cvSeqPush( cnt_list, &cnt);
    }

    claster_num = cvSeqPartition( cnt_list, storage, &clasters, CompareContour, NULL );

    for(claster_cur=0; claster_cur<claster_num; ++claster_cur)
    {
        int         cnt_cur;
        CvBlob      NewBlob;
        double      M00,X,Y,XX,YY; /* image moments */
        CvMoments   m;
        CvRect      rect_res = cvRect(-1,-1,-1,-1);
        CvMat       mat;

        for(cnt_cur=0; cnt_cur<clasters->total; ++cnt_cur)
        {
            CvRect  rect;
            CvSeq*  cnt;
            int k = *(int*)cvGetSeqElem( clasters, cnt_cur );
            if(k!=claster_cur) continue;
            cnt = *(CvSeq**)cvGetSeqElem( cnt_list, cnt_cur );
            rect = ((CvContour*)cnt)->rect;

            if(rect_res.height<0)
            {
                rect_res = rect;
            }
            else
            {   /* Unite rects: */
                int x0,x1,y0,y1;
                x0 = MIN(rect_res.x,rect.x);
                y0 = MIN(rect_res.y,rect.y);
                x1 = MAX(rect_res.x+rect_res.width,rect.x+rect.width);
                y1 = MAX(rect_res.y+rect_res.height,rect.y+rect.height);
                rect_res.x = x0;
                rect_res.y = y0;
                rect_res.width = x1-x0;
                rect_res.height = y1-y0;
            }
        }

        if(rect_res.height < 1 || rect_res.width < 1)
        {
            X = 0;
            Y = 0;
            XX = 0;
            YY = 0;
        }
        else
        {
            cvMoments( cvGetSubRect(pFG,&mat,rect_res), &m, 0 );
            M00 = cvGetSpatialMoment( &m, 0, 0 );
            if(M00 <= 0 ) continue;
            X = cvGetSpatialMoment( &m, 1, 0 )/M00;
            Y = cvGetSpatialMoment( &m, 0, 1 )/M00;
            XX = (cvGetSpatialMoment( &m, 2, 0 )/M00) - X*X;
            YY = (cvGetSpatialMoment( &m, 0, 2 )/M00) - Y*Y;
        }
        NewBlob = cvBlob(rect_res.x+(float)X,rect_res.y+(float)Y,(float)(4*sqrt(XX)),(float)(4*sqrt(YY)));
        pBlobs->AddBlob(&NewBlob);

    }   /* Next cluster. */

    #if 0
    {   // Debug info:
        IplImage* pI = cvCreateImage(cvSize(pFG->width,pFG->height),IPL_DEPTH_8U,3);
        cvZero(pI);
        for(claster_cur=0; claster_cur<claster_num; ++claster_cur)
        {
            int         cnt_cur;
            CvScalar    color = CV_RGB(rand()%256,rand()%256,rand()%256);

            for(cnt_cur=0; cnt_cur<clasters->total; ++cnt_cur)
            {
                CvSeq*  cnt;
                int k = *(int*)cvGetSeqElem( clasters, cnt_cur );
                if(k!=claster_cur) continue;
                cnt = *(CvSeq**)cvGetSeqElem( cnt_list, cnt_cur );
                cvDrawContours( pI, cnt, color, color, 0, 1, 8);
            }

            CvBlob* pB = pBlobs->GetBlob(claster_cur);
            int x = cvRound(CV_BLOB_RX(pB)), y = cvRound(CV_BLOB_RY(pB));
            cvEllipse( pI,
                cvPointFrom32f(CV_BLOB_CENTER(pB)),
                cvSize(MAX(1,x), MAX(1,y)),
                0, 0, 360,
                color, 1 );
        }

        cvNamedWindow( "Clusters", 0);
        cvShowImage( "Clusters",pI );

        cvReleaseImage(&pI);

    }   /* Debug info. */
    #endif

}   /* cvFindBlobsByCCClasters */

/* Simple blob detector.  */
/* Number of successive frame to analyse: */
#define EBD_FRAME_NUM   5
class CvBlobDetectorSimple:public CvBlobDetector
{
public:
    CvBlobDetectorSimple();
   ~CvBlobDetectorSimple();
    int DetectNewBlob(IplImage* pImg, IplImage* pFGMask, CvBlobSeq* pNewBlobList, CvBlobSeq* pOldBlobList);
    void Release(){delete this;};

protected:
    IplImage*       m_pMaskBlobNew;
    IplImage*       m_pMaskBlobExist;
    /* Lists of connected components detected on previous frames: */
    CvBlobSeq*      m_pBlobLists[EBD_FRAME_NUM];
};

/* Blob detector creator (sole interface function for this file) */
CvBlobDetector* cvCreateBlobDetectorSimple(){return new CvBlobDetectorSimple;};

/* Constructor of BlobDetector: */
CvBlobDetectorSimple::CvBlobDetectorSimple()
{
    int i = 0;
    m_pMaskBlobNew = NULL;
    m_pMaskBlobExist = NULL;
    for(i=0;i<EBD_FRAME_NUM;++i)m_pBlobLists[i] = NULL;

    SetModuleName("Simple");
}

/* Destructor of BlobDetector: */
CvBlobDetectorSimple::~CvBlobDetectorSimple()
{
    int i;
    if(m_pMaskBlobExist) cvReleaseImage(&m_pMaskBlobExist);
    if(m_pMaskBlobNew) cvReleaseImage(&m_pMaskBlobNew);
    for(i=0; i<EBD_FRAME_NUM; ++i) delete m_pBlobLists[i];
}   /* cvReleaseBlobDetector */

/* cvDetectNewBlobs
 * return 1 and fill blob pNewBlob by blob parameters
 * if new blob is detected:
 */
int CvBlobDetectorSimple::DetectNewBlob(IplImage* /*pImg*/, IplImage* pFGMask, CvBlobSeq* pNewBlobList, CvBlobSeq* pOldBlobList)
{
    int         result = 0;
    CvSize      S = cvSize(pFGMask->width,pFGMask->height);
    if(m_pMaskBlobNew == NULL ) m_pMaskBlobNew = cvCreateImage(S,IPL_DEPTH_8U,1);
    if(m_pMaskBlobExist == NULL ) m_pMaskBlobExist = cvCreateImage(S,IPL_DEPTH_8U,1);

    /* Shift blob list: */
    {
        int     i;
        if(m_pBlobLists[0]) delete m_pBlobLists[0];
        for(i=1;i<EBD_FRAME_NUM;++i)m_pBlobLists[i-1]=m_pBlobLists[i];
        m_pBlobLists[EBD_FRAME_NUM-1] = new CvBlobSeq;
    }   /* Shift blob list. */

    /* Create exist blob mask: */
    cvCopy(pFGMask, m_pMaskBlobNew);

    /* Create contours and add new blobs to blob list: */
    {   /* Create blobs: */
        CvBlobSeq       Blobs;
        CvMemStorage*   storage = cvCreateMemStorage();

#if 1
        {   /* Glue contours: */
            cvFindBlobsByCCClasters(m_pMaskBlobNew, &Blobs, storage );
        }   /* Glue contours. */
#else
        {   /**/
            IplImage*       pIB = cvCloneImage(m_pMaskBlobNew);
            CvSeq*          cnts = NULL;
            CvSeq*          cnt = NULL;
            cvThreshold(pIB,pIB,128,255,CV_THRESH_BINARY);
            cvFindContours(pIB,storage, &cnts, sizeof(CvContour), CV_RETR_EXTERNAL);

            /* Process each contour: */
            for(cnt = cnts; cnt; cnt=cnt->h_next)
            {
                CvBlob  NewBlob;

                /* Image moments: */
                double      M00,X,Y,XX,YY;
                CvMoments   m;
                CvRect      r = ((CvContour*)cnt)->rect;
                CvMat       mat;

                if(r.height < S.height*0.02 || r.width < S.width*0.02) continue;

                cvMoments( cvGetSubRect(m_pMaskBlobNew,&mat,r), &m, 0 );
                M00 = cvGetSpatialMoment( &m, 0, 0 );

                if(M00 <= 0 ) continue;

                X  = cvGetSpatialMoment( &m, 1, 0 )/M00;
                Y  = cvGetSpatialMoment( &m, 0, 1 )/M00;

                XX = (cvGetSpatialMoment( &m, 2, 0 )/M00) - X*X;
                YY = (cvGetSpatialMoment( &m, 0, 2 )/M00) - Y*Y;

                NewBlob = cvBlob(r.x+(float)X,r.y+(float)Y,(float)(4*sqrt(XX)),(float)(4*sqrt(YY)));

                Blobs.AddBlob(&NewBlob);

            }   /* Next contour. */

            cvReleaseImage(&pIB);

        }   /* One contour - one blob. */
#endif

        {   /* Delete small and intersected blobs: */
            int i;
            for(i=Blobs.GetBlobNum(); i>0; i--)
            {
                CvBlob* pB = Blobs.GetBlob(i-1);

                if(pB->h < S.height*0.02 || pB->w < S.width*0.02)
                {
                    Blobs.DelBlob(i-1);
                    continue;
                }
                if(pOldBlobList)
                {
                    int j;
                    for(j=pOldBlobList->GetBlobNum(); j>0; j--)
                    {
                        CvBlob* pBOld = pOldBlobList->GetBlob(j-1);
                        if((fabs(pBOld->x-pB->x) < (CV_BLOB_RX(pBOld)+CV_BLOB_RX(pB))) &&
                           (fabs(pBOld->y-pB->y) < (CV_BLOB_RY(pBOld)+CV_BLOB_RY(pB))))
                        {   /* Intersection is present, so delete blob from list: */
                            Blobs.DelBlob(i-1);
                            break;
                        }
                    }   /* Check next old blob. */
                }   /*  if pOldBlobList */
            }   /* Check next blob. */
        }   /*  Delete small and intersected blobs. */

        {   /* Bubble-sort blobs by size: */
            int N = Blobs.GetBlobNum();
            int i,j;
            for(i=1; i<N; ++i)
            {
                for(j=i; j>0; --j)
                {
                    CvBlob  temp;
                    float   AreaP, AreaN;
                    CvBlob* pP = Blobs.GetBlob(j-1);
                    CvBlob* pN = Blobs.GetBlob(j);
                    AreaP = CV_BLOB_WX(pP)*CV_BLOB_WY(pP);
                    AreaN = CV_BLOB_WX(pN)*CV_BLOB_WY(pN);
                    if(AreaN < AreaP)break;
                    temp = pN[0];
                    pN[0] = pP[0];
                    pP[0] = temp;
                }
            }

            /* Copy only first 10 blobs: */
            for(i=0; i<MIN(N,10); ++i)
            {
                m_pBlobLists[EBD_FRAME_NUM-1]->AddBlob(Blobs.GetBlob(i));
            }

        }   /* Sort blobs by size. */

        cvReleaseMemStorage(&storage);

    }   /* Create blobs. */

    /* Analyze blob list to find best blob trajectory: */
    {
        int     Count = 0;
        int     pBLIndex[EBD_FRAME_NUM];
        int     pBL_BEST[EBD_FRAME_NUM];
        int     i;
        int     finish = 0;
        double  BestError = -1;
        int     Good = 1;

        for(i=0; i<EBD_FRAME_NUM; ++i)
        {
            pBLIndex[i] = 0;
            pBL_BEST[i] = 0;
        }

        /* Check configuration exist: */
        for(i=0; Good && (i<EBD_FRAME_NUM); ++i)
            if(m_pBlobLists[i] == NULL || m_pBlobLists[i]->GetBlobNum() == 0)
                Good = 0;

        if(Good)
        do{ /* For each configuration: */
            CvBlob* pBL[EBD_FRAME_NUM];
            int     Good = 1;
            double  Error = 0;
            CvBlob* pBNew = m_pBlobLists[EBD_FRAME_NUM-1]->GetBlob(pBLIndex[EBD_FRAME_NUM-1]);

            for(i=0; i<EBD_FRAME_NUM; ++i)  pBL[i] = m_pBlobLists[i]->GetBlob(pBLIndex[i]);

            Count++;

            /* Check intersection last blob with existed: */
            if(Good && pOldBlobList)
            {   /* Check intersection last blob with existed: */
                int     k;
                for(k=pOldBlobList->GetBlobNum(); k>0; --k)
                {
                    CvBlob* pBOld = pOldBlobList->GetBlob(k-1);
                    if((fabs(pBOld->x-pBNew->x) < (CV_BLOB_RX(pBOld)+CV_BLOB_RX(pBNew))) &&
                       (fabs(pBOld->y-pBNew->y) < (CV_BLOB_RY(pBOld)+CV_BLOB_RY(pBNew))))
                        Good = 0;
                }
            }   /* Check intersection last blob with existed. */

            /* Check distance to image border: */
            if(Good)
            {   /* Check distance to image border: */
                CvBlob*  pB = pBNew;
                float    dx = MIN(pB->x,S.width-pB->x)/CV_BLOB_RX(pB);
                float    dy = MIN(pB->y,S.height-pB->y)/CV_BLOB_RY(pB);

                if(dx < 1.1 || dy < 1.1) Good = 0;
            }   /* Check distance to image border. */

            /* Check uniform motion: */
            if(Good)
            {
                int     N = EBD_FRAME_NUM;
                float   sum[2] = {0,0};
                float   jsum[2] = {0,0};
                float   a[2],b[2]; /* estimated parameters of moving x(t) = a*t+b*/

                int j;
                for(j=0; j<N; ++j)
                {
                    float   x = pBL[j]->x;
                    float   y = pBL[j]->y;
                    sum[0] += x;
                    jsum[0] += j*x;
                    sum[1] += y;
                    jsum[1] += j*y;
                }

                a[0] = 6*((1-N)*sum[0]+2*jsum[0])/(N*(N*N-1));
                b[0] = -2*((1-2*N)*sum[0]+3*jsum[0])/(N*(N+1));
                a[1] = 6*((1-N)*sum[1]+2*jsum[1])/(N*(N*N-1));
                b[1] = -2*((1-2*N)*sum[1]+3*jsum[1])/(N*(N+1));

                for(j=0; j<N; ++j)
                {
                    Error +=
                        pow(a[0]*j+b[0]-pBL[j]->x,2)+
                        pow(a[1]*j+b[1]-pBL[j]->y,2);
                }

                Error = sqrt(Error/N);

                if( Error > S.width*0.01 ||
                    fabs(a[0])>S.width*0.1 ||
                    fabs(a[1])>S.height*0.1)
                    Good = 0;

            }   /* Check configuration. */


            /* New best trajectory: */
            if(Good && (BestError == -1 || BestError > Error))
            {
                for(i=0; i<EBD_FRAME_NUM; ++i)
                {
                    pBL_BEST[i] = pBLIndex[i];
                }
                BestError = Error;
            }   /* New best trajectory. */

            /* Set next configuration: */
            for(i=0; i<EBD_FRAME_NUM; ++i)
            {
                pBLIndex[i]++;
                if(pBLIndex[i] != m_pBlobLists[i]->GetBlobNum()) break;
                pBLIndex[i]=0;
            }   /* Next time shift. */

            if(i==EBD_FRAME_NUM)finish=1;

        } while(!finish);	/* Check next time configuration of connected components. */

        #if 0
        {/**/
            printf("BlobDetector configurations = %d [",Count);
            int i;
            for(i=0; i<EBD_FRAME_NUM; ++i)
            {
                printf("%d,",m_pBlobLists[i]?m_pBlobLists[i]->GetBlobNum():0);
            }
            printf("]\n");

        }
        #endif

        if(BestError != -1)
        {   /* Write new blob to output and delete from blob list: */
            CvBlob* pNewBlob = m_pBlobLists[EBD_FRAME_NUM-1]->GetBlob(pBL_BEST[EBD_FRAME_NUM-1]);
            pNewBlobList->AddBlob(pNewBlob);

            for(i=0; i<EBD_FRAME_NUM; ++i)
            {   /* Remove blob from each list: */
                m_pBlobLists[i]->DelBlob(pBL_BEST[i]);
            }   /* Remove blob from each list. */

            result = 1;

        }   /* Write new blob to output and delete from blob list. */
    }   /*  Analyze blob list to find best blob trajectory. */

    return result;

}   /* cvDetectNewBlob */




/* Simple blob detector2.  */
/* Number of successive frames to analyse: */
#define SEQ_SIZE_MAX    30
#define SEQ_NUM         1000
typedef struct
{
    int     size;
    CvBlob* pBlobs[SEQ_SIZE_MAX];
} DefSeq;

class CvBlobDetectorCC:public CvBlobDetector
{
public:
    CvBlobDetectorCC();
   ~CvBlobDetectorCC();
    int DetectNewBlob(IplImage* pImg, IplImage* pFGMask, CvBlobSeq* pNewBlobList, CvBlobSeq* pOldBlobList);
    void Release(){delete this;};

    virtual void ParamUpdate()
    {
        if(SEQ_SIZE<1)SEQ_SIZE=1;
        if(SEQ_SIZE>SEQ_SIZE_MAX)SEQ_SIZE=SEQ_SIZE_MAX;

#ifdef USE_OBJECT_DETECTOR
        if( m_param_split_detector_file_name )
        {
            m_split_detector = new CvObjectDetector();
            if( !m_split_detector->Load( m_param_split_detector_file_name ) )
            {
                delete m_split_detector;
                m_split_detector = 0;
            }
            else
            {
                m_min_window_size = m_split_detector->GetMinWindowSize();
                m_max_border = m_split_detector->GetMaxBorderSize();
            }
        }
#endif
    }

private:
    /* Lists of connected components detected on previous frames: */
    CvBlobSeq*      m_pBlobLists[SEQ_SIZE_MAX];
    DefSeq          m_TrackSeq[SEQ_NUM];
    int             m_TrackNum;
    float           m_HMin;
    float           m_WMin;
    float           m_MinDistToBorder;
    int             m_Clastering;
    int             SEQ_SIZE;

    /* If not 0 then the detector is loaded from the specified file
     * and it is applied for splitting blobs which actually correspond
     * to groups of objects:
     */
    char*           m_param_split_detector_file_name;
    float           m_param_roi_scale;
    int             m_param_only_roi;

    CvObjectDetector* m_split_detector;
    CvSize          m_min_window_size;
    int             m_max_border;

    CvBlobSeq       m_detected_blob_seq;
    CvSeq*          m_roi_seq;

    CvBlobSeq       m_debug_blob_seq;
};

/* Blob detector creator (sole interface function for this file): */
CvBlobDetector* cvCreateBlobDetectorCC(){return new CvBlobDetectorCC;}

/* Constructor for BlobDetector: */
CvBlobDetectorCC::CvBlobDetectorCC() :
    m_split_detector(0),
    m_detected_blob_seq(sizeof(CvDetectedBlob)),
    m_roi_seq(0),
    m_debug_blob_seq(sizeof(CvDetectedBlob))
{
    /*CvDrawShape shapes[] =
    {
        { CvDrawShape::RECT,    {{255,255,255}} },
        { CvDrawShape::RECT,    {{0,0,255}} },
        { CvDrawShape::ELLIPSE, {{0,255,0}} }
    };
    int num_shapes = sizeof(shapes) / sizeof(shapes[0]);*/

    int i = 0;
    SEQ_SIZE = 10;
    AddParam("Latency",&SEQ_SIZE);
    for(i=0;i<SEQ_SIZE_MAX;++i)m_pBlobLists[i] = NULL;
    for(i=0;i<SEQ_NUM;++i)m_TrackSeq[i].size = 0;
    m_TrackNum = 0;

    m_HMin = 0.02f;
    m_WMin = 0.01f;
    AddParam("HMin",&m_HMin);
    AddParam("WMin",&m_WMin);
    m_MinDistToBorder = 1.1f;
    AddParam("MinDistToBorder",&m_MinDistToBorder);
    CommentParam("MinDistToBorder","Minimal allowed distance from blob center to image border in blob sizes");

    m_Clastering=1;
    AddParam("Clastering",&m_Clastering);
    CommentParam("Clastering","Minimal allowed distance from blob center to image border in blob sizes");

    m_param_split_detector_file_name = 0;
#ifdef USE_OBJECT_DETECTOR
    AddParam("Detector", &m_param_split_detector_file_name);
    CommentParam("Detector", "Detector file name");
#endif

    m_param_roi_scale = 1.5F;
    AddParam("ROIScale", &m_param_roi_scale);
    CommentParam("ROIScale", "Determines the size of search window around a blob");

    m_param_only_roi = 1;
    AddParam("OnlyROI", &m_param_only_roi);
    CommentParam("OnlyROI", "Shows the whole debug image (0) or only ROIs where the detector was applied (1)");

    m_min_window_size = cvSize(0,0);
    m_max_border = 0;
    m_roi_seq = cvCreateSeq( 0, sizeof(*m_roi_seq), sizeof(CvRect), cvCreateMemStorage() );

    SetModuleName("CC");
}

/* Destructor for BlobDetector: */
CvBlobDetectorCC::~CvBlobDetectorCC()
{
    int i;
    for(i=0; i<SEQ_SIZE_MAX; ++i)
    {
        if(m_pBlobLists[i])
            delete m_pBlobLists[i];
    }

    if( m_roi_seq )
    {
        cvReleaseMemStorage( &m_roi_seq->storage );
        m_roi_seq = 0;
    }
    //cvDestroyWindow( "EnteringBlobDetectionDebug" );
}   /* cvReleaseBlobDetector */


/* cvDetectNewBlobs
 * Return 1 and fill blob pNewBlob  with
 * blob parameters if new blob is detected:
 */
int CvBlobDetectorCC::DetectNewBlob(IplImage* /*pImg*/, IplImage* pFGMask, CvBlobSeq* pNewBlobList, CvBlobSeq* pOldBlobList)
{
    int         result = 0;
    CvSize      S = cvSize(pFGMask->width,pFGMask->height);

    /* Shift blob list: */
    {
        int     i;
        if(m_pBlobLists[SEQ_SIZE-1]) delete m_pBlobLists[SEQ_SIZE-1];

        for(i=SEQ_SIZE-1; i>0; --i)  m_pBlobLists[i] = m_pBlobLists[i-1];

        m_pBlobLists[0] = new CvBlobSeq;

    }   /* Shift blob list. */

    /* Create contours and add new blobs to blob list: */
    {   /* Create blobs: */
        CvBlobSeq       Blobs;
        CvMemStorage*   storage = cvCreateMemStorage();

        if(m_Clastering)
        {   /* Glue contours: */
            cvFindBlobsByCCClasters(pFGMask, &Blobs, storage );
        }   /* Glue contours. */
        else
        { /**/
            IplImage*       pIB = cvCloneImage(pFGMask);
            CvSeq*          cnts = NULL;
            CvSeq*          cnt = NULL;
            cvThreshold(pIB,pIB,128,255,CV_THRESH_BINARY);
            cvFindContours(pIB,storage, &cnts, sizeof(CvContour), CV_RETR_EXTERNAL);

            /* Process each contour: */
            for(cnt = cnts; cnt; cnt=cnt->h_next)
            {
                CvBlob  NewBlob;
                /* Image moments: */
                double      M00,X,Y,XX,YY;
                CvMoments   m;
                CvRect      r = ((CvContour*)cnt)->rect;
                CvMat       mat;
                if(r.height < S.height*m_HMin || r.width < S.width*m_WMin) continue;
                cvMoments( cvGetSubRect(pFGMask,&mat,r), &m, 0 );
                M00 = cvGetSpatialMoment( &m, 0, 0 );
                if(M00 <= 0 ) continue;
                X = cvGetSpatialMoment( &m, 1, 0 )/M00;
                Y = cvGetSpatialMoment( &m, 0, 1 )/M00;
                XX = (cvGetSpatialMoment( &m, 2, 0 )/M00) - X*X;
                YY = (cvGetSpatialMoment( &m, 0, 2 )/M00) - Y*Y;
                NewBlob = cvBlob(r.x+(float)X,r.y+(float)Y,(float)(4*sqrt(XX)),(float)(4*sqrt(YY)));
                Blobs.AddBlob(&NewBlob);

            }   /* Next contour. */

            cvReleaseImage(&pIB);

        }   /* One contour - one blob. */

        {   /* Delete small and intersected blobs: */
            int i;
            for(i=Blobs.GetBlobNum(); i>0; i--)
            {
                CvBlob* pB = Blobs.GetBlob(i-1);

                if(pB->h < S.height*m_HMin || pB->w < S.width*m_WMin)
                {
                    Blobs.DelBlob(i-1);
                    continue;
                }

                if(pOldBlobList)
                {
                    int j;
                    for(j=pOldBlobList->GetBlobNum(); j>0; j--)
                    {
                        CvBlob* pBOld = pOldBlobList->GetBlob(j-1);
                        if((fabs(pBOld->x-pB->x) < (CV_BLOB_RX(pBOld)+CV_BLOB_RX(pB))) &&
                           (fabs(pBOld->y-pB->y) < (CV_BLOB_RY(pBOld)+CV_BLOB_RY(pB))))
                        {   /* Intersection detected, delete blob from list: */
                            Blobs.DelBlob(i-1);
                            break;
                        }
                    }   /* Check next old blob. */
                }   /*  if pOldBlobList. */
            }   /*  Check next blob. */
        }   /*  Delete small and intersected blobs. */

        {   /* Bubble-sort blobs by size: */
            int N = Blobs.GetBlobNum();
            int i,j;
            for(i=1; i<N; ++i)
            {
                for(j=i; j>0; --j)
                {
                    CvBlob  temp;
                    float   AreaP, AreaN;
                    CvBlob* pP = Blobs.GetBlob(j-1);
                    CvBlob* pN = Blobs.GetBlob(j);
                    AreaP = CV_BLOB_WX(pP)*CV_BLOB_WY(pP);
                    AreaN = CV_BLOB_WX(pN)*CV_BLOB_WY(pN);
                    if(AreaN < AreaP)break;
                    temp = pN[0];
                    pN[0] = pP[0];
                    pP[0] = temp;
                }
            }

            /* Copy only first 10 blobs: */
            for(i=0; i<MIN(N,10); ++i)
            {
                m_pBlobLists[0]->AddBlob(Blobs.GetBlob(i));
            }

        }   /* Sort blobs by size. */

        cvReleaseMemStorage(&storage);

    }   /* Create blobs. */

    {   /* Shift each track: */
        int j;
        for(j=0; j<m_TrackNum; ++j)
        {
            int     i;
            DefSeq* pTrack = m_TrackSeq+j;

            for(i=SEQ_SIZE-1; i>0; --i)
                pTrack->pBlobs[i] = pTrack->pBlobs[i-1];

            pTrack->pBlobs[0] = NULL;
            if(pTrack->size == SEQ_SIZE)pTrack->size--;
        }
    }   /* Shift each track. */

    /* Analyze blob list to find best blob trajectory: */
    {
        double      BestError = -1;
        int         BestTrack = -1;;
        CvBlobSeq*  pNewBlobs = m_pBlobLists[0];
        int         i;
        int         NewTrackNum = 0;
        for(i=pNewBlobs->GetBlobNum(); i>0; --i)
        {
            CvBlob* pBNew = pNewBlobs->GetBlob(i-1);
            int     j;
            int     AsignedTrack = 0;
            for(j=0; j<m_TrackNum; ++j)
            {
                double  dx,dy;
                DefSeq* pTrack = m_TrackSeq+j;
                CvBlob* pLastBlob = pTrack->size>0?pTrack->pBlobs[1]:NULL;
                if(pLastBlob == NULL) continue;
                dx = fabs(CV_BLOB_X(pLastBlob)-CV_BLOB_X(pBNew));
                dy = fabs(CV_BLOB_Y(pLastBlob)-CV_BLOB_Y(pBNew));
                if(dx > 2*CV_BLOB_WX(pLastBlob) || dy > 2*CV_BLOB_WY(pLastBlob)) continue;
                AsignedTrack++;

                if(pTrack->pBlobs[0]==NULL)
                {   /* Fill existed track: */
                    pTrack->pBlobs[0] = pBNew;
                    pTrack->size++;
                }
                else if((m_TrackNum+NewTrackNum)<SEQ_NUM)
                {   /* Duplicate existed track: */
                    m_TrackSeq[m_TrackNum+NewTrackNum] = pTrack[0];
                    m_TrackSeq[m_TrackNum+NewTrackNum].pBlobs[0] = pBNew;
                    NewTrackNum++;
                }
            }   /* Next track. */

            if(AsignedTrack==0 && (m_TrackNum+NewTrackNum)<SEQ_NUM )
            {   /* Initialize new track: */
                m_TrackSeq[m_TrackNum+NewTrackNum].size = 1;
                m_TrackSeq[m_TrackNum+NewTrackNum].pBlobs[0] = pBNew;
                NewTrackNum++;
            }
        }   /* Next new blob. */

        m_TrackNum += NewTrackNum;

        /* Check each track: */
        for(i=0; i<m_TrackNum; ++i)
        {
            int     Good = 1;
            DefSeq* pTrack = m_TrackSeq+i;
            CvBlob* pBNew = pTrack->pBlobs[0];
            if(pTrack->size != SEQ_SIZE) continue;
            if(pBNew == NULL ) continue;

            /* Check intersection last blob with existed: */
            if(Good && pOldBlobList)
            {
                int k;
                for(k=pOldBlobList->GetBlobNum(); k>0; --k)
                {
                    CvBlob* pBOld = pOldBlobList->GetBlob(k-1);
                    if((fabs(pBOld->x-pBNew->x) < (CV_BLOB_RX(pBOld)+CV_BLOB_RX(pBNew))) &&
                       (fabs(pBOld->y-pBNew->y) < (CV_BLOB_RY(pBOld)+CV_BLOB_RY(pBNew))))
                        Good = 0;
                }
            }   /* Check intersection last blob with existed. */

            /* Check distance to image border: */
            if(Good)
            {   /* Check distance to image border: */
                float    dx = MIN(pBNew->x,S.width-pBNew->x)/CV_BLOB_RX(pBNew);
                float    dy = MIN(pBNew->y,S.height-pBNew->y)/CV_BLOB_RY(pBNew);
                if(dx < m_MinDistToBorder || dy < m_MinDistToBorder) Good = 0;
            }   /* Check distance to image border. */

            /* Check uniform motion: */
            if(Good)
            {   /* Check uniform motion: */
                double      Error = 0;
                int         N = pTrack->size;
                CvBlob**    pBL = pTrack->pBlobs;
                float       sum[2] = {0,0};
                float       jsum[2] = {0,0};
                float       a[2],b[2]; /* estimated parameters of moving x(t) = a*t+b*/
                int         j;

                for(j=0; j<N; ++j)
                {
                    float   x = pBL[j]->x;
                    float   y = pBL[j]->y;
                    sum[0] += x;
                    jsum[0] += j*x;
                    sum[1] += y;
                    jsum[1] += j*y;
                }

                a[0] = 6*((1-N)*sum[0]+2*jsum[0])/(N*(N*N-1));
                b[0] = -2*((1-2*N)*sum[0]+3*jsum[0])/(N*(N+1));
                a[1] = 6*((1-N)*sum[1]+2*jsum[1])/(N*(N*N-1));
                b[1] = -2*((1-2*N)*sum[1]+3*jsum[1])/(N*(N+1));

                for(j=0; j<N; ++j)
                {
                    Error +=
                        pow(a[0]*j+b[0]-pBL[j]->x,2)+
                        pow(a[1]*j+b[1]-pBL[j]->y,2);
                }

                Error = sqrt(Error/N);

                if( Error > S.width*0.01 ||
                    fabs(a[0])>S.width*0.1 ||
                    fabs(a[1])>S.height*0.1)
                    Good = 0;

                /* New best trajectory: */
                if(Good && (BestError == -1 || BestError > Error))
                {   /* New best trajectory: */
                    BestTrack = i;
                    BestError = Error;
                }   /* New best trajectory. */
            }   /*  Check uniform motion. */
        }   /*  Next track. */

        #if 0
        {   /**/
            printf("BlobDetector configurations = %d [",m_TrackNum);
            int i;
            for(i=0; i<SEQ_SIZE; ++i)
            {
                printf("%d,",m_pBlobLists[i]?m_pBlobLists[i]->GetBlobNum():0);
            }
            printf("]\n");
        }
        #endif

        if(BestTrack >= 0)
        {   /* Put new blob to output and delete from blob list: */
            assert(m_TrackSeq[BestTrack].size == SEQ_SIZE);
            assert(m_TrackSeq[BestTrack].pBlobs[0]);
            pNewBlobList->AddBlob(m_TrackSeq[BestTrack].pBlobs[0]);
            m_TrackSeq[BestTrack].pBlobs[0] = NULL;
            m_TrackSeq[BestTrack].size--;
            result = 1;
        }   /* Put new blob to output and mark in blob list to delete. */
    }   /*  Analyze blod list to find best blob trajectory. */

    {   /* Delete bad tracks: */
        int i;
        for(i=m_TrackNum-1; i>=0; --i)
        {   /* Delete bad tracks: */
            if(m_TrackSeq[i].pBlobs[0]) continue;
            if(m_TrackNum>0)
                m_TrackSeq[i] = m_TrackSeq[--m_TrackNum];
        }   /* Delete bad tracks: */
    }

#ifdef USE_OBJECT_DETECTOR
    if( m_split_detector && pNewBlobList->GetBlobNum() > 0 )
    {
        int num_new_blobs = pNewBlobList->GetBlobNum();
        int i = 0;

        if( m_roi_seq ) cvClearSeq( m_roi_seq );
        m_debug_blob_seq.Clear();
        for( i = 0; i < num_new_blobs; ++i )
        {
            CvBlob* b = pNewBlobList->GetBlob(i);
            CvMat roi_stub;
            CvMat* roi_mat = 0;
            CvMat* scaled_roi_mat = 0;

            CvDetectedBlob d_b = cvDetectedBlob( CV_BLOB_X(b), CV_BLOB_Y(b), CV_BLOB_WX(b), CV_BLOB_WY(b), 0 );
            m_debug_blob_seq.AddBlob(&d_b);

            float scale = m_param_roi_scale * m_min_window_size.height / CV_BLOB_WY(b);

            float b_width =   MAX(CV_BLOB_WX(b), m_min_window_size.width / scale)
                            + (m_param_roi_scale - 1.0F) * (m_min_window_size.width / scale)
                            + 2.0F * m_max_border / scale;
            float b_height = CV_BLOB_WY(b) * m_param_roi_scale + 2.0F * m_max_border / scale;

            CvRect roi = cvRectIntersection( cvRect( cvFloor(CV_BLOB_X(b) - 0.5F*b_width),
                                                     cvFloor(CV_BLOB_Y(b) - 0.5F*b_height),
                                                     cvCeil(b_width), cvCeil(b_height) ),
                                             cvRect( 0, 0, pImg->width, pImg->height ) );
            if( roi.width <= 0 || roi.height <= 0 )
                continue;

            if( m_roi_seq ) cvSeqPush( m_roi_seq, &roi );

            roi_mat = cvGetSubRect( pImg, &roi_stub, roi );
            scaled_roi_mat = cvCreateMat( cvCeil(scale*roi.height), cvCeil(scale*roi.width), CV_8UC3 );
            cvResize( roi_mat, scaled_roi_mat );

            m_detected_blob_seq.Clear();
            m_split_detector->Detect( scaled_roi_mat, &m_detected_blob_seq );
            cvReleaseMat( &scaled_roi_mat );

            for( int k = 0; k < m_detected_blob_seq.GetBlobNum(); ++k )
            {
                CvDetectedBlob* b = (CvDetectedBlob*) m_detected_blob_seq.GetBlob(k);

                /* scale and shift each detected blob back to the original image coordinates */
                CV_BLOB_X(b) = CV_BLOB_X(b) / scale + roi.x;
                CV_BLOB_Y(b) = CV_BLOB_Y(b) / scale + roi.y;
                CV_BLOB_WX(b) /= scale;
                CV_BLOB_WY(b) /= scale;

                CvDetectedBlob d_b = cvDetectedBlob( CV_BLOB_X(b), CV_BLOB_Y(b), CV_BLOB_WX(b), CV_BLOB_WY(b), 1,
                        b->response );
                m_debug_blob_seq.AddBlob(&d_b);
            }

            if( m_detected_blob_seq.GetBlobNum() > 1 )
            {
                /*
                 * Split blob.
                 * The original blob is replaced by the first detected blob,
                 * remaining detected blobs are added to the end of the sequence:
                 */
                CvBlob* first_b = m_detected_blob_seq.GetBlob(0);
                CV_BLOB_X(b)  = CV_BLOB_X(first_b);  CV_BLOB_Y(b)  = CV_BLOB_Y(first_b);
                CV_BLOB_WX(b) = CV_BLOB_WX(first_b); CV_BLOB_WY(b) = CV_BLOB_WY(first_b);

                for( int j = 1; j < m_detected_blob_seq.GetBlobNum(); ++j )
                {
                    CvBlob* detected_b = m_detected_blob_seq.GetBlob(j);
                    pNewBlobList->AddBlob(detected_b);
                }
            }
        }   /* For each new blob. */

        for( i = 0; i < pNewBlobList->GetBlobNum(); ++i )
        {
            CvBlob* b = pNewBlobList->GetBlob(i);
            CvDetectedBlob d_b = cvDetectedBlob( CV_BLOB_X(b), CV_BLOB_Y(b), CV_BLOB_WX(b), CV_BLOB_WY(b), 2 );
            m_debug_blob_seq.AddBlob(&d_b);
        }
    }   // if( m_split_detector )
#endif

    return result;

}   /* cvDetectNewBlob */


