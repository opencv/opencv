/* Original code has been submitted by Liu Liu. Here is the copyright.
----------------------------------------------------------------------------------
 * An OpenCV Implementation of SURF
 * Further Information Refer to "SURF: Speed-Up Robust Feature"
 * Author: Liu Liu
 * liuliu.1987+opencv@gmail.com
 *
 * There are still serveral lacks for this experimental implementation:
 * 1.The interpolation of sub-pixel mentioned in article was not implemented yet;
 * 2.A comparision with original libSurf.so shows that the hessian detector is not a 100% match to their implementation;
 * 3.Due to above reasons, I recommanded the original one for study and reuse;
 *
 * However, the speed of this implementation is something comparable to original one.
 *
 * Copyright© 2008, Liu Liu All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *  Redistributions of source code must retain the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer.
 *  Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer in the documentation and/or other materials
 *  provided with the distribution.
 *  The name of Contributor may not be used to endorse or
 *  promote products derived from this software without
 *  specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 */

#include "precomp.hpp"

using namespace cv;



CV_IMPL CvSURFParams cvSURFParams(double threshold, int extended)
{
    CvSURFParams params;
    params.hessianThreshold = threshold;
    params.extended = extended;
    params.upright = 0;
    params.nOctaves = 4;
    params.nOctaveLayers = 2;
    return params;
}

CV_IMPL void
cvExtractSURF( const CvArr* _img, const CvArr* _mask,
               CvSeq** _keypoints, CvSeq** _descriptors,
               CvMemStorage* storage, CvSURFParams params,
               int useProvidedKeyPts)
{
    Mat img = cvarrToMat(_img), mask;
    if(_mask)
        mask = cvarrToMat(_mask);
    std::vector<KeyPoint> kpt;
    Mat descr;

    Ptr<Feature2D> surf = Algorithm::create<Feature2D>("Feature2D.SURF");
    if( surf.empty() )
        CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support");

    surf->set("hessianThreshold", params.hessianThreshold);
    surf->set("nOctaves", params.nOctaves);
    surf->set("nOctaveLayers", params.nOctaveLayers);
    surf->set("upright", params.upright != 0);
    surf->set("extended", params.extended != 0);

    surf->operator()(img, mask, kpt, _descriptors ? _OutputArray(descr) : noArray(),
                     useProvidedKeyPts != 0);

    if( _keypoints )
        *_keypoints = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvSURFPoint), storage);

    if( _descriptors )
        *_descriptors = cvCreateSeq(0, sizeof(CvSeq), surf->descriptorSize() * CV_ELEM_SIZE(surf->descriptorType()), storage);

    for( size_t i = 0; i < kpt.size(); i++ )
    {
        if( _keypoints )
        {
            CvSURFPoint pt = cvSURFPoint(kpt[i].pt, kpt[i].class_id, cvRound(kpt[i].size), kpt[i].angle, kpt[i].response);
            cvSeqPush(*_keypoints, &pt);
        }
        if( _descriptors )
            cvSeqPush(*_descriptors, descr.ptr((int)i));
    }
}

CV_IMPL CvSeq*
cvGetStarKeypoints( const CvArr* _img, CvMemStorage* storage,
                    CvStarDetectorParams params )
{
    Ptr<StarDetector> star = new StarDetector(params.maxSize, params.responseThreshold,
                                              params.lineThresholdProjected,
                                              params.lineThresholdBinarized,
                                              params.suppressNonmaxSize);
    std::vector<KeyPoint> kpts;
    star->detect(cvarrToMat(_img), kpts, Mat());

    CvSeq* seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvStarKeypoint), storage);
    for( size_t i = 0; i < kpts.size(); i++ )
    {
        CvStarKeypoint kpt = cvStarKeypoint(kpts[i].pt, cvRound(kpts[i].size), kpts[i].response);
        cvSeqPush(seq, &kpt);
    }
    return seq;
}
