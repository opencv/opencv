/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

///////////// Haar ////////////////////////
namespace cv
{
namespace ocl
{

struct getRect
{
    Rect operator()(const CvAvgComp &e) const
    {
        return e.rect;
    }
};

class CascadeClassifier_GPU : public OclCascadeClassifier
{
public:
    void detectMultiScale(oclMat &image,
                          CV_OUT std::vector<cv::Rect>& faces,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size())
    {
        (void)maxSize;
        MemStorage storage(cvCreateMemStorage(0));
        //CvMat img=image;
        CvSeq *objs = oclHaarDetectObjects(image, storage, scaleFactor, minNeighbors, flags, minSize);
        vector<CvAvgComp> vecAvgComp;
        Seq<CvAvgComp>(objs).copyTo(vecAvgComp);
        faces.resize(vecAvgComp.size());
        std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());
    }

};

}
}
TEST(Haar)
{
    Mat img = imread(abspath("basketball1.png"), CV_LOAD_IMAGE_GRAYSCALE);

    if (img.empty())
    {
        throw runtime_error("can't open basketball1.png");
    }

    CascadeClassifier faceCascadeCPU;

    if (!faceCascadeCPU.load(abspath("haarcascade_frontalface_alt.xml")))
    {
        throw runtime_error("can't load haarcascade_frontalface_alt.xml");
    }

    vector<Rect> faces;

    SUBTEST << img.cols << "x" << img.rows << "; scale image";
    CPU_ON;
    faceCascadeCPU.detectMultiScale(img, faces,
                                    1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    CPU_OFF;

    ocl::CascadeClassifier_GPU faceCascade;

    if (!faceCascade.load(abspath("haarcascade_frontalface_alt.xml")))
    {
        throw runtime_error("can't load haarcascade_frontalface_alt.xml");
    }

    ocl::oclMat d_img(img);

    faces.clear();

    WARMUP_ON;
    faceCascade.detectMultiScale(d_img, faces,
                                 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    WARMUP_OFF;

    faces.clear();

    GPU_ON;
    faceCascade.detectMultiScale(d_img, faces,
                                 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
     ;
    GPU_OFF;

    GPU_FULL_ON;
    d_img.upload(img);
    faceCascade.detectMultiScale(d_img, faces,
                                 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
    GPU_FULL_OFF;
}