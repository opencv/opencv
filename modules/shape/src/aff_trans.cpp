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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

namespace cv
{

class AffineTransformerImpl : public AffineTransformer
{
public:
    /* Constructors */
    AffineTransformerImpl()
    {
        fullAffine = true;
        name_ = "ShapeTransformer.AFF";
    }

    AffineTransformerImpl(bool _fullAffine)
    {
        fullAffine = _fullAffine;
        name_ = "ShapeTransformer.AFF";
    }

    /* Destructor */
    ~AffineTransformerImpl()
    {
    }

    //! the main operator
    virtual void estimateTransformation(InputArray transformingShape, InputArray targetShape, std::vector<DMatch> &matches);
    virtual float applyTransformation(InputArray input, OutputArray output=noArray());
    virtual void warpImage(InputArray transformingImage, OutputArray output,
                           int flags, int borderMode, const Scalar& borderValue) const;

    //! Setters/Getters
    virtual void setFullAffine(bool _fullAffine) {fullAffine=_fullAffine;}
    virtual bool getFullAffine() const {return fullAffine;}

    //! write/read
    virtual void write(FileStorage& fs) const
    {
        writeFormat(fs);
        fs << "name" << name_
           << "affine_type" << int(fullAffine);
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        fullAffine = int(fn["affine_type"])?true:false;
    }

private:
    bool fullAffine;
    Mat affineMat;
    float transformCost;

protected:
    String name_;
};

void AffineTransformerImpl::warpImage(InputArray transformingImage, OutputArray output,
                                      int flags, int borderMode, const Scalar& borderValue) const
{
    CV_INSTRUMENT_REGION()

    CV_Assert(!affineMat.empty());
    warpAffine(transformingImage, output, affineMat, transformingImage.getMat().size(), flags, borderMode, borderValue);
}


static Mat _localAffineEstimate(const std::vector<Point2f>& shape1, const std::vector<Point2f>& shape2,
                                bool fullAfine)
{
    Mat out(2,3,CV_32F);
    int siz=2*(int)shape1.size();

    if (fullAfine)
    {
        Mat matM(siz, 6, CV_32F);
        Mat matP(siz,1,CV_32F);
        int contPt=0;
        for (int ii=0; ii<siz; ii++)
        {
            Mat therow = Mat::zeros(1,6,CV_32F);
            if (ii%2==0)
            {
                therow.at<float>(0,0)=shape1[contPt].x;
                therow.at<float>(0,1)=shape1[contPt].y;
                therow.at<float>(0,2)=1;
                therow.row(0).copyTo(matM.row(ii));
                matP.at<float>(ii,0) = shape2[contPt].x;
            }
            else
            {
                therow.at<float>(0,3)=shape1[contPt].x;
                therow.at<float>(0,4)=shape1[contPt].y;
                therow.at<float>(0,5)=1;
                therow.row(0).copyTo(matM.row(ii));
                matP.at<float>(ii,0) = shape2[contPt].y;
                contPt++;
            }
        }
        Mat sol;
        solve(matM, matP, sol, DECOMP_SVD);
        out = sol.reshape(0,2);
    }
    else
    {
        Mat matM(siz, 4, CV_32F);
        Mat matP(siz,1,CV_32F);
        int contPt=0;
        for (int ii=0; ii<siz; ii++)
        {
            Mat therow = Mat::zeros(1,4,CV_32F);
            if (ii%2==0)
            {
                therow.at<float>(0,0)=shape1[contPt].x;
                therow.at<float>(0,1)=shape1[contPt].y;
                therow.at<float>(0,2)=1;
                therow.row(0).copyTo(matM.row(ii));
                matP.at<float>(ii,0) = shape2[contPt].x;
            }
            else
            {
                therow.at<float>(0,0)=-shape1[contPt].y;
                therow.at<float>(0,1)=shape1[contPt].x;
                therow.at<float>(0,3)=1;
                therow.row(0).copyTo(matM.row(ii));
                matP.at<float>(ii,0) = shape2[contPt].y;
                contPt++;
            }
        }
        Mat sol;
        solve(matM, matP, sol, DECOMP_SVD);
        out.at<float>(0,0)=sol.at<float>(0,0);
        out.at<float>(0,1)=sol.at<float>(1,0);
        out.at<float>(0,2)=sol.at<float>(2,0);
        out.at<float>(1,0)=-sol.at<float>(1,0);
        out.at<float>(1,1)=sol.at<float>(0,0);
        out.at<float>(1,2)=sol.at<float>(3,0);
    }
    return out;
}

void AffineTransformerImpl::estimateTransformation(InputArray _pts1, InputArray _pts2, std::vector<DMatch>& _matches)
{
    CV_INSTRUMENT_REGION()

    Mat pts1 = _pts1.getMat();
    Mat pts2 = _pts2.getMat();
    CV_Assert((pts1.channels()==2) && (pts1.cols>0) && (pts2.channels()==2) && (pts2.cols>0));
    CV_Assert(_matches.size()>1);

    if (pts1.type() != CV_32F)
        pts1.convertTo(pts1, CV_32F);
    if (pts2.type() != CV_32F)
        pts2.convertTo(pts2, CV_32F);

    // Use only valid matchings //
    std::vector<DMatch> matches;
    for (size_t i=0; i<_matches.size(); i++)
    {
        if (_matches[i].queryIdx<pts1.cols &&
            _matches[i].trainIdx<pts2.cols)
        {
            matches.push_back(_matches[i]);
        }
    }

    // Organizing the correspondent points in vector style //
    std::vector<Point2f> shape1; // transforming shape
    std::vector<Point2f> shape2; // target shape
    for (size_t i=0; i<matches.size(); i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,matches[i].queryIdx);
        shape1.push_back(pt1);

        Point2f pt2=pts2.at<Point2f>(0,matches[i].trainIdx);
        shape2.push_back(pt2);
    }

    // estimateRigidTransform //
    Mat affine;
    estimateRigidTransform(shape1, shape2, fullAffine).convertTo(affine, CV_32F);

    if (affine.empty())
        affine=_localAffineEstimate(shape1, shape2, fullAffine); //In case there is not good solution, just give a LLS based one

    affineMat = affine;
}

float AffineTransformerImpl::applyTransformation(InputArray inPts, OutputArray outPts)
{
    CV_INSTRUMENT_REGION()

    Mat pts1 = inPts.getMat();
    CV_Assert((pts1.channels()==2) && (pts1.cols>0));

    //Apply transformation in the complete set of points
    Mat fAffine;
    transform(pts1, fAffine, affineMat);

    // Ensambling output //
    if (outPts.needed())
    {
        outPts.create(1,fAffine.cols, CV_32FC2);
        Mat outMat = outPts.getMat();
        for (int i=0; i<fAffine.cols; i++)
            outMat.at<Point2f>(0,i)=fAffine.at<Point2f>(0,i);
    }

    // Updating Transform Cost //
    Mat Af(2, 2, CV_32F);
    Af.at<float>(0,0)=affineMat.at<float>(0,0);
    Af.at<float>(0,1)=affineMat.at<float>(1,0);
    Af.at<float>(1,0)=affineMat.at<float>(0,1);
    Af.at<float>(1,1)=affineMat.at<float>(1,1);
    SVD mysvd(Af, SVD::NO_UV);
    Mat singVals=mysvd.w;
    transformCost=std::log((singVals.at<float>(0,0)+FLT_MIN)/(singVals.at<float>(1,0)+FLT_MIN));

    return transformCost;
}

Ptr <AffineTransformer> createAffineTransformer(bool fullAffine)
{
    return Ptr<AffineTransformer>( new AffineTransformerImpl(fullAffine) );
}

} // cv
