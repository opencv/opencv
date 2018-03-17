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

class ThinPlateSplineShapeTransformerImpl : public ThinPlateSplineShapeTransformer
{
public:
    /* Constructors */
    ThinPlateSplineShapeTransformerImpl()
    {
        regularizationParameter=0;
        name_ = "ShapeTransformer.TPS";
        tpsComputed=false;
        transformCost = 0;
    }

    ThinPlateSplineShapeTransformerImpl(double _regularizationParameter)
    {
        regularizationParameter=_regularizationParameter;
        name_ = "ShapeTransformer.TPS";
        tpsComputed=false;
        transformCost = 0;
    }

    /* Destructor */
    ~ThinPlateSplineShapeTransformerImpl()
    {
    }

    //! the main operators
    virtual void estimateTransformation(InputArray transformingShape, InputArray targetShape, std::vector<DMatch> &matches);
    virtual float applyTransformation(InputArray inPts, OutputArray output=noArray());
    virtual void warpImage(InputArray transformingImage, OutputArray output,
                           int flags, int borderMode, const Scalar& borderValue) const;

    //! Setters/Getters
    virtual void setRegularizationParameter(double _regularizationParameter) {regularizationParameter=_regularizationParameter;}
    virtual double getRegularizationParameter() const {return regularizationParameter;}

    //! write/read
    virtual void write(FileStorage& fs) const
    {
        writeFormat(fs);
        fs << "name" << name_
           << "regularization" << regularizationParameter;
    }

    virtual void read(const FileNode& fn)
    {
        CV_Assert( (String)fn["name"] == name_ );
        regularizationParameter = (int)fn["regularization"];
    }

private:
    bool tpsComputed;
    double regularizationParameter;
    float transformCost;
    Mat tpsParameters;
    Mat shapeReference;

protected:
    String name_;
};

static float distance(Point2f p, Point2f q)
{
    Point2f diff = p - q;
    float norma = diff.x*diff.x + diff.y*diff.y;// - 2*diff.x*diff.y;
    if (norma<0) norma=0;
    //else norma = std::sqrt(norma);
    norma = norma*std::log(norma+FLT_EPSILON);
    return norma;
}

static Point2f _applyTransformation(const Mat &shapeRef, const Point2f point, const Mat &tpsParameters)
{
    Point2f out;
    for (int i=0; i<2; i++)
    {
        float a1=tpsParameters.at<float>(tpsParameters.rows-3,i);
        float ax=tpsParameters.at<float>(tpsParameters.rows-2,i);
        float ay=tpsParameters.at<float>(tpsParameters.rows-1,i);

        float affine=a1+ax*point.x+ay*point.y;
        float nonrigid=0;
        for (int j=0; j<shapeRef.rows; j++)
        {
            nonrigid+=tpsParameters.at<float>(j,i)*
                    distance(Point2f(shapeRef.at<float>(j,0),shapeRef.at<float>(j,1)),
                            point);
        }
        if (i==0)
        {
            out.x=affine+nonrigid;
        }
        if (i==1)
        {
            out.y=affine+nonrigid;
        }
    }
    return out;
}

/* public methods */
void ThinPlateSplineShapeTransformerImpl::warpImage(InputArray transformingImage, OutputArray output,
                                      int flags, int borderMode, const Scalar& borderValue) const
{
    CV_INSTRUMENT_REGION()

    CV_Assert(tpsComputed==true);

    Mat theinput = transformingImage.getMat();
    Mat mapX(theinput.rows, theinput.cols, CV_32FC1);
    Mat mapY(theinput.rows, theinput.cols, CV_32FC1);

    for (int row = 0; row < theinput.rows; row++)
    {
        for (int col = 0; col < theinput.cols; col++)
        {
            Point2f pt = _applyTransformation(shapeReference, Point2f(float(col), float(row)), tpsParameters);
            mapX.at<float>(row, col) = pt.x;
            mapY.at<float>(row, col) = pt.y;
        }
    }
    remap(transformingImage, output, mapX, mapY, flags, borderMode, borderValue);
}

float ThinPlateSplineShapeTransformerImpl::applyTransformation(InputArray inPts, OutputArray outPts)
{
    CV_INSTRUMENT_REGION()

    CV_Assert(tpsComputed);
    Mat pts1 = inPts.getMat();
    CV_Assert((pts1.channels()==2) && (pts1.cols>0));

    //Apply transformation in the complete set of points
    // Ensambling output //
    if (outPts.needed())
    {
        outPts.create(1,pts1.cols, CV_32FC2);
        Mat outMat = outPts.getMat();
        for (int i=0; i<pts1.cols; i++)
        {
            Point2f pt=pts1.at<Point2f>(0,i);
            outMat.at<Point2f>(0,i)=_applyTransformation(shapeReference, pt, tpsParameters);
        }
    }

    return transformCost;
}

void ThinPlateSplineShapeTransformerImpl::estimateTransformation(InputArray _pts1, InputArray _pts2,
                                                               std::vector<DMatch>& _matches )
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

    // Organizing the correspondent points in matrix style //
    Mat shape1((int)matches.size(),2,CV_32F); // transforming shape
    Mat shape2((int)matches.size(),2,CV_32F); // target shape
    for (int i=0, end = (int)matches.size(); i<end; i++)
    {
        Point2f pt1=pts1.at<Point2f>(0,matches[i].queryIdx);
        shape1.at<float>(i,0) = pt1.x;
        shape1.at<float>(i,1) = pt1.y;

        Point2f pt2=pts2.at<Point2f>(0,matches[i].trainIdx);
        shape2.at<float>(i,0) = pt2.x;
        shape2.at<float>(i,1) = pt2.y;
    }
    shape1.copyTo(shapeReference);

    // Building the matrices for solving the L*(w|a)=(v|0) problem with L={[K|P];[P'|0]}

    //Building K and P (Needed to build L)
    Mat matK((int)matches.size(),(int)matches.size(),CV_32F);
    Mat matP((int)matches.size(),3,CV_32F);
    for (int i=0, end=(int)matches.size(); i<end; i++)
    {
        for (int j=0; j<end; j++)
        {
            if (i==j)
            {
                matK.at<float>(i,j)=float(regularizationParameter);
            }
            else
            {
                matK.at<float>(i,j) = distance(Point2f(shape1.at<float>(i,0),shape1.at<float>(i,1)),
                                               Point2f(shape1.at<float>(j,0),shape1.at<float>(j,1)));
            }
        }
        matP.at<float>(i,0) = 1;
        matP.at<float>(i,1) = shape1.at<float>(i,0);
        matP.at<float>(i,2) = shape1.at<float>(i,1);
    }

    //Building L
    Mat matL=Mat::zeros((int)matches.size()+3,(int)matches.size()+3,CV_32F);
    Mat matLroi(matL, Rect(0,0,(int)matches.size(),(int)matches.size())); //roi for K
    matK.copyTo(matLroi);
    matLroi = Mat(matL,Rect((int)matches.size(),0,3,(int)matches.size())); //roi for P
    matP.copyTo(matLroi);
    Mat matPt;
    transpose(matP,matPt);
    matLroi = Mat(matL,Rect(0,(int)matches.size(),(int)matches.size(),3)); //roi for P'
    matPt.copyTo(matLroi);

    //Building B (v|0)
    Mat matB = Mat::zeros((int)matches.size()+3,2,CV_32F);
    for (int i=0, end = (int)matches.size(); i<end; i++)
    {
        matB.at<float>(i,0) = shape2.at<float>(i,0); //x's
        matB.at<float>(i,1) = shape2.at<float>(i,1); //y's
    }

    //Obtaining transformation params (w|a)
    solve(matL, matB, tpsParameters, DECOMP_LU);
    //tpsParameters = matL.inv()*matB;

    //Setting transform Cost and Shape reference
    Mat w(tpsParameters, Rect(0,0,2,tpsParameters.rows-3));
    Mat Q=w.t()*matK*w;
    transformCost=fabs(Q.at<float>(0,0)*Q.at<float>(1,1));//fabs(mean(Q.diag(0))[0]);//std::max(Q.at<float>(0,0),Q.at<float>(1,1));
    tpsComputed=true;
}

Ptr <ThinPlateSplineShapeTransformer> createThinPlateSplineShapeTransformer(double regularizationParameter)
{
    return Ptr<ThinPlateSplineShapeTransformer>( new ThinPlateSplineShapeTransformerImpl(regularizationParameter) );
}

} // cv
