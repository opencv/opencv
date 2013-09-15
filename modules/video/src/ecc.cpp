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


/****************************************************************************************\
*                                       Image Alignment (ECC algorithm)                  *
\****************************************************************************************/

using namespace cv;

static void image_jacobian_homo_ECC(const Mat& src1, const Mat& src2,
                                    const Mat& src3, const Mat& src4,
                                    const Mat& src5, Mat& dst)
{


    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.size() == src3.size());
    CV_Assert(src1.size() == src4.size());

    CV_Assert( src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols*8));
    CV_Assert(dst.type() == CV_32FC1);

    CV_Assert(src5.isContinuous());


    const float* hptr = src5.ptr<float>(0);

    const float h0_ = hptr[0];
    const float h1_ = hptr[3];
    const float h2_ = hptr[6];
    const float h3_ = hptr[1];
    const float h4_ = hptr[4];
    const float h5_ = hptr[7];
    const float h6_ = hptr[2];
    const float h7_ = hptr[5];

    const int w = src1.cols;


    //create denominator for all points as a block
    Mat den_ = src3*h2_ + src4*h5_ + 1.0;//check the time of this! otherwise use addWeighted

    //create projected points
    Mat hatX_ = -src3*h0_ - src4*h3_ - h6_;
    divide(hatX_, den_, hatX_);
    Mat hatY_ = -src3*h1_ - src4*h4_ - h7_;
    divide(hatY_, den_, hatY_);


    //instead of dividing each block with den,
    //just pre-devide the block of gradients (it's more efficient)

    Mat src1Divided_;
    Mat src2Divided_;

    divide(src1, den_, src1Divided_);
    divide(src2, den_, src2Divided_);


    //compute Jacobian blocks (8 blocks)

    dst.colRange(0, w) = src1Divided_.mul(src3);//1

    dst.colRange(w,2*w) = src2Divided_.mul(src3);//2

    Mat temp_ = (hatX_.mul(src1Divided_)+hatY_.mul(src2Divided_));
    dst.colRange(2*w,3*w) = temp_.mul(src3);//3

    hatX_.release();
    hatY_.release();

    dst.colRange(3*w, 4*w) = src1Divided_.mul(src4);//4

    dst.colRange(4*w, 5*w) = src2Divided_.mul(src4);//5

    dst.colRange(5*w, 6*w) = temp_.mul(src4);//6

    src1Divided_.copyTo(dst.colRange(6*w, 7*w));//7

    src2Divided_.copyTo(dst.colRange(7*w, 8*w));//8
}

static void image_jacobian_euclidean_ECC(const Mat& src1, const Mat& src2,
                                         const Mat& src3, const Mat& src4,
                                         const Mat& src5, Mat& dst)
{

    CV_Assert( src1.size()==src2.size());
    CV_Assert( src1.size()==src3.size());
    CV_Assert( src1.size()==src4.size());

    CV_Assert( src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols*3));
    CV_Assert(dst.type() == CV_32FC1);

    CV_Assert(src5.isContinuous());

    const float* hptr = src5.ptr<float>(0);

    const float h0 = hptr[0];//cos(theta)
    const float h1 = hptr[3];//sin(theta)

    const int w = src1.cols;

    //create -sin(theta)*X -cos(theta)*Y for all points as a block -> hatX
    Mat hatX = -(src3*h1) - (src4*h0);

    //create cos(theta)*X -sin(theta)*Y for all points as a block -> hatY
    Mat hatY = (src3*h0) - (src4*h1);


    //compute Jacobian blocks (3 blocks)
    dst.colRange(0, w) = (src1.mul(hatX))+(src2.mul(hatY));//1

    src1.copyTo(dst.colRange(w, 2*w));//2
    src2.copyTo(dst.colRange(2*w, 3*w));//3
}


static void image_jacobian_affine_ECC(const Mat& src1, const Mat& src2,
                                      const Mat& src3, const Mat& src4,
                                      Mat& dst)
{

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.size() == src3.size());
    CV_Assert(src1.size() == src4.size());

    CV_Assert(src1.rows == dst.rows);
    CV_Assert(dst.cols == (6*src1.cols));

    CV_Assert(dst.type() == CV_32FC1);


    const int w = src1.cols;

    //compute Jacobian blocks (6 blocks)

    dst.colRange(0,w) = src1.mul(src3);//1
    dst.colRange(w,2*w) = src2.mul(src3);//2
    dst.colRange(2*w,3*w) = src1.mul(src4);//3
    dst.colRange(3*w,4*w) = src2.mul(src4);//4
    src1.copyTo(dst.colRange(4*w,5*w));//5
    src2.copyTo(dst.colRange(5*w,6*w));//6
}


static void image_jacobian_translation_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{

    CV_Assert( src1.size()==src2.size());

    CV_Assert( src1.rows == dst.rows);
    CV_Assert(dst.cols == (src1.cols*2));
    CV_Assert(dst.type() == CV_32FC1);

    const int w = src1.cols;

    //compute Jacobian blocks (2 blocks)
    src1.copyTo(dst.colRange(0, w));
    src2.copyTo(dst.colRange(w, 2*w));
}


static void project_onto_jacobian_ECC(const Mat& src1, const Mat& src2, Mat& dst)
{
    /* this functions is used for two types of projections. If src1.cols ==src.cols
    it does a blockwise multiplication (like in the outer product of vectors)
    of the blocks in matrices src1 and src2 and dst
    has size (number_of_blcks x number_of_blocks), otherwise dst is a vector of size
    (number_of_blocks x 1) since src2 is "multiplied"(dot) with each block of src1.

    The number_of_blocks is equal to the number of parameters we are lloking for
    (i.e. rtanslation:2, euclidean: 3, affine: 6, homography: 8)

    */
    CV_Assert(src1.rows == src2.rows);
    CV_Assert((src1.cols % src2.cols) == 0);
    int w;

    float* dstPtr = dst.ptr<float>(0);

    if (src1.cols !=src2.cols){//dst.cols==1
        w  = src2.cols;
        for (int i=0; i<dst.rows; i++){
            dstPtr[i] = (float) src2.dot(src1.colRange(i*w,(i+1)*w));
        }
    }

    else {
        CV_Assert(dst.cols == dst.rows); //dst is square (and symmetric)
        w = src2.cols/dst.cols;
        Mat mat;
        for (int i=0; i<dst.rows; i++){

            mat = Mat(src1.colRange(i*w, (i+1)*w));
            dstPtr[i*(dst.rows+1)] = (float) pow(norm(mat),2); //diagonal elements

            for (int j=i+1; j<dst.cols; j++){ //j starts from i+1
                dstPtr[i*dst.cols+j] = (float) mat.dot(src2.colRange(j*w, (j+1)*w));
                dstPtr[j*dst.cols+i] = dstPtr[i*dst.cols+j]; //due to symmetry
            }
        }
    }
}


static void update_warping_matrix_ECC (Mat& map_matrix, const Mat& update, const int motionType)
{
    CV_Assert (map_matrix.type() == CV_32FC1);
    CV_Assert (update.type() == CV_32FC1);

    CV_Assert (motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN ||
        motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY);

    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert (map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert (map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert (map_matrix.rows == 2 && update.rows == 2);

    CV_Assert (update.cols == 1);

    CV_Assert( map_matrix.isContinuous());
    CV_Assert( update.isContinuous() );


    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);


    if (motionType == MOTION_TRANSLATION){
        mapPtr[2] += updatePtr[0];
        mapPtr[5] += updatePtr[1];
    }
    if (motionType == MOTION_AFFINE) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[1] += updatePtr[2];
        mapPtr[4] += updatePtr[3];
        mapPtr[2] += updatePtr[4];
        mapPtr[5] += updatePtr[5];
    }
    if (motionType == MOTION_HOMOGRAPHY) {
        mapPtr[0] += updatePtr[0];
        mapPtr[3] += updatePtr[1];
        mapPtr[6] += updatePtr[2];
        mapPtr[1] += updatePtr[3];
        mapPtr[4] += updatePtr[4];
        mapPtr[7] += updatePtr[5];
        mapPtr[2] += updatePtr[6];
        mapPtr[5] += updatePtr[7];
    }
    if (motionType == MOTION_EUCLIDEAN) {
        double new_theta = acos(mapPtr[0]) + updatePtr[0];

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = (float) cos(new_theta);
        mapPtr[3] = (float) sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}


double cv::findTransformECC(InputArray templateImage,
                            InputArray inputImage,
                            InputOutputArray warpMatrix,
                            int motionType,
                            TermCriteria criteria)
{


    Mat src = templateImage.getMat();//template iamge
    Mat dst = inputImage.getMat(); //input image (to be warped)
    Mat map = warpMatrix.getMat(); //warp (transformation)

    CV_Assert(!src.empty());
    CV_Assert(!dst.empty());


    if( ! (src.type()==dst.type()))
        CV_Error( Error::StsUnmatchedFormats, "Both input images must have the same data type" );

    //accept only 1-channel images
    if( src.type() != CV_8UC1 && src.type()!= CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "Images must have 8uC1 or 32fC1 type");

    if( map.type() != CV_32FC1)
        CV_Error( Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

    CV_Assert (map.cols == 3);
    CV_Assert (map.rows == 2 || map.rows ==3);

    CV_Assert (motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY ||
        motionType == MOTION_EUCLIDEAN || motionType == MOTION_TRANSLATION);

    if (motionType == MOTION_HOMOGRAPHY){
        CV_Assert (map.rows ==3);
    }

    CV_Assert (criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int    numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps    = (criteria.type & TermCriteria::EPS)   ? criteria.epsilon  :  -1;

    int paramTemp = 6;//default: affine
    switch (motionType){
      case MOTION_TRANSLATION:
          paramTemp = 2;
          break;
      case MOTION_EUCLIDEAN:
          paramTemp = 3;
          break;
      case MOTION_HOMOGRAPHY:
          paramTemp = 8;
          break;
    }


    const int numberOfParameters = paramTemp;

    const int ws = src.cols;
    const int hs = src.rows;
    const int wd = dst.cols;
    const int hd = dst.rows;

    Mat Xcoord = Mat(1, ws, CV_32F);
    Mat Ycoord = Mat(hs, 1, CV_32F);
    Mat Xgrid = Mat(hs, ws, CV_32F);
    Mat Ygrid = Mat(hs, ws, CV_32F);

    float* XcoPtr = Xcoord.ptr<float>(0);
    float* YcoPtr = Ycoord.ptr<float>(0);
    int j;
    for (j=0; j<ws; j++)
        XcoPtr[j] = (float) j;
    for (j=0; j<hs; j++)
        YcoPtr[j] = (float) j;

    repeat(Xcoord, hs, 1, Xgrid);
    repeat(Ycoord, 1, ws, Ygrid);

    Xcoord.release();
    Ycoord.release();

    Mat templateZM    = Mat(hs, ws, CV_32F);// to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, CV_32F);// to store the (smoothed) template
    Mat imageFloat    = Mat(hd, wd, CV_32F);// to store the (smoothed) input image
    Mat imageWarped   = Mat(hs, ws, CV_32F);// to store the warped zero-mean input image
    Mat allOnes		= Mat::ones(hd, wd, CV_8U); //to use it for mask warping
    Mat imageMask		= Mat(hs, ws, CV_8U); //to store the final mask

    //gaussian filtering is optional
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(5, 5), 0, 0);//is in-place filtering slower?

    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(5, 5), 0, 0);

    // needed matrices for gradients and warped gradients
    Mat gradientX = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientY = Mat::zeros(hd, wd, CV_32FC1);
    Mat gradientXWarped = Mat(hs, ws, CV_32FC1);
    Mat gradientYWarped = Mat(hs, ws, CV_32FC1);


    // calculate first order image derivatives
    Matx13f dx(-0.5f, 0.0f, 0.5f);

    filter2D(imageFloat, gradientX, -1, dx);
    filter2D(imageFloat, gradientY, -1, dx.t());


    // matrices needed for solving linear equation system for maximizing ECC
    Mat jacobian                = Mat(hs, ws*numberOfParameters, CV_32F);
    Mat hessian                 = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv              = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection         = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection      = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian  = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection         = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);//transformation parameter correction
    Mat error = Mat(hs, ws, CV_32F);//error as 2D matrix

    const int imageFlags = INTER_LINEAR  + WARP_INVERSE_MAP;
    const int maskFlags  = INTER_NEAREST + WARP_INVERSE_MAP;


    // iteratively update map_matrix
    double rho      = -1;
    double last_rho = - termination_eps;
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho-last_rho)>= termination_eps); i++)
    {

        // warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        if (motionType != MOTION_HOMOGRAPHY)
        {
            warpAffine(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);
            warpAffine(gradientX,  gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpAffine(gradientY,  gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpAffine(allOnes,    imageMask,       map, imageMask.size(),       maskFlags);
        }
        else
        {
            warpPerspective(imageFloat, imageWarped,     map, imageWarped.size(),     imageFlags);
            warpPerspective(gradientX,  gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpPerspective(gradientY,  gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpPerspective(allOnes,    imageMask,       map, imageMask.size(),       maskFlags);
        }


        Scalar imgMean, imgStd, tmpMean, tmpStd;
        meanStdDev(imageWarped,   imgMean, imgStd, imageMask);
        meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

        subtract(imageWarped,   imgMean, imageWarped, imageMask);//zero-mean input
        subtract(templateFloat, tmpMean, templateZM,  imageMask);//zero-mean template

        const double tmpNorm = std::sqrt(countNonZero(imageMask)*(tmpStd.val[0])*(tmpStd.val[0]));
        const double imgNorm = std::sqrt(countNonZero(imageMask)*(imgStd.val[0])*(imgStd.val[0]));

        // calculate jacobian of image wrt parameters
        switch (motionType){
            case MOTION_AFFINE:
                image_jacobian_affine_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, jacobian);
                break;
            case MOTION_HOMOGRAPHY:
                image_jacobian_homo_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
            case MOTION_TRANSLATION:
                image_jacobian_translation_ECC(gradientXWarped, gradientYWarped, jacobian);
                break;
            case MOTION_EUCLIDEAN:
                image_jacobian_euclidean_ECC(gradientXWarped, gradientYWarped, Xgrid, Ygrid, map, jacobian);
                break;
        }

        // calculate Hessian and its inverse
        project_onto_jacobian_ECC(jacobian, jacobian, hessian);

        hessianInv = hessian.inv();

        const double correlation = templateZM.dot(imageWarped);

        // calculate enhanced correlation coefficiont (ECC)->rho
        last_rho = rho;
        rho = correlation/(imgNorm*tmpNorm);

        // project images into jacobian
        project_onto_jacobian_ECC( jacobian, imageWarped, imageProjection);
        project_onto_jacobian_ECC(jacobian, templateZM, templateProjection);


        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv*imageProjection;
        const double lambda_n = (imgNorm*imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0)
        {
            rho = -1;
            CV_Error(Error::StsNoConv, "The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped");

        }
        const double lambda = (lambda_n/lambda_d);

        // estimate the update step delta_p
        error = lambda*templateZM - imageWarped;
        project_onto_jacobian_ECC(jacobian, error, errorProjection);
        deltaP = hessianInv * errorProjection;

        // update warping matrix
        update_warping_matrix_ECC( map, deltaP, motionType);


    }

    // return final correlation coefficient
    return rho;
}


/* End of file. */
