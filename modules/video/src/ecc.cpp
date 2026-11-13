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

static double accumulateECCSystem(const Mat& gradX, const Mat& gradY,
                                  const Mat& imgWarped, const Mat& tmplZM,
                                  const Mat& map, int motionType, int numParams,
                                  Mat& hessian, Mat& imageProjection, Mat& templateProjection)
{
    const int hs = gradX.rows;
    const int ws = gradX.cols;
    const int ch = gradX.channels();
    CV_Assert(gradY.size() == gradX.size() && imgWarped.size() == gradX.size() && tmplZM.size() == gradX.size());
    CV_Assert(gradX.type() == CV_MAKETYPE(CV_32F, ch));
    CV_Assert(map.isContinuous());

    const float* hptr = map.ptr<float>(0);
    const float h0_ = hptr[0];
    const float h1_ = hptr[3];
    const float h2_ = (motionType == MOTION_HOMOGRAPHY) ? hptr[6] : 0.f;
    const float h3_ = hptr[1];
    const float h4_ = hptr[4];
    const float h5_ = (motionType == MOTION_HOMOGRAPHY) ? hptr[7] : 0.f;
    const float h6_ = hptr[2];
    const float h7_ = hptr[5];

    double totalH[8][8] = {};
    double totalImgP[8] = {};
    double totalTmplP[8] = {};
    double totalCorr = 0.0;
    Mutex accMutex;

    parallel_for_(Range(0, hs), [&](const Range& range) {
        double H[8][8] = {};
        double imgP[8] = {};
        double tmplP[8] = {};
        double corr = 0.0;

        for (int y = range.start; y < range.end; ++y)
        {
            const float fy = (float)y;
            const float* gxRow = gradX.ptr<float>(y);
            const float* gyRow = gradY.ptr<float>(y);
            const float* iwRow = imgWarped.ptr<float>(y);
            const float* tzRow = tmplZM.ptr<float>(y);

            for (int x = 0; x < ws; ++x)
            {
                const float fx = (float)x;
                float hatX = 0.f, hatY = 0.f, den = 1.f;
                if (motionType == MOTION_EUCLIDEAN)
                {
                    hatX = -(fx * h1_) - (fy * h0_);
                    hatY = (fx * h0_) - (fy * h1_);
                }
                else if (motionType == MOTION_HOMOGRAPHY)
                {
                    den = fx * h2_ + fy * h5_ + 1.0f;
                    hatX = -(fx * h0_ + fy * h3_ + h6_) / den;
                    hatY = -(fx * h1_ + fy * h4_ + h7_) / den;
                }

                const int base = x * ch;
                for (int c = 0; c < ch; ++c)
                {
                    const int idx = base + c;
                    const float gx = gxRow[idx];
                    const float gy = gyRow[idx];
                    const float iw = iwRow[idx];
                    const float tz = tzRow[idx];
                    corr += (double)tz * (double)iw;

                    float j[8];
                    switch (motionType)
                    {
                        case MOTION_TRANSLATION:
                            j[0] = gx;
                            j[1] = gy;
                            break;
                        case MOTION_EUCLIDEAN:
                            j[0] = gx * hatX + gy * hatY;
                            j[1] = gx;
                            j[2] = gy;
                            break;
                        case MOTION_AFFINE:
                            j[0] = gx * fx;
                            j[1] = gy * fx;
                            j[2] = gx * fy;
                            j[3] = gy * fy;
                            j[4] = gx;
                            j[5] = gy;
                            break;
                        case MOTION_HOMOGRAPHY:
                        default:
                        {
                            const float gxd = gx / den;
                            const float gyd = gy / den;
                            const float t = hatX * gxd + hatY * gyd;
                            j[0] = gxd * fx;
                            j[1] = gyd * fx;
                            j[2] = t * fx;
                            j[3] = gxd * fy;
                            j[4] = gyd * fy;
                            j[5] = t * fy;
                            j[6] = gxd;
                            j[7] = gyd;
                            break;
                        }
                    }

                    for (int p = 0; p < numParams; ++p)
                    {
                        const double jp = (double)j[p];
                        imgP[p] += jp * (double)iw;
                        tmplP[p] += jp * (double)tz;
                        for (int q = p; q < numParams; ++q)
                            H[p][q] += jp * (double)j[q];
                    }
                }
            }
        }

        AutoLock lock(accMutex);
        totalCorr += corr;
        for (int p = 0; p < numParams; ++p)
        {
            totalImgP[p] += imgP[p];
            totalTmplP[p] += tmplP[p];
            for (int q = p; q < numParams; ++q)
                totalH[p][q] += H[p][q];
        }
    });

    float* hOut = hessian.ptr<float>(0);
    float* imgOut = imageProjection.ptr<float>(0);
    float* tmplOut = templateProjection.ptr<float>(0);
    for (int p = 0; p < numParams; ++p)
    {
        imgOut[p] = (float)totalImgP[p];
        tmplOut[p] = (float)totalTmplP[p];
        hOut[p * numParams + p] = (float)totalH[p][p];
        for (int q = p + 1; q < numParams; ++q)
        {
            const float val = (float)totalH[p][q];
            hOut[p * numParams + q] = val;
            hOut[q * numParams + p] = val;
        }
    }
    return totalCorr;
}

static void update_warping_matrix_ECC(Mat& map_matrix, const Mat& update, const int motionType) {
    CV_Assert(map_matrix.type() == CV_32FC1);
    CV_Assert(update.type() == CV_32FC1);

    CV_Assert(motionType == MOTION_TRANSLATION || motionType == MOTION_EUCLIDEAN || motionType == MOTION_AFFINE ||
              motionType == MOTION_HOMOGRAPHY);

    if (motionType == MOTION_HOMOGRAPHY)
        CV_Assert(map_matrix.rows == 3 && update.rows == 8);
    else if (motionType == MOTION_AFFINE)
        CV_Assert(map_matrix.rows == 2 && update.rows == 6);
    else if (motionType == MOTION_EUCLIDEAN)
        CV_Assert(map_matrix.rows == 2 && update.rows == 3);
    else
        CV_Assert(map_matrix.rows == 2 && update.rows == 2);

    CV_Assert(update.cols == 1);

    CV_Assert(map_matrix.isContinuous());
    CV_Assert(update.isContinuous());

    float* mapPtr = map_matrix.ptr<float>(0);
    const float* updatePtr = update.ptr<float>(0);

    if (motionType == MOTION_TRANSLATION) {
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
        double new_theta = updatePtr[0];
        new_theta += asin(mapPtr[3]);

        mapPtr[2] += updatePtr[1];
        mapPtr[5] += updatePtr[2];
        mapPtr[0] = mapPtr[4] = (float)cos(new_theta);
        mapPtr[3] = (float)sin(new_theta);
        mapPtr[1] = -mapPtr[3];
    }
}

/** Function that computes enhanced corelation coefficient from Georgios et.al. 2008
 *   See https://github.com/opencv/opencv/issues/12432
 */
double cv::computeECC(InputArray templateImage, InputArray inputImage, InputArray inputMask) {
    CV_Assert(!templateImage.empty());
    CV_Assert(!inputImage.empty());

    CV_Assert(templateImage.channels() == 1 || templateImage.channels() == 3);

    if (!(templateImage.type() == inputImage.type()))
        CV_Error(Error::StsUnmatchedFormats, "Both input images must have the same data type");

    Scalar meanTemplate, sdTemplate;

    int active_pixels = inputMask.empty() ? templateImage.size().area() : countNonZero(inputMask);
    int type = templateImage.type();
    meanStdDev(templateImage, meanTemplate, sdTemplate, inputMask);
    Mat templateMat = templateImage.getMat();
    Mat inputMat = inputImage.getMat();

    /*
     * For unsigned ints, when the mean is computed and subtracted, any values less than the mean
     * will be set to 0 (since there are no negatives values). This impacts the norm and dot product, which
     * ultimately results in an incorrect ECC. To circumvent this problem, if unsigned ints are provided,
     * we convert them to a signed ints with larger resolution for the subtraction step.
     */
    const int depth = templateImage.depth();
    const int zmType = (depth == CV_8U) ? CV_MAKETYPE(CV_16S, templateMat.channels())
                     : (depth == CV_16U) ? CV_MAKETYPE(CV_32S, templateMat.channels())
                     : type;
    if (depth == CV_8U || depth == CV_16U) {
        Mat templateMatConverted, inputMatConverted;
        templateMat.convertTo(templateMatConverted, zmType);
        cv::swap(templateMat, templateMatConverted);
        inputMat.convertTo(inputMatConverted, zmType);
        cv::swap(inputMat, inputMatConverted);
    }
    Mat templateImage_zeromean;
    if (!inputMask.empty())
        templateImage_zeromean = Mat::zeros(templateMat.size(), zmType);
    subtract(templateMat, meanTemplate, templateImage_zeromean, inputMask);
    double templateImagenorm = std::sqrt(active_pixels * cv::norm(sdTemplate, NORM_L2SQR));

    Scalar meanInput, sdInput;

    Mat inputImage_zeromean;
    if (!inputMask.empty())
        inputImage_zeromean = Mat::zeros(inputMat.size(), zmType);
    meanStdDev(inputImage, meanInput, sdInput, inputMask);
    subtract(inputMat, meanInput, inputImage_zeromean, inputMask);
    double inputImagenorm = std::sqrt(active_pixels * norm(sdInput, NORM_L2SQR));

    return templateImage_zeromean.dot(inputImage_zeromean) / (templateImagenorm * inputImagenorm);
}


double cv::findTransformECCWithMask( InputArray templateImage,
                                 InputArray inputImage,
                                 InputArray templateMask,
                                 InputArray inputMask,
                                 InputOutputArray warpMatrix,
                                 int motionType,
                                 TermCriteria criteria,
                                 int gaussFiltSize) {
    Mat src = templateImage.getMat();  // template image
    Mat dst = inputImage.getMat();     // input image (to be warped)
    Mat map = warpMatrix.getMat();     // warp (transformation)

    CV_Assert(!src.empty());
    CV_Assert(!dst.empty());

    CV_Assert(src.channels() == 1 || src.channels() == 3);
    CV_Assert(src.channels() == dst.channels());
    CV_Assert(src.depth() == dst.depth());
    CV_Assert(src.depth() == CV_8U || src.depth() == CV_16U || src.depth() == CV_32F || src.depth() == CV_64F);

    // If the user passed an un-initialized warpMatrix, initialize to identity
    if (map.empty()) {
        int rowCount = 2;
        if (motionType == MOTION_HOMOGRAPHY)
            rowCount = 3;

        warpMatrix.create(rowCount, 3, CV_32FC1);
        map = warpMatrix.getMat();
        map = Mat::eye(rowCount, 3, CV_32F);
    }

    if (!(src.type() == dst.type()))
        CV_Error(Error::StsUnmatchedFormats, "Both input images must have the same data type");

    if (map.type() != CV_32FC1)
        CV_Error(Error::StsUnsupportedFormat, "warpMatrix must be single-channel floating-point matrix");

    CV_Assert(map.cols == 3);
    CV_Assert(map.rows == 2 || map.rows == 3);

    CV_Assert(motionType == MOTION_AFFINE || motionType == MOTION_HOMOGRAPHY || motionType == MOTION_EUCLIDEAN ||
              motionType == MOTION_TRANSLATION);

    if (motionType == MOTION_HOMOGRAPHY) {
        CV_Assert(map.rows == 3);
    }

    CV_Assert(criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

    int paramTemp = 6;  // default: affine
    switch (motionType) {
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

    const int channels = src.channels();
    int type = CV_MAKETYPE(CV_32F, channels);

    Mat templateZM = Mat(hs, ws, type);     // to store the (smoothed)zero-mean version of template
    Mat templateFloat = Mat(hs, ws, type);  // to store the (smoothed) template
    Mat imageFloat = Mat(hd, wd, type);     // to store the (smoothed) input image
    Mat imageWarped = Mat(hs, ws, type);    // to store the warped zero-mean input image
    Mat imageMask = Mat(hs, ws, CV_8U);     // to store the final mask

    // Gaussian filtering is optional
    src.convertTo(templateFloat, templateFloat.type());
    GaussianBlur(templateFloat, templateFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    dst.convertTo(imageFloat, imageFloat.type());
    GaussianBlur(imageFloat, imageFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);

    // needed matrices for gradients and warped gradients
    Mat gradientX = Mat::zeros(hd, wd, type);
    Mat gradientY = Mat::zeros(hd, wd, type);
    Mat gradientXWarped = Mat(hs, ws, type);
    Mat gradientYWarped = Mat(hs, ws, type);

    // calculate first order image derivatives
    Matx13f dx(-0.5f, 0.0f, 0.5f);

    filter2D(imageFloat, gradientX, -1, dx);
    filter2D(imageFloat, gradientY, -1, dx.t());

    // To use in mask warping
    Mat templtMask;
    if(templateMask.empty())
    {
        templtMask = Mat::ones(hs, ws, CV_8U);
    }
    else
    {
        threshold(templateMask, templtMask, 0, 1, THRESH_BINARY);
        templtMask.convertTo(templtMask, CV_32F);
        GaussianBlur(templtMask, templtMask, Size(gaussFiltSize, gaussFiltSize), 0, 0);
        templtMask *= (0.5/0.95);
        templtMask.convertTo(templtMask, CV_8U);
    }

    //to use it for mask warping
    Mat preMask;
    if(inputMask.empty())
    {
        preMask = Mat::ones(hd, wd, CV_8U);
    }
    else
    {
        Mat preMaskFloat;
        threshold(inputMask, preMask, 0, 1, THRESH_BINARY);

        preMask.convertTo(preMaskFloat, CV_32F);
        GaussianBlur(preMaskFloat, preMaskFloat, Size(gaussFiltSize, gaussFiltSize), 0, 0);
        // Change threshold.
        preMaskFloat *= (0.5/0.95);
        // Rounding conversion.
        preMaskFloat.convertTo(preMask, CV_8U);

        // If there's no template mask, we can apply image masks to gradients only once.
        // Otherwise, we'll need to combine the template and image masks at each iteration.
        if (templateMask.empty())
        {
            cv::Mat zeroMask = (preMask == 0);
            gradientX.setTo(0, zeroMask);
            gradientY.setTo(0, zeroMask);
        }
    }

    // matrices needed for solving linear equation system for maximizing ECC
    Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);  // transformation parameter correction
    Mat zeroMask;

    const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
    const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;

    // iteratively update map_matrix
    double rho = -1;
    double last_rho = -termination_eps;
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++) {
        // warp-back portion of the inputImage and gradients to the coordinate space of the templateImage
        if (motionType != MOTION_HOMOGRAPHY) {
            warpAffine(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
            warpAffine(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpAffine(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpAffine(preMask, imageMask, map, imageMask.size(), maskFlags);
        } else {
            warpPerspective(imageFloat, imageWarped, map, imageWarped.size(), imageFlags);
            warpPerspective(gradientX, gradientXWarped, map, gradientXWarped.size(), imageFlags);
            warpPerspective(gradientY, gradientYWarped, map, gradientYWarped.size(), imageFlags);
            warpPerspective(preMask, imageMask, map, imageMask.size(), maskFlags);
        }

        if (!templateMask.empty())
        {
            cv::bitwise_and(imageMask, templtMask, imageMask);

            cv::compare(imageMask, 0, zeroMask, CMP_EQ);
            gradientXWarped.setTo(0, zeroMask);
            gradientYWarped.setTo(0, zeroMask);
        }

        Scalar imgMean, imgStd, tmpMean, tmpStd;
        meanStdDev(imageWarped, imgMean, imgStd, imageMask);
        meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

        subtract(imageWarped, imgMean, imageWarped, imageMask);  // zero-mean input
        templateZM.setTo(Scalar::all(0));
        subtract(templateFloat, tmpMean, templateZM, imageMask);  // zero-mean template

        int validPixels = countNonZero(imageMask);
        double tmpNorm = std::sqrt(validPixels * cv::norm(tmpStd, cv::NORM_L2SQR));
        double imgNorm = std::sqrt(validPixels * cv::norm(imgStd, cv::NORM_L2SQR));

        // compute Hessian, image/template Jacobian projections, and correlation without materializing Jacobian/Xgrid/Ygrid
        const double correlation = accumulateECCSystem(
            gradientXWarped, gradientYWarped, imageWarped, templateZM,
            map, motionType, numberOfParameters,
            hessian, imageProjection, templateProjection);

        hessianInv = hessian.inv();

        // calculate enhanced correlation coefficient (ECC)->rho
        last_rho = rho;
        rho = correlation / (imgNorm * tmpNorm);
        if (cvIsNaN(rho)) {
            CV_Error(Error::StsNoConv, "NaN encountered.");
        }

        // calculate the parameter lambda to account for illumination variation
        gemm(hessianInv, imageProjection, 1.0, Mat(), 0.0, imageProjectionHessian);
        const double lambda_n = (imgNorm * imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0) {
            rho = -1;
            CV_Error(Error::StsNoConv,
                     "The algorithm stopped before its convergence. The correlation is going to be minimized. Images "
                     "may be uncorrelated or non-overlapped");
        }
        const double lambda = (lambda_n / lambda_d);

        // estimate the update step delta_p via linearity: J^T (lambda * templateZM - imageWarped) = lambda * templateProjection - imageProjection
        const float* tmplP = templateProjection.ptr<float>(0);
        const float* imgP = imageProjection.ptr<float>(0);
        float* errP = errorProjection.ptr<float>(0);
        for (int p = 0; p < numberOfParameters; ++p)
            errP[p] = (float)lambda * tmplP[p] - imgP[p];
        gemm(hessianInv, errorProjection, 1.0, Mat(), 0.0, deltaP);

        // update warping matrix
        update_warping_matrix_ECC(map, deltaP, motionType);
    }

    // return final correlation coefficient
    return rho;
}

double cv::findTransformECC(InputArray templateImage,
                            InputArray inputImage,
                            InputOutputArray warpMatrix,
                            int motionType,
                            TermCriteria criteria,
                            InputArray inputMask,
                            int gaussFiltSize
                            ) {
    return findTransformECCWithMask(templateImage, inputImage, noArray(), inputMask,
            warpMatrix, motionType, criteria, gaussFiltSize);
}

double cv::findTransformECC(InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix,
                            int motionType, TermCriteria criteria, InputArray inputMask) {
    // Use default value of 5 for gaussFiltSize to maintain backward compatibility.
    return findTransformECC(templateImage, inputImage, warpMatrix, motionType, criteria, inputMask, 5);
}

/* End of file. */
