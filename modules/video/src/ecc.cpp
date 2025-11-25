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

class Jacobian {
    int paramsNum;
    Mat m;
    Size size;
public:
    Jacobian(int num, int h, int w, int type): paramsNum(num), size(w, h) {
        m = Mat(h, w * paramsNum, type);
    }

     // Access block dF/dp_i
    Mat dp(const int i) {
        CV_Assert(i < paramsNum);
        return m.colRange(i * size.width, (i + 1) * size.width);
    }

    // Performs dst = J^T * e
    void project(const Mat& e, Mat dst) {
        CV_Assert(dst.rows == paramsNum);
        CV_Assert(e.size() == size);
        for (int i = 0; i < paramsNum; ++i) {
            dst.at<float>(i) = (float)(e.dot(dp(i)));
        }
    }

    // Performs  J^T*J
    void getHessian(Mat H) {
        float* dstPtr = H.ptr<float>(0);

        CV_Assert(H.cols == H.rows);  // H is square (and symmetric)
        CV_Assert(H.cols == paramsNum);
        Mat mat;
        for (int i = 0; i < H.rows; i++) {
            mat = dp(i);
            dstPtr[i * (H.rows + 1)] = (float)pow(norm(mat), 2);  // diagonal elements

            for (int j = i + 1; j < H.cols; j++) {  // j starts from i+1
                dstPtr[i * H.cols + j] = (float)mat.dot(dp(j));
                dstPtr[j * H.cols + i] = dstPtr[i * H.cols + j];  // due to symmetry
            }
        }
    }
};

/* Generic transformation interface. The derived class should define:
 * The warp transformation from image to template coordinates.
 * Method for calculating partial derivatives of intensity dI/dp with respect to model parameters (Jacobian)
 * */
class WarpModel {
    // A row vector of parameters of the model being optimized
    Mat T;
protected:
    Mat X;
    Mat Y;
public:
    WarpModel(const Mat map, const Size size, const int type): T(map) {

        const int ws = size.width;
        const int hs = size.height;

        Mat Xcoord = Mat(1, ws, CV_32F);
        Mat Ycoord = Mat(hs, 1, CV_32F);
        Mat Xgrid = Mat(hs, ws, CV_32F);
        Mat Ygrid = Mat(hs, ws, CV_32F);

        float* XcoPtr = Xcoord.ptr<float>(0);
        float* YcoPtr = Ycoord.ptr<float>(0);
        int j;
        for (j = 0; j < ws; j++) XcoPtr[j] = (float)j;
        for (j = 0; j < hs; j++) YcoPtr[j] = (float)j;

        repeat(Xcoord, hs, 1, Xgrid);
        repeat(Ycoord, 1, ws, Ygrid);

        Xcoord.release();
        Ycoord.release();

        const int channels = CV_MAT_CN(type);

        std::vector<cv::Mat> XgridCh(channels, Xgrid);
        cv::merge(XgridCh, X);

        std::vector<cv::Mat> YgridCh(channels, Ygrid);
        cv::merge(YgridCh, Y);

        Xgrid.release();
        Ygrid.release();
    }

    // Returns the number of model parameters
    int getNumParams() const {
        return (int)T.total();
    }

    /* This method returns the transformation matrix in the canonical form in which it would be returned
     * from the findTransformECC method, for example, affine or perspective.
     * The internal parameters of the model T must be properly packed into the output matrix.
     * Direct model parameter modification is prohibited to avoid inconsistency.
    */
    virtual Mat getMat() const {
        return T;
    }

    // Returns model parameter with index i.
    // Direct parameter modification is prohibited to avoid inconsistency.
    float p(int idx) const {
        return T.at<float>(idx, 0);
    }

    // Default affine warp transform
    virtual void warp(const Mat src, Mat dst, const int flags) const {
        warpAffine(src, dst, getMat(), dst.size(), flags);
    }

    // Calculates spatial derivatives of pixel intensities I(x, y, c) w.r.t model parameters.
    // dI/dp = dI/dx * dx/dp * dI/dy * dy/dp = gradX * dx/dp + gradY * dy/dp
    // gradX - X image gradient.
    // gradY - Y image gradient.
    // dI - image intensity spatial derivatives (Jacobian)  with respect to model parameters
    virtual void calcJacobian(const Mat& gradX, const Mat& gradY, Jacobian& dI) const = 0;

    // Parameter models can only be updated using this method to avoid inconsistency.
    void updateTransform(const Mat& update) {
        T += update;
    }

    virtual ~WarpModel() {}

    static cv::Ptr<WarpModel> make(const int motionType, const Mat map, Size size, int type);
};

class WarpTranslation : public WarpModel {
    enum {tx = 0, ty = 1};
public:
    WarpTranslation(const Mat map, const Size size, const int type): WarpModel(map.col(2), size, type) {
    }

    Mat getMat() const override {
        return  (cv::Mat_<float>(2, 3) << 1, 0, p(tx),
                                          0, 1, p(ty));
    }

    void calcJacobian(const Mat& gradX, const Mat& gradY, Jacobian& dI) const override
    {
        // x' = x + tx
        // y' = y + ty
        // dx'/dtx = 1; dx'/dty = 0
        // dy'/dtx = 0; dy'/dty = 1
        gradX.copyTo(dI.dp(tx)); // dI/dtx = gradX * 1 + gradY * 0
        gradY.copyTo(dI.dp(ty)); // dI/dty = gradX * 0 + gradY * 1
    }
};


class WarpEuclidean: public WarpModel {
    enum {theta = 0, tx = 1, ty = 2};

    static Mat extract(const Mat &map){
        // [ a   b   tx ]
        // [ c   d   ty ]
        // Best-fit rotation angle in Frobenius norm:
        // theta = atan2(c - b, a + d)
        float X = map.at<float>(0, 0) + map.at<float>(1, 1);
        float Y = map.at<float>(1, 0) - map.at<float>(0, 1);
        float theta = std::atan2(Y, X);

        float tx = map.at<float>(0,2);
        float ty = map.at<float>(1,2);
        return Mat_<float>(3, 1) << theta, tx, ty;
    }
public:

    WarpEuclidean(const Mat &map, const Size size, const int type): WarpModel(extract(map), size, type) {
    }

    Mat getMat() const override{
        const float c = std::cos(p(theta));
        const float s = std::sin(p(theta));

        return (cv::Mat_<float>(2,3) << c, -s, p(tx),
                                        s, c,  p(ty));
    }

    void calcJacobian(const Mat& gradX, const Mat& gradY, Jacobian& dI) const override {
        const float c = std::cos(p(theta));
        const float s = std::sin(p(theta));

        // hatX = -sin(theta)*X - cos(theta)*Y
        // hatY =  cos(theta)*X - sin(theta)*Y
        Mat hatX = -(X * s) - (Y * c);
        Mat hatY =  (X * c) - (Y * s);

        // compute Jacobian blocks (3 blocks)
        // dI/dtheta = Ix * dx/dtheta + Iy * dy/dtheta
        dI.dp(theta) = (gradX.mul(hatX)) + (gradY.mul(hatY));
        gradX.copyTo(dI.dp(tx)); // dI/dtx = Ix  (dx/dtx = 1, dy/dtx = 0)
        gradY.copyTo(dI.dp(ty)); // dI/dty = Iy  (dx/dty = 0, dy/dty = 1)
    }
};

class WarpAffine: public WarpModel {
public:
    WarpAffine(const Mat map, const Size size, const int type): WarpModel (map.reshape(1,6), size, type) {
    }

    virtual Mat getMat() const override {
        return WarpModel::getMat().reshape(1, 2);
    }

    void calcJacobian(const Mat& gradX, const Mat& gradY, Jacobian& dI) const override {
        // compute Jacobian blocks (6 blocks)
        dI.dp(0) = gradX.mul(X);  // dI/da00
        dI.dp(1) = gradX.mul(Y);  // dI/da01
        gradX.copyTo(dI.dp(2));   // dI/da12
        dI.dp(3) = gradY.mul(X);  // dI/da10
        dI.dp(4) = gradY.mul(Y);  // dI/da11
        gradY.copyTo(dI.dp(5));   // dI/da12
    }
};

class WarpHomography : public WarpModel {
public:
    WarpHomography(const Mat& map, const Size size, const int type):
        WarpModel(map.reshape(1, 9).rowRange(0, 8), size, type) {
    }

    Mat getMat() const override {

        return (Mat_<float>(3, 3) << p(0), p(1), p(2),
                                     p(3), p(4), p(5),
                                     p(6), p(7), 1.0f);
    }

    void warp(const Mat src, Mat dst, const int flags) const override {
        warpPerspective(src, dst, getMat(), dst.size(), flags);
    }

    void calcJacobian(const Mat& gradX, const Mat& gradY, Jacobian& dI) const override
    {
        const float H00 = p(0);
        const float H01 = p(1);
        const float H02 = p(2);
        const float H10 = p(3);
        const float H11 = p(4);
        const float H12 = p(5);
        const float H20 = p(6);
        const float H21 = p(7);

        Mat den_;
        addWeighted(X, H20, Y, H21, 1.0, den_);

        // hatX numerator = h0 * X + h3 * Y + h6
        //                = H00 * X + H01 * Y + H02
        Mat hatX_;
        Mat hatY_;
        addWeighted(X, H00, Y, H01, 0.0, hatX_);
        hatX_ += H02;

        addWeighted(X, H10, Y, H11, 0.0, hatY_);
        hatY_ += H12;

        // -x'/w, -y'/w
        divide(-hatX_, den_, hatX_);
        divide(-hatY_, den_, hatY_);

        // Pre-divide gradients by denominator
        Mat gradXDivided_;
        Mat gradYDivided_;
        divide(gradX, den_, gradXDivided_);
        divide(gradY, den_, gradYDivided_);

        Mat temp_ = hatX_.mul(gradXDivided_) + hatY_.mul(gradYDivided_);

        dI.dp(0) = gradXDivided_.mul(X); // dI/dp0 = dI/dh0
        dI.dp(1) = gradXDivided_.mul(Y); // dI/dp1 = dI/dh3
        gradXDivided_.copyTo(dI.dp(2));  // dI/dp2 = dI/dh6
        dI.dp(3) = gradYDivided_.mul(X); // dI/dp3 = dI/dh1
        dI.dp(4) = gradYDivided_.mul(Y); // dI/dp4 = dI/dh4
        gradYDivided_.copyTo(dI.dp(5));  // dI/dp5 = dI/dh7
        dI.dp(6) = temp_.mul(X);         // dI/dp6 = dI/dh2
        dI.dp(7) = temp_.mul(Y);         // dI/dp7 = dI/dh5
    }
};

cv::Ptr<WarpModel> WarpModel::make(const int motionType,const Mat map, const Size size, const int type) {

    if (!map.empty()) {
        if (map.type() != CV_32FC1)
            CV_Error(Error::StsUnsupportedFormat,"warpMatrix must be single-channel floating-point matrix");
    }

    switch (motionType) {
    case MOTION_TRANSLATION:
        if (!map.empty()){
            CV_Assert(map.cols == 3 && map.rows == 2);
            return cv::makePtr<WarpTranslation>(map, size, type);
        } else
            return cv::makePtr<WarpTranslation>(Mat::eye(2,3,CV_32F), size, type);
    case MOTION_EUCLIDEAN:
        if (!map.empty()){
            CV_Assert(map.cols == 3 && map.rows == 2);
            return cv::makePtr<WarpEuclidean>(map, size, type);
        } else
            return cv::makePtr<WarpEuclidean>(Mat::eye(2,3,CV_32F), size, type);
    case MOTION_AFFINE:
        if (!map.empty()){
            CV_Assert(map.cols == 3 && map.rows == 2);
            return cv::makePtr<WarpAffine>(map, size, type);
        } else
            return cv::makePtr<WarpAffine>(Mat::eye(2,3,CV_32F), size, type);
    case MOTION_HOMOGRAPHY:
        if (!map.empty()){
            CV_Assert(map.cols == 3 && map.rows == 3);
            return cv::makePtr<WarpHomography>(map, size, type);
        } else
            return cv::makePtr<WarpHomography>(Mat::eye(3,3,CV_32F), size, type);
    default:
        CV_Error(cv::Error::StsBadArg,"Unsupported motion type");
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
    Mat templateImage_zeromean = Mat::zeros(templateImage.size(), templateImage.type());
    Mat templateMat = templateImage.getMat();
    Mat inputMat = inputImage.getMat();

    /*
     * For unsigned ints, when the mean is computed and subtracted, any values less than the mean
     * will be set to 0 (since there are no negatives values). This impacts the norm and dot product, which
     * ultimately results in an incorrect ECC. To circumvent this problem, if unsigned ints are provided,
     * we convert them to a signed ints with larger resolution for the subtraction step.
     */
    if (type == CV_8U || type == CV_16U) {
        int newType = type == CV_8U ? CV_16S : CV_32S;
        Mat templateMatConverted, inputMatConverted;
        templateMat.convertTo(templateMatConverted, newType);
        cv::swap(templateMat, templateMatConverted);
        inputMat.convertTo(inputMatConverted, newType);
        cv::swap(inputMat, inputMatConverted);
    }
    subtract(templateMat, meanTemplate, templateImage_zeromean, inputMask);
    double templateImagenorm = std::sqrt(active_pixels * cv::norm(sdTemplate, NORM_L2SQR));

    Scalar meanInput, sdInput;

    Mat inputImage_zeromean = Mat::zeros(inputImage.size(), inputImage.type());
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

    if (!(src.type() == dst.type()))
        CV_Error(Error::StsUnmatchedFormats, "Both input images must have the same data type");

    auto model = WarpModel::make(motionType, map, src.size(), src.type());

    if (motionType == MOTION_HOMOGRAPHY) {
        CV_Assert(map.rows == 3);
    }

    CV_Assert(criteria.type & TermCriteria::COUNT || criteria.type & TermCriteria::EPS);
    const int numberOfIterations = (criteria.type & TermCriteria::COUNT) ? criteria.maxCount : 200;
    const double termination_eps = (criteria.type & TermCriteria::EPS) ? criteria.epsilon : -1;

    const int numberOfParameters = model->getNumParams();

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
    Jacobian jacobian(numberOfParameters, hs, ws, type);
    Mat hessian = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat hessianInv = Mat(numberOfParameters, numberOfParameters, CV_32F);
    Mat imageProjection = Mat(numberOfParameters, 1, CV_32F);
    Mat templateProjection = Mat(numberOfParameters, 1, CV_32F);
    Mat imageProjectionHessian = Mat(numberOfParameters, 1, CV_32F);
    Mat errorProjection = Mat(numberOfParameters, 1, CV_32F);

    Mat deltaP = Mat(numberOfParameters, 1, CV_32F);  // transformation parameter correction
    Mat error = Mat(hs, ws, CV_32F);                  // error as 2D matrix

    const int imageFlags = INTER_LINEAR + WARP_INVERSE_MAP;
    const int maskFlags = INTER_NEAREST + WARP_INVERSE_MAP;

    // Iteratively update map_matrix using Gauss–Newton algorithm
    double rho = -1;
    double last_rho = -termination_eps;
    for (int i = 1; (i <= numberOfIterations) && (fabs(rho - last_rho) >= termination_eps); i++) {
        //
        model->warp(imageFloat, imageWarped, imageFlags);
        model->warp(gradientX, gradientXWarped, imageFlags);
        model->warp(gradientY, gradientYWarped, imageFlags);
        model->warp(preMask, imageMask, maskFlags);

        if (!templateMask.empty())
        {
            cv::bitwise_and(imageMask, templtMask, imageMask);

            cv::Mat zeroMask = (imageMask == 0);
            gradientXWarped.setTo(0, zeroMask);
            gradientYWarped.setTo(0, zeroMask);
        }

        Scalar imgMean, imgStd, tmpMean, tmpStd;
        meanStdDev(imageWarped, imgMean, imgStd, imageMask);
        meanStdDev(templateFloat, tmpMean, tmpStd, imageMask);

        subtract(imageWarped, imgMean, imageWarped, imageMask);  // zero-mean input
        templateZM = Mat::zeros(templateZM.rows, templateZM.cols, templateZM.type());
        subtract(templateFloat, tmpMean, templateZM, imageMask);  // zero-mean template

        int validPixels = countNonZero(imageMask);
        double tmpNorm = std::sqrt(validPixels * cv::norm(tmpStd, cv::NORM_L2SQR));
        double imgNorm = std::sqrt(validPixels * cv::norm(imgStd, cv::NORM_L2SQR));

        model->calcJacobian(gradientXWarped, gradientYWarped, jacobian);

        // calculate Hessian and its inverse
        jacobian.getHessian(hessian);

        hessianInv = hessian.inv();

        const double correlation = templateZM.dot(imageWarped);

        // calculate enhanced correlation coefficient (ECC)->rho
        last_rho = rho;
        rho = correlation / (imgNorm * tmpNorm);
        if (cvIsNaN(rho)) {
            CV_Error(Error::StsNoConv, "NaN encountered.");
        }

        // project images into jacobian
        jacobian.project(imageWarped, imageProjection);
        jacobian.project(templateZM, templateProjection);

        // calculate the parameter lambda to account for illumination variation
        imageProjectionHessian = hessianInv * imageProjection;
        const double lambda_n = (imgNorm * imgNorm) - imageProjection.dot(imageProjectionHessian);
        const double lambda_d = correlation - templateProjection.dot(imageProjectionHessian);
        if (lambda_d <= 0.0) {
            rho = -1;
            CV_Error(Error::StsNoConv,
                     "The algorithm stopped before its convergence. The correlation is going to be minimized. Images "
                     "may be uncorrelated or non-overlapped");
        }
        const double lambda = (lambda_n / lambda_d);

        // estimate the update step delta_p
        error = lambda * templateZM - imageWarped;
        jacobian.project(error, errorProjection);
        deltaP = hessianInv * errorProjection;

        // Update warping matrix. w <- w + deltaP
        model->updateTransform(deltaP);
    }

    model->getMat().copyTo(map);
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
