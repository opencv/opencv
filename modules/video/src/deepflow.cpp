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

using namespace cv;

namespace {

class OpticalFlowDeepFlow: public DenseOpticalFlow
{
public:
    OpticalFlowDeepFlow();

    void calc( InputArray I0, InputArray I1, InputOutputArray flow );
    void collectGarbage();

    AlgorithmInfo* info() const;

protected:
    float sigma; // Gaussian smoothing parameter
    int minSize; // minimal dimension of an image in the pyramid
    float downscaleFactor; // scaling factor in the pyramid
    int fixedPointIterations; // during each level of the pyramid

private:
    void calcOneLevel( const Mat I0, const Mat I1, Mat W );
    Mat remapRelative( const Mat input, const Mat flow);
    void dataTerm(const Mat W, const Mat dW, const Mat tempW, const Mat Ix, const Mat Iy,
            const Mat Iz, const Mat Ixx, const Mat Ixy, const Mat Iyy, const Mat Ixz,
            const Mat Iyz, Mat a11, Mat a12, Mat a22, Mat b1, Mat b2);
    void smoothnessTerm(const Mat W, const Mat tempW, Mat b1, Mat b2);
    void sorSolve(const Mat a11, const Mat a12, const Mat a22,
            const Mat b1, const Mat b2, Mat dW);
    std::vector<Mat> buildPyramid( const Mat& src);

    int interpolationType;

};

OpticalFlowDeepFlow::OpticalFlowDeepFlow()
{
    // parameters
    sigma = 0.5f;
    minSize = 25;
    downscaleFactor = 0.95;
    fixedPointIterations = 20;

    //consts
    interpolationType = INTER_LINEAR;
}

std::vector<Mat> OpticalFlowDeepFlow::buildPyramid( const Mat& src )
{
    std::vector<Mat> pyramid;
    pyramid.push_back(src);
    Mat prev = pyramid[0];
    while ( prev.cols > minSize && prev.rows > minSize)
    {
        Mat next;
        resize(prev, next, Size(floor(prev.cols * downscaleFactor), floor(prev.rows * downscaleFactor)), 0, 0,
                interpolationType);
        pyramid.push_back(next);
        prev = next;
    }
    return pyramid;
}
Mat OpticalFlowDeepFlow::remapRelative( const Mat input, const Mat flow)
{
    Mat output;
    // TODO: implement, based on remap()
    return output;
}
void OpticalFlowDeepFlow::calc( InputArray _I0, InputArray _I1, InputOutputArray _flow )
{

    Mat I0 = _I0.getMat();
    Mat I1 = _I1.getMat();
    Mat W = _flow.getMat(); // if any data present - will be discarded

    CV_Assert(I0.size() == I1.size());
    CV_Assert(I0.type() == I1.type());
    // TODO: ensure the right format ( or conversion )

    // pre-smooth images
    int kernelLen = (floor(3 * sigma) * 2) + 1;
    Size kernelSize(kernelLen, kernelLen);
    GaussianBlur(I0, I0, kernelSize, sigma);
    GaussianBlur(I1, I1, kernelSize, sigma);
    // build down-sized pyramids
    std::vector<Mat> pyramid_I0 = buildPyramid(I0);
    std::vector<Mat> pyramid_I1 = buildPyramid(I1);
    int levelCount = pyramid_I0.size();

    // initialize the first version of flow estimate to zeros
    Size smallestSize = pyramid_I0[levelCount-1].size();
    W = Mat::zeros(smallestSize, CV_32FC2);

    for ( int level = levelCount-1; level >= 0; --level)
    { //iterate through  all levels, beginning with the most coarse
        calcOneLevel(pyramid_I0[level], pyramid_I1[level], W);
        if(level > 0) //not the last level
        {
            Mat temp;
            Size newSize = pyramid_I0[level-1].size();
            resize(W, temp, newSize, 0, 0, interpolationType); //resize calculated flow
            W =  W * (1.0f / downscaleFactor); //scale values
        }
    }
}

void OpticalFlowDeepFlow::calcOneLevel( const Mat I0, const Mat I1, Mat W )
{
    CV_DbgAssert( I0.size() == I1.size() );
    CV_DbgAssert( I0.type() == I1.type() );
    CV_DbgAssert( W.size() == I0.size() );

    // linear equation systems
    Mat a11, a12, a22, b1, b2;

    Mat warpedI1 = remapRelative(I1, W); // warped second image
    Mat averageFrame = 0.5 * (I0 + warpedI1); // mean value of 2 frames - to compute derivatives on


    //computing derivatives, notation as in Brox's paper
    Mat Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz;
    int ddepth = -1; //as source image
    int kernel_size = 1; //
    //FIXME: if source image is has 8-bit depth output may be truncated
    Sobel(averageFrame, Ix, ddepth, 1, 0, kernel_size);
    Sobel(averageFrame, Iy, ddepth, 0, 1, kernel_size);
    Iz = I1 - I0; // FIXME: should the warped I1 be used?
    Sobel(Ix, Ixx, ddepth, 1, 0, kernel_size);
    Sobel(Ix, Ixy, ddepth, 0, 1, kernel_size);
    Sobel(Iy, Iyy, ddepth, 0, 1, kernel_size);
    Sobel(Ixz, Iz, ddepth, 1, 0, kernel_size); // should a difference of derivatives be used instead?
    Sobel(Iyz, Iz, ddepth, 0, 1, kernel_size);

    Mat tempW = W.clone(); // flow version to be modified in each iteration
    Mat dW = Mat::zeros(W.size(), W.type()); // flow increment

    //fixed-point iterations
    for(int i=0; i < fixedPointIterations; ++i)
    {
        dataTerm(W, dW, tempW, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, a11, a12, a22, b1, b2);
        smoothnessTerm(W, tempW, b1, b2);
        sorSolve(a11, a12, a22, b1, b2, dW);
        tempW = W + dW;
    }
    W = tempW;
}
void OpticalFlowDeepFlow::dataTerm(const Mat W, const Mat dW, const Mat tempW, const Mat Ix,
        const Mat Iy, const Mat Iz, const Mat Ixx, const Mat Ixy, const Mat Iyy, const Mat Ixz,
        const Mat Iyz, Mat a11, Mat a12, Mat a22, Mat b1, Mat b2)
{
    //TODO: data term implementation
}
void OpticalFlowDeepFlow::smoothnessTerm(const Mat W, const Mat tempW, Mat b1, Mat b2)
{
    //TODO: smoothness term implementation -> b1, b2
}
void OpticalFlowDeepFlow::sorSolve(const Mat a11, const Mat a12, const Mat a22,
        const Mat b1, const Mat b2, Mat dW)
{
    //TODO: implement the solver
}
void OpticalFlowDeepFlow::collectGarbage()
{

}

CV_INIT_ALGORITHM(OpticalFlowDeepFlow, "DenseOpticalFlow.DeepFlow",
        obj.info()->addParam(obj, "sigma", obj.sigma, false, 0, 0, "Gaussian blur parameter");
        obj.info()->addParam(obj, "minSize", obj.minSize, false, 0, 0, "Min. image size in the pyramid");
        obj.info()->addParam(obj, "fixedPointIterations", obj.fixedPointIterations, false, 0, 0, "Fixed point iterations");
        obj.info()->addParam(obj, "downscaleFactor", obj.downscaleFactor, false, 0, 0,"Downscale factor"))

} // namespace

Ptr<DenseOpticalFlow> cv::createOptFlow_DeepFlow()
{
    return makePtr<OpticalFlowDeepFlow>();
}
