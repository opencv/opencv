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
    int sorIterations; // iterations of SOR
    float alpha; // smoothness assumption weight
    float delta; // color constancy weight
    float gamma; // gradient constancy weight
    float omega; // relaxation factor in SOR

    float zeta; // added to the denomimnator of theta_0 (normaliation of the data term)
    float epsilon; // robust penalizer const

private:
    void calcOneLevel( const Mat I0, const Mat I1, Mat W );
    Mat remapRelative( const Mat input, const Mat flow );
    void dataTerm( const Mat W, const Mat dW, const Mat tempW, const Mat Ix, const Mat Iy,
            const Mat Iz, const Mat Ixx, const Mat Ixy, const Mat Iyy, const Mat Ixz, const Mat Iyz,
            Mat a11, Mat a12, Mat a22, Mat b1, Mat b2 );
    void smoothnessWeights( const Mat W, Mat weightsX, Mat weightsY );
    void smoothnessTerm( const Mat W, const Mat weightsX, const Mat weightsY, Mat b1, Mat b2 );
    void sorSolve( const Mat a11, const Mat a12, const Mat a22, const Mat b1, const Mat b2,
            const Mat smoothX, const Mat smoothY, Mat dW );
    std::vector<Mat> buildPyramid( const Mat& src );

    int interpolationType;

};

OpticalFlowDeepFlow::OpticalFlowDeepFlow()
{
    // parameters
    sigma = 0.5f;
    minSize = 25;
    downscaleFactor = 0.95f;
    fixedPointIterations = 5;
    sorIterations = 25;
    alpha = 6.0f;
    delta = 0.5f;
    gamma = 5.0f;
    omega = 1.6f;

    //consts
    interpolationType = INTER_LINEAR;
    zeta = 0.1f;
    epsilon = 0.001f;
}

std::vector<Mat> OpticalFlowDeepFlow::buildPyramid( const Mat& src )
{
    std::vector<Mat> pyramid;
    pyramid.push_back(src);
    Mat prev = pyramid[0];
    while ( prev.cols > minSize && prev.rows > minSize )
    {
        Mat next; //FIXME: filtering at each level?
        resize(prev, next,
                Size(floor(prev.cols * downscaleFactor), floor(prev.rows * downscaleFactor)), 0, 0,
                interpolationType);
        pyramid.push_back(next);
        prev = next;
    }
    return pyramid;
}
Mat OpticalFlowDeepFlow::remapRelative( const Mat input, const Mat flow )
{
    Mat output;
    Mat mapX = Mat(flow.size(), CV_32FC1);
    Mat mapY = Mat(flow.size(), CV_32FC1);

    const float *pFlow;
    float *pMapX, *pMapY;
    for ( int j = 0; j < flow.rows; ++j )
    {
        pFlow = flow.ptr<float>(j);
        pMapX = mapX.ptr<float>(j);
        pMapY = mapY.ptr<float>(j);
        for ( int i = 0; i < flow.cols; ++i )
        {
            pMapX[i] = i + pFlow[2 * i];
            pMapY[i] = j + pFlow[2 * i + 1];
        }
    }
    remap(input, output, mapX, mapY, interpolationType);
    return output;
}
void OpticalFlowDeepFlow::calc( InputArray _I0, InputArray _I1, InputOutputArray _flow )
{

    Mat I0temp = _I0.getMat();
    Mat I1temp = _I1.getMat();

    CV_Assert(I0temp.size() == I1temp.size());
    CV_Assert(I0temp.type() == I1temp.type());
    CV_Assert(I0temp.channels() == 1); // FIXME: only grayscale - to be generalised

    Mat I0, I1;

    I0temp.convertTo(I0, CV_32F);
    I1temp.convertTo(I1, CV_32F);
    // TODO: ensure the right format ( or conversion )

    _flow.create(I0.size(), CV_32FC2);
    Mat W = _flow.getMat(); // if any data present - will be discarded

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
    Size smallestSize = pyramid_I0[levelCount - 1].size();
    W = Mat::zeros(smallestSize, CV_32FC2);

    for ( int level = levelCount - 1; level >= 0; --level )
    { //iterate through  all levels, beginning with the most coarse
        calcOneLevel(pyramid_I0[level], pyramid_I1[level], W);
        if ( level > 0 ) //not the last level
        {
            Mat temp;
            Size newSize = pyramid_I0[level - 1].size();
            resize(W, temp, newSize, 0, 0, interpolationType); //resize calculated flow
            W = temp * (1.0f / downscaleFactor); //scale values
        }
    }
}

void OpticalFlowDeepFlow::calcOneLevel( const Mat I0, const Mat I1, Mat W )
{
    CV_DbgAssert( I0.size() == I1.size() );CV_DbgAssert( I0.type() == I1.type() );CV_DbgAssert( W.size() == I0.size() );

    // linear equation systems
    Size s = I0.size();
    int t = CV_32F; // data type
    Mat a11, a12, a22, b1, b2;
    a11.create(s, t);
    a12.create(s, t);
    a22.create(s, t);
    b1.create(s, t);
    b2.create(s, t);
    // diffusivity coeffs
    Mat weightsX, weightsY;
    weightsX.create(s, t);
    weightsY.create(s, t);

    Mat warpedI1 = remapRelative(I1, W); // warped second image
    Mat averageFrame = 0.5 * (I0 + warpedI1); // mean value of 2 frames - to compute derivatives on

    //computing derivatives, notation as in Brox's paper
    Mat Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz;
    int ddepth = -1; //as source image
    int kernel_size = 1; //
    //FIXME: if source image is has 8-bit depth output may be truncated
    Sobel(averageFrame, Ix, ddepth, 1, 0, kernel_size);
    Sobel(averageFrame, Iy, ddepth, 0, 1, kernel_size);
    Iz.create(I1.size(), I1.type());
    Iz = I1 - I0; // FIXME: should the warped I1 be used?
    Sobel(Ix, Ixx, ddepth, 1, 0, kernel_size);
    Sobel(Ix, Ixy, ddepth, 0, 1, kernel_size);
    Sobel(Iy, Iyy, ddepth, 0, 1, kernel_size);
    Sobel(Iz, Ixz, ddepth, 1, 0, kernel_size); // should a difference of derivatives be used instead?
    Sobel(Iz, Iyz, ddepth, 0, 1, kernel_size);

    Mat tempW = W.clone(); // flow version to be modified in each iteration
    Mat dW = Mat::zeros(W.size(), W.type()); // flow increment

    //fixed-point iterations
    for ( int i = 0; i < fixedPointIterations; ++i )
    {
        dataTerm(W, dW, tempW, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, a11, a12, a22, b1, b2);
        smoothnessWeights(W, weightsX, weightsY);
        smoothnessTerm(W, weightsX, weightsY, b1, b2);
        sorSolve(a11, a12, a22, b1, b2, weightsX, weightsY, dW);
        tempW = W + dW;
    }
    tempW.copyTo(W);
}
void OpticalFlowDeepFlow::dataTerm( const Mat W, const Mat dW, const Mat tempW, const Mat Ix,
        const Mat Iy, const Mat Iz, const Mat Ixx, const Mat Ixy, const Mat Iyy, const Mat Ixz,
        const Mat Iyz, Mat a11, Mat a12, Mat a22, Mat b1, Mat b2 )
{
    //TODO: data term implementation
    const float zeta_squared = zeta * zeta; // added in normalization factor to be non-zero
    const float epsilon_squared = epsilon * epsilon;
    if ( Ix.channels() == 1 )
    {
        const float *pIx, *pIy, *pIz;
        const float *pIxx, *pIxy, *pIyy, *pIxz, *pIyz;
        const float *pdU, *pdV; // accessing 2 layers of dW. Succesive columns interleave u and v
        float *pa11, *pa12, *pa22, *pb1, *pb2; // linear equation sys. coeffs for each pixel

        float derivNorm; //denominator of the spatial-derivative normalizing factor (theta_0)
        float derivNorm2;
        float Ik1z, Ik1zx, Ik1zy; // approximations of I^(k+1) values by Taylor expansions
        float temp;
        for ( int j = 0; j < W.rows; j++ ) //for each row
        {
            pIx = Ix.ptr<float>(j);
            pIy = Iy.ptr<float>(j);
            pIz = Iz.ptr<float>(j);
            pIxx = Ixx.ptr<float>(j);
            pIxy = Ixy.ptr<float>(j);
            pIyy = Iyy.ptr<float>(j);
            pIxz = Ixz.ptr<float>(j);
            pIyz = Iyz.ptr<float>(j);

            pa11 = a11.ptr<float>(j);
            pa12 = a12.ptr<float>(j);
            pa22 = a22.ptr<float>(j);
            pb1 = b1.ptr<float>(j);
            pb2 = b2.ptr<float>(j);

            pdU = dW.ptr<float>(j);
            pdV = pdU + 1;
            for ( int i = 0; i < W.cols; i++ ) //for each pixel in the row
            { // TODO: implement masking of points warped out of the image
              //color constancy component
                derivNorm = (*pIx) * (*pIx) + (*pIy) * (*pIy) + zeta_squared;
                Ik1z = *pIz + (*pIx * *pdU) + (*pIy * *pdV);
                temp = delta / (sqrt(Ik1z * Ik1z / derivNorm + epsilon_squared) * derivNorm);
                *pa11 = *pIx * *pIx * temp;
                *pa12 = *pIx * *pIy * temp;
                *pa22 = *pIy * *pIy * temp;
                *pb1 = -*pIz * *pIx * temp;
                *pb2 = -*pIz * *pIy * temp;

                // gradient constancy component

                derivNorm = *pIxx * *pIxx + *pIxy * *pIxy + zeta_squared;
                derivNorm2 = *pIyy * *pIyy + *pIxy * *pIxy + zeta_squared;
                Ik1zx = *pIxz + *pIxx * *pdU + *pIxy * *pdV;
                Ik1zy = *pIyz + *pIyz * *pdU + *pIyy * *pdV;

                temp = gamma
                        / sqrt(
                                Ik1zx * Ik1zx / derivNorm + Ik1zy * Ik1zy / derivNorm2
                                        + epsilon_squared);
                *pa11 += temp * (*pIxx * *pIxx / derivNorm + *pIxy * *pIxy / derivNorm2);
                *pa12 += temp * (*pIxx * *pIxy / derivNorm + *pIxy * *pIyy / derivNorm2);
                *pa22 += temp * (*pIxy * *pIxy / derivNorm + *pIyy * *pIyy / derivNorm2);
                *pb1 += -temp * (*pIxx * *pIxz / derivNorm + *pIxy * *pIyz / derivNorm2);
                *pb2 += -temp * (*pIyy * *pIyz / derivNorm + *pIxy * *pIxz / derivNorm2);

                ++pIx;
                ++pIy;
                ++pIz;
                ++pIxx;
                ++pIxy;
                ++pIyy;
                ++pIxz;
                ++pIyz;
                pdU += 2;
                pdV += 2;
                ++pa11;
                ++pa12;
                ++pa22;
                ++pb1;
                ++pb2;

            }
        }

    } else if ( Ix.channels() == 3 )
    {
        //TODO: implement 3-channel version of data-term computation
        CV_Assert(false);
    }

}
void OpticalFlowDeepFlow::smoothnessWeights( const Mat W, Mat weightsX, Mat weightsY )
{
    float k[] = { -0.5, 0, 0.5 };
    const float epsilon_squared = epsilon * epsilon;
    Mat kernel_h = Mat(1, 3, CV_32FC1, k);
    Mat kernel_v = Mat(3, 1, CV_32FC1, k);
    Mat Wx, Wy; // partial derivatives of the flow
    Mat S = Mat(W.size(), CV_32FC1); // sum of squared derivatives
    weightsX = Mat::zeros(W.size(), CV_32FC1); //output - weights of smoothness terms in x and y directions
    weightsY = Mat::zeros(W.size(), CV_32FC1);

    filter2D(W, Wx, -1, kernel_h);
    filter2D(W, Wy, -1, kernel_v);

    const float * ux, *uy, *vx, *vy;
    float * pS, *pWeight, *temp;

    for ( int j = 0; j < S.rows; ++j )
    {
        ux = Wx.ptr<float>(j);
        vx = ux + 1;
        uy = Wx.ptr<float>(j);
        vy = uy + 1;
        pS = S.ptr<float>(j);
        for ( int i = 0; i < S.cols; ++i )
        {
            *pS = alpha / sqrt(*ux * *ux + *vx * *vx + *uy * *uy + *vy * *vy + epsilon_squared);
            ux += 2;
            vx += 2;
            uy += 2;
            vy += 2;
            ++pS;
        }
    }
    // horizontal weights
    for ( int j = 0; j < S.rows; ++j )
    {
        pWeight = weightsX.ptr<float>(j);
        pS = S.ptr<float>(j);
        for ( int i = 0; i < S.cols - 1; ++i )
        {
            *pWeight = *pS + *(pS + 1);
            ++pS;
        }
    }
    //vertical weights
    for ( int j = 0; j < S.rows - 1; ++j )
    {
        pWeight = weightsY.ptr<float>(j);
        pS = S.ptr<float>(j);
        temp = S.ptr<float>(j + 1); // next row pointer for easy access
        for ( int i = 0; i < S.cols; ++i )
        {
            *pWeight = *(pS++) + *(temp++);
        }
    }
}
void OpticalFlowDeepFlow::smoothnessTerm( const Mat W, const Mat weightsX, const Mat weightsY,
        Mat b1, Mat b2 )
{
    float *pB1, *pB2;
    const float *pU, *pV, *pWeight;
    float iB1, iB2; // increments of b1 and b2
    //horizontal direction - both U and V (b1 and b2)
    for ( int j = 0; j < W.rows; j++ )
    {
        pB1 = b1.ptr<float>(j);
        pB2 = b2.ptr<float>(j);
        pU = W.ptr<float>(j);
        pV = pU + 1;
        pWeight = weightsX.ptr<float>(j);
        for ( int i = 0; i < W.cols - 1; i++ )
        {
            iB1 = (*(pU + 2) - *pU) * *pWeight;
            iB2 = (*(pV + 2) - *pV) * *pWeight;
            *pB1 += iB1;
            *(pB1 + 1) -= iB2;
            *pB2 += iB2;
            *(pB2 + 1) -= iB2;

            pB1++;
            pB2++;
            pU += 2;
            pV += 2;
            pWeight++;
        }
    }
    const float *pUnext, *pVnext; // temp pointers for next row
    float *pB1next, *pB2next;
    //vertical direction - both U and V
    for ( int j = 0; j < W.rows - 1; j++ )
    {
        pB1 = b1.ptr<float>(j);
        pB2 = b2.ptr<float>(j);
        pU = W.ptr<float>(j);
        pV = pU + 1;
        pUnext = W.ptr<float>(j + 1);
        pVnext = pUnext + 1;
        pB1next = b1.ptr<float>(j + 1);
        pB2next = b2.ptr<float>(j + 1);
        pWeight = weightsY.ptr<float>(j);
        for ( int i = 0; i < W.cols; i++ )
        {
            iB1 = (*pUnext - *pU) * *pWeight;
            iB2 = (*pVnext - *pV) * *pWeight;
            *pB1 += iB1;
            *pB1next -= iB2;
            *pB2 += iB2;
            *pB2next -= iB2;

            pB1++;
            pB2++;
            pU += 2;
            pV += 2;
            pWeight++;
            pUnext += 2;
            pVnext += 2;
            pB1next++;
            pB2next++;
        }
    }
}
void OpticalFlowDeepFlow::sorSolve( const Mat a11, const Mat a12, const Mat a22, const Mat b1,
        const Mat b2, const Mat smoothX, const Mat smoothY, Mat dW )
{
    //FIXME: TEMPORARY VERSION TO BE REPLACED
    // FOR TESTING PURPOSES ONLY
    // CODE ADAPTED FROM http://lear.inrialpes.fr/src/deepmatching/
    // shared for non-commercial use under GPL license
    CV_Assert(a11.isContinuous());
    CV_Assert(a12.isContinuous());
    CV_Assert(a22.isContinuous());
    CV_Assert(b1.isContinuous());
    CV_Assert(b2.isContinuous());
    CV_Assert(smoothX.isContinuous());
    CV_Assert(smoothY.isContinuous());

    std::vector<Mat> dWChannels(2);
    split(dW, dWChannels);

    Mat du = dWChannels[0];
    Mat dv = dWChannels[1];

    CV_Assert(du.isContinuous());
    CV_Assert(dv.isContinuous());

    const float *pa11, *pa12, *pa22, *pb1, *pb2, *psmoothX, *psmoothY;
    float *pdu, *pdv;
    pa11 = a11.ptr<float>(0);
    pa12 = a12.ptr<float>(0);
    pa22 = a22.ptr<float>(0);
    pb1 = b1.ptr<float>(0);
    pb2 = b2.ptr<float>(0);
    psmoothX = smoothX.ptr<float>(0);
    psmoothY = smoothY.ptr<float>(0);
    pdu = du.ptr<float>(0);
    pdv = dv.ptr<float>(0);

    float sigma_u, sigma_v, sum_dpsis, A11, A22, A12, B1, B2, det;

    int s = dW.cols; // step between rows

    for ( int iter = 0; iter < sorIterations; ++iter )
    {
        for ( int j = 0; j < dW.rows; ++j )
        {
            for ( int i = 0; i < dW.cols; ++i )
            {
                int o = j * s + i;
                if ( i == 0 && j == 0 )
                {
                    sum_dpsis = psmoothX[o] + psmoothY[o];
                    sigma_u = psmoothX[o] * pdu[o + 1] + psmoothY[o] * pdu[o + s];
                    sigma_v = psmoothX[o] * pdv[o + 1] + psmoothY[o] * pdv[o + s];
                } else if ( i == du.cols - 1 && j == 0 )
                {
                    sum_dpsis = psmoothX[o - 1] + psmoothY[o];
                    sigma_u = psmoothX[o - 1] * pdu[o - 1]
                            + psmoothY[o] * pdu[o + s];
                    sigma_v = psmoothX[o - 1] * pdv[o - 1]
                            + psmoothY[o] * pdv[o + s];
                } else if ( j == 0 )
                {
                    sum_dpsis = psmoothX[o - 1] + psmoothX[o] + psmoothY[o];
                    sigma_u = psmoothX[o - 1] * pdu[o - 1]
                            + psmoothX[o] * pdu[o + 1] + psmoothY[o] * pdu[o + s];
                    sigma_v = psmoothX[o - 1] * pdv[o - 1]
                            + psmoothX[o] * pdv[o + 1] + psmoothY[o] * pdv[o + s];
                } else if ( i == 0 && j == du.rows - 1 )
                {
                    sum_dpsis = psmoothX[o] + psmoothY[o - s];
                    sigma_u = psmoothX[o] * pdu[o + 1]
                            + psmoothY[o - s] * pdu[o - s];
                    sigma_v = psmoothX[o] * pdv[o + 1]
                            + psmoothY[o - s] * pdv[o - s];
                } else if ( i == du.cols - 1 && j == du.rows - 1 )
                {
                    sum_dpsis = psmoothX[o - 1] + psmoothY[o - s];
                    sigma_u = psmoothX[o - 1] * pdu[o - 1]
                            + psmoothY[o - s] * pdu[o - s];
                    sigma_v = psmoothX[o - 1] * pdv[o - 1]
                            + psmoothY[o - s] * pdv[o - s];
                } else if ( j == du.rows - 1 )
                {
                    sum_dpsis = psmoothX[o - 1] + psmoothX[o] + psmoothY[o - s];
                    sigma_u = psmoothX[o - 1] * pdu[o - 1]
                            + psmoothX[o] * pdu[o + 1]
                            + psmoothY[o - s] * pdu[o - s];
                    sigma_v = psmoothX[o - 1] * pdv[o - 1]
                            + psmoothX[o] * pdv[o + 1]
                            + psmoothY[o - s] * pdv[o - s];
                } else if ( i == 0 )
                {
                    sum_dpsis = psmoothX[o] + psmoothY[o - s] + psmoothY[o];
                    sigma_u = psmoothX[o] * pdu[o + 1]
                            + psmoothY[o - s] * pdu[o - s]
                            + psmoothY[o] * pdu[o + s];
                    sigma_v = psmoothX[o] * pdv[o + 1]
                            + psmoothY[o - s] * pdv[o - s]
                            + psmoothY[o] * pdv[o + s];
                } else if ( i == du.cols - 1 )
                {
                    sum_dpsis = psmoothX[o - 1] + psmoothY[o - s] + psmoothY[o];
                    sigma_u = psmoothX[o - 1] * pdu[o - 1]
                            + psmoothY[o - s] * pdu[o - s]
                            + psmoothY[o] * pdu[o + s];
                    sigma_v = psmoothX[o - 1] * pdv[o - 1]
                            + psmoothY[o - s] * pdv[o - s]
                            + psmoothY[o] * pdv[o + s];
                } else
                {
                    sum_dpsis = psmoothX[o - 1] + psmoothX[o] + psmoothY[o - s]
                            + psmoothY[o];
                    sigma_u = psmoothX[o - 1] * pdu[o - 1]
                            + psmoothX[o] * pdu[o + 1]
                            + psmoothY[o - s] * pdu[o - s]
                            + psmoothY[o] * pdu[o + s];
                    sigma_v = psmoothX[o - 1] * pdv[o - 1]
                            + psmoothX[o] * pdv[o + 1]
                            + psmoothY[o - s] * pdv[o - s]
                            + psmoothY[o] * pdv[o + s];
                }
                A22 = pa11[o] + sum_dpsis;
                A11 = pa22[o] + sum_dpsis;
                A12 = -pa12[o];
                det = A11 * A22 - A12 * A12;
                A22 /= det;
                A11 /= det;
                A12 /= det;
                B1 = pb1[o] + sigma_u;
                B2 = pb2[o] + sigma_v;
                pdu[o] += omega * (A11 * B1 + A12 * B2 - pdu[o]);
                pdv[o] += omega * (A12 * B1 + A22 * B2 - pdv[o]);

            }
        }
    }
    merge(dWChannels, dW);
}
void OpticalFlowDeepFlow::collectGarbage()
{

}

CV_INIT_ALGORITHM(OpticalFlowDeepFlow, "DenseOpticalFlow.DeepFlow",
        obj.info()->addParam(obj, "sigma", obj.sigma, false, 0, 0, "Gaussian blur parameter"); obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0, "Smoothness assumption weight"); obj.info()->addParam(obj, "delta", obj.delta, false, 0, 0, "Color constancy weight"); obj.info()->addParam(obj, "gamma", obj.gamma, false, 0, 0, "Gradient constancy weight"); obj.info()->addParam(obj, "omega", obj.omega, false, 0, 0, "Relaxation factor in SOR"); obj.info()->addParam(obj, "minSize", obj.minSize, false, 0, 0, "Min. image size in the pyramid"); obj.info()->addParam(obj, "fixedPointIterations", obj.fixedPointIterations, false, 0, 0, "Fixed point iterations"); obj.info()->addParam(obj, "sorIterations", obj.sorIterations, false, 0, 0, "SOR iterations"); obj.info()->addParam(obj, "downscaleFactor", obj.downscaleFactor, false, 0, 0,"Downscale factor"))

} // namespace

Ptr<DenseOpticalFlow> cv::createOptFlow_DeepFlow()
{
    return makePtr<OpticalFlowDeepFlow>();
}
