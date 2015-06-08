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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
#ifndef __OPENCV_OMNIDIR_HPP__
#define __OPENCV_OMNIDIR_HPP__
#ifdef __cplusplus
#include "precomp.hpp"

namespace cv
{
namespace omnidir
{
//! @addtogroup calib3d_omnidir
//! @{

    /** @brief Projects points for omnidirectional camera using CMei's model

    @param objectPoints Object points in world coordiante, 1xN/Nx1 3-channel of type CV_64F and N 
    is the number of points.
    @param imagePoints Output array of image points, 1xN/Nx1 2-channel of type CV_64F
    @param rvec vector of rotation between world coordinate and camera coordinate, i.e., om
    @param tvec vector of translation between pattern coordinate and camera coordinate
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param jacobian Optional output 2Nx16 of type CV_64F jacobian matrix, constains the derivatives of 
    image pixel points wrt parametes including \f$om, T, f_x, f_y, s, c_x, c_y, xi, k_1, k_2, p_1, p_2\f$.
    This matrix will be used in calibration by optimization.

    The function projects object 3D points of world coordiante to image pixels, parametered by intrinsic 
    and extrinsic parameters. Also, it optionaly compute a by-product: the jacobian matrix containing 
    onstains the derivatives of image pixel points wrt intrinsic and extrinsic parametes.
     */
    CV_EXPORTS_W void projectPoints(InputArray objectPoints, OutputArray imagePoints, InputArray rvec, InputArray tvec, 
        InputArray K, InputArray D, double xi, OutputArray jacobian = noArray());

    /** @brief Undistort 2D image points for omnidirectional camera using CMei's model

    @param Array of distorted image points, 1xN/Nx1 2-channel of tyep CV_64F 
    
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param R Rotation trainsform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param undistorted array of normalized object points, 1xN/Nx1 2-channel of type CV_64F
     */
    CV_EXPORTS_W void undistortPoints(InputArray distorted, OutputArray undistorted, InputArray K, InputArray D, 
        double xi, InputArray R);

    /** @brief Distorts 2D object points to image points, similar to projectPoints

    @param undistorted Array of undistorted object points, 1xN/Nx1 2-channel with type CV_64F
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param distorted Array of distorted image points of tyep CV_64F
     */
    CV_EXPORTS_W void distortPoints(InputArray undistorted, OutputArray distorted, InputArray K, InputArray D, double xi);

    /** @brief Computes undistortion and rectification maps for omnidirectional camera image transform by cv::remap(). 
    If D is empty zero distortion is used, if R or P is empty identity matrixes are used.

    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param R Rotation trainsform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
    for details.
    @param map1 The first output map.
    @param map2 The second output map.
     */
    CV_EXPORTS_W void initUndistortRectifyMap(InputArray K, InputArray D, double xi, InputArray R, InputArray P, 
        const cv::Size& size, int mltype, OutputArray map1, OutputArray map2);

    /** @brief Undistort omnidirectional images to perspective images

    @param distorted omnidirectional image with very large distortion
    @param undistorted The output undistorted image 
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param Knew Camera matrix of the distorted image. By default, it is just K.
    @param new_size The new image size. By default, it is the size of distorted.
    */
    CV_EXPORTS_W void undistortImage(InputArray distorted, OutputArray undistorted,
        InputArray K, InputArray D, double xi, InputArray Knew = cv::noArray(), const Size& new_size = Size());

    /** @brief Perform omnidirectional camera calibration

    @param objectPoints Vector of vector of pattern points in world (pattern) coordiante, 1xN/Nx1 3-channel
    @param imagePoints Vector of vector of correspoinding image points of objectPoints
    @param size Image size of calibration images.
    @param K Output calibrated camera matrix. If you want to initialize K by yourself, input a non-empty K.
    @param xi Ouput parameter xi for CMei's model
    @param D Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$
    @param omAll Output rotations for each calibration images
    @param tAll Output translation for each calibration images
    @param flags The flags of some features that will added
    @param criteria Termination criteria for optimization
    */
    CV_EXPORTS_W double calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size size, 
        InputOutputArray K, double& xi, InputOutputArray D, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll,
        int flags, TermCriteria criteria);

    /** @brief Stereo calibration for omnidirectional camera model. It computes the intrinsic parameters for two 
    cameras and the extrinsic parameters between two cameras
    
    @param objectPoints Vector of vector of pattern points in world (pattern) coordiante, 1xN/Nx1 3-channel
    @param imagePoints1 Vector of vector of correspoinding image points of the first camera
    @param imagePoints2 Vector of vector of correspoinding image points of the second camera
    @param imageSize Image size of calibration images.
    @param K1 Output calibrated camera matrix. If you want to initialize K1 by yourself, input a non-empty K1.
    @param xi1 Ouput parameter xi for the first camera for CMei's model
    @param D1 Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the first camera
    @param K2 Output calibrated camera matrix. If you want to initialize K2 by yourself, input a non-empty K2.
    @param xi2 Ouput parameter xi for the second camera for CMei's model
    @param D2 Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the second camera
    @param R Output rotation between the first and second camera
    @param T Output translation between the first and second camera
    @param flags The flags of some features that will added
    @param criteria Termination criteria for optimization
    @
    */
    CV_EXPORTS_W double stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, 
        Size imageSize, InputOutputArray K1, double& xi1, InputOutputArray D1, InputOutputArray K2, double& xi2, 
        InputOutputArray D2, OutputArray R, OutputArray T, int flags, TermCriteria criteria);

    /** @brief Stereo rectification for omnidirectional camera model. It computes the rectification rotations for two cameras
    
    @param K1 Input camera matrix of the first camera
    @param D1 Input distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the first camera
    @param xi1 Input parameter xi for the first camera for CMei's model
    @param K2 Input camera matrix of the second camera
    @param D2 Input distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the second camera
    @param xi2 Input parameter xi for the second camera for CMei's model
    @param imageSize Image size of calibration images.
    @param R Rotation between the first and second camera
    @param tvec Translation between the first and second camera
    @param R1 Output 3x3 rotation matrix for the first camera
    @param R2 Output 3x3 rotation matrix for the second camera
    @param P1 Output 3x4 projection matrix in the rectified coordinate systems for the first
    camera.
    @param P2 Output 3x4 projection matrix in the rectified coordinate systems for the second
    camera.
    @param Q Output 4x4 disparity-to-depth mapping matrix
    @param newImageSize New image size of rectified images. When it is (0,0), the new image size is 
    equivalent to imageSize
    */
    CV_EXPORTS_W void stereoRectify(InputArray K1, InputArray D1, double xi1, InputArray K2, InputArray D2, double xi2, const Size imageSize,
        InputArray R, InputArray tvec, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags,
        const Size& newImageSize);


    
//! @} calib3d_omnidir

namespace internal
{
    void initializeCalibration(InputOutputArrayOfArrays objectPoints, InputOutputArrayOfArrays imagePoints, Size size, 
        OutputArrayOfArrays omAll, OutputArrayOfArrays tAll, OutputArray K, double& xi);

    void computeJacobian(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters,
        Mat& JTJ_inv, Mat& JTE);

    void encodeParameters(InputArray K, InputArrayOfArrays omAll, InputArrayOfArrays tAll, InputArray distortion,
        double xi, OutputArray parameters);

    void decodeParameters(InputArray paramsters, OutputArray K, OutputArrayOfArrays omAll, OutputArrayOfArrays tAll,
        OutputArray distortion, double& xi);

    void estimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, InputArray parameters,
        Vec2d& std_error, double& rms);

    double computeMeanReproErr(InputArrayOfArrays imagePoints, InputArrayOfArrays proImagePoints);

    double median(InputArray row);

} // internal
} // omnidir
} //cv
#endif
#endif