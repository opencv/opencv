// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on file issued with the following license:

/*============================================================================

Copyright 2017 Toby Collins
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef OPENCV_CALIB3D_IPPE_HPP
#define OPENCV_CALIB3D_IPPE_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace IPPE {

class PoseSolver {
public:
    /**
     * @brief PoseSolver constructor
     */
    PoseSolver();

    /**
     * @brief                Finds the two possible poses of a planar object given a set of correspondences and their respective reprojection errors.
     *                       The poses are sorted with the first having the lowest reprojection error.
     * @param objectPoints   Array of 4 or more coplanar object points defined in object coordinates.
     *                       1xN/Nx1 3-channel (float or double) where N is the number of points
     * @param imagePoints    Array of corresponding image points, 1xN/Nx1 2-channel. Points are in normalized pixel coordinates.
     * @param rvec1          First rotation solution (3x1 rotation vector)
     * @param tvec1          First translation solution (3x1 vector)
     * @param reprojErr1     Reprojection error of first solution
     * @param rvec2          Second rotation solution (3x1 rotation vector)
     * @param tvec2          Second translation solution (3x1 vector)
     * @param reprojErr2     Reprojection error of second solution
     */
    void solveGeneric(InputArray objectPoints, InputArray imagePoints, OutputArray rvec1, OutputArray tvec1,
                      float& reprojErr1, OutputArray rvec2, OutputArray tvec2, float& reprojErr2);

    /**
     * @brief                   Finds the two possible poses of a square planar object and their respective reprojection errors using IPPE.
     *                          The poses are sorted so that the first one is the one with the lowest reprojection error.
     *
     * @param objectPoints      Array of 4 coplanar object points defined in the following object coordinates:
     *                            - point 0: [-squareLength / 2.0, squareLength / 2.0, 0]
     *                            - point 1: [squareLength / 2.0, squareLength / 2.0, 0]
     *                            - point 2: [squareLength / 2.0, -squareLength / 2.0, 0]
     *                            - point 3: [-squareLength / 2.0, -squareLength / 2.0, 0]
     *                          1xN/Nx1 3-channel (float or double) where N is the number of points
     * @param imagePoints       Array of corresponding image points, 1xN/Nx1 2-channel. Points are in normalized pixel coordinates.
     * @param rvec1             First rotation solution (3x1 rotation vector)
     * @param tvec1             First translation solution (3x1 vector)
     * @param reprojErr1        Reprojection error of first solution
     * @param rvec2             Second rotation solution (3x1 rotation vector)
     * @param tvec2             Second translation solution (3x1 vector)
     * @param reprojErr2        Reprojection error of second solution
     */
    void solveSquare(InputArray objectPoints, InputArray imagePoints, OutputArray rvec1, OutputArray tvec1,
                     float& reprojErr1, OutputArray rvec2, OutputArray tvec2, float& reprojErr2);

private:
    /**
     * @brief                         Finds the two possible poses of a planar object given a set of correspondences in normalized pixel coordinates.
     *                                These poses are **NOT** sorted on reprojection error. Note that the returned poses are object-to-camera transforms, and not camera-to-object transforms.
     * @param objectPoints            Array of 4 or more coplanar object points defined in object coordinates. 1xN/Nx1 3-channel (float or double).
     * @param normalizedImagePoints   Array of corresponding image points in normalized pixel coordinates, 1xN/Nx1 2-channel (float or double).
     * @param Ma                      First pose solution (unsorted)
     * @param Mb                      Second pose solution (unsorted)
     */
    void solveGeneric(InputArray objectPoints, InputArray normalizedImagePoints, OutputArray Ma, OutputArray Mb);

    /**
     * @brief                         Finds the two possible poses of a planar object in its canonical position, given a set of correspondences in normalized pixel coordinates.
     *                                These poses are **NOT** sorted on reprojection error. Note that the returned poses are object-to-camera transforms, and not camera-to-object transforms.
     * @param canonicalObjPoints      Array of 4 or more coplanar object points defined in object coordinates. 1xN/Nx1 3-channel (double) where N is the number of points
     * @param normalizedInputPoints   Array of corresponding image points in normalized pixel coordinates, 1xN/Nx1 2-channel (double) where N is the number of points
     * @param H                       Homography mapping canonicalObjPoints to normalizedInputPoints.
     * @param Ma
     * @param Mb
     */
    void solveCanonicalForm(InputArray canonicalObjPoints, InputArray normalizedInputPoints, const Matx33d& H,
                            OutputArray Ma, OutputArray Mb);

    /**
     * @brief                           Computes the translation solution for a given rotation solution
     * @param objectPoints              Array of corresponding object points, 1xN/Nx1 3-channel where N is the number of points
     * @param normalizedImagePoints     Array of corresponding image points (undistorted), 1xN/Nx1 2-channel where N is the number of points
     * @param R                         Rotation solution (3x1 rotation vector)
     * @param t                         Translation solution (3x1 rotation vector)
     */
    void computeTranslation(InputArray objectPoints, InputArray normalizedImgPoints, InputArray R, OutputArray t);

    /**
     * @brief                           Computes the two rotation solutions from the Jacobian of a homography matrix H at a point (ux,uy) on the object plane.
     *                                  For highest accuracy the Jacobian should be computed at the centroid of the point correspondences (see the IPPE paper for the explanation of this).
     *                                  For a point (ux,uy) on the object plane, suppose the homography H maps (ux,uy) to a point (p,q) in the image (in normalized pixel coordinates).
     *                                  The Jacobian matrix [J00, J01; J10,J11] is the Jacobian of the mapping evaluated at (ux,uy).
     * @param j00                       Homography jacobian coefficient at (ux,uy)
     * @param j01                       Homography jacobian coefficient at (ux,uy)
     * @param j10                       Homography jacobian coefficient at (ux,uy)
     * @param j11                       Homography jacobian coefficient at (ux,uy)
     * @param p                         The x coordinate of point (ux,uy) mapped into the image (undistorted and normalized position)
     * @param q                         The y coordinate of point (ux,uy) mapped into the image (undistorted and normalized position)
    */
    void computeRotations(double j00, double j01, double j10, double j11, double p, double q, OutputArray _R1, OutputArray _R2);

    /**
     * @brief                         Closed-form solution for the homography mapping with four corner correspondences of a square (it maps source points to target points).
     *                                The source points are the four corners of a zero-centred squared defined by:
     *                                  - point 0: [-squareLength / 2.0, squareLength / 2.0]
     *                                  - point 1: [squareLength / 2.0, squareLength / 2.0]
     *                                  - point 2: [squareLength / 2.0, -squareLength / 2.0]
     *                                  - point 3: [-squareLength / 2.0, -squareLength / 2.0]
     *
     * @param targetPoints            Array of four corresponding target points, 1x4/4x1 2-channel. Note that the points should be ordered to correspond with points 0, 1, 2 and 3.
     * @param halfLength              The square's half length (i.e. squareLength/2.0)
     * @param H                       Homograhy mapping the source points to the target points, 3x3 single channel
    */
    void homographyFromSquarePoints(InputArray targetPoints, double halfLength, OutputArray H);

    /**
     * @brief                  Fast conversion from a rotation matrix to a rotation vector using Rodrigues' formula
     * @param R                Input rotation matrix, 3x3 1-channel (double)
     * @param r                Output rotation vector, 3x1/1x3 1-channel (double)
     */
    void rot2vec(InputArray R, OutputArray r);

    /**
     * @brief                         Takes a set of planar object points and transforms them to 'canonical' object coordinates This is when they have zero mean and are on the plane z=0
     * @param objectPoints            Array of 4 or more coplanar object points defined in object coordinates. 1xN/Nx1 3-channel (float or double) where N is the number of points
     * @param canonicalObjectPoints   Object points in canonical coordinates 1xN/Nx1 2-channel (double)
     * @param MobjectPoints2Canonical Transform matrix mapping _objectPoints to _canonicalObjectPoints: 4x4 1-channel (double)
     */
    void makeCanonicalObjectPoints(InputArray objectPoints, OutputArray canonicalObjectPoints, OutputArray MobjectPoints2Canonical);

    /**
     * @brief                         Evaluates the Root Mean Squared (RMS) reprojection error of a pose solution.
     * @param objectPoints            Array of 4 or more coplanar object points defined in object coordinates. 1xN/Nx1 3-channel (float or double) where N is the number of points
     * @param imagePoints             Array of corresponding image points, 1xN/Nx1 2-channel. This can either be in pixel coordinates or normalized pixel coordinates.
     * @param M                       Pose matrix from 3D object to camera coordinates: 4x4 1-channel (double)
     * @param err                     RMS reprojection error
     */
    void evalReprojError(InputArray objectPoints, InputArray imagePoints, InputArray M, float& err);

    /**
     * @brief                         Sorts two pose solutions according to their RMS reprojection error (lowest first).
     * @param objectPoints            Array of 4 or more coplanar object points defined in object coordinates. 1xN/Nx1 3-channel (float or double) where N is the number of points
     * @param imagePoints             Array of corresponding image points, 1xN/Nx1 2-channel.  This can either be in pixel coordinates or normalized pixel coordinates.
     * @param Ma                      Pose matrix 1: 4x4 1-channel
     * @param Mb                      Pose matrix 2: 4x4 1-channel
     * @param M1                      Member of (Ma,Mb} with lowest RMS reprojection error. Performs deep copy.
     * @param M2                      Member of (Ma,Mb} with highest RMS reprojection error. Performs deep copy.
     * @param err1                    RMS reprojection error of _M1
     * @param err2                    RMS reprojection error of _M2
     */
    void sortPosesByReprojError(InputArray objectPoints, InputArray imagePoints, InputArray Ma, InputArray Mb, OutputArray M1, OutputArray M2, float& err1, float& err2);

    /**
     * @brief                         Finds the rotation _Ra that rotates a vector _a to the z axis (0,0,1)
     * @param a                       vector: 3x1 mat (double)
     * @param Ra                      Rotation: 3x3 mat (double)
     */
    void rotateVec2ZAxis(const Matx31d& a, Matx33d& Ra);

    /**
     * @brief                         Computes the rotation _R that rotates the object points to the plane z=0. This uses the cross-product method with the first three object points.
     * @param objectPoints            Array of N>=3 coplanar object points defined in object coordinates. 1xN/Nx1 3-channel (float or double) where N is the number of points
     * @param R                       Rotation Mat: 3x3 (double)
     * @return                        Success (true) or failure (false)
     */
    bool computeObjextSpaceR3Pts(InputArray objectPoints, Matx33d& R);

    /**
     * @brief computeObjextSpaceRSvD   Computes the rotation _R that rotates the object points to the plane z=0. This uses the cross-product method with the first three object points.
     * @param objectPointsZeroMean     Zero-meaned coplanar object points: 3xN matrix (double) where N>=3
     * @param R                        Rotation Mat: 3x3 (double)
     */
    void computeObjextSpaceRSvD(InputArray objectPointsZeroMean, OutputArray R);

    /**
     * @brief                   Generates the 4 object points of a square planar object
     * @param squareLength      The square's length (which is also it's width) in object coordinate units (e.g. millimeters, meters, etc.)
     * @param objectPoints      Set of 4 object points (1x4 3-channel double)
     */
    void generateSquareObjectCorners3D(double squareLength, OutputArray objectPoints);

    /**
     * @brief                   Generates the 4 object points of a square planar object, without including the z-component (which is z=0 for all points).
     * @param squareLength      The square's length (which is also it's width) in object coordinate units (e.g. millimeters, meters, etc.)
     * @param objectPoints      Set of 4 object points (1x4 2-channel double)
     */
    void generateSquareObjectCorners2D(double squareLength, OutputArray objectPoints);

    /**
     * @brief                   Computes the average depth of an object given its pose in camera coordinates
     * @param objectPoints:     Object points defined in 3D object space
     * @param rvec:             Rotation component of pose
     * @param tvec:             Translation component of pose
     * @return:                 average depth of the object
     */
    double meanSceneDepth(InputArray objectPoints, InputArray rvec, InputArray tvec);

    //! a small constant used to test 'small' values close to zero.
    double IPPE_SMALL;
};
} //namespace IPPE

namespace HomographyHO {

/**
* @brief                   Computes the best-fitting homography matrix from source to target points using Harker and O'Leary's method:
*                          Harker, M., O'Leary, P., Computation of Homographies, Proceedings of the British Machine Vision Conference 2005, Oxford, England.
*                          This is not the author's implementation.
* @param srcPoints         Array of source points: 1xN/Nx1 2-channel (float or double) where N is the number of points
* @param targPoints        Array of target points: 1xN/Nx1 2-channel (float or double)
* @param H                 Homography from source to target: 3x3 1-channel (double)
*/
void homographyHO(InputArray srcPoints, InputArray targPoints, Matx33d& H);

/**
* @brief                      Performs data normalization before homography estimation. For details see Hartley, R., Zisserman, A., Multiple View Geometry in Computer Vision,
*                             Cambridge University Press, Cambridge, 2001
* @param Data                 Array of source data points: 1xN/Nx1 2-channel (float or double) where N is the number of points
* @param DataN                Normalized data points: 1xN/Nx1 2-channel (float or double) where N is the number of points
* @param T                    Homogeneous transform from source to normalized: 3x3 1-channel (double)
* @param Ti                   Homogeneous transform from normalized to source: 3x3 1-channel (double)
*/
void normalizeDataIsotropic(InputArray Data, OutputArray DataN, OutputArray T, OutputArray Ti);

}
} //namespace cv
#endif
