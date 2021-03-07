// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on file issued with the following license:

/*
BSD 3-Clause License

Copyright (c) 2020, George Terzakis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef OPENCV_CALIB3D_SQPNP_HPP
#define OPENCV_CALIB3D_SQPNP_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace sqpnp {


class PoseSolver {
public:
    /**
    * @brief PoseSolver constructor
    */
    PoseSolver();

    /**
    * @brief                Finds the possible poses of a camera given a set of 3D points
    *                       and their corresponding 2D image projections. The poses are
    *                       sorted by lowest squared error (which corresponds to lowest
    *                       reprojection error).
    * @param objectPoints   Array or vector of 3 or more 3D points defined in object coordinates.
    *                       1xN/Nx1 3-channel (float or double) where N is the number of points.
    * @param imagePoints    Array or vector of corresponding 2D points, 1xN/Nx1 2-channel.
    * @param rvec           The output rotation solutions (up to 18 3x1 rotation vectors)
    * @param tvec           The output translation solutions (up to 18 3x1 vectors)
    */
    void solve(InputArray objectPoints, InputArray imagePoints, OutputArrayOfArrays rvec,
        OutputArrayOfArrays tvec);

private:
    struct SQPSolution
    {
        cv::Matx<double, 9, 1> r_hat;
        cv::Matx<double, 3, 1> t;
        double sq_error;
        SQPSolution() : sq_error(0) {}
    };

    /*
    * @brief                Computes the 9x9 PSD Omega matrix and supporting matrices.
    * @param objectPoints   Array or vector of 3 or more 3D points defined in object coordinates.
    *                       1xN/Nx1 3-channel (float or double) where N is the number of points.
    * @param imagePoints    Array or vector of corresponding 2D points, 1xN/Nx1 2-channel.
    */
    void computeOmega(InputArray objectPoints, InputArray imagePoints);

    /*
    * @brief                Computes the 9x9 PSD Omega matrix and supporting matrices.
    */
    void solveInternal();

    /*
    * @brief                Produces the distance from being orthogonal for a given 3x3 matrix
    *                       in row-major form.
    * @param e              The vector to test representing a 3x3 matrix in row major form.
    * @return               The distance the matrix is from being orthogonal.
    */
    static double orthogonalityError(const cv::Matx<double, 9, 1>& e);

    /*
    * @brief                Processes a solution and sorts it by error.
    * @param solution       The solution to evaluate.
    * @param min_error          The current minimum error.
    */
    void checkSolution(SQPSolution& solution, double& min_error);

    /*
    * @brief                Computes the determinant of a matrix stored in row-major format.
    * @param e              Vector representing a 3x3 matrix stored in row-major format.
    * @return               The determinant of the matrix.
    */
    static double det3x3(const cv::Matx<double, 9, 1>& e);

    /*
    * @brief                Tests the cheirality for a given solution.
    * @param solution       The solution to evaluate.
    */
    inline bool positiveDepth(const SQPSolution& solution) const;

    /*
    * @brief                Determines the nearest rotation matrix to a given rotaiton matrix.
    *                       Input and output are 9x1 vector representing a vector stored in row-major
    *                       form.
    * @param e              The input 3x3 matrix stored in a vector in row-major form.
    * @param r              The nearest rotation matrix to the input e (again in row-major form).
    */
    static void nearestRotationMatrix(const cv::Matx<double, 9, 1>& e,
        cv::Matx<double, 9, 1>& r);

    /*
    * @brief                Runs the sequential quadratic programming on orthogonal matrices.
    * @param r0             The start point of the solver.
    */
    SQPSolution runSQP(const cv::Matx<double, 9, 1>& r0);

    /*
    * @brief                Steps down the gradient for the given matrix r to solve the SQP system.
    * @param r              The current matrix step.
    * @param delta          The next step down the gradient.
    */
    void solveSQPSystem(const cv::Matx<double, 9, 1>& r, cv::Matx<double, 9, 1>& delta);

    /*
    * @brief                Analytically computes the inverse of a symmetric 3x3 matrix using the
    *                       lower triangle.
    * @param Q              The matrix to invert.
    * @param Qinv           The inverse of Q.
    * @param threshold      The threshold to determine if Q is singular and non-invertible.
    */
    bool analyticalInverse3x3Symm(const cv::Matx<double, 3, 3>& Q,
        cv::Matx<double, 3, 3>& Qinv,
        const double& threshold = 1e-8);

    /*
    * @brief                Computes the 3D null space and 6D normal space of the constraint Jacobian
    *                       at a 9D vector r (representing a rank-3 matrix). Note that K is lower
    *                       triangular so upper triangle is undefined.
    * @param r              9D vector representing a rank-3 matrix.
    * @param H              6D row space of the constraint Jacobian at r.
    * @param N              3D null space of the constraint Jacobian at r.
    * @param K              The constraint Jacobian at r.
    * @param norm_threshold Threshold for column vector norm of Pn (the projection onto the null space
    *                       of the constraint Jacobian).
    */
    void computeRowAndNullspace(const cv::Matx<double, 9, 1>& r,
        cv::Matx<double, 9, 6>& H,
        cv::Matx<double, 9, 3>& N,
        cv::Matx<double, 6, 6>& K,
        const double& norm_threshold = 0.1);

    static const double RANK_TOLERANCE;
    static const double SQP_SQUARED_TOLERANCE;
    static const double SQP_DET_THRESHOLD;
    static const double ORTHOGONALITY_SQUARED_ERROR_THRESHOLD;
    static const double EQUAL_VECTORS_SQUARED_DIFF;
    static const double EQUAL_SQUARED_ERRORS_DIFF;
    static const double POINT_VARIANCE_THRESHOLD;
    static const int SQP_MAX_ITERATION;
    static const double SQRT3;

    cv::Matx<double, 9, 9> omega_;
    cv::Vec<double, 9> s_;
    cv::Matx<double, 9, 9> u_;
    cv::Matx<double, 3, 9> p_;
    cv::Vec3d point_mean_;
    int num_null_vectors_;

    SQPSolution solutions_[18];
    int num_solutions_;

};

}
}

#endif
