Camera Calibration and 3D Reconstruction
========================================

.. highlight:: cpp



cuda::solvePnPRansac
--------------------
Finds the object pose from 3D-2D point correspondences.

.. ocv:function:: void cuda::solvePnPRansac(const Mat& object, const Mat& image, const Mat& camera_mat, const Mat& dist_coef, Mat& rvec, Mat& tvec, bool use_extrinsic_guess=false, int num_iters=100, float max_dist=8.0, int min_inlier_count=100, vector<int>* inliers=NULL)

    :param object: Single-row matrix of object points.

    :param image: Single-row matrix of image points.

    :param camera_mat: 3x3 matrix of intrinsic camera parameters.

    :param dist_coef: Distortion coefficients. See :ocv:func:`undistortPoints` for details.

    :param rvec: Output 3D rotation vector.

    :param tvec: Output 3D translation vector.

    :param use_extrinsic_guess: Flag to indicate that the function must use ``rvec`` and ``tvec`` as an initial transformation guess. It is not supported for now.

    :param num_iters: Maximum number of RANSAC iterations.

    :param max_dist: Euclidean distance threshold to detect whether point is inlier or not.

    :param min_inlier_count: Flag to indicate that the function must stop if greater or equal number of inliers is achieved. It is not supported for now.

    :param inliers: Output vector of inlier indices.

.. seealso:: :ocv:func:`solvePnPRansac`
