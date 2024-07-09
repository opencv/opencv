// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2019 Czech Technical University.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of Czech Technical University nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Daniel Barath (barath.daniel@sztaki.mta.hu)
// Modification: Maksym Ivashechkin (ivashmak@cmp.felk.cvut.cz)

#include "../precomp.hpp"
#include "../usac.hpp"
#if defined(HAVE_EIGEN)
#include <Eigen/Eigen>
#elif defined(HAVE_LAPACK)
#include "opencv_lapack.h"
#endif

namespace cv { namespace usac {
class DLSPnPImpl : public DLSPnP {
#if defined(HAVE_LAPACK) || defined(HAVE_EIGEN)
private:
    Mat points_mat, calib_norm_points_mat;
    const Matx33d K;
public:
    explicit DLSPnPImpl (const Mat &points_, const Mat &calib_norm_points_, const Mat &K_)
        : points_mat(points_), calib_norm_points_mat(calib_norm_points_), K(K_)
    {
        CV_DbgAssert(!points_mat.empty() && points_mat.isContinuous());
        CV_DbgAssert(!calib_norm_points_mat.empty() && calib_norm_points_mat.isContinuous());
    }
#else
public:
    explicit DLSPnPImpl (const Mat &, const Mat &, const Mat &) {}
#endif

    // return minimal sample size required for non-minimal estimation.
    int getMinimumRequiredSampleSize() const override { return 3; }
    // return maximum number of possible solutions.
    int getMaxNumberOfSolutions () const override { return 27; }
#if defined(HAVE_LAPACK) || defined(HAVE_EIGEN)
    int estimate(const std::vector<int> &sample, int sample_number,
        std::vector<Mat> &models_, const std::vector<double> &/*weights_*/) const override {
        if (sample_number < getMinimumRequiredSampleSize())
            return 0;

        // Estimate the model parameters from the given point sample
        // using weighted fitting if possible.

        // Holds the normalized feature positions cross multiplied with itself
        // i.e. n * n^t. This value is used multiple times so it is efficient to
        // pre-compute it.
        std::vector<Matx33d> normalized_feature_cross(sample_number);
        std::vector<Vec3d> world_points(sample_number);
        const Matx33d eye = Matx33d::eye();

        // The bottom-right symmetric block matrix of inverse(A^T * A). Matrix H from
        // Eq. 25 in the Appendix of the DLS paper.
        Matx33d h_inverse = sample_number * eye;

        // Compute V*W*b with the rotation parameters factored out. This is the
        // translation parameterized by the 9 entries of the rotation matrix.
        Matx<double, 3, 9> translation_factor = Matx<double, 3, 9>::zeros();
        const float * points = points_mat.ptr<float>();
        const float * calib_norm_points = calib_norm_points_mat.ptr<float>();
        for (int i = 0; i < sample_number; i++) {
            const int idx_world = 5 * sample[i], idx_calib = 3 * sample[i];
            Vec3d normalized_feature_pos(calib_norm_points[idx_calib],
                                         calib_norm_points[idx_calib+1],
                                         calib_norm_points[idx_calib+2]);
            normalized_feature_cross[i] = normalized_feature_pos * normalized_feature_pos.t();
            world_points[i] = Vec3d(points[idx_world + 2], points[idx_world + 3], points[idx_world + 4]);

            h_inverse -= normalized_feature_cross[i];
            translation_factor += (normalized_feature_cross[i] - eye) * leftMultiplyMatrix(world_points[i]);
        }

        const Matx33d h_matrix = h_inverse.inv();
        translation_factor = h_matrix * translation_factor;

        // Compute the cost function J' of Eq. 17 in DLS paper. This is a factorized
        // version where the rotation matrix parameters have been pulled out. The
        // entries to this equation are the coefficients to the cost function which is
        // a quartic in the rotation parameters.
        Matx<double, 9, 9> ls_cost_coefficients = Matx<double, 9, 9>::zeros();
        for (int i = 0; i < sample_number; i++)
            ls_cost_coefficients +=
                    (leftMultiplyMatrix(world_points[i]) + translation_factor).t() *
                    (eye - normalized_feature_cross[i]) *
                    (leftMultiplyMatrix(world_points[i]) + translation_factor);

        // Extract the coefficients of the jacobian (Eq. 18) from the
        // ls_cost_coefficients matrix. The jacobian represent 3 monomials in the
        // rotation parameters. Each entry of the jacobian will be 0 at the roots of
        // the polynomial, so we can arrange a system of polynomials from these
        // equations.
        double f1_coeff[20], f2_coeff[20], f3_coeff[20];
        extractJacobianCoefficients(ls_cost_coefficients.val, f1_coeff, f2_coeff, f3_coeff);

        // We create one equation with random terms that is generally non-zero at the
        // roots of our system.
        RNG rng;
        const double macaulay_term[4] = { 100 * rng.uniform(-1.,1.), 100 * rng.uniform(-1.,1.),
                                          100 * rng.uniform(-1.,1.), 100 * rng.uniform(-1.,1.) };

        // Create Macaulay matrix that will be used to solve our polynonomial system.
        Mat macaulay_matrix = Mat_<double>::zeros(120, 120);
        createMacaulayMatrix(f1_coeff, f2_coeff, f3_coeff, macaulay_term, (double*)macaulay_matrix.data);

        // Via the Schur complement trick, the top-left of the Macaulay matrix
        // contains a multiplication matrix whose eigenvectors correspond to solutions
        // to our system of equations.
        Mat sol;
        if (!solve(macaulay_matrix.colRange(27, 120).rowRange(27, 120),
                   macaulay_matrix.colRange(0 ,  27).rowRange(27, 120), sol, DECOMP_LU))
            return 0;

        const Mat solution_polynomial = macaulay_matrix.colRange(0,27).rowRange(0,27) -
                (macaulay_matrix.colRange(27,120).rowRange(0,27) * sol);

        // Extract eigenvectors of the solution polynomial to obtain the roots which
        // are contained in the entries of the eigenvectors.
#ifdef HAVE_EIGEN
        Eigen::Map<Eigen::Matrix<double, 27, 27>> sol_poly((double*)solution_polynomial.data);
        const Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(sol_poly);
        const auto &eigen_vectors = eigen_solver.eigenvectors();
        const auto &eigen_values = eigen_solver.eigenvalues();
#else

#if defined (ACCELERATE_NEW_LAPACK) && defined (ACCELERATE_LAPACK_ILP64)
        long mat_order = 27, info, lda = 27, ldvl = 1, ldvr = 27, lwork = 500;
#else
        int mat_order = 27, info, lda = 27, ldvl = 1, ldvr = 27, lwork = 500;
#endif
        double wr[27], wi[27] = {0}; // 27 = mat_order
        std::vector<double> work(lwork), eig_vecs(729);
        char jobvl = 'N', jobvr = 'V'; // only left eigen vectors are computed
        OCV_LAPACK_FUNC(dgeev)(&jobvl, &jobvr, &mat_order, (double*)solution_polynomial.data, &lda, wr, wi, nullptr, &ldvl,
               &eig_vecs[0], &ldvr, &work[0], &lwork, &info);
        if (info != 0) return 0;
#endif
        models_ = std::vector<Mat>(); models_.reserve(3);
        const int max_pts_to_eval = std::min(sample_number, 100);
        std::vector<int> pts_random_shuffle(sample_number);
        for (int i = 0; i < sample_number; i++)
            pts_random_shuffle[i] = i;
        randShuffle(pts_random_shuffle);
        for (int i = 0; i < 27; i++) {
            // If the rotation solutions are real, treat this as a valid candidate
            // rotation.
            // The first entry of the eigenvector should equal 1 according to our
            // polynomial, so we must divide each solution by the first entry.

#ifdef HAVE_EIGEN
            if (eigen_values(i).imag() != 0)
                continue;
            const double eigen_vec_1i = 1 / eigen_vectors(0, i).real();
            const double s1 = eigen_vectors(9, i).real() * eigen_vec_1i,
                         s2 = eigen_vectors(3, i).real() * eigen_vec_1i,
                         s3 = eigen_vectors(1, i).real() * eigen_vec_1i;
#else
            if (wi[i] != 0)
                continue;
            const double eigen_vec_1i = 1 / eig_vecs[mat_order*i];
            const double s1 = eig_vecs[mat_order*i+9] * eigen_vec_1i,
                         s2 = eig_vecs[mat_order*i+3] * eigen_vec_1i,
                         s3 = eig_vecs[mat_order*i+1] * eigen_vec_1i;
#endif
            // Compute the rotation (which is the transpose rotation of our solution)
            // and translation.
            const double qi = s1, qi2 = qi*qi, qj = s2, qj2 = qj*qj, qk = s3, qk2 = qk*qk;
            const double s = 1 / (1 + qi2 + qj2 + qk2);
            const Matx33d rot_mat (1-2*s*(qj2+qk2), 2*s*(qi*qj+qk), 2*s*(qi*qk-qj),
                                   2*s*(qi*qj-qk), 1-2*s*(qi2+qk2), 2*s*(qj*qk+qi),
                                   2*s*(qi*qk+qj), 2*s*(qj*qk-qi), 1-2*s*(qi2+qj2));
            const Matx31d soln_translation = translation_factor * rot_mat.reshape<9,1>();

            // Check that all points are in front of the camera. Discard the solution
            // if this is not the case.
            bool all_points_in_front_of_camera = true;
            const Vec3d r3 (rot_mat(2,0),rot_mat(2,1),rot_mat(2,2));
            const double z = soln_translation(2);
            for (int pt = 0; pt < max_pts_to_eval; pt++) {
                if (r3.dot(world_points[pts_random_shuffle[pt]]) + z < 0) {
                    all_points_in_front_of_camera = false;
                    break;
                }
            }

            if (all_points_in_front_of_camera) {
                Mat model;
                hconcat(Math::rotVec2RotMat(Math::rotMat2RotVec(rot_mat)), soln_translation, model);
                models_.emplace_back(K * model);
            }
        }
        return static_cast<int>(models_.size());
#else
    int estimate(const std::vector<int> &/*sample*/, int /*sample_number*/,
        std::vector<Mat> &/*models_*/, const std::vector<double> &/*weights_*/) const override {
        return 0;
#endif
    }

    int estimate (const std::vector<bool> &/*mask*/, std::vector<Mat> &/*models*/,
            const std::vector<double> &/*weights*/) override {
        return 0;
    }
    void enforceRankConstraint (bool /*enforce*/) override {}

protected:
#if defined(HAVE_LAPACK) || defined(HAVE_EIGEN)
    const int indices[1968] = {
            0, 35, 83, 118, 120, 121, 154, 155, 174, 203, 219, 238, 241, 242, 274, 275,
            291, 294, 305, 323, 329, 339, 358, 360, 363, 395, 409, 436, 443, 478, 479,
            481, 483, 484, 514, 515, 523, 529, 534, 551, 556, 563, 579, 580, 598, 599,
            602, 604, 605, 634, 635, 641, 643, 649, 651, 654, 662, 665, 671, 676, 683,
            689, 699, 700, 711, 718, 719, 723, 726, 750, 755, 769, 795, 796, 803, 827,
            838, 839, 844, 846, 847, 870, 874, 875, 883, 885, 889, 894, 903, 911, 915,
            916, 923, 939, 940, 947, 952, 958, 959, 965, 967, 968, 990, 994, 1001, 1003,
            1005, 1006, 1009, 1011, 1014, 1022, 1023, 1025, 1026, 1031, 1035, 1036,
            1049, 1059, 1060, 1062, 1067, 1071, 1072, 1079, 1080, 1089, 1115, 1116,
            1163, 1164, 1168, 1198, 1201, 1209, 1210, 1233, 1234, 1235, 1236, 1254,
            1259, 1283, 1284, 1288, 1299, 1317, 1318, 1322, 1330, 1331, 1348, 1353,
            1354, 1355, 1356, 1371, 1374, 1377, 1379, 1385, 1403, 1404, 1408, 1409,
            1419, 1434, 1437, 1438, 1443, 1449, 1452, 1475, 1476, 1479, 1489, 1516,
            1519, 1523, 1524, 1528, 1536, 1558, 1559, 1564, 1570, 1572, 1573, 1593,
            1594, 1595, 1596, 1599, 1603, 1607, 1609, 1614, 1619, 1620, 1631, 1636,
            1639, 1643, 1644, 1648, 1650, 1656, 1659, 1660, 1677, 1678, 1679, 1685,
            1691, 1693, 1694, 1708, 1713, 1714, 1716, 1719, 1721, 1722, 1723, 1727,
            1729, 1731, 1734, 1736, 1737, 1739, 1740, 1742, 1745, 1751, 1756, 1759,
            1764, 1768, 1769, 1770, 1776, 1779, 1780, 1786, 1791, 1794, 1797, 1799,
            1806, 1812, 1815, 1829, 1830, 1835, 1836, 1839, 1849, 1874, 1875, 1876,
            1879, 1883, 1884, 1888, 1894, 1896, 1907, 1918, 1919, 1927, 1933, 1935,
            1936, 1949, 1950, 1953, 1954, 1956, 1959, 1963, 1964, 1965, 1967, 1969,
            1974, 1979, 1980, 1983, 1988, 1991, 1994, 1995, 1996, 1999, 2004, 2008,
            2010, 2014, 2016, 2017, 2019, 2020, 2027, 2032, 2037, 2039, 2048, 2054,
            2056, 2057, 2068, 2069, 2070, 2073, 2079, 2081, 2082, 2083, 2084, 2085,
            2086, 2087, 2091, 2096, 2097, 2099, 2100, 2102, 2103, 2105, 2106, 2108,
            2111, 2114, 2115, 2119, 2129, 2130, 2134, 2136, 2137, 2140, 2142, 2146,
            2147, 2151, 2152, 2154, 2157, 2169, 2178, 2195, 2196, 2213, 2242, 2243,
            2244, 2247, 2248, 2278, 2290, 2298, 2299, 2312, 2313, 2314, 2315, 2316,
            2333, 2334, 2339, 2341, 2362, 2363, 2364, 2367, 2368, 2379, 2396, 2397,
            2398, 2411, 2419, 2420, 2427, 2428, 2432, 2433, 2434, 2436, 2451, 2453,
            2454, 2455, 2457, 2459, 2461, 2465, 2482, 2484, 2487, 2488, 2489, 2499,
            2513, 2514, 2516, 2517, 2532, 2538, 2541, 2555, 2556, 2558, 2559, 2569,
            2573, 2596, 2598, 2599, 2602, 2603, 2604, 2607, 2608, 2612, 2616, 2638,
            2639, 2653, 2659, 2661, 2662, 2672, 2673, 2674, 2676, 2678, 2679, 2680,
            2683, 2687, 2689, 2693, 2694, 2699, 2700, 2701, 2711, 2712, 2716, 2718,
            2719, 2722, 2724, 2727, 2728, 2730, 2732, 2735, 2736, 2739, 2740, 2756,
            2757, 2759, 2774, 2780, 2782, 2783, 2787, 2788, 2792, 2793, 2798, 2799,
            2800, 2801, 2802, 2803, 2807, 2811, 2813, 2815, 2816, 2817, 2819, 2820,
            2821, 2822, 2825, 2831, 2832, 2838, 2839, 2842, 2847, 2849, 2850, 2852,
            2855, 2856, 2860, 2866, 2871, 2873, 2874, 2876, 2877, 2895, 2901, 2904,
            2909, 2910, 2916, 2918, 2919, 2929, 2932, 2933, 2953, 2954, 2955, 2956,
            2958, 2959, 2962, 2964, 2967, 2968, 2972, 2973, 2974, 2976, 2987, 2999,
            3016, 3022, 3024, 3025, 3029, 3030, 3032, 3033, 3038, 3039, 3040, 3043,
            3044, 3045, 3047, 3052, 3053, 3059, 3060, 3061, 3063, 3068, 3071, 3072,
            3073, 3074, 3075, 3078, 3079, 3082, 3087, 3090, 3092, 3093, 3094, 3095,
            3096, 3097, 3100, 3107, 3112, 3116, 3117, 3137, 3143, 3145, 3146, 3147,
            3148, 3149, 3152, 3158, 3160, 3161, 3162, 3164, 3165, 3166, 3167, 3172,
            3175, 3176, 3177, 3180, 3181, 3182, 3183, 3186, 3188, 3192, 3193, 3194,
            3198, 3210, 3212, 3213, 3214, 3215, 3217, 3222, 3226, 3231, 3232, 3233,
            3234, 3236, 3255, 3269, 3270, 3276, 3279, 3289, 3309, 3310, 3314, 3315,
            3316, 3319, 3324, 3328, 3331, 3334, 3336, 3347, 3350, 3359, 3366, 3390,
            3395, 3409, 3429, 3435, 3436, 3443, 3467, 3470, 3478, 3479, 3504, 3509,
            3510, 3518, 3519, 3532, 3533, 3549, 3550, 3553, 3554, 3555, 3558, 3559,
            3562, 3567, 3571, 3572, 3573, 3574, 3576, 3587, 3590, 3637, 3648, 3652,
            3670, 3673, 3677, 3681, 3685, 3691, 3693, 3698, 3749, 3757, 3758, 3770,
            3772, 3789, 3790, 3793, 3794, 3797, 3798, 3800, 3806, 3811, 3812, 3813,
            3814, 3818, 3830, 3888, 3890, 3893, 3920, 3921, 3922, 3925, 3926, 3927,
            3989, 3990, 3999, 4024, 4029, 4030, 4034, 4035, 4039, 4051, 4054, 4056,
            4063, 4067, 4070, 4109, 4118, 4132, 4144, 4149, 4150, 4153, 4154, 4158,
            4171, 4172, 4173, 4174, 4183, 4190, 4237, 4252, 4264, 4270, 4273, 4277,
            4291, 4293, 4298, 4303, 4325, 4354, 4361, 4363, 4369, 4371, 4374, 4382,
            4385, 4391, 4396, 4409, 4419, 4420, 4421, 4429, 4431, 4439, 4442, 4474,
            4475, 4491, 4494, 4505, 4523, 4529, 4539, 4549, 4558, 4590, 4609, 4624,
            4629, 4635, 4636, 4663, 4667, 4670, 4679, 4708, 4713, 4731, 4737, 4739,
            4745, 4769, 4785, 4788, 4789, 4794, 4797, 4827, 4828, 4832, 4855, 4857,
            4861, 4905, 4908, 4909, 4913, 4914, 4916, 4950, 4984, 4989, 4995, 5023,
            5027, 5030, 5067, 5071, 5095, 5098, 5145, 5148, 5153, 5155, 5189, 5224,
            5229, 5230, 5234, 5251, 5254, 5263, 5270, 5308, 5337, 5385, 5388, 5389,
            5394, 5427, 5455, 5505, 5508, 5513, 5572, 5584, 5590, 5593, 5611, 5613,
            5623, 5680, 5684, 5692, 5704, 5707, 5708, 5710, 5712, 5713, 5731, 5733,
            5735, 5737, 5743, 5744, 5790, 5803, 5805, 5823, 5824, 5827, 5829, 5831,
            5835, 5860, 5863, 5864, 5867, 5870, 5872, 5921, 5925, 5926, 5942, 5943,
            5946, 5981, 5982, 5985, 5989, 5991, 5992, 6041, 6062, 6101, 6105, 6109,
            6111, 6184, 6190, 6211, 6223, 6281, 6285, 6286, 6302, 6303, 6306, 6307,
            6309, 6341, 6342, 6344, 6349, 6350, 6351, 6352, 6424, 6429, 6463, 6470,
            6585, 6589, 6644, 6664, 6667, 6668, 6670, 6691, 6697, 6703, 6704, 6825,
            6828, 6904, 6907, 6943, 6944, 7006, 7024, 7026, 7027, 7062, 7063, 7064,
            7088, 7110, 7121, 7123, 7125, 7126, 7131, 7142, 7143, 7145, 7146, 7151,
            7155, 7169, 7180, 7181, 7182, 7187, 7189, 7191, 7192, 7208, 7230, 7241,
            7243, 7245, 7246, 7251, 7262, 7263, 7265, 7266, 7267, 7269, 7271, 7275,
            7289, 7300, 7302, 7304, 7307, 7310, 7311, 7312, 7362, 7376, 7421, 7425,
            7426, 7428, 7504, 7543, 7665, 7726, 7746, 7747, 7781, 7782, 7784, 7785,
            7846, 7864, 7866, 7867, 7901, 7902, 7903, 7904, 7966, 7986, 8021, 8022,
            8025, 8141, 8145, 8201, 8203, 8211, 8222, 8225, 8231, 8249, 8260, 8261,
            8265, 8269, 8271, 8317, 8328, 8332, 8353, 8357, 8361, 8365, 8373, 8378,
            8420, 8427, 8428, 8431, 8432, 8433, 8450, 8451, 8453, 8455, 8457, 8458,
            8459, 8461, 8465, 8480, 8482, 8486, 8487, 8489, 8513, 8514, 8515, 8516,
            8517, 8565, 8583, 8584, 8587, 8589, 8623, 8624, 8630, 8632, 8681, 8685,
            8686, 8702, 8703, 8704, 8706, 8707, 8709, 8742, 8743, 8744, 8750, 8751,
            8752, 8808, 8810, 8840, 8841, 8845, 8846, 8905, 8909, 8912, 8918, 8920,
            8924, 8925, 8927, 8932, 8940, 8941, 8943, 8947, 8948, 8949, 8950, 8952,
            8953, 8954, 8958, 8970, 8971, 8972, 8973, 8974, 8975, 8977, 8984, 8990,
            8992, 8996, 9021, 9036, 9037, 9038, 9039, 9049, 9050, 9053, 9076, 9077,
            9078, 9079, 9080, 9082, 9084, 9086, 9087, 9088, 9092, 9096, 9098, 9119,
            9168, 9201, 9205, 9274, 9291, 9294, 9305, 9329, 9339, 9345, 9349, 9387,
            9391, 9397, 9400, 9402, 9415, 9416, 9418, 9432, 9437, 9455, 9458, 9461,
            9466, 9468, 9473, 9475, 9522, 9524, 9526, 9536, 9546, 9548, 9577, 9581,
            9582, 9585, 9586, 9588, 9614, 9628, 9633, 9639, 9641, 9642, 9643, 9647,
            9651, 9656, 9657, 9659, 9660, 9662, 9665, 9671, 9679, 9689, 9690, 9696,
            9700, 9701, 9706, 9708, 9709, 9711, 9714, 9717, 9751, 9752, 9757, 9758,
            9760, 9767, 9768, 9770, 9778, 9780, 9781, 9792, 9797, 9798, 9800, 9801,
            9805, 9806, 9810, 9812, 9815, 9818, 9835, 9836, 9869, 9884, 9885, 9887,
            9900, 9903, 9904, 9907, 9908, 9909, 9910, 9914, 9930, 9931, 9934, 9937,
            9943, 9944, 9950, 9952, 9986, 9987, 9991, 9997, 10000, 10002, 10004, 10006,
            10012, 10015, 10016, 10018, 10026, 10028, 10032, 10033, 10037, 10053, 10055,
            10057, 10058, 10062, 10066, 10073, 10075, 10096, 10109, 10110, 10113, 10119,
            10123, 10124, 10125, 10127, 10139, 10140, 10143, 10147, 10148, 10149, 10150,
            10151, 10154, 10155, 10159, 10170, 10171, 10174, 10176, 10177, 10180, 10184,
            10187, 10190, 10192, 10197, 10225, 10229, 10231, 10232, 10237, 10238, 10240,
            10244, 10245, 10247, 10250, 10252, 10258, 10260, 10261, 10263, 10268, 10272,
            10273, 10274, 10277, 10278, 10280, 10286, 10290, 10292, 10293, 10294, 10295,
            10297, 10298, 10312, 10315, 10316, 10351, 10357, 10360, 10364, 10368, 10372,
            10378, 10388, 10392, 10393, 10397, 10401, 10405, 10413, 10415, 10417, 10418,
            10435, 10462, 10471, 10472, 10473, 10477, 10478, 10479, 10480, 10483, 10487,
            10490, 10493, 10498, 10499, 10500, 10501, 10511, 10512, 10517, 10518, 10519,
            10520, 10522, 10526, 10527, 10530, 10532, 10535, 10536, 10538, 10540, 10555,
            10556, 10557, 10587, 10591, 10597, 10600, 10602, 10608, 10615, 10616, 10618,
            10632, 10637, 10641, 10645, 10655, 10658, 10666, 10673, 10675, 10711, 10717,
            10720, 10724, 10732, 10738, 10747, 10748, 10750, 10752, 10753, 10757, 10771,
            10773, 10775, 10777, 10778, 10784, 10795, 10827, 10840, 10842, 10855, 10856,
            10872, 10895, 10901, 10905, 10906, 10908, 10913, 10943, 10947, 10948, 10951,
            10952, 10957, 10958, 10960, 10961, 10962, 10967, 10970, 10975, 10976, 10977,
            10978, 10980, 10981, 10982, 10992, 10997, 10998, 11000, 11006, 11010, 11012,
            11015, 11018, 11026, 11031, 11033, 11034, 11035, 11036, 11057, 11068, 11069,
            11081, 11082, 11084, 11085, 11086, 11087, 11096, 11097, 11100, 11102, 11103,
            11106, 11108, 11114, 11130, 11134, 11137, 11141, 11142, 11146, 11148, 11149,
            11151, 11152, 11154, 11177, 11188, 11189, 11201, 11202, 11204, 11205, 11206,
            11207, 11216, 11217, 11220, 11222, 11223, 11226, 11227, 11228, 11229, 11230,
            11234, 11250, 11251, 11254, 11257, 11262, 11264, 11266, 11270, 11271, 11272,
            11274, 11311, 11317, 11320, 11328, 11338, 11352, 11357, 11361, 11365, 11375,
            11378, 11395, 11426, 11427, 11440, 11442, 11444, 11446, 11452, 11455, 11456,
            11466, 11468, 11472, 11473, 11493, 11495, 11497, 11501, 11502, 11506, 11508,
            11513, 11543, 11547, 11548, 11552, 11558, 11560, 11561, 11562, 11567, 11575,
            11576, 11577, 11580, 11581, 11582, 11592, 11598, 11610, 11612, 11615, 11621,
            11626, 11628, 11629, 11631, 11633, 11634, 11636, 11682, 11684, 11686, 11696,
            11706, 11707, 11708, 11710, 11731, 11737, 11741, 11742, 11744, 11746, 11748,
            11788, 11801, 11802, 11807, 11816, 11817, 11820, 11822, 11850, 11861, 11865,
            11866, 11868, 11869, 11871, 11874, 11922, 11924, 11926, 11936, 11944, 11946,
            11947, 11948, 11950, 11971, 11977, 11982, 11983, 11984, 11986, 12051, 12065,
            12089, 12105, 12109, 12157, 12158, 12159, 12168, 12170, 12173, 12197, 12198,
            12199, 12200, 12201, 12202, 12205, 12206, 12207, 12212, 12216, 12218, 12277,
            12278, 12288, 12290, 12317, 12318, 12320, 12321, 12325, 12326, 12332, 12338,
            12397, 12408, 12437, 12441, 12445, 12458, 12491, 12508, 12513, 12514, 12516,
            12531, 12534, 12537, 12539, 12545, 12564, 12568, 12569, 12579, 12588, 12589,
            12594, 12597, 12620, 12627, 12628, 12632, 12633, 12651, 12653, 12655, 12657,
            12659, 12661, 12665, 12682, 12687, 12689, 12708, 12709, 12713, 12714, 12716,
            12717, 12747, 12748, 12751, 12752, 12770, 12775, 12777, 12778, 12781, 12800,
            12806, 12828, 12829, 12833, 12834, 12835, 12836, 12867, 12871, 12888, 12895,
            12898, 12921, 12925, 12948, 12953, 12955, 12996, 13008, 13010, 13013, 13040,
            13041, 13042, 13044, 13045, 13046, 13047, 13048, 13106, 13107, 13120, 13122,
            13124, 13126, 13132, 13135, 13136, 13146, 13147, 13148, 13150, 13152, 13153,
            13171, 13173, 13175, 13177, 13182, 13184, 13186, 13193, 13207, 13230, 13234,
            13243, 13245, 13249, 13254, 13263, 13267, 13269, 13271, 13275, 13276, 13299,
            13300, 13304, 13307, 13310, 13312, 13319, 13338, 13355, 13356, 13370, 13373,
            13400, 13402, 13403, 13404, 13406, 13407, 13408, 13438, 13459, 13471, 13472,
            13473, 13474, 13476, 13490, 13493, 13494, 13498, 13499, 13501, 13520, 13522,
            13524, 13526, 13527, 13528, 13539, 13555, 13556, 13557, 13591, 13592, 13593,
            13608, 13610, 13613, 13618, 13619, 13621, 13640, 13641, 13642, 13645, 13646,
            13647, 13675, 13676, 13677, 13711, 13712, 13728, 13730, 13738, 13741, 13760,
            13761, 13765, 13766, 13795, 13796, 13831, 13848, 13858, 13881, 13885, 13915,
            13944, 13949, 13950, 13957, 13958, 13959, 13970, 13972, 13973, 13993, 13994,
            13995, 13997, 13998, 13999, 14000, 14002, 14006, 14007, 14012, 14013, 14014,
            14016, 14018, 14027, 14069, 14077, 14078, 14088, 14090, 14092, 14113, 14114,
            14117, 14118, 14120, 14121, 14125, 14126, 14132, 14133, 14134, 14138, 14187,
            14188, 14191, 14192, 14208, 14210, 14215, 14217, 14218, 14221, 14240, 14241,
            14245, 14246, 14273, 14274, 14275, 14276, 14307, 14311, 14328, 14335, 14338,
            14361, 14365, 14393, 14395
    };
    void createMacaulayMatrix(const double a[20], const double b[20],
            const double c[20], const double u[4], double * macaulay_matrix) const {
        // The matrix is very large (14400 elements!) and sparse (1968 non-zero
        // elements) so we load it from pre-computed values calculated in matlab.

        const double values[1968] = {
                u[0], a[0], b[0], c[0], u[3], u[0], a[0], a[9], b[0], b[9], c[0], c[9],
                u[3], u[0], a[9], a[13], a[0], b[9], b[0], b[13], c[0], c[9], c[13], u[2],
                u[0], a[10], a[0], b[0], b[10], c[10], c[0], u[2], u[3], u[0], a[10], a[4],
                a[0], a[9], b[10], b[0], b[9], b[4], c[10], c[0], c[4], c[9], u[2], u[3],
                u[0], a[4], a[11], a[0], a[9], a[13], a[10], b[4], b[0], b[10], b[9], b[13],
                b[11], c[10], c[4], c[9], c[0], c[11], c[13], u[2], u[0], a[0], a[14],
                a[10], b[0], b[10], b[14], c[0], c[14], c[10], u[2], u[3], u[0], a[9],
                a[14], a[5], a[10], a[0], a[4], b[14], b[0], b[10], b[9], b[4], b[5], c[14],
                c[10], c[9], c[0], c[5], c[4], u[2], u[3], u[0], a[13], a[5], a[10], a[4],
                a[9], a[0], a[11], a[14], b[5], b[10], b[9], b[14], b[0], b[4], b[13],
                b[11], c[14], c[5], c[4], c[0], c[13], c[10], c[9], c[11], u[1], u[0], a[8],
                a[0], b[8], b[0], c[0], c[8], u[1], u[3], u[0], a[0], a[8], a[3], a[9],
                b[8], b[0], b[3], b[9], c[9], c[8], c[0], c[3], u[1], u[3], u[0], a[0],
                a[9], a[3], a[7], a[13], a[8], b[3], b[0], b[9], b[8], b[7], b[13], c[13],
                c[8], c[3], c[0], c[9], c[7], u[1], u[2], u[0], a[2], a[10], a[0], a[8],
                b[8], b[0], b[2], b[10], c[10], c[0], c[2], c[8], u[1], u[2], u[3], u[0],
                a[10], a[2], a[16], a[4], a[9], a[8], a[0], a[3], b[2], b[10], b[0], b[8],
                b[3], b[9], b[16], b[4], c[4], c[0], c[9], c[2], c[8], c[10], c[16], c[3],
                u[1], u[2], u[3], u[0], a[10], a[4], a[16], a[11], a[13], a[8], a[0], a[3],
                a[9], a[7], a[2], b[16], b[0], b[10], b[4], b[9], b[8], b[2], b[3], b[7],
                b[13], b[11], c[11], c[2], c[9], c[13], c[16], c[3], c[0], c[8], c[10],
                c[4], c[7], u[1], u[2], u[0], a[0], a[8], a[17], a[14], a[10], a[2], b[0],
                b[8], b[2], b[10], b[17], b[14], c[14], c[0], c[10], c[8], c[17], c[2],
                u[1], u[2], u[3], u[0], a[9], a[3], a[14], a[17], a[5], a[4], a[2], a[0],
                a[8], a[10], a[16], b[17], b[14], b[10], b[8], b[0], b[2], b[9], b[3],
                b[16], b[4], b[5], c[5], c[10], c[9], c[4], c[0], c[17], c[2], c[3], c[8],
                c[14], c[16], u[1], u[2], u[3], u[0], a[14], a[13], a[7], a[5], a[11], a[2],
                a[10], a[16], a[9], a[3], a[8], a[4], a[17], b[10], b[14], b[5], b[4], b[2],
                b[3], b[17], b[8], b[9], b[16], b[13], b[7], b[11], c[17], c[4], c[13],
                c[11], c[9], c[16], c[8], c[10], c[7], c[2], c[3], c[14], c[5], u[1], u[0],
                a[12], a[8], a[0], b[0], b[12], b[8], c[0], c[8], c[12], u[1], u[3], u[0],
                a[0], a[8], a[12], a[18], a[3], a[9], b[12], b[8], b[0], b[9], b[18], b[3],
                c[9], c[3], c[12], c[0], c[8], c[18], u[1], u[3], u[0], a[0], a[8], a[9],
                a[3], a[18], a[7], a[12], a[13], b[18], b[0], b[8], b[3], b[9], b[12],
                b[13], b[7], c[13], c[7], c[12], c[18], c[0], c[8], c[9], c[3], u[1], u[2],
                u[0], a[1], a[2], a[0], a[8], a[12], a[10], b[12], b[0], b[8], b[10], b[1],
                b[2], c[10], c[2], c[0], c[8], c[1], c[12], u[1], u[2], u[3], u[0], a[10],
                a[2], a[1], a[16], a[9], a[3], a[0], a[12], a[8], a[18], a[4], b[1], b[2],
                b[8], b[10], b[12], b[0], b[18], b[9], b[3], b[4], b[16], c[4], c[16], c[8],
                c[9], c[0], c[3], c[1], c[12], c[10], c[2], c[18], u[1], u[2], u[3], u[0],
                a[10], a[2], a[4], a[16], a[13], a[7], a[9], a[12], a[8], a[18], a[3], a[1],
                a[11], b[10], b[8], b[2], b[16], b[3], b[4], b[12], b[1], b[18], b[9],
                b[13], b[7], b[11], c[11], c[1], c[3], c[13], c[9], c[7], c[18], c[8],
                c[12], c[10], c[2], c[4], c[16], u[1], u[2], u[0], a[8], a[12], a[17],
                a[10], a[2], a[1], a[0], a[14], b[0], b[8], b[12], b[1], b[10], b[2], b[14],
                b[17], c[14], c[17], c[10], c[0], c[8], c[2], c[12], c[1], u[1], u[2], u[3],
                u[0], a[3], a[18], a[14], a[17], a[4], a[16], a[10], a[1], a[8], a[12],
                a[2], a[9], a[5], b[17], b[2], b[14], b[12], b[8], b[1], b[10], b[9], b[3],
                b[18], b[4], b[16], b[5], c[5], c[2], c[4], c[9], c[3], c[10], c[16], c[8],
                c[1], c[18], c[12], c[14], c[17], u[1], u[2], u[3], u[0], a[14], a[17],
                a[7], a[5], a[11], a[4], a[1], a[2], a[3], a[18], a[12], a[16], a[13],
                b[14], b[2], b[17], b[16], b[5], b[1], b[18], b[12], b[3], b[4], b[13],
                b[7], b[11], c[16], c[11], c[13], c[7], c[4], c[3], c[12], c[2], c[1],
                c[18], c[14], c[17], c[5], u[2], a[10], a[2], a[6], a[14], a[17], b[8],
                b[0], b[10], b[2], b[17], b[14], b[6], c[6], c[0], c[10], c[14], c[2], c[8],
                c[17], u[2], a[10], a[6], a[14], b[0], b[10], b[14], b[6], c[10], c[0],
                c[6], c[14], u[2], a[2], a[1], a[14], a[17], a[10], a[6], b[12], b[8],
                b[10], b[2], b[1], b[14], b[17], b[6], c[6], c[8], c[14], c[10], c[2],
                c[17], c[1], c[12], a[17], a[6], a[1], b[19], b[1], b[17], b[6], c[6],
                c[19], c[1], c[17], a[1], a[14], a[17], a[6], a[2], b[19], b[12], b[2],
                b[1], b[14], b[17], b[6], c[6], c[12], c[17], c[2], c[1], c[14], c[19],
                a[8], a[12], a[19], b[12], b[8], b[19], c[8], c[12], c[19], a[14], a[17],
                a[6], b[8], b[2], b[10], b[14], b[17], b[6], c[10], c[14], c[6], c[8],
                c[17], c[2], a[17], a[6], a[14], b[12], b[1], b[2], b[14], b[17], b[6],
                c[2], c[6], c[14], c[17], c[12], c[1], a[6], a[17], b[19], b[1], b[17],
                b[6], c[1], c[17], c[6], c[19], u[3], a[11], a[9], a[13], a[15], a[4],
                b[11], b[9], b[4], b[13], b[15], c[4], c[11], c[13], c[0], c[10], c[9],
                c[15], u[3], a[13], a[15], a[9], b[13], b[9], b[15], c[9], c[13], c[0],
                c[15], a[14], a[6], b[0], b[10], b[14], b[6], c[0], c[14], c[10], c[6],
                a[13], a[15], a[7], b[13], b[15], b[7], c[7], c[8], c[9], c[3], c[13],
                c[15], a[13], a[7], a[15], b[13], b[7], b[15], c[12], c[3], c[18], c[13],
                c[7], c[15], a[6], b[10], b[14], b[6], c[10], c[6], c[14], a[7], a[15],
                b[7], b[15], c[19], c[18], c[7], c[15], a[6], b[2], b[17], b[14], b[6],
                c[14], c[6], c[2], c[17], a[15], b[15], c[3], c[13], c[7], c[15], a[15],
                b[15], c[18], c[7], c[15], a[6], b[1], b[17], b[6], c[17], c[6], c[1], a[6],
                a[17], a[5], b[18], b[1], b[17], b[16], b[6], b[5], c[16], c[5], c[6],
                c[17], c[18], c[1], a[5], a[6], a[14], b[14], b[9], b[10], b[4], b[6], b[5],
                c[6], c[9], c[10], c[5], c[4], c[14], a[11], a[15], a[13], b[11], b[15],
                b[13], c[4], c[13], c[14], c[5], c[11], c[15], a[15], b[15], c[13], c[4],
                c[11], c[15], b[17], b[6], c[6], c[17], a[5], a[11], a[4], b[5], b[11],
                b[4], b[13], b[15], c[14], c[4], c[13], c[6], c[15], c[5], c[11], b[14],
                b[6], c[14], c[6], c[13], c[15], a[6], b[16], b[17], b[6], b[5], c[5], c[6],
                c[16], c[17], c[7], c[15], b[5], b[6], c[5], c[6], a[6], b[11], b[6], b[5],
                c[6], c[11], c[5], u[3], a[15], a[4], a[11], a[13], a[9], a[5], b[4], b[13],
                b[5], b[9], b[11], b[15], c[5], c[11], c[10], c[9], c[15], c[14], c[4],
                c[13], u[2], a[11], a[14], a[5], a[4], a[10], a[6], b[14], b[4], b[6],
                b[10], b[9], b[13], b[5], b[11], c[6], c[5], c[10], c[9], c[11], c[13],
                c[14], c[4], a[15], b[15], c[7], c[16], c[15], c[11], b[6], c[6], c[15],
                a[11], b[11], b[15], c[5], c[11], c[15], c[6], a[5], b[15], b[5], b[11],
                c[6], c[5], c[15], c[11], a[15], b[15], c[11], c[15], c[5], c[15], c[11],
                a[13], a[15], a[11], b[13], b[11], b[15], c[11], c[15], c[9], c[10], c[4],
                c[13], a[1], a[17], a[19], b[19], b[1], b[17], c[17], c[19], c[1], u[1],
                a[8], a[12], a[9], a[3], a[18], a[13], a[19], a[7], b[8], b[12], b[9],
                b[18], b[3], b[19], b[13], b[7], c[13], c[7], c[19], c[8], c[12], c[9],
                c[3], c[18], a[6], b[6], b[4], b[14], b[5], c[4], c[14], c[5], c[6], a[6],
                a[5], a[14], b[6], b[5], b[13], b[14], b[4], b[11], c[14], c[13], c[4],
                c[11], c[6], c[5], a[12], a[19], b[19], b[12], c[12], c[19], u[2], a[16],
                a[6], a[5], a[14], a[2], a[1], a[17], a[4], b[17], b[6], b[1], b[12], b[2],
                b[18], b[3], b[14], b[4], b[16], b[5], c[17], c[3], c[5], c[4], c[16],
                c[14], c[2], c[12], c[18], c[1], c[6], u[1], a[1], a[0], a[8], a[12], a[19],
                a[10], a[2], b[19], b[0], b[8], b[12], b[10], b[2], b[1], c[10], c[2], c[1],
                c[8], c[12], c[0], c[19], a[19], b[19], c[19], a[15], a[13], b[15], b[13],
                c[13], c[15], c[0], c[9], a[16], a[11], a[15], a[7], a[18], b[16], b[18],
                b[11], b[7], b[15], c[7], c[15], c[19], c[18], c[1], c[16], c[11], a[11],
                a[15], a[7], b[11], b[7], b[15], c[15], c[16], c[7], c[17], c[11], c[5],
                u[3], a[4], a[11], a[15], a[3], a[9], a[7], a[13], a[16], b[9], b[4], b[11],
                b[13], b[3], b[16], b[7], b[15], c[16], c[13], c[15], c[7], c[8], c[9],
                c[10], c[2], c[3], c[4], c[11], a[2], a[1], a[3], a[18], a[12], a[19], a[4],
                a[16], b[2], b[19], b[1], b[12], b[3], b[18], b[16], b[4], c[4], c[16],
                c[19], c[18], c[12], c[3], c[2], c[1], a[5], a[14], a[17], a[6], b[6],
                b[17], b[3], b[2], b[14], b[16], b[4], b[5], c[6], c[4], c[5], c[14], c[3],
                c[2], c[16], c[17], u[1], a[17], a[5], a[11], a[16], a[1], a[18], a[19],
                a[7], b[17], b[1], b[5], b[19], b[18], b[16], b[7], b[11], c[7], c[16],
                c[18], c[11], c[19], c[1], c[17], c[5], u[2], a[4], a[16], a[6], a[5],
                a[17], a[10], a[2], a[14], b[6], b[14], b[2], b[8], b[10], b[3], b[9],
                b[17], b[4], b[16], b[5], c[14], c[9], c[4], c[5], c[10], c[17], c[8],
                c[16], c[3], c[2], c[6], u[1], a[18], a[14], a[17], a[4], a[16], a[2],
                a[12], a[19], a[1], a[5], a[3], b[14], b[1], b[17], b[19], b[12], b[2],
                b[3], b[18], b[4], b[16], b[5], c[5], c[1], c[16], c[3], c[18], c[2], c[12],
                c[4], c[19], c[14], c[17], a[17], a[16], a[1], a[19], a[5], a[18], b[17],
                b[19], b[1], b[18], b[16], b[5], c[5], c[18], c[1], c[19], c[16], c[17],
                u[1], a[10], a[2], a[1], a[9], a[3], a[18], a[8], a[19], a[12], a[4], a[16],
                b[10], b[1], b[12], b[2], b[19], b[8], b[9], b[3], b[18], b[4], b[16], c[4],
                c[16], c[12], c[3], c[8], c[18], c[9], c[19], c[10], c[2], c[1], a[1],
                a[16], a[7], a[18], a[19], a[11], b[1], b[19], b[16], b[18], b[7], b[11],
                c[11], c[18], c[7], c[19], c[1], c[16], a[6], a[5], a[17], a[1], a[16],
                b[6], b[19], b[1], b[18], b[17], b[16], b[5], c[18], c[16], c[17], c[1],
                c[5], c[19], c[6], a[11], a[15], a[7], b[11], b[7], b[15], c[15], c[18],
                c[1], c[7], c[16], c[11], u[1], a[2], a[1], a[4], a[16], a[13], a[7], a[3],
                a[19], a[12], a[18], a[11], b[2], b[12], b[1], b[4], b[18], b[16], b[19],
                b[3], b[13], b[7], b[11], c[11], c[18], c[7], c[3], c[13], c[12], c[19],
                c[2], c[1], c[4], c[16], u[3], a[5], a[15], a[16], a[4], a[13], a[7], a[3],
                a[11], b[4], b[5], b[11], b[16], b[7], b[3], b[13], b[15], c[11], c[15],
                c[13], c[2], c[3], c[4], c[14], c[17], c[16], c[7], c[5], u[2], a[6], a[11],
                a[17], a[14], a[4], a[16], a[2], a[5], b[14], b[6], b[5], b[17], b[16],
                b[2], b[3], b[4], b[7], b[13], b[11], c[5], c[13], c[11], c[4], c[2], c[3],
                c[14], c[7], c[17], c[16], c[6], a[1], a[18], a[19], a[16], b[1], b[19],
                b[18], b[16], c[16], c[19], c[18], c[1], u[3], a[5], a[11], a[16], a[7],
                a[18], a[15], b[5], b[16], b[18], b[7], b[11], b[15], c[15], c[11], c[7],
                c[1], c[18], c[16], c[17], c[5], u[3], a[4], a[16], a[11], a[15], a[13],
                a[18], a[3], a[7], b[4], b[3], b[16], b[7], b[11], b[18], b[13], b[15],
                c[7], c[15], c[13], c[12], c[3], c[2], c[1], c[18], c[4], c[16], c[11],
                a[5], a[11], a[16], b[5], b[16], b[7], b[11], b[15], c[15], c[11], c[17],
                c[16], c[7], c[5], c[6], a[11], a[7], a[13], a[15], b[13], b[11], b[15],
                b[7], c[15], c[3], c[2], c[13], c[4], c[16], c[7], c[11], a[6], a[5], a[17],
                b[6], b[7], b[17], b[16], b[5], b[11], c[11], c[5], c[17], c[7], c[16],
                c[6], a[15], b[15], c[15], c[9], c[13], a[8], a[12], a[19], a[10], a[2],
                a[1], b[8], b[12], b[19], b[2], b[10], b[1], c[10], c[2], c[1], c[12],
                c[19], c[8], a[12], a[19], a[2], a[1], b[12], b[19], b[1], b[2], c[2], c[1],
                c[19], c[12], a[19], a[1], b[19], b[1], c[1], c[19], u[3], a[9], a[13],
                a[7], a[15], a[3], b[7], b[9], b[13], b[3], b[15], c[15], c[3], c[7], c[0],
                c[8], c[9], c[13], u[3], a[9], a[3], a[13], a[7], a[18], a[15], b[9], b[3],
                b[7], b[13], b[18], b[15], c[15], c[18], c[8], c[12], c[9], c[3], c[13],
                c[7], a[3], a[18], a[13], a[7], a[15], b[3], b[18], b[13], b[7], b[15],
                c[15], c[12], c[19], c[3], c[18], c[13], c[7], a[18], a[7], a[15], b[18],
                b[7], b[15], c[15], c[19], c[18], c[7], a[19], a[0], a[8], a[12], b[8],
                b[0], b[12], b[19], c[0], c[8], c[12], c[19], u[2], a[6], a[5], a[17],
                a[16], a[1], a[11], b[6], b[17], b[1], b[18], b[16], b[7], b[5], b[11],
                c[7], c[11], c[5], c[16], c[1], c[18], c[17], c[6], u[2], a[4], a[6], a[14],
                a[10], a[5], b[6], b[10], b[0], b[9], b[14], b[4], b[5], c[6], c[14], c[0],
                c[4], c[9], c[10], c[5], u[1], a[19], a[12], a[0], a[8], b[0], b[8], b[19],
                b[12], c[0], c[8], c[12], c[19], u[1], a[0], a[8], a[12], a[19], a[18],
                a[9], a[3], b[19], b[0], b[12], b[8], b[9], b[3], b[18], c[9], c[3], c[18],
                c[19], c[0], c[8], c[12], a[8], a[12], a[19], a[9], a[3], a[18], b[8],
                b[19], b[12], b[3], b[9], b[18], c[9], c[3], c[18], c[8], c[12], c[19],
                a[12], a[19], a[3], a[18], b[12], b[19], b[18], b[3], c[3], c[18], c[12],
                c[19], a[19], a[18], b[19], b[18], c[18], c[19], u[1], a[12], a[19], a[10],
                a[2], a[1], a[14], a[8], a[17], b[8], b[12], b[19], b[10], b[2], b[1],
                b[14], b[17], c[14], c[17], c[2], c[8], c[12], c[1], c[10], c[19], a[19],
                a[2], a[1], a[14], a[17], a[12], b[12], b[19], b[2], b[1], b[17], b[14],
                c[14], c[17], c[1], c[12], c[19], c[2], a[12], a[19], a[3], a[18], a[13],
                a[7], b[12], b[19], b[3], b[18], b[7], b[13], c[13], c[7], c[12], c[19],
                c[3], c[18], a[19], a[18], a[7], b[19], b[18], b[7], c[7], c[19], c[18]
        };
        for (int i = 0; i < 1968; i++)
            macaulay_matrix[indices[i]] = values[i];
    }
#endif

    // Transforms a 3 - vector in a 3x9 matrix such that :
    // R * v = leftMultiplyMatrix(v) * vec(R)
    // Where R is a rotation matrix and vec(R) converts R to a 9x1 vector.
    Matx<double, 3, 9> leftMultiplyMatrix(const Vec3d& v) const {
        Matx<double, 3, 9> left_mult_mat = Matx<double, 3, 9>::zeros();
        left_mult_mat(0,0) = v[0]; left_mult_mat(0,1) = v[1]; left_mult_mat(0,2) = v[2];
        left_mult_mat(1,3) = v[0]; left_mult_mat(1,4) = v[1]; left_mult_mat(1,5) = v[2];
        left_mult_mat(2,6) = v[0]; left_mult_mat(2,7) = v[1]; left_mult_mat(2,8) = v[2];
        return left_mult_mat;
    }

    // Extracts the coefficients of the Jacobians of the LS cost function (which is
    // parameterized by the 3 rotation coefficients s1, s2, s3).
    void extractJacobianCoefficients(const double * const D,
            double f1_coeff[20], double f2_coeff[20], double f3_coeff[20]) const {
        f1_coeff[0] =
                2 * D[5] - 2 * D[7] + 2 * D[41] - 2 * D[43] + 2 * D[45] +
                2 * D[49] + 2 * D[53] - 2 * D[63] - 2 * D[67] - 2 * D[71] +
                2 * D[77] - 2 * D[79];             // constant term
        f1_coeff[1] =
                (6 * D[1] + 6 * D[3] + 6 * D[9] - 6 * D[13] - 6 * D[17] +
                 6 * D[27] - 6 * D[31] - 6 * D[35] - 6 * D[37] - 6 * D[39] -
                 6 * D[73] - 6 * D[75]);           // s1^2  * s2
        f1_coeff[2] =
                (4 * D[6] - 4 * D[2] + 8 * D[14] - 8 * D[16] - 4 * D[18] +
                 4 * D[22] + 4 * D[26] + 8 * D[32] - 8 * D[34] + 4 * D[38] -
                 4 * D[42] + 8 * D[46] + 8 * D[48] + 4 * D[54] - 4 * D[58] -
                 4 * D[62] - 8 * D[64] - 8 * D[66] + 4 * D[74] -
                 4 * D[78]);                         // s1 * s2
        f1_coeff[3] =
                (4 * D[1] - 4 * D[3] + 4 * D[9] - 4 * D[13] - 4 * D[17] +
                 8 * D[23] - 8 * D[25] - 4 * D[27] + 4 * D[31] + 4 * D[35] -
                 4 * D[37] + 4 * D[39] + 8 * D[47] + 8 * D[51] + 8 * D[59] -
                 8 * D[61] - 8 * D[65] - 8 * D[69] - 4 * D[73] +
                 4 * D[75]);                         // s1 * s3
        f1_coeff[4] = (8 * D[10] - 8 * D[20] - 8 * D[30] + 8 * D[50] +
                       8 * D[60] - 8 * D[70]);  // s2 * s3
        f1_coeff[5] =
                (4 * D[14] - 2 * D[6] - 2 * D[2] + 4 * D[16] - 2 * D[18] +
                 2 * D[22] - 2 * D[26] + 4 * D[32] + 4 * D[34] + 2 * D[38] +
                 2 * D[42] + 4 * D[46] + 4 * D[48] - 2 * D[54] + 2 * D[58] -
                 2 * D[62] + 4 * D[64] + 4 * D[66] - 2 * D[74] -
                 2 * D[78]);                         // s2^2 * s3
        f1_coeff[6] = (2 * D[13] - 2 * D[3] - 2 * D[9] - 2 * D[1] -
                       2 * D[17] - 2 * D[27] + 2 * D[31] - 2 * D[35] +
                       2 * D[37] + 2 * D[39] - 2 * D[73] - 2 * D[75]);  // s2^3
        f1_coeff[7] =
                (4 * D[8] - 4 * D[0] + 8 * D[20] + 8 * D[24] + 4 * D[40] +
                 8 * D[56] + 8 * D[60] + 4 * D[72] - 4 * D[80]);  // s1 * s3^2
        f1_coeff[8] =
                (4 * D[0] - 4 * D[40] - 4 * D[44] + 8 * D[50] - 8 * D[52] -
                 8 * D[68] + 8 * D[70] - 4 * D[76] - 4 * D[80]);  // s1
        f1_coeff[9] = (2 * D[2] + 2 * D[6] + 4 * D[14] - 4 * D[16] +
                       2 * D[18] + 2 * D[22] + 2 * D[26] - 4 * D[32] +
                       4 * D[34] + 2 * D[38] + 2 * D[42] + 4 * D[46] -
                       4 * D[48] + 2 * D[54] + 2 * D[58] + 2 * D[62] -
                       4 * D[64] + 4 * D[66] + 2 * D[74] + 2 * D[78]);   // s3
        f1_coeff[10] = (2 * D[1] + 2 * D[3] + 2 * D[9] + 2 * D[13] +
                        2 * D[17] - 4 * D[23] + 4 * D[25] + 2 * D[27] +
                        2 * D[31] + 2 * D[35] + 2 * D[37] + 2 * D[39] -
                        4 * D[47] + 4 * D[51] + 4 * D[59] - 4 * D[61] +
                        4 * D[65] - 4 * D[69] + 2 * D[73] + 2 * D[75]);  // s2
        f1_coeff[11] =
                (2 * D[17] - 2 * D[3] - 2 * D[9] - 2 * D[13] - 2 * D[1] +
                 4 * D[23] + 4 * D[25] - 2 * D[27] - 2 * D[31] + 2 * D[35] -
                 2 * D[37] - 2 * D[39] + 4 * D[47] + 4 * D[51] + 4 * D[59] +
                 4 * D[61] + 4 * D[65] + 4 * D[69] + 2 * D[73] +
                 2 * D[75]);                                            // s2 * s3^2
        f1_coeff[12] =
                (6 * D[5] - 6 * D[7] - 6 * D[41] + 6 * D[43] + 6 * D[45] -
                 6 * D[49] - 6 * D[53] - 6 * D[63] + 6 * D[67] + 6 * D[71] -
                 6 * D[77] + 6 * D[79]);                              // s1^2
        f1_coeff[13] =
                (2 * D[7] - 2 * D[5] + 4 * D[11] + 4 * D[15] + 4 * D[19] -
                 4 * D[21] - 4 * D[29] - 4 * D[33] - 2 * D[41] + 2 * D[43] -
                 2 * D[45] - 2 * D[49] + 2 * D[53] + 4 * D[55] - 4 * D[57] +
                 2 * D[63] + 2 * D[67] - 2 * D[71] + 2 * D[77] -
                 2 * D[79]);                                            // s3^2
        f1_coeff[14] =
                (2 * D[7] - 2 * D[5] - 4 * D[11] + 4 * D[15] - 4 * D[19] -
                 4 * D[21] - 4 * D[29] + 4 * D[33] + 2 * D[41] - 2 * D[43] -
                 2 * D[45] + 2 * D[49] - 2 * D[53] + 4 * D[55] + 4 * D[57] +
                 2 * D[63] - 2 * D[67] + 2 * D[71] - 2 * D[77] +
                 2 * D[79]);                                            // s2^2
        f1_coeff[15] =
                (2 * D[26] - 2 * D[6] - 2 * D[18] - 2 * D[22] - 2 * D[2] -
                 2 * D[38] - 2 * D[42] - 2 * D[54] - 2 * D[58] + 2 * D[62] +
                 2 * D[74] + 2 * D[78]);                              // s3^3
        f1_coeff[16] =
                (4 * D[5] + 4 * D[7] + 8 * D[11] + 8 * D[15] + 8 * D[19] +
                 8 * D[21] + 8 * D[29] + 8 * D[33] - 4 * D[41] - 4 * D[43] +
                 4 * D[45] - 4 * D[49] - 4 * D[53] + 8 * D[55] + 8 * D[57] +
                 4 * D[63] - 4 * D[67] - 4 * D[71] - 4 * D[77] -
                 4 * D[79]);                                            // s1 * s2 * s3
        f1_coeff[17] =
                (4 * D[4] - 4 * D[0] + 8 * D[10] + 8 * D[12] + 8 * D[28] +
                 8 * D[30] + 4 * D[36] - 4 * D[40] + 4 * D[80]);  // s1 * s2^2
        f1_coeff[18] =
                (6 * D[2] + 6 * D[6] + 6 * D[18] - 6 * D[22] - 6 * D[26] -
                 6 * D[38] - 6 * D[42] + 6 * D[54] - 6 * D[58] - 6 * D[62] -
                 6 * D[74] - 6 * D[78]);                              // s1^2 * s3
        f1_coeff[19] =
                (4 * D[0] - 4 * D[4] - 4 * D[8] - 4 * D[36] + 4 * D[40] +
                 4 * D[44] - 4 * D[72] + 4 * D[76] + 4 * D[80]);  // s1^3

        f2_coeff[0] =
                -2 * D[2] + 2 * D[6] - 2 * D[18] - 2 * D[22] - 2 * D[26] -
                2 * D[38] + 2 * D[42] + 2 * D[54] + 2 * D[58] + 2 * D[62] -
                2 * D[74] + 2 * D[78];                                // constant term
        f2_coeff[1] =
                (4 * D[4] - 4 * D[0] + 8 * D[10] + 8 * D[12] + 8 * D[28] +
                 8 * D[30] + 4 * D[36] - 4 * D[40] + 4 * D[80]);  // s1^2  * s2
        f2_coeff[2] =
                (4 * D[7] - 4 * D[5] - 8 * D[11] + 8 * D[15] - 8 * D[19] -
                 8 * D[21] - 8 * D[29] + 8 * D[33] + 4 * D[41] - 4 * D[43] -
                 4 * D[45] + 4 * D[49] - 4 * D[53] + 8 * D[55] + 8 * D[57] +
                 4 * D[63] - 4 * D[67] + 4 * D[71] - 4 * D[77] +
                 4 * D[79]);                                            // s1 * s2
        f2_coeff[3] = (8 * D[10] - 8 * D[20] - 8 * D[30] + 8 * D[50] +
                       8 * D[60] - 8 * D[70]);                     // s1 * s3
        f2_coeff[4] =
                (4 * D[3] - 4 * D[1] - 4 * D[9] + 4 * D[13] - 4 * D[17] -
                 8 * D[23] - 8 * D[25] + 4 * D[27] - 4 * D[31] + 4 * D[35] +
                 4 * D[37] - 4 * D[39] - 8 * D[47] + 8 * D[51] + 8 * D[59] +
                 8 * D[61] - 8 * D[65] + 8 * D[69] - 4 * D[73] +
                 4 * D[75]);                                            // s2 * s3
        f2_coeff[5] =
                (6 * D[41] - 6 * D[7] - 6 * D[5] + 6 * D[43] - 6 * D[45] +
                 6 * D[49] - 6 * D[53] - 6 * D[63] + 6 * D[67] - 6 * D[71] -
                 6 * D[77] - 6 * D[79]);                              // s2^2 * s3
        f2_coeff[6] =
                (4 * D[0] - 4 * D[4] + 4 * D[8] - 4 * D[36] + 4 * D[40] -
                 4 * D[44] + 4 * D[72] - 4 * D[76] + 4 * D[80]);  // s2^3
        f2_coeff[7] =
                (2 * D[17] - 2 * D[3] - 2 * D[9] - 2 * D[13] - 2 * D[1] +
                 4 * D[23] + 4 * D[25] - 2 * D[27] - 2 * D[31] + 2 * D[35] -
                 2 * D[37] - 2 * D[39] + 4 * D[47] + 4 * D[51] + 4 * D[59] +
                 4 * D[61] + 4 * D[65] + 4 * D[69] + 2 * D[73] +
                 2 * D[75]);                                            // s1 * s3^2
        f2_coeff[8] = (2 * D[1] + 2 * D[3] + 2 * D[9] + 2 * D[13] +
                       2 * D[17] - 4 * D[23] + 4 * D[25] + 2 * D[27] +
                       2 * D[31] + 2 * D[35] + 2 * D[37] + 2 * D[39] -
                       4 * D[47] + 4 * D[51] + 4 * D[59] - 4 * D[61] +
                       4 * D[65] - 4 * D[69] + 2 * D[73] + 2 * D[75]);  // s1
        f2_coeff[9] = (2 * D[5] + 2 * D[7] - 4 * D[11] + 4 * D[15] -
                       4 * D[19] + 4 * D[21] + 4 * D[29] - 4 * D[33] +
                       2 * D[41] + 2 * D[43] + 2 * D[45] + 2 * D[49] +
                       2 * D[53] + 4 * D[55] - 4 * D[57] + 2 * D[63] +
                       2 * D[67] + 2 * D[71] + 2 * D[77] + 2 * D[79]);  // s3
        f2_coeff[10] =
                (8 * D[20] - 4 * D[8] - 4 * D[0] - 8 * D[24] + 4 * D[40] -
                 8 * D[56] + 8 * D[60] - 4 * D[72] - 4 * D[80]);           // s2
        f2_coeff[11] =
                (4 * D[0] - 4 * D[40] + 4 * D[44] + 8 * D[50] + 8 * D[52] +
                 8 * D[68] + 8 * D[70] + 4 * D[76] - 4 * D[80]);  // s2 * s3^2
        f2_coeff[12] =
                (2 * D[6] - 2 * D[2] + 4 * D[14] - 4 * D[16] - 2 * D[18] +
                 2 * D[22] + 2 * D[26] + 4 * D[32] - 4 * D[34] + 2 * D[38] -
                 2 * D[42] + 4 * D[46] + 4 * D[48] + 2 * D[54] - 2 * D[58] -
                 2 * D[62] - 4 * D[64] - 4 * D[66] + 2 * D[74] -
                 2 * D[78]);                                            // s1^2
        f2_coeff[13] =
                (2 * D[2] - 2 * D[6] + 4 * D[14] + 4 * D[16] + 2 * D[18] +
                 2 * D[22] - 2 * D[26] - 4 * D[32] - 4 * D[34] + 2 * D[38] -
                 2 * D[42] + 4 * D[46] - 4 * D[48] - 2 * D[54] - 2 * D[58] +
                 2 * D[62] + 4 * D[64] - 4 * D[66] - 2 * D[74] +
                 2 * D[78]);                                            // s3^2
        f2_coeff[14] =
                (6 * D[2] - 6 * D[6] + 6 * D[18] - 6 * D[22] + 6 * D[26] -
                 6 * D[38] + 6 * D[42] - 6 * D[54] + 6 * D[58] - 6 * D[62] +
                 6 * D[74] - 6 * D[78]);                              // s2^2
        f2_coeff[15] =
                (2 * D[53] - 2 * D[7] - 2 * D[41] - 2 * D[43] - 2 * D[45] -
                 2 * D[49] - 2 * D[5] - 2 * D[63] - 2 * D[67] + 2 * D[71] +
                 2 * D[77] + 2 * D[79]);                              // s3^3
        f2_coeff[16] =
                (8 * D[14] - 4 * D[6] - 4 * D[2] + 8 * D[16] - 4 * D[18] +
                 4 * D[22] - 4 * D[26] + 8 * D[32] + 8 * D[34] + 4 * D[38] +
                 4 * D[42] + 8 * D[46] + 8 * D[48] - 4 * D[54] + 4 * D[58] -
                 4 * D[62] + 8 * D[64] + 8 * D[66] - 4 * D[74] -
                 4 * D[78]);                                            // s1 * s2 * s3
        f2_coeff[17] =
                (6 * D[13] - 6 * D[3] - 6 * D[9] - 6 * D[1] - 6 * D[17] -
                 6 * D[27] + 6 * D[31] - 6 * D[35] + 6 * D[37] + 6 * D[39] -
                 6 * D[73] - 6 * D[75]);                              // s1 * s2^2
        f2_coeff[18] =
                (2 * D[5] + 2 * D[7] + 4 * D[11] + 4 * D[15] + 4 * D[19] +
                 4 * D[21] + 4 * D[29] + 4 * D[33] - 2 * D[41] - 2 * D[43] +
                 2 * D[45] - 2 * D[49] - 2 * D[53] + 4 * D[55] + 4 * D[57] +
                 2 * D[63] - 2 * D[67] - 2 * D[71] - 2 * D[77] -
                 2 * D[79]);                                            // s1^2 * s3
        f2_coeff[19] =
                (2 * D[1] + 2 * D[3] + 2 * D[9] - 2 * D[13] - 2 * D[17] +
                 2 * D[27] - 2 * D[31] - 2 * D[35] - 2 * D[37] - 2 * D[39] -
                 2 * D[73] - 2 * D[75]);                              // s1^3

        f3_coeff[0] =
                2 * D[1] - 2 * D[3] + 2 * D[9] + 2 * D[13] + 2 * D[17] -
                2 * D[27] - 2 * D[31] - 2 * D[35] + 2 * D[37] - 2 * D[39] +
                2 * D[73] - 2 * D[75];                                // constant term
        f3_coeff[1] =
                (2 * D[5] + 2 * D[7] + 4 * D[11] + 4 * D[15] + 4 * D[19] +
                 4 * D[21] + 4 * D[29] + 4 * D[33] - 2 * D[41] - 2 * D[43] +
                 2 * D[45] - 2 * D[49] - 2 * D[53] + 4 * D[55] + 4 * D[57] +
                 2 * D[63] - 2 * D[67] - 2 * D[71] - 2 * D[77] -
                 2 * D[79]);                                            // s1^2  * s2
        f3_coeff[2] = (8 * D[10] - 8 * D[20] - 8 * D[30] + 8 * D[50] +
                       8 * D[60] - 8 * D[70]);                     // s1 * s2
        f3_coeff[3] =
                (4 * D[7] - 4 * D[5] + 8 * D[11] + 8 * D[15] + 8 * D[19] -
                 8 * D[21] - 8 * D[29] - 8 * D[33] - 4 * D[41] + 4 * D[43] -
                 4 * D[45] - 4 * D[49] + 4 * D[53] + 8 * D[55] - 8 * D[57] +
                 4 * D[63] + 4 * D[67] - 4 * D[71] + 4 * D[77] -
                 4 * D[79]);                                            // s1 * s3
        f3_coeff[4] =
                (4 * D[2] - 4 * D[6] + 8 * D[14] + 8 * D[16] + 4 * D[18] +
                 4 * D[22] - 4 * D[26] - 8 * D[32] - 8 * D[34] + 4 * D[38] -
                 4 * D[42] + 8 * D[46] - 8 * D[48] - 4 * D[54] - 4 * D[58] +
                 4 * D[62] + 8 * D[64] - 8 * D[66] - 4 * D[74] +
                 4 * D[78]);                                            // s2 * s3
        f3_coeff[5] =
                (4 * D[0] - 4 * D[40] + 4 * D[44] + 8 * D[50] + 8 * D[52] +
                 8 * D[68] + 8 * D[70] + 4 * D[76] - 4 * D[80]);  // s2^2 * s3
        f3_coeff[6] = (2 * D[41] - 2 * D[7] - 2 * D[5] + 2 * D[43] -
                       2 * D[45] + 2 * D[49] - 2 * D[53] - 2 * D[63] +
                       2 * D[67] - 2 * D[71] - 2 * D[77] - 2 * D[79]);  // s2^3
        f3_coeff[7] =
                (6 * D[26] - 6 * D[6] - 6 * D[18] - 6 * D[22] - 6 * D[2] -
                 6 * D[38] - 6 * D[42] - 6 * D[54] - 6 * D[58] + 6 * D[62] +
                 6 * D[74] + 6 * D[78]);  // s1 * s3^2
        f3_coeff[8] = (2 * D[2] + 2 * D[6] + 4 * D[14] - 4 * D[16] +
                       2 * D[18] + 2 * D[22] + 2 * D[26] - 4 * D[32] +
                       4 * D[34] + 2 * D[38] + 2 * D[42] + 4 * D[46] -
                       4 * D[48] + 2 * D[54] + 2 * D[58] + 2 * D[62] -
                       4 * D[64] + 4 * D[66] + 2 * D[74] + 2 * D[78]);   // s1
        f3_coeff[9] =
                (8 * D[10] - 4 * D[4] - 4 * D[0] - 8 * D[12] - 8 * D[28] +
                 8 * D[30] - 4 * D[36] - 4 * D[40] + 4 * D[80]);            // s3
        f3_coeff[10] = (2 * D[5] + 2 * D[7] - 4 * D[11] + 4 * D[15] -
                        4 * D[19] + 4 * D[21] + 4 * D[29] - 4 * D[33] +
                        2 * D[41] + 2 * D[43] + 2 * D[45] + 2 * D[49] +
                        2 * D[53] + 4 * D[55] - 4 * D[57] + 2 * D[63] +
                        2 * D[67] + 2 * D[71] + 2 * D[77] + 2 * D[79]);  // s2
        f3_coeff[11] =
                (6 * D[53] - 6 * D[7] - 6 * D[41] - 6 * D[43] - 6 * D[45] -
                 6 * D[49] - 6 * D[5] - 6 * D[63] - 6 * D[67] + 6 * D[71] +
                 6 * D[77] + 6 * D[79]);                              // s2 * s3^2
        f3_coeff[12] =
                (2 * D[1] - 2 * D[3] + 2 * D[9] - 2 * D[13] - 2 * D[17] +
                 4 * D[23] - 4 * D[25] - 2 * D[27] + 2 * D[31] + 2 * D[35] -
                 2 * D[37] + 2 * D[39] + 4 * D[47] + 4 * D[51] + 4 * D[59] -
                 4 * D[61] - 4 * D[65] - 4 * D[69] - 2 * D[73] +
                 2 * D[75]);                                            // s1^2
        f3_coeff[13] =
                (6 * D[3] - 6 * D[1] - 6 * D[9] - 6 * D[13] + 6 * D[17] +
                 6 * D[27] + 6 * D[31] - 6 * D[35] - 6 * D[37] + 6 * D[39] +
                 6 * D[73] - 6 * D[75]);                              // s3^2
        f3_coeff[14] =
                (2 * D[3] - 2 * D[1] - 2 * D[9] + 2 * D[13] - 2 * D[17] -
                 4 * D[23] - 4 * D[25] + 2 * D[27] - 2 * D[31] + 2 * D[35] +
                 2 * D[37] - 2 * D[39] - 4 * D[47] + 4 * D[51] + 4 * D[59] +
                 4 * D[61] - 4 * D[65] + 4 * D[69] - 2 * D[73] +
                 2 * D[75]);                                            // s2^2
        f3_coeff[15] =
                (4 * D[0] + 4 * D[4] - 4 * D[8] + 4 * D[36] + 4 * D[40] -
                 4 * D[44] - 4 * D[72] - 4 * D[76] + 4 * D[80]);  // s3^3
        f3_coeff[16] =
                (4 * D[17] - 4 * D[3] - 4 * D[9] - 4 * D[13] - 4 * D[1] +
                 8 * D[23] + 8 * D[25] - 4 * D[27] - 4 * D[31] + 4 * D[35] -
                 4 * D[37] - 4 * D[39] + 8 * D[47] + 8 * D[51] + 8 * D[59] +
                 8 * D[61] + 8 * D[65] + 8 * D[69] + 4 * D[73] +
                 4 * D[75]);                                            // s1 * s2 * s3
        f3_coeff[17] =
                (4 * D[14] - 2 * D[6] - 2 * D[2] + 4 * D[16] - 2 * D[18] +
                 2 * D[22] - 2 * D[26] + 4 * D[32] + 4 * D[34] + 2 * D[38] +
                 2 * D[42] + 4 * D[46] + 4 * D[48] - 2 * D[54] + 2 * D[58] -
                 2 * D[62] + 4 * D[64] + 4 * D[66] - 2 * D[74] -
                 2 * D[78]);                                            // s1 * s2^2
        f3_coeff[18] =
                (4 * D[8] - 4 * D[0] + 8 * D[20] + 8 * D[24] + 4 * D[40] +
                 8 * D[56] + 8 * D[60] + 4 * D[72] - 4 * D[80]);  // s1^2 * s3
        f3_coeff[19] =
                (2 * D[2] + 2 * D[6] + 2 * D[18] - 2 * D[22] - 2 * D[26] -
                 2 * D[38] - 2 * D[42] + 2 * D[54] - 2 * D[58] - 2 * D[62] -
                 2 * D[74] - 2 * D[78]);                              // s1^3
    }
};
Ptr<DLSPnP> DLSPnP::create(const Mat &points_, const Mat &calib_norm_pts, const Mat &K) {
    return makePtr<DLSPnPImpl>(points_, calib_norm_pts, K);
}
}}
