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
#include "opencv2/core/utils/logger.hpp"

namespace cv {
namespace multiview {
class RobustFunction : public Algorithm {
public:
    virtual float getError(float err) const = 0;
};

class RobustExpFunction : public RobustFunction {
private:
    const float over_scale, pow_23 = 1 << 23;
public:
    explicit RobustExpFunction (float scale_=30.0f) : over_scale(-1.442695040f /scale_) {}
    // err > 0
    float getError(float err) const override {
        const auto under_exp = err * over_scale;
        if (under_exp < -20) return 0; // prevent overflow further
        // http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
        union { uint32_t i; float f; } exp_val = { static_cast<uint32_t>(pow_23 * (under_exp + 126.94269504f)) };
        return err * exp_val.f;
    }
};

static double robustWrapper (InputArray errors, const RobustFunction &fnc) {
    Mat errs = errors.getMat();
    errs.convertTo(errs, CV_32F);
    auto * errs_ptr = (float *) errs.data;
    double robust_sum_sqr_errs = 0.0;
    for (int pt = 0; pt < (int)errs.total()*2; pt++) {
        auto sqr_err = errs_ptr[pt];
        sqr_err *= sqr_err;
        robust_sum_sqr_errs += fnc.getError(sqr_err);
    }
    return sqrt(robust_sum_sqr_errs);
}

static double computeReprojectionRMSE(const Mat &obj_points_, const Mat &img_points_, const Matx33d &K, const Mat &distortion,
               const Mat &rvec, const Mat &tvec, InputArray rvec2, InputArray tvec2, bool is_fisheye) {
    Mat r, t;
    if (!rvec2.empty() && !tvec2.empty()) {
        composeRT(rvec, tvec, rvec2, tvec2, r, t);
    } else {
        r = rvec; t = tvec;
    }
    Mat tmpImagePoints, obj_points = obj_points_, img_points = img_points_;
    if (is_fisheye) {
        obj_points = obj_points.reshape(3); // must be 3 channels
        fisheye::projectPoints(obj_points, tmpImagePoints, r, t, K, distortion);
    } else
        projectPoints(obj_points, r, t, K, distortion, tmpImagePoints);
    if (img_points.channels() != tmpImagePoints.channels())
        img_points = img_points.reshape(tmpImagePoints.channels());
    if (img_points.rows != tmpImagePoints.rows)
        img_points = img_points.t();
    subtract (tmpImagePoints, img_points, tmpImagePoints);
    return sqrt(norm(tmpImagePoints, NORM_L2SQR) / tmpImagePoints.rows);
}

static bool maximumSpanningTree (int NUM_CAMERAS, int NUM_FRAMES, const std::vector<std::vector<bool>> &visibility,
          std::vector<int> &parent, std::vector<std::vector<int>> &overlap,
          std::vector<std::vector<Vec3d>> &opt_axes,
          const std::vector<std::vector<bool>> &is_valid_angle2pattern,
          const std::vector<std::vector<float>> &points_area_ratio,
          double WEIGHT_ANGLE_PATTERN, double WEIGHT_CAMERAS_ANGLES) {
    const double THR_CAMERAS_ANGLES = 160*M_PI/180;
    // build weights matrix
    overlap = std::vector<std::vector<int>>(NUM_CAMERAS, std::vector<int>(NUM_CAMERAS, 0));
    std::vector<std::vector<double>> weights(NUM_CAMERAS, std::vector<double>(NUM_CAMERAS, DBL_MIN));
    for (int c1 = 0; c1 < NUM_CAMERAS; c1++) {
        for (int c2 = c1+1; c2 < NUM_CAMERAS; c2++) {
            double weight = 0;
            int overlaps = 0;
            for (int f = 0; f < NUM_FRAMES; f++) {
                if (visibility[c1][f] && visibility[c2][f]) {
                    overlaps += 1;
                    weight += points_area_ratio[c1][f] + points_area_ratio[c2][f];
                    weight += WEIGHT_ANGLE_PATTERN * ((int)is_valid_angle2pattern[c1][f] + (int)is_valid_angle2pattern[c2][f]);
                    if (WEIGHT_CAMERAS_ANGLES > 0) {
                        // angle between cameras optical axes
                        weight += WEIGHT_CAMERAS_ANGLES * int(acos(opt_axes[c1][f].dot(opt_axes[c2][f])) < THR_CAMERAS_ANGLES);
                    }
                }
            }
            if (overlaps > 0) {
                overlap[c1][c2] = overlap[c2][c1] = overlaps;
                weights[c1][c2] = weights[c2][c1] = overlaps + weight;
            }
        }
    }

    // find maximum spanning tree using Prim's algorithm
    std::vector<bool> visited(NUM_CAMERAS, false);
    std::vector<double> weight(NUM_CAMERAS, DBL_MIN);
    parent = std::vector<int>(NUM_CAMERAS, -1);
    weight[0] = DBL_MAX;
    for (int cam =  0; cam < NUM_CAMERAS-1; cam++) {
        int max_weight_idx = -1;
        auto max_weight = DBL_MIN;
        for (int cam2 = 0; cam2 < NUM_CAMERAS; cam2++) {
            if (!visited[cam2] && max_weight < weight[cam2]) {
                max_weight = weight[cam2];
                max_weight_idx = cam2;
            }
        }
        if (max_weight_idx == -1)
            return false;
        visited[max_weight_idx] = true;
        for (int cam2 = 0; cam2 < NUM_CAMERAS; cam2++) {
            if (!visited[cam2] && overlap[max_weight_idx][cam2] > 0) {
                if (weight[cam2] < weights[max_weight_idx][cam2]) {
                    weight[cam2] = weights[max_weight_idx][cam2];
                    parent[cam2] = max_weight_idx;
                }
            }
        }
    }
    return true;
}

static void imagePointsArea (const std::vector<Size> &imageSize, const std::vector<std::vector<bool>> &visibility_mat,
          const std::vector<std::vector<Mat>> &imagePoints, std::vector<std::vector<float>> &points_ratio_area) {
    const int NUM_CAMERAS = (int) imageSize.size(), NUM_FRAMES = (int)visibility_mat[0].size();
    for (int c = 0; c < NUM_CAMERAS; c++) {
        const auto img_area = (float)(imageSize[c].width * imageSize[c].height);
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (!visibility_mat[c][f])
                continue;
            CV_Assert((imagePoints[c][f].type() == CV_32F && imagePoints[c][f].cols == 2) || imagePoints[c][f].type() == CV_32FC2);
            std::vector<int> hull;
            const auto img_pts = imagePoints[c][f];
            const auto * const image_pts_ptr = (float *) img_pts.data;
            convexHull(img_pts, hull, true/*has to be clockwise*/, false/*indices*/);
            float area = 0;
            int j = hull.back();
            // http://alienryderflex.com/polygon_area/
            for (int i : hull) {
                area += (image_pts_ptr[j*2] + image_pts_ptr[i*2])*(image_pts_ptr[j*2+1] - image_pts_ptr[i*2+1]);
                j = i;
            }
            points_ratio_area[c][f] = area*.5f / img_area;
        }
    }
}

static void selectPairsBFS (std::vector<std::pair<int,int>> &pairs, int NUM_CAMERAS, const std::vector<int> &parent) {
    // find pairs using Breadth First Search graph traversing
    // it is important to keep this order of pairs, since it is easier
    // to find relative views wrt to 0-th camera.
    std::vector<int> nodes = {0};
    pairs.reserve(NUM_CAMERAS-1);
    while (!nodes.empty()) {
        std::vector<int> new_nodes;
        for (int n : nodes) {
            for (int c = 0; c < NUM_CAMERAS; c++) {
                if (parent[c] == n) {
                    pairs.emplace_back(std::make_pair(n, c));
                    new_nodes.emplace_back(c);
                }
            }
        }
        nodes = new_nodes;
    }
}

static double getScaleOfObjPoints (int NUM_PATTERN_PTS, const Mat &obj_pts, bool obj_points_in_rows) {
    double scale_3d_pts = 0.0;
    // compute scale of 3D points as the maximum pairwise distance
    for (int i = 0; i < NUM_PATTERN_PTS; i++) {
        for (int j = i+1; j < NUM_PATTERN_PTS; j++) {
            double dist;
            if (obj_points_in_rows) {
                dist = norm(obj_pts.row(i)-obj_pts.row(j), NORM_L2SQR);
            } else {
                dist = norm(obj_pts.col(i)-obj_pts.col(j), NORM_L2SQR);
            }
            if (scale_3d_pts < dist) {
                scale_3d_pts = dist;
            }
        }
    }
    return scale_3d_pts;
}

static void thresholdPatternCameraAngles (int NUM_PATTERN_PTS, double THR_PATTERN_CAMERA_ANGLES,
        const std::vector<Mat> &objPoints_norm, const std::vector<std::vector<Vec3d>> &rvecs_all,
        std::vector<std::vector<Vec3d>> &opt_axes, std::vector<std::vector<bool>> &is_valid_angle2pattern) {
    const int NUM_FRAMES = (int)objPoints_norm.size(), NUM_CAMERAS = (int)rvecs_all.size();
    is_valid_angle2pattern = std::vector<std::vector<bool>>(NUM_CAMERAS, std::vector<bool>(NUM_FRAMES, true));
    int pattern1 = -1, pattern2 = -1, pattern3 = -1;
    for (int f = 0; f < NUM_FRAMES; f++) {
        double norm_normal = 0;
        if (pattern1 == -1) {
            // take non colinear 3 points and save them
            for (int p1 = 0; p1 < NUM_PATTERN_PTS; p1++) {
                for (int p2 = p1+1; p2 < NUM_PATTERN_PTS; p2++) {
                    for (int p3 = NUM_PATTERN_PTS-1; p3 > p2; p3--) { // start from the last point
                        Mat pattern_normal = (objPoints_norm[f].row(p2)-objPoints_norm[f].row(p1))
                                    .cross(objPoints_norm[f].row(p3)-objPoints_norm[f].row(p1));
                        norm_normal = norm(pattern_normal, NORM_L2SQR);
                        if (norm_normal > 1e-6) {
                            pattern1 = p1;
                            pattern2 = p2;
                            pattern3 = p3;
                            norm_normal = sqrt(norm_normal);
                            break;
                        }
                    }
                    if (pattern1 != -1) break;
                }
                if (pattern1 != -1) break;
            }
            if (pattern1 == -1) {
                CV_Error(Error::StsBadArg, "Pattern points are collinear!");
            }
        }
        Vec3d pattern_normal = (objPoints_norm[f].row(pattern2)-objPoints_norm[f].row(pattern1)).
                  cross(objPoints_norm[f].row(pattern3)-objPoints_norm[f].row(pattern1));
        norm_normal = norm(pattern_normal);
        pattern_normal /= norm_normal;

        for (int c = 0; c < NUM_CAMERAS; c++) {
            Matx33d R;
            Rodrigues(rvecs_all[c][f], R);
            opt_axes[c][f] = Vec3d(Mat(R.row(2)));
            const double angle = acos(opt_axes[c][f].dot(pattern_normal));
            is_valid_angle2pattern[c][f] = min(M_PI-angle, angle) < THR_PATTERN_CAMERA_ANGLES;
        }
    }
}

static void pairwiseStereoCalibration (const std::vector<std::pair<int,int>> &pairs,
        const std::vector<bool> &is_fisheye_vec, const std::vector<Mat> &objPoints_norm,
        const std::vector<std::vector<Mat>> &imagePoints, const std::vector<std::vector<int>> &overlaps,
        const std::vector<std::vector<bool>> &visibility_mat, const std::vector<Mat> &Ks,
        const std::vector<Mat> &distortions, std::vector<Matx33d> &Rs_vec, std::vector<Vec3d> &Ts_vec,
        int flags_extrinsics) {
    const int NUM_FRAMES = (int) objPoints_norm.size();
    for (const auto &pair : pairs) {
        const int c1 = pair.first, c2 = pair.second, overlap = overlaps[c1][c2];
        // prepare image points of two cameras and grid points
        std::vector<Mat> image_points1, image_points2, grid_points;
        grid_points.reserve(overlap);
        image_points1.reserve(overlap);
        image_points2.reserve(overlap);
        const bool are_fisheye_cams = is_fisheye_vec[c1] && is_fisheye_vec[c2];
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (visibility_mat[c1][f] && visibility_mat[c2][f]) {
                grid_points.emplace_back((are_fisheye_cams && objPoints_norm[f].channels() != 3) ?
                                         objPoints_norm[f].reshape(3): objPoints_norm[f]);
                image_points1.emplace_back((are_fisheye_cams && imagePoints[c1][f].channels() != 2) ?
                                           imagePoints[c1][f].reshape(2) : imagePoints[c1][f]);
                image_points2.emplace_back((are_fisheye_cams && imagePoints[c2][f].channels() != 2) ?
                                         imagePoints[c2][f].reshape(2) : imagePoints[c2][f]);
            }
        }
        Matx33d R;
        Vec3d T;
        // image size does not matter since intrinsics are used
        if (are_fisheye_cams) {
            fisheye::stereoCalibrate(grid_points, image_points1, image_points2,
                            Ks[c1], distortions[c1],
                            Ks[c2], distortions[c2],
                            Size(), R, T, flags_extrinsics);
        } else {
            stereoCalibrate(grid_points, image_points1, image_points2,
                            Ks[c1], distortions[c1],
                            Ks[c2], distortions[c2],
                            Size(), R, T, noArray(), noArray(), noArray(), flags_extrinsics);
        }

        // R_0 = I
        // R_ij = R_i R_j^T     =>  R_i = R_ij R_j
        // t_ij = ti - R_ij tj  =>  t_i = t_ij + R_ij t_j
        if (c1 == 0) {
            Rs_vec[c2] = R;
            Ts_vec[c2] = T;
        } else {
            Rs_vec[c2] = Matx33d(Mat(R * Rs_vec[c1]));
            Ts_vec[c2] = Vec3d(Mat(T + R * Ts_vec[c1]));
        }
    }
}

static void optimizeLM (std::vector<double> &param, const RobustFunction &robust_fnc, const TermCriteria &termCrit,
         const std::vector<bool> &valid_frames, const std::vector<std::vector<bool>> &visibility_mat,
         const std::vector<Mat> &objPoints_norm, const std::vector<std::vector<Mat>> &imagePoints,
         const std::vector<Mat> &Ks, const std::vector<Mat> &distortions,
         const std::vector<bool> &is_fisheye_vec, int NUM_PATTERN_PTS) {
    const int NUM_FRAMES = (int) objPoints_norm.size(), NUM_CAMERAS = (int)visibility_mat.size();
    int iters_lm = 0, cnt_valid_frame = 0;
    auto lmcallback = [&](InputOutputArray _param, OutputArray JtErr_, OutputArray JtJ_, double& errnorm) {
        auto * param_p = _param.getMat().ptr<double>();
        errnorm = 0;
        cnt_valid_frame = 0;
        for (int i = 0; i < NUM_FRAMES; i++ ) {
            if (!valid_frames[i]) continue;
            for (int k = 1; k < NUM_CAMERAS; k++ ) { // skip first camera as there is nothing to optimize
                if (!visibility_mat[k][i]) continue;
                const int cam_idx = (k-1)*6; // extrinsics
                const auto * const pose_k = param_p + cam_idx;
                Vec3d om_0ToK(pose_k[0], pose_k[1], pose_k[2]), om[2];
                Vec3d T_0ToK(pose_k[3], pose_k[4], pose_k[5]), T[2];
                Matx33d dr3dr1, dr3dr2, dt3dr2, dt3dt1, dt3dt2;

                auto * pi = param_p + (cnt_valid_frame+NUM_CAMERAS-1)*6; // get rvecs / tvecs
                om[0] = Vec3d(pi[0], pi[1], pi[2]);
                T[0] = Vec3d(pi[3], pi[4], pi[5]);

                if( JtJ_.needed() || JtErr_.needed() )
                    composeRT( om[0], T[0], om_0ToK, T_0ToK, om[1], T[1], dr3dr1, noArray(),
                               dr3dr2, noArray(), noArray(), dt3dt1, dt3dr2, dt3dt2 );
                else
                    composeRT( om[0], T[0], om_0ToK, T_0ToK, om[1], T[1] );

                // get object points
                Mat objpt_i = objPoints_norm[i].reshape(3, 1);
                objpt_i.convertTo(objpt_i, CV_64FC3);

                Mat err( NUM_PATTERN_PTS*2, 1, CV_64F ), tmpImagePoints = err.reshape(2, 1);
                Mat Je( NUM_PATTERN_PTS*2, 6, CV_64F ), J_0ToK( NUM_PATTERN_PTS*2, 6, CV_64F );
                Mat dpdrot = Je.colRange(0, 3), dpdt = Je.colRange(3, 6); // num_points*2 x 3 each
                // get image points
                Mat imgpt_ik = imagePoints[k][i].reshape(2, 1);
                imgpt_ik.convertTo(imgpt_ik, CV_64FC2);

                if (is_fisheye_vec[k]) {
                    if( JtJ_.needed() || JtErr_.needed() ) {
                        Mat jacobian; // of size num_points*2  x  15 (2 + 2 + 1 + 4 + 3 + 3; // f, c, alpha, k, om, T)
                        fisheye::projectPoints(objpt_i, tmpImagePoints, om[1], T[1], Ks[k], distortions[k], 0, jacobian);
                        jacobian.colRange(8,11).copyTo(dpdrot);
                        jacobian.colRange(11,14).copyTo(dpdt);
                    } else
                        fisheye::projectPoints(objpt_i, tmpImagePoints, om[1], T[1], Ks[k], distortions[k]);
                } else {
                    if( JtJ_.needed() || JtErr_.needed() )
                        projectPoints(objpt_i, om[1], T[1], Ks[k], distortions[k],
                                        tmpImagePoints, dpdrot, dpdt, noArray(), noArray(), noArray(), noArray());
                    else
                        projectPoints(objpt_i, om[1], T[1], Ks[k], distortions[k], tmpImagePoints);
                }
                subtract( tmpImagePoints, imgpt_ik, tmpImagePoints);
                const double robust_l2_norm = multiview::robustWrapper(tmpImagePoints, robust_fnc);
                errnorm += robust_l2_norm;

                if (JtJ_.needed()) {
                    Mat JtErr = JtErr_.getMat(), JtJ = JtJ_.getMat();
                    const int eofs = (cnt_valid_frame+NUM_CAMERAS-1)*6;
                    assert( JtJ_.needed() && JtErr_.needed() );
                    // JtJ : NUM_PARAMS x NUM_PARAMS, JtErr : NUM_PARAMS x 1

                    if( k != 0 ) { // k == 1 for stereoCalibrate
                        // d(err_{x|y}R) ~ de3
                        // convert de3/{dr3,dt3} => de3{dr1,dt1} & de3{dr2,dt2}
                        for (int p = 0; p < NUM_PATTERN_PTS*2; p++ ) {
                            Mat de3dr3( 1, 3, CV_64F, Je.ptr(p));
                            Mat de3dt3( 1, 3, CV_64F, de3dr3.ptr<double>() + 3 );
                            Mat de3dr2( 1, 3, CV_64F, J_0ToK.ptr(p) );
                            Mat de3dt2( 1, 3, CV_64F, de3dr2.ptr<double>() + 3 );
                            Matx13d de3dr1, de3dt1;

                            gemm(de3dr3, dr3dr1, 1, noArray(), 0, de3dr1);
                            gemm(de3dt3, dt3dt1, 1, noArray(), 0, de3dt1);

                            gemm(de3dr3, dr3dr2, 1, noArray(), 0, de3dr2);
                            gemm(de3dt3, dt3dr2, 1, de3dr2, 1, de3dr2);
                            gemm(de3dt3, dt3dt2, 1, noArray(), 0, de3dt2);

                            Mat(de3dr1).copyTo(de3dr3);
                            Mat(de3dt1).copyTo(de3dt3);
                        }

                        JtJ(Rect((k-1)*6, (k-1)*6, 6, 6)) += J_0ToK.t()*J_0ToK; // 6 x ni * ni x 6
                        JtJ(Rect(eofs, (k-1)*6, 6, 6)) = J_0ToK.t()*Je; // 6 x ni * ni x 6
                        JtErr.rowRange((k-1)*6, (k-1)*6+6) += J_0ToK.t()*err;
                    }
                    JtJ(Rect(eofs, eofs, 6, 6)) += Je.t()*Je;
                    JtErr.rowRange(eofs, eofs + 6) += Je.t()*err;
                }
            }
            cnt_valid_frame++;
        }
        iters_lm += 1;
        return true;
    };
    LevMarq solver(param, lmcallback,
       LevMarq::Settings()
               .setMaxIterations(termCrit.maxCount)
               .setStepNormTolerance(termCrit.epsilon)
               .setSmallEnergyTolerance(termCrit.epsilon * termCrit.epsilon),
           noArray()/*mask, all variables to optimize*/);
    solver.optimize();
}

static void checkConnected (const std::vector<std::vector<bool>> &visibility_mat) {
    const int NUM_CAMERAS = (int)visibility_mat.size(), NUM_FRAMES = (int)visibility_mat[0].size();
    std::vector<bool> visited(NUM_CAMERAS, false);
    std::function<void(int)> dfs_search;
    dfs_search = [&] (int cam) {
        visited[cam] = true;
        for (int cam2 = 0; cam2 < NUM_CAMERAS; cam2++) {
            if (!visited[cam2]) {
                for (int f = 0; f < NUM_FRAMES; f++) {
                    if (visibility_mat[cam][f] && visibility_mat[cam2][f]) {
                        dfs_search(cam2);
                        break;
                    }
                }
            }
        }
    };
    dfs_search(0);
    for (int c = 0; c < NUM_CAMERAS; c++) {
        if (! visited[c]) {
            std::string isolated_cameras = "", visited = "";
            for (int i = 0; i < NUM_CAMERAS; i++) {
                if (!visited[i]) {
                    if (isolated_cameras != "")
                        isolated_cameras += ", ";
                    isolated_cameras += std::to_string(i);
                } else {
                    if (visited != "")
                        visited += ", ";
                    visited += std::to_string(i);
                }
            }
            CV_Error(Error::StsBadArg, "Isolated cameras (or components) "+isolated_cameras+" from the connected component "+visited+"!");
        }
    }
}
}

bool calibrateMultiview (InputArrayOfArrays objPoints, const std::vector<std::vector<Mat>> &imagePoints,
        const std::vector<Size> &imageSize, const Mat &visibility,
        OutputArrayOfArrays Rs, OutputArrayOfArrays Ts, std::vector<Mat> &Ks, std::vector<Mat> &distortions,
        OutputArrayOfArrays rvecs0, OutputArrayOfArrays tvecs0, InputArray is_fisheye,
        OutputArray errors_per_frame, OutputArray output_pairs, bool USE_INTRINSICS_GUESS, int flags_intrinsics) {

    CV_CheckEQ((int)objPoints.empty(), 0, "Objects points must not be empty!");
    CV_CheckEQ((int)imagePoints.empty(), 0, "Image points must not be empty!");
    CV_CheckEQ((int)imageSize.empty(), 0, "Image size per camera must not be empty!");
    CV_CheckEQ((int)visibility.empty(), 0, "Visibility matrix must not be empty!");
    CV_CheckEQ((int)is_fisheye.empty(), 0, "Fisheye mask must not be empty!");
    // equal number of cameras
    CV_Assert(imageSize.size() == imagePoints.size());
    CV_Assert(visibility.rows == std::max(is_fisheye.rows(), is_fisheye.cols()));
    CV_Assert(visibility.rows == (int)imageSize.size());
    CV_Assert(visibility.cols == std::max(objPoints.rows(), objPoints.cols())); // equal number of frames
    CV_Assert(Rs.isMatVector() == Ts.isMatVector());
    if (USE_INTRINSICS_GUESS) {
        CV_Assert(Ks.size() == distortions.size() && Ks.size() == imageSize.size());
    }
    // normalize object points
    const Mat obj_pts_0 = objPoints.getMat(0);
    CV_Assert((obj_pts_0.type() == CV_32F && (obj_pts_0.rows == 3 || obj_pts_0.cols == 3)) ||
              (obj_pts_0.type() == CV_32FC3 && (obj_pts_0.rows == 1 || obj_pts_0.cols == 1)));
    const bool obj_points_in_rows = obj_pts_0.cols == 3;
    const int NUM_CAMERAS = (int)visibility.rows, NUM_FRAMES = (int)visibility.cols;
    const int NUM_PATTERN_PTS = obj_points_in_rows ? obj_pts_0.rows : obj_pts_0.cols;
    const double scale_3d_pts = multiview::getScaleOfObjPoints(NUM_PATTERN_PTS, obj_pts_0, obj_points_in_rows);

    std::vector<Mat> objPoints_norm;
    objPoints_norm.reserve(NUM_FRAMES);
    for (int i = 0; i < NUM_FRAMES; i++) {
        if (obj_points_in_rows)
            objPoints_norm.emplace_back(objPoints.getMat(i)*(1/scale_3d_pts));
        else
            objPoints_norm.emplace_back(objPoints.getMat(i).t()*(1/scale_3d_pts));
        objPoints_norm[i] = objPoints_norm[i].reshape(1);
    }

    ////////////////////////////////////////////////
    std::vector<int> num_visible_frames_per_camera(NUM_CAMERAS);
    std::vector<bool> valid_frames(NUM_FRAMES, false);
    // process input and count all visible frames and points

    std::vector<bool> is_fisheye_vec(NUM_CAMERAS);
    std::vector<std::vector<bool>> visibility_mat(NUM_CAMERAS, std::vector<bool>(NUM_FRAMES));
    // convert to boolean
    Mat visibility_ = visibility, is_fisheye_mat = is_fisheye.getMat();
    visibility_.convertTo(visibility_, CV_8U);
    is_fisheye_mat.convertTo(is_fisheye_mat, CV_8U);
    const auto * const visibility_ptr = visibility_.data, * const is_fisheye_ptr = is_fisheye_mat.data;
    int num_fisheye_cameras = 0;
    for (int c = 0; c < NUM_CAMERAS; c++) {
        is_fisheye_vec[c] = is_fisheye_ptr[c];
        if (is_fisheye_vec[c]) num_fisheye_cameras++;
        int num_visible_frames = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            visibility_mat[c][f] = visibility_ptr[c*NUM_FRAMES + f];
            if (visibility_mat[c][f]) {
                num_visible_frames++;
                valid_frames[f] = true; // if frame is visible by at least one camera then count is as a valid one
            }
        }
        if (num_visible_frames == 0) {
            CV_Error(Error::StsBadArg, "camera "+std::to_string(c)+" has no visible frames!");
            // return false;
        }
        num_visible_frames_per_camera[c] = num_visible_frames;
    }
    multiview::checkConnected(visibility_mat);
    int flags_extrinsics = CALIB_FIX_INTRINSIC;
    if (num_fisheye_cameras != 0 && num_fisheye_cameras != NUM_CAMERAS) {
        // cameras are mixed (fisheye and pinhole)
        // use standard pinhole calibration for this case
        for (int i = 0; i < NUM_CAMERAS; i++) {
            is_fisheye_vec[i] = false;
        }
        // update flags, use rational model and no tangential coefficients
        flags_intrinsics = CALIB_RATIONAL_MODEL+CALIB_ZERO_TANGENT_DIST+CALIB_FIX_K5+CALIB_FIX_K6;
        flags_extrinsics += CALIB_RATIONAL_MODEL;
    }

    std::vector<std::vector<float>> points_ratio_area(NUM_CAMERAS, std::vector<float>(NUM_FRAMES));
    multiview::imagePointsArea(imageSize, visibility_mat, imagePoints, points_ratio_area);

    const double THR_PATTERN_CAMERA_ANGLES = 160*M_PI/180;
    std::vector<std::vector<Vec3d>> rvecs_all(NUM_CAMERAS, std::vector<Vec3d>(NUM_FRAMES)),
        tvecs_all(NUM_CAMERAS, std::vector<Vec3d>(NUM_FRAMES)),
        opt_axes(NUM_CAMERAS, std::vector<Vec3d>(NUM_FRAMES));

    std::vector<int> camera_rt_best(NUM_FRAMES, -1);
    std::vector<double> camera_rt_errors(NUM_FRAMES, std::numeric_limits<double>::max());
    const double WARNING_RMSE = 15.;
    if (!USE_INTRINSICS_GUESS) {
        // calibrate each camera independently to find intrinsic parameters - K and distortion coefficients
        distortions = std::vector<Mat>(NUM_CAMERAS);
        Ks = std::vector<Mat>(NUM_CAMERAS);
        for (int camera = 0; camera < NUM_CAMERAS; camera++) {
            Mat rvecs, tvecs;
            std::vector<Mat> obj_points_, img_points_;
            std::vector<double> errors_per_view;
            obj_points_.reserve(num_visible_frames_per_camera[camera]);
            img_points_.reserve(num_visible_frames_per_camera[camera]);
            for (int f = 0; f < NUM_FRAMES; f++) {
                if (visibility_mat[camera][f]) {
                    obj_points_.emplace_back((is_fisheye_vec[camera] && objPoints_norm[f].channels() != 3) ?
                        objPoints_norm[f].reshape(3): objPoints_norm[f]);
                    img_points_.emplace_back((is_fisheye_vec[camera] && imagePoints[camera][f].channels() != 2) ?
                        imagePoints[camera][f].reshape(2) : imagePoints[camera][f]);
                }
            }
            double repr_err;
            if (is_fisheye_vec[camera]) {
                repr_err = fisheye::calibrate(obj_points_, img_points_, imageSize[camera],
                    Ks[camera], distortions[camera], rvecs, tvecs, flags_intrinsics);
                // calibrate does not compute error per view, so compute it manually
                errors_per_view = std::vector<double>(obj_points_.size());
                for (int f = 0; f < (int) obj_points_.size(); f++) {
                    errors_per_view[f] = multiview::computeReprojectionRMSE(obj_points_[f],
                        img_points_[f], Ks[camera], distortions[camera], rvecs.row(f), tvecs.row(f), noArray(), noArray(), true);
                }
            } else {
                repr_err = calibrateCamera(obj_points_, img_points_, imageSize[camera], Ks[camera], distortions[camera],
                   rvecs, tvecs, noArray(), noArray(), errors_per_view, flags_intrinsics);
            }
            CV_LOG_IF_WARNING(NULL, repr_err > WARNING_RMSE, "Warning! Mean RMSE of intrinsics calibration is higher than "+std::to_string(WARNING_RMSE));
            int cnt_visible_frame = 0;
            for (int f = 0; f < NUM_FRAMES; f++) {
                if (visibility_mat[camera][f]) {
                    rvecs_all[camera][f] = Vec3d(Mat(3, 1, CV_64F, rvecs.row(cnt_visible_frame).data));
                    tvecs_all[camera][f] = Vec3d(Mat(3, 1, CV_64F, tvecs.row(cnt_visible_frame).data));
                    if (camera_rt_errors[f] > errors_per_view[cnt_visible_frame]) {
                        camera_rt_errors[f] = errors_per_view[cnt_visible_frame];
                        camera_rt_best[f] = camera;
                    }
                    cnt_visible_frame++;
                }
            }
        }
    } else {
        // use PnP to compute rvecs and tvecs
        for (int i = 0; i < NUM_FRAMES; i++) {
            for (int k = 0; k < NUM_CAMERAS; k++) {
                if (!visibility_mat[k][i]) continue;
                Vec3d rvec, tvec;
                solvePnP(objPoints_norm[i], imagePoints[k][i], Ks[k], distortions[k], rvec, tvec, false, SOLVEPNP_ITERATIVE);
                rvecs_all[k][i] = rvec;
                tvecs_all[k][i] = tvec;
                const auto err = multiview::computeReprojectionRMSE(objPoints_norm[i], imagePoints[k][i], Ks[k], distortions[k], Mat(rvec), Mat(tvec), noArray(), noArray(), is_fisheye_vec[k]);
                if (camera_rt_errors[i] > err) {
                    camera_rt_errors[i] = err;
                    camera_rt_best[i] = k;
                }
            }
        }
    }

    std::vector<std::vector<bool>> is_valid_angle2pattern;
    multiview::thresholdPatternCameraAngles(NUM_PATTERN_PTS, THR_PATTERN_CAMERA_ANGLES, objPoints_norm, rvecs_all, opt_axes, is_valid_angle2pattern);

    std::vector<Matx33d> Rs_vec(NUM_CAMERAS);
    std::vector<Vec3d> Ts_vec(NUM_CAMERAS);
    Rs_vec[0] = Matx33d ::eye();
    Ts_vec[0] = Vec3d::zeros();

    if (NUM_CAMERAS == 1)
        return true;

    std::vector<int> parent;
    std::vector<std::vector<int>> overlaps;
    if (! multiview::maximumSpanningTree(NUM_CAMERAS, NUM_FRAMES, visibility_mat, parent, overlaps, opt_axes,
            is_valid_angle2pattern, points_ratio_area, .5, 1.0)) {
        // failed to find suitable pairs with constraints!
        CV_Error(Error::StsInternal, "Failed to build tree for stereo calibration.");
//        return false;
    }

    std::vector<std::pair<int,int>> pairs;
    multiview::selectPairsBFS (pairs, NUM_CAMERAS, parent);

    if ((int)pairs.size() != NUM_CAMERAS-1) {
        CV_Error(Error::StsInternal, "Failed to build tree for stereo calibration. Incorrect number of pairs.");
//        return false;
    }
    if (output_pairs.needed()) {
        Mat pairs_mat = Mat_<int>(NUM_CAMERAS-1, 2);
        auto * pairs_ptr = (int *) pairs_mat.data;
        for (const auto &p : pairs) {
            (*pairs_ptr++) = p.first;
            (*pairs_ptr++) = p.second;
        }
        pairs_mat.copyTo(output_pairs);
    }
    multiview::pairwiseStereoCalibration(pairs, is_fisheye_vec, objPoints_norm, imagePoints,
         overlaps, visibility_mat, Ks, distortions, Rs_vec, Ts_vec, flags_extrinsics);

    const int NUM_VALID_FRAMES = countNonZero(valid_frames);
    const int nparams = (NUM_VALID_FRAMES + NUM_CAMERAS - 1) * 6; // rvecs + tvecs (6)
    std::vector<double> param(nparams, 0.);

    // use found relative extrinsics to initialize parameters
    for (int c = 1; c < NUM_CAMERAS; c++) {
        Vec3d rvec;
        Rodrigues(Rs_vec[c], rvec);
        memcpy(&param[0]+(c-1)*6  , rvec.val, 3*sizeof(double));
        memcpy(&param[0]+(c-1)*6+3, Ts_vec[c].val, 3*sizeof(double));
    }

    // use found rvecs / tvecs or estimate them to initialize rest of parameters
    int cnt_valid_frame = 0;
    for (int i = 0; i < NUM_FRAMES; i++ ) {
        if (!valid_frames[i]) continue;
        Vec3d rvec_0, tvec_0;
        if (camera_rt_best[i] != 0) {
            // convert rvecs / tvecs from k-th camera to the first one

            // formulas for relative rotation / translation
            // R = R_k R0^T       => R_k = R R_0
            // t = t_k - R t_0    => t_k = t + R t_0

            // initial camera R_0 = I, t_0 = 0 is fixed to R(rvec_0) and tvec_0
            // R_0 = R(rvec_0)
            // t_0 = tvec_0

            // R'_k = R(rvec_k) = R_k R_0       => R_0 = R_k^T R(rvec_k)
            // t'_k = tvec_k = t_k + R_k t_0    => t_0 = R_k^T (tvec_k - t_k)
            const int rt_best_idx = camera_rt_best[i];
            Matx33d R_k;
            Rodrigues(rvecs_all[rt_best_idx][i], R_k);
            tvec_0 = Rs_vec[rt_best_idx].t() * (tvecs_all[rt_best_idx][i] - Ts_vec[rt_best_idx]);
            Rodrigues(Rs_vec[rt_best_idx].t() * R_k, rvec_0);
        } else {
            rvec_0 = rvecs_all[0][i];
            tvec_0 = tvecs_all[0][i];
        }

        // save rvecs0 / tvecs0 parameters
        memcpy(&param[0]+(cnt_valid_frame+NUM_CAMERAS-1)*6  , rvec_0.val, 3*sizeof(double));
        memcpy(&param[0]+(cnt_valid_frame+NUM_CAMERAS-1)*6+3, tvec_0.val, 3*sizeof(double));
        cnt_valid_frame++;
    }

    TermCriteria termCrit (TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-6);
    const float RBS_FNC_SCALE = 30;
    multiview::RobustExpFunction robust_fnc(RBS_FNC_SCALE);
    multiview::optimizeLM(param, robust_fnc, termCrit, valid_frames, visibility_mat, objPoints_norm, imagePoints, Ks, distortions, is_fisheye_vec, NUM_PATTERN_PTS);
    const auto * const params = &param[0];

    // extract extrinsics (R_i, t_i) for i = 1 ... NUM_CAMERAS:
    const bool rt_mat_vec = Rs.isMatVector();
    Mat rs, ts;
    if (rt_mat_vec) {
        Rs.create(NUM_CAMERAS, 1, CV_64F);
        Ts.create(NUM_CAMERAS, 1, CV_64F);
    } else {
        rs = Mat_<double>(NUM_CAMERAS, 3);
        ts = Mat_<double>(NUM_CAMERAS, 3);
    }
    for (int c = 0; c < NUM_CAMERAS; c++) {
        Mat r_store, t_store;
        if (rt_mat_vec) {
            Rs.create(3, 1, CV_64F, c, true);
            Ts.create(3, 1, CV_64F, c, true);
            r_store = Rs.getMat(c);
            t_store = Ts.getMat(c);
        } else {
            r_store = rs.row(c);
            t_store = ts.row(c);
        }
        if (c == 0) {
            memcpy(r_store.ptr(), Vec3d(0,0,0).val, 3*sizeof(double));
            memcpy(t_store.ptr(), Vec3d(0,0,0).val, 3*sizeof(double));
        } else {
            memcpy(r_store.ptr(), params + (c-1)*6, 3*sizeof(double));
            memcpy(t_store.ptr(), params + (c-1)*6+3, 3*sizeof(double)); // and de-normalize translation
            t_store *= scale_3d_pts;
        }
        Mat R;
        Rodrigues(r_store, R);
    }
    if (! rt_mat_vec) {
        rs.copyTo(Rs);
        ts.copyTo(Ts);
    }
    Mat rvecs0_, tvecs0_;
    if (rvecs0.needed() || errors_per_frame.needed()) {
        const bool is_mat_vec = rvecs0.needed() && rvecs0.isMatVector();
        if (is_mat_vec) {
            rvecs0.create(NUM_FRAMES, 1, CV_64F);
        } else {
            rvecs0_ = Mat_<double>(NUM_FRAMES, 3);
        }
        cnt_valid_frame = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (!valid_frames[f]) continue;
            if (is_mat_vec)
                rvecs0.create(3, 1, CV_64F, f, true);
            Mat store = is_mat_vec ? rvecs0.getMat(f) : rvecs0_.row(f);
            memcpy(store.ptr(), params + (cnt_valid_frame + NUM_CAMERAS - 1)*6, 3*sizeof(double));
            cnt_valid_frame += 1;
        }
        if (!is_mat_vec && rvecs0.needed())
            rvecs0_.copyTo(rvecs0);
    }

    if (tvecs0.needed() || errors_per_frame.needed()) {
        const bool is_mat_vec = tvecs0.needed() && tvecs0.isMatVector();
        if (is_mat_vec) {
            tvecs0.create(NUM_FRAMES, 1, CV_64F);
        } else {
            tvecs0_ = Mat_<double>(NUM_FRAMES, 3);
        }
        cnt_valid_frame = 0;
        for (int f = 0; f < NUM_FRAMES; f++) {
            if (!valid_frames[f]) continue;
            if (is_mat_vec)
                tvecs0.create(3, 1, CV_64F, f, true);
            Mat store = is_mat_vec ? tvecs0.getMat(f) : tvecs0_.row(f);
            memcpy(store.ptr(), params + (cnt_valid_frame + NUM_CAMERAS - 1)*6+3, 3*sizeof(double));
            store *= scale_3d_pts;
            cnt_valid_frame += 1;
        }
        if (!is_mat_vec && tvecs0.needed())
            tvecs0_.copyTo(tvecs0);
    }
    if (errors_per_frame.needed()) {
        const bool rvecs_mat_vec = rvecs0.needed() && rvecs0.isMatVector(), tvecs_mat_vec = tvecs0.needed() && tvecs0.isMatVector();
        const bool r_mat_vec = Rs.isMatVector(), t_mat_vec = Ts.isMatVector();
        Mat errs = Mat_<double>(NUM_CAMERAS, NUM_FRAMES);
        auto * errs_ptr = (double *) errs.data;
        for (int c = 0; c < NUM_CAMERAS; c++) {
            const Mat rvec = r_mat_vec ? Rs.getMat(c) : Rs.getMat().row(c).t();
            const Mat tvec = t_mat_vec ? Ts.getMat(c) : Ts.getMat().row(c).t();
            for (int f = 0; f < NUM_FRAMES; f++) {
                if (visibility_mat[c][f]) {
                    const Mat rvec0 = rvecs_mat_vec ? rvecs0.getMat(f) : rvecs0_.row(f).t();
                    const Mat tvec0 = tvecs_mat_vec ? tvecs0.getMat(f) : tvecs0_.row(f).t();
                    (*errs_ptr++) = multiview::computeReprojectionRMSE(objPoints.getMat(f), imagePoints[c][f], Ks[c],
                         distortions[c], rvec0, tvec0, rvec, tvec, is_fisheye_vec[c]);
                } else (*errs_ptr++) = -1.0;
            }
        }
        errs.copyTo(errors_per_frame);
    }
    return true;
}
}
