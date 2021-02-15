// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
static double getError2EpipLines (const Mat &F, const Mat &pts1, const Mat &pts2, const Mat &mask) {
    Mat points1, points2;
    vconcat(pts1, Mat::ones(1, pts1.cols, pts1.type()), points1);
    vconcat(pts2, Mat::ones(1, pts2.cols, pts2.type()), points2);

    double mean_error = 0;
    for (int pt = 0; pt < (int) mask.total(); pt++)
        if (mask.at<uchar>(pt)) {
            const Mat l2 = F     * points1.col(pt);
            const Mat l1 = F.t() * points2.col(pt);
            mean_error += (fabs(points1.col(pt).dot(l1)) / sqrt(pow(l1.at<double>(0), 2) + pow(l1.at<double>(1), 2)) +
                           fabs(points2.col(pt).dot(l2) / sqrt(pow(l2.at<double>(0), 2) + pow(l2.at<double>(1), 2)))) / 2;
        }
    return mean_error / mask.total();
}
static int sgn(double val) { return (0 < val) - (val < 0); }

/*
 * @points3d - vector of Point3 or Mat of size Nx3
 * @planes - vector of found planes
 * @labels - vector of size point3d. Every point which has non-zero label is classified to this plane.
 */
static void getPlanes (InputArray points3d_, std::vector<int> &labels, std::vector<Vec4d> &planes, int desired_num_planes, double thr_, double conf_, int max_iters_) {
    Mat points3d = points3d_.getMat();
    points3d.convertTo(points3d, CV_64F); // convert points to have double precision
    if (points3d_.isVector())
        points3d = Mat((int)points3d.total(), 3, CV_64F, points3d.data);
    else {
        if (points3d.type() != CV_64F)
            points3d = points3d.reshape(1, (int)points3d.total()); // convert point to have 1 channel
        if (points3d.rows < points3d.cols)
            transpose(points3d, points3d); // transpose so points will be in rows
        CV_CheckEQ(points3d.cols, 3, "Invalid dimension of point");
    }

    /*
     * 3D plane fitting with RANSAC
     * @best_model contains coefficients [a b c d] s.t. ax + by + cz = d
     *
     */
    auto plane_ransac = [] (const Mat &pts, double thr, double conf, int max_iters, Vec4d &best_model, std::vector<bool> &inliers) {
        const int pts_size = pts.rows, max_lo_inliers = 15, max_lo_iters = 10;
        int best_inls = 0;
        if (pts_size < 3) return false;
        RNG rng;
        const auto * const points = (double *) pts.data;
        std::vector<int> min_sample(3);
        inliers = std::vector<bool>(pts_size);
        const double log_conf = log(1-conf);
        Vec4d model, lo_model;
        std::vector<int> random_pool (pts_size);
        for (int p = 0; p < pts_size; p++)
            random_pool[p] = p;

        // estimate plane coefficients using covariance matrix
        auto estimate = [&] (const std::vector<int> &sample, Vec4d &model_) {
            // https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
            const int n = static_cast<int>(sample.size());
            if (n < 3) return false;
            double sum_x = 0, sum_y = 0, sum_z = 0;
            for (int s : sample) {
                sum_x += points[3*s  ];
                sum_y += points[3*s+1];
                sum_z += points[3*s+2];
            }
            const double c_x = sum_x / n, c_y = sum_y / n, c_z = sum_z / n;
            double xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
            for (int s : sample) {
                const double x_ = points[3*s] - c_x, y_ = points[3*s+1] - c_y, z_ = points[3*s+2] - c_z;
                xx += x_*x_; yy += y_*y_; zz += z_*z_; xy += x_*y_; yz += y_*z_; xz += x_*z_;
            }
            xx /= n; yy /= n; zz /= n; xy /= n; yz /= n; xz /= n;
            Vec3d weighted_normal(0,0,0);
            const double det_x = yy*zz - yz*yz, det_y = xx*zz - xz*xz, det_z = xx*yy - xy*xy;
            Vec3d axis_x (det_x, xz*xz-xy*zz, xy*yz-xz*yy);
            Vec3d axis_y (xz*yz-xy*zz, det_y, xy*xz-yz*xx);
            Vec3d axis_z (xy*yz-xz*yy, xy*xz-yz*xx, det_z);
            weighted_normal += axis_x * det_x * det_x;
            weighted_normal += sgn(weighted_normal.dot(axis_y)) * axis_y * det_y * det_y;
            weighted_normal += sgn(weighted_normal.dot(axis_z)) * axis_z * det_z * det_z;
            weighted_normal /= norm(weighted_normal);
            if (std::isinf(weighted_normal(0)) ||
                std::isinf(weighted_normal(1)) ||
                std::isinf(weighted_normal(2))) return false;
            // find plane model from normal and centroid
            model_ = Vec4d(weighted_normal(0), weighted_normal(1), weighted_normal(2),
                           weighted_normal.dot(Vec3d(c_x, c_y, c_z)));
            return true;
        };

        // calculate number of inliers
        auto getInliers = [&] (const Vec4d &model_) {
            const double a = model_(0), b = model_(1), c = model_(2), d = model_(3);
            int num_inliers = 0;
            std::fill(inliers.begin(), inliers.end(), false);
            for (int p = 0; p < pts_size; p++) {
                inliers[p] = fabs(a * points[3*p] + b * points[3*p+1] + c * points[3*p+2] - d) < thr;
                if (inliers[p]) num_inliers++;
                if (num_inliers + pts_size - p < best_inls) break;
            }
            return num_inliers;
        };
        // main RANSAC loop
        for (int iters = 0; iters < max_iters; iters++) {
            // find minimal sample: 3 points
            min_sample[0] = rng.uniform(0, pts_size);
            min_sample[1] = rng.uniform(0, pts_size);
            min_sample[2] = rng.uniform(0, pts_size);
            if (! estimate(min_sample, model))
                continue;
            int num_inliers = getInliers(model);
            if (num_inliers > best_inls) {
                // store so-far-the-best
                std::vector<bool> best_inliers = inliers;
                // do Local Optimization
                for (int lo_iter = 0; lo_iter < max_lo_iters; lo_iter++) {
                    std::vector<int> inliers_idx; inliers_idx.reserve(max_lo_inliers);
                    randShuffle(random_pool);
                    for (int p : random_pool) {
                        if (best_inliers[p]) {
                            inliers_idx.emplace_back(p);
                            if ((int)inliers_idx.size() >= max_lo_inliers)
                                break;
                        }
                    }
                    if (! estimate(inliers_idx, lo_model))
                        continue;
                    int lo_inls = getInliers(lo_model);
                    if (best_inls < lo_inls) {
                        best_model = lo_model;
                        best_inls = lo_inls;
                        best_inliers = inliers;
                    }
                }
                if (best_inls < num_inliers) {
                    best_model = model;
                    best_inls = num_inliers;
                }
                // update max iters
                // because points are quite noisy we need more iterations
                const double max_hyp = 3 * log_conf / log(1 - pow(double(best_inls) / pts_size, 3));
                if (! std::isinf(max_hyp) && max_hyp < max_iters)
                    max_iters = static_cast<int>(max_hyp);
            }
        }
        getInliers(best_model);
        return best_inls != 0;
    };

    labels = std::vector<int>(points3d.rows, 0);
    Mat pts3d_plane_fit = points3d.clone();
    // keep array of indices of points corresponding to original points3d
    std::vector<int> to_orig_pts_arr(pts3d_plane_fit.rows);
    for (int i = 0; i < (int) to_orig_pts_arr.size(); i++)
        to_orig_pts_arr[i] = i;
    for (int num_planes = 1; num_planes <= desired_num_planes; num_planes++) {
        Vec4d model;
        std::vector<bool> inl;
        if (!plane_ransac(pts3d_plane_fit, thr_, conf_, max_iters_, model, inl))
            break;
        planes.emplace_back(model);

        const int pts3d_size = pts3d_plane_fit.rows;
        pts3d_plane_fit = Mat();
        pts3d_plane_fit.reserve(points3d.rows);

        int cnt = 0;
        for (int p = 0; p < pts3d_size; p++) {
            if (! inl[p]) {
                // if point is not inlier to found plane - add it to next run
                to_orig_pts_arr[cnt] = to_orig_pts_arr[p];
                pts3d_plane_fit.push_back(points3d.row(to_orig_pts_arr[cnt]));
                cnt++;
            } else labels[to_orig_pts_arr[p]] = num_planes; // otherwise label this point
        }
    }
}

int main(int args, char** argv) {
    std::string data_file, image_dir;
    if (args < 3) {
       CV_Error(Error::StsBadArg,
                "Path to data file and directory to image files are missing!\nData file must have"
                " format:\n--------------\n image_name_1\nimage_name_2\nk11 k12 k13\n0   k22 k23\n"
                "0   0   1\n--------------\nIf image_name_{1,2} are not in the same directory as "
                "the data file then add argument with directory to image files.\nFor example: "
                "./essential_mat_reconstr essential_mat_data.txt ./");
    } else {
       data_file = argv[1];
       image_dir = argv[2];
    }
    std::ifstream file(data_file, std::ios_base::in);
    CV_CheckEQ((int)file.is_open(), 1, "Data file is not found!");
    std::string filename1, filename2;
    std::getline(file, filename1);
    std::getline(file, filename2);
    Mat image1 = imread(image_dir+filename1);
    Mat image2 = imread(image_dir+filename2);
    CV_CheckEQ((int)image1.empty(), 0, "Image 1 is not found!");
    CV_CheckEQ((int)image2.empty(), 0, "Image 2 is not found!");

    // read calibration
    Matx33d K;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            file >> K(i,j);
    file.close();

    Mat descriptors1, descriptors2;
    std::vector<KeyPoint> keypoints1, keypoints2;

    // detect points with SIFT
    Ptr<SIFT> detector = SIFT::create();
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);
    detector->compute(image1, keypoints1, descriptors1);
    detector->compute(image2, keypoints2, descriptors2);

    FlannBasedMatcher matcher(makePtr<flann::KDTreeIndexParams>(5), makePtr<flann::SearchParams>(32));

    // get k=2 best match that we can apply ratio test explained by D.Lowe
    std::vector<std::vector<DMatch>> matches_vector;
    matcher.knnMatch(descriptors1, descriptors2, matches_vector, 2);

    // filter keypoints with Lowe ratio test
    std::vector<Point2d> pts1, pts2;
    pts1.reserve(matches_vector.size()); pts2.reserve(matches_vector.size());
    for (const auto &m : matches_vector) {
        // compare best and second match using Lowe ratio test
        if (m[0].distance / m[1].distance < 0.75) {
            pts1.emplace_back(keypoints1[m[0].queryIdx].pt);
            pts2.emplace_back(keypoints2[m[0].trainIdx].pt);
        }
    }

    Mat inliers;
    const int pts_size = (int) pts1.size();
    const auto begin_time = std::chrono::steady_clock::now();
    // fine essential matrix
    const Mat E = findEssentialMat(pts1, pts2, Mat(K), RANSAC, 0.99, 1.0, inliers);
    std::cout << "RANSAC essential matrix time " << std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::steady_clock::now() - begin_time).count() <<
            "mcs.\nNumber of inliers " << countNonZero(inliers) << "\n";

    Mat points1 = Mat((int)pts1.size(), 2, CV_64F, pts1.data());
    Mat points2 = Mat((int)pts2.size(), 2, CV_64F, pts2.data());
    points1 = points1.t(); points2 = points2.t();

    std::cout << "Mean error to epipolar lines " <<
        getError2EpipLines(K.inv().t() * E * K.inv(), points1, points2, inliers) << "\n";

    // decompose essential into rotation and translation
    Mat R1, R2, t;
    decomposeEssentialMat(E, R1, R2, t);

    // Create two relative pose
    // P1 = K [  I    |   0  ]
    // P2 = K [R{1,2} | {+-}t]
    Mat P1;
    hconcat(K, Vec3d::zeros(), P1);
    std::vector<Mat> P2s(4);
    hconcat(K * R1,  K * t, P2s[0]);
    hconcat(K * R1, -K * t, P2s[1]);
    hconcat(K * R2,  K * t, P2s[2]);
    hconcat(K * R2, -K * t, P2s[3]);

    // find objects point by enumerating over 4 different projection matrices of second camera
    // vector to keep object points
    std::vector<std::vector<Vec3d>> obj_pts_per_cam(4);
    // vector to keep indices of image points corresponding to object points
    std::vector<std::vector<int>> img_idxs_per_cam(4);
    int cam_idx = 0, best_cam_idx = 0, max_obj_pts = 0;
    for (const auto &P2 : P2s) {
        obj_pts_per_cam[cam_idx].reserve(pts_size);
        img_idxs_per_cam[cam_idx].reserve(pts_size);
        for (int i = 0; i < pts_size; i++) {
            // process only inliers
            if (! inliers.at<uchar>(i))
                continue;

            Vec4d obj_pt;
            // find object point using triangulation
            triangulatePoints(P1, P2, points1.col(i), points2.col(i), obj_pt);
            obj_pt /= obj_pt(3); // normalize 4d point
            if (obj_pt(2) > 0) { // check if projected point has positive depth
                obj_pts_per_cam[cam_idx].emplace_back(Vec3d(obj_pt(0), obj_pt(1), obj_pt(2)));
                img_idxs_per_cam[cam_idx].emplace_back(i);
            }
        }
        if (max_obj_pts < (int) obj_pts_per_cam[cam_idx].size()) {
            max_obj_pts = (int) obj_pts_per_cam[cam_idx].size();
            best_cam_idx = cam_idx;
        }
        cam_idx++;
    }

    std::cout << "Number of object points " << max_obj_pts << "\n";

    const int circle_sz = 7;
    // draw image points that are inliers on two images
    std::vector<int> labels;
    std::vector<Vec4d> planes;
    getPlanes (obj_pts_per_cam[best_cam_idx], labels, planes, 4, 0.002, 0.99, 10000);
    const int num_found_planes = (int) planes.size();
    RNG rng;
    std::vector<Scalar> plane_colors (num_found_planes);
    for (int pl = 0; pl < num_found_planes; pl++)
        plane_colors[pl] = Scalar (rng.uniform(0,256), rng.uniform(0,256), rng.uniform(0,256));
    for (int obj_pt = 0; obj_pt < max_obj_pts; obj_pt++) {
        const int pt = img_idxs_per_cam[best_cam_idx][obj_pt];
        if (labels[obj_pt] > 0) { // plot plane points
            circle (image1, pts1[pt], circle_sz, plane_colors[labels[obj_pt]-1], -1);
            circle (image2, pts2[pt], circle_sz, plane_colors[labels[obj_pt]-1], -1);
        } else { // plot inliers
            circle (image1, pts1[pt], circle_sz, Scalar(0,0,0), -1);
            circle (image2, pts2[pt], circle_sz, Scalar(0,0,0), -1);
        }
    }

    // concatenate two images
    hconcat(image1, image2, image1);
    const int new_img_size = 1200 * 800; // for example
    // resize with the same aspect ratio
    resize(image1, image1, Size((int)sqrt ((double) image1.cols * new_img_size / image1.rows),
                                (int)sqrt ((double) image1.rows * new_img_size / image1.cols)));
    imshow("image 1-2", image1);
    imwrite("planes.png", image1);
    waitKey(0);
}
