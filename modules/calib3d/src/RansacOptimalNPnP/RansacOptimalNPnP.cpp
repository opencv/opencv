#include "RansacOptimalNPnP.h"
#include <iostream>
#include <opencv2/calib3d.hpp>

// -----------------------------------------
// Find camera pose from rotation matrix and translation vector
// -----------------------------------------
bool RansacOptimalPnP::calculate_camera_pose(const Eigen::Matrix3d &rmat,
                                             const Eigen::Vector3d &tvec,
                                             Point3d &camera_pose)
{
    auto pos = -(rmat.transpose()) * tvec;

    if (isnan(pos[0]))
    {
        return false;
    }
    camera_pose = Point3d(pos[0], pos[1], pos[2]);
    return true;
}

// -----------------------------------------
// Optimal PnP Solver
// -----------------------------------------
void RansacOptimalPnP::optimal_pnp_solver(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<Eigen::Vector3d> &lines,
    const std::vector<double> &weights, const std::vector<int> &indices,
    Eigen::Matrix3d &rmat, Eigen::Vector3d &tvec)
{
    // pnp fucntion call and output
    auto pnp_input = PnpInput::init(points, lines, weights, indices);
    auto pnp_objective = PnpObjective::init(pnp_input);
    auto pnp_res =
        this->pnp->solve_pnp(pnp_objective, this->barrier_method_settings);
    rmat = pnp_res.rotation_matrix().transpose();
    tvec = pnp_res.translation_vector();
}

// -----------------------------------------
// Reproject points
// -----------------------------------------
void RansacOptimalPnP::reproject_points(const vector<Point3d> &obj_points,
                                        const Eigen::Matrix3d &rmat,
                                        const Eigen::Vector3d &tvec,
                                        vector<Point2d> &image_points)
{
    auto R = rmat;
    auto t = tvec;

    uint nPoints = obj_points.size();

    image_points.reserve(nPoints);
    for (uint i = 0; i < nPoints; i++)
    {
        const cv::Point3d &p3d = obj_points[i];
        double Xc = R.coeff(0, 0) * p3d.x + R.coeff(0, 1) * p3d.y +
                    R.coeff(0, 2) * p3d.z + t.coeff(0);
        double Yc = R.coeff(1, 0) * p3d.x + R.coeff(1, 1) * p3d.y +
                    R.coeff(1, 2) * p3d.z + t.coeff(1);
        double Zc = R.coeff(2, 0) * p3d.x + R.coeff(2, 1) * p3d.y +
                    R.coeff(2, 2) * p3d.z + t.coeff(2);
        double invZc = 1 / Zc;

        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;
        image_points.emplace_back(ue, ve);
    }
}

// -----------------------------------------
// Find inliers by distance from reprojection
// -----------------------------------------
int RansacOptimalPnP::check_inliers(const vector<Point2d> &scene,
                                    const vector<Point2d> &scene_reproj,
                                    vector<bool> &is_inliers)
{
    int inliers_cnt = 0;
    is_inliers.resize(scene.size());

    for (uint i = 0; i < scene.size(); i++)
    {
        const cv::Point2d &p2d = scene[i];
        const cv::Point2d &p2d_reproj = scene_reproj[i];
        double distX = p2d.x - p2d_reproj.x;
        double distY = p2d.y - p2d_reproj.y;

        double err = distX * distX + distY * distY;

        if (err <= this->max_error_threshold)
        {
            inliers_cnt++;
            is_inliers[i] = true;
        }
        else
        {
            is_inliers[i] = false;
        }
    }
    return inliers_cnt;
}

// -----------------------------------------
// PnP caller:
// -----------------------------------------
bool RansacOptimalPnP::optimal_pnp(const vector<Point2d> &scene,
                                   const vector<Point3d> &obj,
                                   vector<bool> &is_inliers,
                                   int &inliers_cnt_result,
                                   FramePose &frame_pose)
{
    // Prepare inliers
    // -----------------------------------------
    vector<Point2d> inliers_scene;
    vector<Point3d> inliers_obj;

    for (uint i = 0; i < scene.size(); i++)
    {
        if (is_inliers[i])
        {
            inliers_scene.push_back(scene[i]);
            inliers_obj.push_back(obj[i]);
        }
    }

    uint nPnts = inliers_scene.size();

    std::vector<double> weights(nPnts, 1);
    std::vector<int> indexes(nPnts);

    //--- fill indexes with 0,1,...,nPnts-1
    std::iota(begin(indexes), end(indexes), 0);
    // -----------------------------------------

    // Convert matched pixels to lines
    // -----------------------------------------
    std::vector<cv::Point3d> lines_double;
    lines_double.reserve(nPnts);

    for (uint i = 0; i < nPnts; i++)
    {
        // 2D point from the scene (from keypoint)
        const cv::Point2f &point2d = inliers_scene[i];
        cv::Mat pt = (cv::Mat_<float>(3, 1) << point2d.x, point2d.y, 1); //
        cv::Mat templine = this->invK * pt;
        templine = templine / norm(templine);
        cv::Point3f v(templine);
        lines_double.push_back(v);
    }
    // -----------------------------------------

    // convert to Eigen:
    // -----------------------------------------
    std::vector<Eigen::Vector3d> points_vector;
    std::vector<Eigen::Vector3d> lines_vector;
    points_vector.reserve(nPnts);
    lines_vector.reserve(nPnts);
    //
    for (uint i = 0; i < nPnts; i++)
    {
        lines_vector.emplace_back(lines_double[i].x, lines_double[i].y,
                                  lines_double[i].z);
        points_vector.emplace_back(inliers_obj[i].x, inliers_obj[i].y,
                                   inliers_obj[i].z);
    } //
    // -----------------------------------------

    // Solver
    // -----------------------------------------
    Eigen::Matrix3d rmat;
    Eigen::Vector3d tvec;
    this->optimal_pnp_solver(points_vector, lines_vector, weights, indexes, rmat, tvec);

    frame_pose.set_values(rmat, tvec);
    if (!calculate_camera_pose(frame_pose.rmat, frame_pose.tvec, frame_pose.cam_pose))
    {
        std::cout << "BAD POSE" << std::endl;
        return false;
    } //

    vector<Point2d> scene_reproject;

    cv::Mat cv_R, cv_t;
    frame_pose.R.convertTo(cv_R, CV_64F);
    frame_pose.t.convertTo(cv_t, CV_64F);
    projectPoints(obj, cv_R, cv_t, this->camera_matrix, cv::Mat(), scene_reproject);

    inliers_cnt_result = this->check_inliers(scene, scene_reproject, is_inliers);
    return true;
}

// -----------------------------------------
// Ransac optimal PnP solver robust to outliers
// -----------------------------------------
bool RansacOptimalPnP::ransac_solve_pnp(const vector<Point2d> &scene,
                                        const vector<Point3d> &obj,
                                        vector<bool> &is_inliers,
                                        int &inliers_cnt_result,
                                        FramePose &frame_pose)
{
    bool result = false;
    int curr_iter = 0;
    int max_inliers_cnt = 0;
    vector<bool> max_inliers;

    int success_inliers_cnt = this->min_inliers;
    int stop_inliers_cnt = success_inliers_cnt * 2;

    uint nPnts = scene.size();

    // fill available_index with [0,1,...,nPnts-1]
    vector<int> available_index(nPnts);
    std::iota(begin(available_index), end(available_index), 0);


    while (curr_iter < this->max_iterations)
    {
        curr_iter++;
        // Solve pnp on 4 random points
        vector<bool> curr_inliers(nPnts, false);

        for (uint i = 0; i < 4; i++)
        { // 4 points per iteration
            int rand_ind = uniform_int_distribution<int>(0, nPnts - 1 - i)(this->rng);
            int idx = available_index[rand_ind];

            curr_inliers[idx] = true;

            available_index[rand_ind] = available_index[nPnts - 1 - i];
            available_index[nPnts - 1 - i] = idx;
        }

        int curr_inliers_cnt;

        bool res = this->optimal_pnp(scene, obj, curr_inliers, curr_inliers_cnt, frame_pose);


        if (res)
        {
            if (max_inliers_cnt < curr_inliers_cnt)
            {
                max_inliers_cnt = curr_inliers_cnt;
                max_inliers = curr_inliers;
            }
        }

        if (max_inliers_cnt >= stop_inliers_cnt &&
            curr_iter >= this->min_iterations)
        {
            break;
        }
    }

    if (max_inliers_cnt >= success_inliers_cnt)
    {

        bool res = this->optimal_pnp(scene, obj, max_inliers, max_inliers_cnt, frame_pose);

        if (res && max_inliers_cnt >= success_inliers_cnt)
        {
            result = true;

            //--- Refine pose
            vector<bool> refine_inliers;
            int refine_inliers_cnt;
            FramePose refined_frame_pose;

            for (int i = 0; i < 2; i++)
            {
                refine_inliers = max_inliers;
                refine_inliers_cnt = max_inliers_cnt;
                bool resu = this->optimal_pnp(scene, obj, refine_inliers, refine_inliers_cnt, refined_frame_pose);

                if (resu && refine_inliers_cnt > max_inliers_cnt)
                {
                    max_inliers = refine_inliers;
                    max_inliers_cnt = refine_inliers_cnt;
                    frame_pose = refined_frame_pose;
                }
                else
                {
                    break;
                }
            }
        }
    }

    is_inliers = max_inliers;
    inliers_cnt_result = max_inliers_cnt;
    return result;
}
