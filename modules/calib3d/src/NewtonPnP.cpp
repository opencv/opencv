#include "NewtonPnP.h"
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>

// -----------------------------------------
// Find camera pose from rotation matrix and translation vector
// -----------------------------------------
bool NewtonPnP::calculate_camera_pose(const Eigen::Matrix3d &rmat,
                                             const Eigen::Vector3d &tvec,
                                             Point3d &camera_pose) {
    auto pos = -(rmat.transpose()) * tvec;

    if (isnan(pos[0])) {
        return false;
    }
    camera_pose = Point3d(pos[0], pos[1], pos[2]);
    return true;
}

// -----------------------------------------
// Optimal PnP Solver
// -----------------------------------------
void NewtonPnP::optimal_pnp_solver(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<Eigen::Vector3d> &lines,
    const std::vector<double> &weights, const std::vector<int> &indices,
    Eigen::Matrix3d &rmat, Eigen::Vector3d &tvec) {
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
void NewtonPnP::reproject_points(const vector<Point3d> &obj_points,
                                        const Eigen::Matrix3d &rmat,
                                        const Eigen::Vector3d &tvec,
                                        vector<Point2d> &image_points) {
    auto R = rmat;
    auto t = tvec;

    uint nPoints = obj_points.size();

    image_points.reserve(nPoints);
    for (uint i = 0; i < nPoints; i++) {
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
// PnP caller:
// -----------------------------------------
bool NewtonPnP::newton_pnp(const vector<Point2d> &scene,
                                   const vector<Point3d> &obj,
                                   FramePose &frame_pose) {
    // Prepare inliers
    // -----------------------------------------
    vector<Point2d> inliers_scene; 
    vector<Point3d> inliers_obj;   

    for (uint i = 0; i < scene.size(); i++) {   
                           
            inliers_scene.push_back(scene[i]);  
            inliers_obj.push_back(obj[i]);      
                                              
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

    for (uint i = 0; i < nPnts; i++) {
        // 2D point from the scene (from keypoint)
        const cv::Point2f &point2d = inliers_scene[i]; 
        cv::Mat pt = (cv::Mat_<float>(3, 1) << point2d.x, point2d.y,1);                                                                 //
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
    for (uint i = 0; i < nPnts; i++) {  
        lines_vector.emplace_back(lines_double[i].x, lines_double[i].y,
                                  lines_double[i].z);  
        points_vector.emplace_back(inliers_obj[i].x, inliers_obj[i].y,
                                   inliers_obj[i].z);  
    }                                                                                                                                       //
    // -----------------------------------------

    // Solver
    // -----------------------------------------
    Eigen::Matrix3d rmat;
    Eigen::Vector3d tvec;
    this->optimal_pnp_solver(points_vector, lines_vector, weights, indexes, rmat, tvec);  

    frame_pose.set_values(rmat, tvec);
    if (!calculate_camera_pose(frame_pose.rmat, frame_pose.tvec, frame_pose.cam_pose)){
        std::cout << "BAD POSE" << std::endl;
        return false;
    }                                                                                                                                                   //
                                      
    vector<Point2d> scene_reproject;  

    cv::Mat cv_R, cv_t;
    frame_pose.R.convertTo(cv_R, CV_64F);
    frame_pose.t.convertTo(cv_t, CV_64F);
    projectPoints(obj, cv_R, cv_t, this->camera_matrix, cv::Mat(), scene_reproject);

    CV_LOG_INFO(NULL, "using Newton PnP");

    return true;                                                  
}
