/*
 * PnPProblem.cpp
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#include <iostream>
#include <sstream>

#include "PnPProblem.h"
#include "Mesh.h"

#include <opencv2/calib3d/calib3d.hpp>

/* Functions headers */
cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2);
double DOT(cv::Point3f v1, cv::Point3f v2);
cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2);
cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin);


/* Functions for Möller-Trumbore intersection algorithm */

cv::Point3f CROSS(cv::Point3f v1, cv::Point3f v2)
{
  cv::Point3f tmp_p;
  tmp_p.x =  v1.y*v2.z - v1.z*v2.y;
  tmp_p.y =  v1.z*v2.x - v1.x*v2.z;
  tmp_p.z =  v1.x*v2.y - v1.y*v2.x;
  return tmp_p;
}

double DOT(cv::Point3f v1, cv::Point3f v2)
{
  return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

cv::Point3f SUB(cv::Point3f v1, cv::Point3f v2)
{
  cv::Point3f tmp_p;
  tmp_p.x =  v1.x - v2.x;
  tmp_p.y =  v1.y - v2.y;
  tmp_p.z =  v1.z - v2.z;
  return tmp_p;
}

/* End functions for Möller-Trumbore intersection algorithm
 *  */

// Function to get the nearest 3D point to the Ray origin
cv::Point3f get_nearest_3D_point(std::vector<cv::Point3f> &points_list, cv::Point3f origin)
{
  cv::Point3f p1 = points_list[0];
  cv::Point3f p2 = points_list[1];

  double d1 = std::sqrt( std::pow(p1.x-origin.x, 2) + std::pow(p1.y-origin.y, 2) + std::pow(p1.z-origin.z, 2) );
  double d2 = std::sqrt( std::pow(p2.x-origin.x, 2) + std::pow(p2.y-origin.y, 2) + std::pow(p2.z-origin.z, 2) );

  if(d1 < d2)
  {
      return p1;
  }
  else
  {
      return p2;
  }
}

// Custom constructor given the intrinsic camera parameters

PnPProblem::PnPProblem(const double params[])
{
  _A_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // intrinsic camera parameters
  _A_matrix.at<double>(0, 0) = params[0];       //      [ fx   0  cx ]
  _A_matrix.at<double>(1, 1) = params[1];       //      [  0  fy  cy ]
  _A_matrix.at<double>(0, 2) = params[2];       //      [  0   0   1 ]
  _A_matrix.at<double>(1, 2) = params[3];
  _A_matrix.at<double>(2, 2) = 1;
  _R_matrix = cv::Mat::zeros(3, 3, CV_64FC1);   // rotation matrix
  _t_matrix = cv::Mat::zeros(3, 1, CV_64FC1);   // translation matrix
  _P_matrix = cv::Mat::zeros(3, 4, CV_64FC1);   // rotation-translation matrix

}

PnPProblem::~PnPProblem()
{
  // TODO Auto-generated destructor stub
}

void PnPProblem::set_P_matrix( const cv::Mat &R_matrix, const cv::Mat &t_matrix)
{
  // Rotation-Translation Matrix Definition
  _P_matrix.at<double>(0,0) = R_matrix.at<double>(0,0);
  _P_matrix.at<double>(0,1) = R_matrix.at<double>(0,1);
  _P_matrix.at<double>(0,2) = R_matrix.at<double>(0,2);
  _P_matrix.at<double>(1,0) = R_matrix.at<double>(1,0);
  _P_matrix.at<double>(1,1) = R_matrix.at<double>(1,1);
  _P_matrix.at<double>(1,2) = R_matrix.at<double>(1,2);
  _P_matrix.at<double>(2,0) = R_matrix.at<double>(2,0);
  _P_matrix.at<double>(2,1) = R_matrix.at<double>(2,1);
  _P_matrix.at<double>(2,2) = R_matrix.at<double>(2,2);
  _P_matrix.at<double>(0,3) = t_matrix.at<double>(0);
  _P_matrix.at<double>(1,3) = t_matrix.at<double>(1);
  _P_matrix.at<double>(2,3) = t_matrix.at<double>(2);
}


// Estimate the pose given a list of 2D/3D correspondences and the method to use
bool PnPProblem::estimatePose( const std::vector<cv::Point3f> &list_points3d,
                               const std::vector<cv::Point2f> &list_points2d,
                               int flags)
{
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

  bool useExtrinsicGuess = false;

  // Pose estimation
  bool correspondence = cv::solvePnP( list_points3d, list_points2d, _A_matrix, distCoeffs, rvec, tvec,
                                      useExtrinsicGuess, flags);

  // Transforms Rotation Vector to Matrix
  Rodrigues(rvec,_R_matrix);
  _t_matrix = tvec;

  // Set projection matrix
  this->set_P_matrix(_R_matrix, _t_matrix);

  return correspondence;
}

// Estimate the pose given a list of 2D/3D correspondences with RANSAC and the method to use

void PnPProblem::estimatePoseRANSAC( const std::vector<cv::Point3f> &list_points3d, // list with model 3D coordinates
                                     const std::vector<cv::Point2f> &list_points2d,     // list with scene 2D coordinates
                                     int flags, cv::Mat &inliers, int iterationsCount,  // PnP method; inliers container
                                     float reprojectionError, double confidence )    // Ransac parameters
{
  cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);  // vector of distortion coefficients
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);    // output translation vector

  bool useExtrinsicGuess = false;   // if true the function uses the provided rvec and tvec values as
            // initial approximations of the rotation and translation vectors

  cv::solvePnPRansac( list_points3d, list_points2d, _A_matrix, distCoeffs, rvec, tvec,
                useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                inliers, flags );

  Rodrigues(rvec,_R_matrix);      // converts Rotation Vector to Matrix
  _t_matrix = tvec;       // set translation matrix

  this->set_P_matrix(_R_matrix, _t_matrix); // set rotation-translation matrix

}

// Given the mesh, backproject the 3D points to 2D to verify the pose estimation
std::vector<cv::Point2f> PnPProblem::verify_points(Mesh *mesh)
{
  std::vector<cv::Point2f> verified_points_2d;
  for( int i = 0; i < mesh->getNumVertices(); i++)
  {
    cv::Point3f point3d = mesh->getVertex(i);
    cv::Point2f point2d = this->backproject3DPoint(point3d);
    verified_points_2d.push_back(point2d);
  }

  return verified_points_2d;
}


// Backproject a 3D point to 2D using the estimated pose parameters

cv::Point2f PnPProblem::backproject3DPoint(const cv::Point3f &point3d)
{
  // 3D point vector [x y z 1]'
  cv::Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
  point3d_vec.at<double>(0) = point3d.x;
  point3d_vec.at<double>(1) = point3d.y;
  point3d_vec.at<double>(2) = point3d.z;
  point3d_vec.at<double>(3) = 1;

  // 2D point vector [u v 1]'
  cv::Mat point2d_vec = cv::Mat(4, 1, CV_64FC1);
  point2d_vec = _A_matrix * _P_matrix * point3d_vec;

  // Normalization of [u v]'
  cv::Point2f point2d;
  point2d.x = (float)(point2d_vec.at<double>(0) / point2d_vec.at<double>(2));
  point2d.y = (float)(point2d_vec.at<double>(1) / point2d_vec.at<double>(2));

  return point2d;
}

// Back project a 2D point to 3D and returns if it's on the object surface
bool PnPProblem::backproject2DPoint(const Mesh *mesh, const cv::Point2f &point2d, cv::Point3f &point3d)
{
  // Triangles list of the object mesh
  std::vector<std::vector<int> > triangles_list = mesh->getTrianglesList();

  double lambda = 8;
  double u = point2d.x;
  double v = point2d.y;

  // Point in vector form
  cv::Mat point2d_vec = cv::Mat::ones(3, 1, CV_64F); // 3x1
  point2d_vec.at<double>(0) = u * lambda;
  point2d_vec.at<double>(1) = v * lambda;
  point2d_vec.at<double>(2) = lambda;

  // Point in camera coordinates
  cv::Mat X_c = _A_matrix.inv() * point2d_vec ; // 3x1

  // Point in world coordinates
  cv::Mat X_w = _R_matrix.inv() * ( X_c - _t_matrix ); // 3x1

  // Center of projection
  cv::Mat C_op = cv::Mat(_R_matrix.inv()).mul(-1) * _t_matrix; // 3x1

  // Ray direction vector
  cv::Mat ray = X_w - C_op; // 3x1
  ray = ray / cv::norm(ray); // 3x1

  // Set up Ray
  Ray R((cv::Point3f)C_op, (cv::Point3f)ray);

  // A vector to store the intersections found
  std::vector<cv::Point3f> intersections_list;

  // Loop for all the triangles and check the intersection
  for (unsigned int i = 0; i < triangles_list.size(); i++)
  {
    cv::Point3f V0 = mesh->getVertex(triangles_list[i][0]);
    cv::Point3f V1 = mesh->getVertex(triangles_list[i][1]);
    cv::Point3f V2 = mesh->getVertex(triangles_list[i][2]);

    Triangle T(i, V0, V1, V2);

    double out;
    if(this->intersect_MollerTrumbore(R, T, &out))
    {
      cv::Point3f tmp_pt = R.getP0() + out*R.getP1(); // P = O + t*D
      intersections_list.push_back(tmp_pt);
    }
  }

  // If there are intersection, find the nearest one
  if (!intersections_list.empty())
  {
    point3d = get_nearest_3D_point(intersections_list, R.getP0());
    return true;
  }
  else
  {
    return false;
  }
}

// Möller-Trumbore intersection algorithm
bool PnPProblem::intersect_MollerTrumbore(Ray &Ray, Triangle &Triangle, double *out)
{
  const double EPSILON = 0.000001;

  cv::Point3f e1, e2;
  cv::Point3f P, Q, T;
  double det, inv_det, u, v;
  double t;

  cv::Point3f V1 = Triangle.getV0();  // Triangle vertices
  cv::Point3f V2 = Triangle.getV1();
  cv::Point3f V3 = Triangle.getV2();

  cv::Point3f O = Ray.getP0(); // Ray origin
  cv::Point3f D = Ray.getP1(); // Ray direction

  //Find vectors for two edges sharing V1
  e1 = SUB(V2, V1);
  e2 = SUB(V3, V1);

  // Begin calculation determinant - also used to calculate U parameter
  P = CROSS(D, e2);

  // If determinant is near zero, ray lie in plane of triangle
  det = DOT(e1, P);

  //NOT CULLING
  if(det > -EPSILON && det < EPSILON) return false;
  inv_det = 1.f / det;

  //calculate distance from V1 to ray origin
  T = SUB(O, V1);

  //Calculate u parameter and test bound
  u = DOT(T, P) * inv_det;

  //The intersection lies outside of the triangle
  if(u < 0.f || u > 1.f) return false;

  //Prepare to test v parameter
  Q = CROSS(T, e1);

  //Calculate V parameter and test bound
  v = DOT(D, Q) * inv_det;

  //The intersection lies outside of the triangle
  if(v < 0.f || u + v  > 1.f) return false;

  t = DOT(e2, Q) * inv_det;

  if(t > EPSILON) { //ray intersection
    *out = t;
    return true;
  }

  // No hit, no win
  return false;
}
