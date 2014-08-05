/*
 * Utils.h
 *
 *  Created on: Mar 28, 2014
 *      Author: Edgar Riba
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <cv.h>

#include "PnPProblem.h"

// Draw a text with the question point
void drawQuestion(cv::Mat image, cv::Point3f point, cv::Scalar color);

// Draw a text in the left of the image
void drawText(cv::Mat image, std::string text, cv::Scalar color);
void drawText2(cv::Mat image, std::string text, cv::Scalar color);
void drawFPS(cv::Mat image, double fps, cv::Scalar color);
void drawConfidence(cv::Mat image, double confidence, cv::Scalar color);

// Draw a text with the number of registered points
void drawCounter(cv::Mat image, int n, int n_max, cv::Scalar color);

// Draw the points and the coordinates
void drawPoints(cv::Mat image, std::vector<cv::Point2f> &list_points_2d, std::vector<cv::Point3f> &list_points_3d, cv::Scalar color);

// Draw only the points
void draw2DPoints(cv::Mat image, std::vector<cv::Point2f> &list_points, cv::Scalar color);

// Draw the object mesh
void drawObjectMesh(cv::Mat image, const Mesh *mesh, PnPProblem *pnpProblem, cv::Scalar color);

// Draw an arrow into the image
void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0);

// Draw the 3D coordinate axes
void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d);

// Compute the 2D points with the esticv::Mated pose
std::vector<cv::Point2f> verification_points(PnPProblem *p);

bool equal_point(const cv::Point2f &p1, const cv::Point2f &p2);

double get_translation_error(const cv::Mat &t_true, const cv::Mat &t);
double get_rotation_error(const cv::Mat &R_true, const cv::Mat &R);
double get_variance(const std::vector<cv::Point3f> list_points3d);
cv::Mat rot2quat(cv::Mat &rotationMatrix);
cv::Mat quat2rot(cv::Mat &q);
cv::Mat euler2rot(const cv::Mat & euler);
cv::Mat rot2euler(const cv::Mat & rotationMatrix);
cv::Mat quat2euler(const cv::Mat & q);
int StringToInt ( const std::string &Text );
std::string FloatToString ( float Number );
std::string IntToString ( int Number );

class Timer
{
public:
  Timer();
    void start() { tstart = (double)clock()/CLOCKS_PER_SEC; }
    void stop()  { tstop = (double)clock()/CLOCKS_PER_SEC; }
    void reset()  { tstart = 0;  tstop = 0;}

    double getTimeMilli() const { return (tstop-tstart)*1000; }
    double getTimeSec()   const {  return tstop-tstart; }

private:
    double tstart, tstop, ttime;
};

#endif /* UTILS_H_ */
