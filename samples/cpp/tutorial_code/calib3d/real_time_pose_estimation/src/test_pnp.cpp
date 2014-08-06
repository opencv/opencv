#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

void generate3DPointCloud(vector<Point3f>& points, Point3f pmin = Point3f(-1,
        -1, 5), Point3f pmax = Point3f(1, 1, 10))
    {
        const Point3f delta = pmax - pmin;
        for (size_t i = 0; i < points.size(); i++)
        {
            Point3f p(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX,
                float(rand()) / RAND_MAX);
            p.x *= delta.x;
            p.y *= delta.y;
            p.z *= delta.z;
            p = p + pmin;
            points[i] = p;
        }
    }

void generateCameraMatrix(Mat& cameraMatrix, RNG& rng)
{
    const double fcMinVal = 1e-3;
    const double fcMaxVal = 100;
    cameraMatrix.create(3, 3, CV_64FC1);
    cameraMatrix.setTo(Scalar(0));
    cameraMatrix.at<double>(0,0) = rng.uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(1,1) = rng.uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(0,2) = rng.uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(1,2) = rng.uniform(fcMinVal, fcMaxVal);
    cameraMatrix.at<double>(2,2) = 1;
}

void generateDistCoeffs(Mat& distCoeffs, RNG& rng)
{
    distCoeffs = Mat::zeros(4, 1, CV_64FC1);
    for (int i = 0; i < 3; i++)
        distCoeffs.at<double>(i,0) = rng.uniform(0.0, 1.0e-6);
}

void generatePose(Mat& rvec, Mat& tvec, RNG& rng)
{
    const double minVal = 1.0e-3;
    const double maxVal = 1.0;
    rvec.create(3, 1, CV_64FC1);
    tvec.create(3, 1, CV_64FC1);
    for (int i = 0; i < 3; i++)
    {
        rvec.at<double>(i,0) = rng.uniform(minVal, maxVal);
        tvec.at<double>(i,0) = rng.uniform(minVal, maxVal/10);
    }
}

void data2file(const string& path, const vector<vector<double> >& data)
{
  std::fstream fs;
  fs.open(path.c_str(), std::fstream::in | std::fstream::out | std::fstream::app);

  for (int method = 0; method < data.size(); ++method)
  {
    for (int i = 0; i < data[method].size(); ++i)
    {
      fs << data[method][i] << " ";
    }
    fs << endl;
  }

  fs.close();
}


int main(int argc, char *argv[])
{

  RNG rng;
 // TickMeter tm;
  vector<vector<double> > error_trans(4), error_rot(4), comp_time(4);

  int maxpoints = 2000;
  for (int npoints = 10; npoints < maxpoints+10; ++npoints)
  {
    // generate 3D point cloud
    vector<Point3f> points;
    points.resize(npoints);
    generate3DPointCloud(points);

    // generate cameramatrix
    Mat rvec, tvec;
    Mat trueRvec, trueTvec;
    Mat intrinsics, distCoeffs;
    generateCameraMatrix(intrinsics, rng);

    // generate distorsion coefficients
    generateDistCoeffs(distCoeffs, rng);

    // generate groud truth pose
    generatePose(trueRvec, trueTvec, rng);

    for (int method = 0; method < 4; ++method)
    {
      std::vector<Point3f> opoints;
      if (method == 2)
      {
        opoints = std::vector<Point3f>(points.begin(), points.begin()+4);
      }
      else
        opoints = points;

      vector<Point2f> projectedPoints;
      projectedPoints.resize(opoints.size());
      projectPoints(Mat(opoints), trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

      //tm.reset(); tm.start();

      solvePnP(opoints, projectedPoints, intrinsics, distCoeffs, rvec, tvec, false, method);

     // tm.stop();

    //  double compTime = tm.getTimeMilli();
      double rvecDiff = norm(rvec-trueRvec), tvecDiff = norm(tvec-trueTvec);

      error_rot[method].push_back(rvecDiff);
      error_trans[method].push_back(tvecDiff);
      //comp_time[method].push_back(compTime);

    }

    //system("clear");
    cout << "Completed " << npoints+1 << "/" << maxpoints << endl;

  }

  data2file("translation_error.txt", error_trans);
  data2file("rotation_error.txt", error_rot);
  data2file("computation_time.txt", comp_time);

}
