#include "../precomp.hpp"

using namespace cv;

class OptimalPnPRansacCallback : public PointSetRegistrator::Callback {

public:
  OptimalPnPRansacCallback(Mat _cameraMatrix = Mat(3, 3, CV_64F))
      : cameraMatrix(_cameraMatrix) {}

  /* Pre: True */
  /* Post: compute _model with given points and return number of found models */
  int runKernel(InputArray scene, InputArray obj,
                OutputArray _model) const CV_OVERRIDE {
    Mat opoints = scene.getMat(), ipoints = obj.getMat();

    bool correspondence = solvePnP(scene, obj, cameraMatrix, distCoeffs, rvec,
                                   tvec, useExtrinsicGuess, flags);

    Mat _local_model;
    hconcat(rvec, tvec, _local_model);
    _local_model.copyTo(_model);

    return correspondence;
  }

  /* Pre: True */
  /* Post: fill _err with projection errors */
  void computeError(InputArray _m1, InputArray _m2, InputArray _model,
                    OutputArray _err) const CV_OVERRIDE {

    Mat opoints = _m1.getMat(), ipoints = _m2.getMat(), model = _model.getMat();

    int i, count = opoints.checkVector(3);
    Mat _rvec = model.col(0);
    Mat _tvec = model.col(1);

    Mat projpoints(count, 2, CV_32FC1);
    projectPoints(opoints, _rvec, _tvec, cameraMatrix, distCoeffs, projpoints);

    const Point2f *ipoints_ptr = ipoints.ptr<Point2f>();
    const Point2f *projpoints_ptr = projpoints.ptr<Point2f>();

    _err.create(count, 1, CV_32FC1);
    float *err = _err.getMat().ptr<float>();

    for (i = 0; i < count; ++i)
      err[i] =
          (float)norm(Matx21f(ipoints_ptr[i] - projpoints_ptr[i]), NORM_L2SQR);
  }

  Mat cameraMatrix;
  Mat distCoeffs;
  int flags;
  bool useExtrinsicGuess;
  Mat rvec;
  Mat tvec;
};
