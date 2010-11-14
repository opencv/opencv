#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

int main()
{
  Mat i = Mat::eye(4, 4, CV_32F);
  cout << "i = " << i << ";" << endl;

  Mat r = Mat(10, 10, CV_8UC1);
  randu(r, Scalar(0), Scalar(255));

  cout << "r = " << r << ";" << endl;

  Point2f p(5, 1);
  cout << "p = " << p << ";" << endl;

  Point3f p3f(2, 6, 7);
  cout << "p3f = " << p3f << ";" << endl;

  vector<Point2f> points(20);
  for (size_t i = 0; i < points.size(); ++i)
  {
    points[i] = Point2f(i * 5, i % 7);
  }
  cout << "points = " << points << ";" << endl;

  cout << "#csv" << endl;

  writeCSV(cout, r);

  return 1;
}
