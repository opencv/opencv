#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
const char* message = "Hello Android!";

int main(int argc, char* argv[])
{
  // print message to console
  printf("%s\n", message);
  
  // put message to simple image
  Size textsize = getTextSize(message, CV_FONT_HERSHEY_COMPLEX, 3, 5, 0);
  Mat img(textsize.height + 20, textsize.width + 20, CV_32FC1, Scalar(230,230,230));
  putText(img, message, Point(10, img.rows - 10), CV_FONT_HERSHEY_COMPLEX, 3, Scalar(0, 0, 0), 5);

  // save\show resulting image
#if ANDROID
  imwrite("/mnt/sdcard/HelloAndroid.png", img);
#else
  imshow("test", img);
  waitKey();
#endif
  return 0;
}

