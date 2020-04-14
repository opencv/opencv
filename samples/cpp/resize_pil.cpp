#include <opencv2/opencv.hpp>
//Include file for every supported OpenCV function

using namespace cv;

int main(int argc, char** argv)
{
  Mat image;
  image = imread("d:\\mask.bmp", IMREAD_GRAYSCALE);  // Read the file

  if (!image.data)  // Check for invalid input
  {
    std::cout << "Could not open or find the image" << std::endl;
    return -1;
  }
  imshow("Source image", image);

  Mat target_image;

  resize(image, target_image, Size(16, 16), 0, 0, INTER_NEAREST_PIL);
  for (int row = 0; row < target_image.rows; ++row) {
      for (int col = 0; col < target_image.cols; ++col) {
        std::cout << abs((int) target_image.at<char>(row, col)) << ' ';
      }
      std::cout << "\n";
  }
  resize(target_image, image, Size(256, 256), 0, 0, INTER_NEAREST_PIL);

  imshow("Target image", image);
  waitKey(0);
  return 0;
}