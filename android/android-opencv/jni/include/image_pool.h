#ifndef IMAGE_POOL_H_ANDROID_KDJFKJ
#define IMAGE_POOL_H_ANDROID_KDJFKJ
#include <opencv2/core/core.hpp>
#include <map>



class image_pool
{

public:
  image_pool();
  ~image_pool();
  cv::Mat getImage(int i);
  cv::Mat getGrey(int i);
  cv::Mat getYUV(int i);

  int getCount()
  {
    return imagesmap.size();
  }

  /** Adds a mat at the given index - will not do a deep copy, just images[i] = mat
   *
   */
  void addImage(int i, cv::Mat mat);

  /** this function stores the given matrix in the the yuvImagesMap. Also,
   * after this call getGrey will work, as the grey image is just the top
   * half of the YUV mat.
   *
   * \param i index to store yuv image at
   * \param mat the yuv matrix to store
   */
  void addYUVMat(int i, cv::Mat mat);

  void convertYUVtoColor(int i, cv::Mat& out);

  //	int addYUV(uchar* buffer, int size, int width, int height, bool grey,int idx);
  //
  //	void getBitmap(int * outintarray, int size, int idx);
private:
  std::map<int, cv::Mat> imagesmap;
  std::map<int, cv::Mat> yuvImagesMap;

};

void copyMatToBuffer(char* buffer, const cv::Mat& mat);
void copyBufferToMat(cv::Mat& mat, const char* buffer);
void RGB2BGR(const cv::Mat& in, cv::Mat& out);
#endif
