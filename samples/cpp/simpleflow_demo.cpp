#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

static void help()
{
    // print a welcome message, and the OpenCV version
    printf("This is a demo of SimpleFlow optical flow algorithm,\n"
           "Using OpenCV version %s\n\n", CV_VERSION);

    printf("Usage: simpleflow_demo frame1 frame2 output_flow"
           "\nApplication will write estimated flow "
           "\nbetween 'frame1' and 'frame2' in binary format"
           "\ninto file 'output_flow'"
           "\nThen one can use code from http://vision.middlebury.edu/flow/data/"
           "\nto convert flow in binary file to image\n");
}

// binary file format for flow data specified here:
// http://vision.middlebury.edu/flow/data/
static void writeOpticalFlowToFile(const Mat& u, const Mat& v, FILE* file) {
  int cols = u.cols;
  int rows = u.rows;
  
  fprintf(file, "PIEH");
   
  if (fwrite(&cols, sizeof(int), 1, file) != 1 ||
      fwrite(&rows, sizeof(int), 1, file) != 1) {
    fprintf(stderr, "writeOpticalFlowToFile : problem writing header\n");
    exit(1);
  }

  for (int i= 0; i < u.rows; ++i) {
    for (int j = 0; j < u.cols; ++j) {
      float uPoint = u.at<double>(i, j);
      float vPoint = v.at<double>(i, j);

      if (fwrite(&uPoint, sizeof(float), 1, file) != 1 ||
          fwrite(&vPoint, sizeof(float), 1, file) != 1) {
        fprintf(stderr, "writeOpticalFlowToFile : problem writing data\n");
        exit(1);
      }
    }
  }
}
int main(int argc, char** argv) {
    help();

    if (argc < 4) {
      fprintf(stderr, "Wrong number of command line arguments : %d (expected %d)\n", argc, 4);
      exit(1);
    }
    
    Mat frame1 = imread(argv[1]);
    Mat frame2 = imread(argv[2]);

    if (frame1.empty() || frame2.empty()) {
      fprintf(stderr, "simpleflow_demo : Images cannot be read\n");
      exit(1);
    }

    if (frame1.rows != frame2.rows && frame1.cols != frame2.cols) {
      fprintf(stderr, "simpleflow_demo : Images should be of equal sizes\n");
      exit(1);
    }

    if (frame1.type() != 16 || frame2.type() != 16) {
      fprintf(stderr, "simpleflow_demo : Images should be of equal type CV_8UC3\n");
      exit(1);
    }

    printf("simpleflow_demo : Read two images of size [rows = %d, cols = %d]\n", 
           frame1.rows, frame1.cols);

    Mat flowX, flowY;

    calcOpticalFlowSF(frame1, frame2, 
                      flowX, flowY,
                      3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);

  FILE* file = fopen(argv[3], "wb");
  if (file == NULL) {
    fprintf(stderr, "simpleflow_demo : Unable to open file '%s' for writing\n", argv[3]); 
    exit(1);
  }
  printf("simpleflow_demo : Writing to file\n");
  writeOpticalFlowToFile(flowX, flowY, file);
  fclose(file);
  return 0;
}
