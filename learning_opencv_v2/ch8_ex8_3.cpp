//
// Example 8-3. Finding and drawing contours on an input image
//
//
/* License:
   July 20, 2011
   Standard BSD

   BOOK: It would be nice if you cited it:
   Learning OpenCV 2: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   Main OpenCV site
   http://opencv.willowgarage.com/wiki/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/
//

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>

using namespace cv;
using namespace std;

struct AreaCmp
{
    AreaCmp(const vector<float>& _areas) : areas(&_areas) {}
    bool operator()(int a, int b) const { return (*areas)[a] > (*areas)[b]; }
    const vector<float>* areas;
};

//  Example 8-3. Finding and drawing contours on an input image
int main(int argc, char* argv[]) {

  Mat img, img_edge, img_color;
  //Changed this a little for safer image loading and help if not
  if( argc != 2 || (img = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE )).empty() ){
      cout << "\nExample 8_3 Drawing Contours\nCall is:\n./ch8_ex8_3 image\n\n";
      return -1;
  }
  
  threshold(img, img_edge, 128, 255, THRESH_BINARY);
  imshow("Image after threshold", img_edge);
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;  
    
  findContours(img_edge, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
  cout << "\n\nHit any key to draw the next contour, ESC to quit\n\n";
  cout << "Total Contours Detected: " << contours.size() << endl;
  vector<int> sortIdx(contours.size());  
  vector<float> areas(contours.size());
  for( int n = 0; n < (int)contours.size(); n++ ) {
      sortIdx[n] = n;
      areas[n] = contourArea(contours[n], false);
  }
  // sort contours so that largest contours go first
  std::sort(sortIdx.begin(), sortIdx.end(), AreaCmp(areas));
    
  for( int n = 0; n < (int)sortIdx.size(); n++ ) {
     int idx = sortIdx[n]; 
     cvtColor( img, img_color, CV_GRAY2BGR );
     drawContours(img_color, contours, idx,
                   Scalar(0,0,255), 2, 8, hierarchy,
                   0 // Try different values of max_level, and see what happens
                   ); 
     cout << "Contour #" << idx << ": area=" << areas[idx] <<
        ", nvertices=" << contours[idx].size() << endl;
     imshow(argv[0], img_color);
     int k;
     if((k = waitKey()&255) == 27)
       break;
  }
  cout << "Finished all contours\n";
  return 0;
}

