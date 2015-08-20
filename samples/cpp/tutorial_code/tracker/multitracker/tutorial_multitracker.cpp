/*----------------------------------------------
 * Usage:
 * example_tracking_multitracker <video_name> [algorithm]
 *
 * example:
 * example_tracking_multitracker Bolt/img/%04d.jpg
 * example_tracking_multitracker faceocc2.webm KCF
 *--------------------------------------------------*/

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

using namespace std;
using namespace cv;

int main( int argc, char** argv ){
  // show help
  if(argc<2){
    cout<<
      " Usage: example_tracking_multitracker <video_name> [algorithm]\n"
      " examples:\n"
      " example_tracking_multitracker Bolt/img/%04d.jpg\n"
      " example_tracking_multitracker faceocc2.webm MEDIANFLOW\n"
      << endl;
    return 0;
  }

  // set the default tracking algorithm
  std::string trackingAlg = "KCF";

  // set the tracking algorithm from parameter
  if(argc>2)
    trackingAlg = argv[2];

  // create the tracker
  //! [create]
  MultiTracker trackers(trackingAlg);
  //! [create]

  // container of the tracked objects
  //! [roi]
  vector<Rect2d> objects;
  //! [roi]

  // set input video
  std::string video = argv[1];
  VideoCapture cap(video);

  Mat frame;

  // get bounding box
  cap >> frame;
  //! [selectmulti]
  selectROI("tracker",frame,objects);
  //! [selectmulti]

  //quit when the tracked object(s) is not provided
  if(objects.size()<1)
    return 0;

  // initialize the tracker
  //! [init]
  trackers.add(frame,objects);
  //! [init]

  // do the tracking
  printf("Start the tracking process, press ESC to quit.\n");
  for ( ;; ){
    // get frame from the video
    cap >> frame;

    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;

    //update the tracking result
    //! [update]
    trackers.update(frame);
    //! [update]

    //! [result]
    // draw the tracked object
    for(unsigned i=0;i<trackers.objects.size();i++)
      rectangle( frame, trackers.objects[i], Scalar( 255, 0, 0 ), 2, 1 );
    //! [result]

    // show image with the tracked object
    imshow("tracker",frame);

    //quit on ESC button
    if(waitKey(1)==27)break;
  }

}
