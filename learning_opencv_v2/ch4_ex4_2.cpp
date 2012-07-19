// An example program in which the
// user can draw boxes on the screen.
//
#include <opencv2/opencv.hpp>
#include <iostream>

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

using namespace cv;
using namespace std;

//
// Using a trackbar to create a "switch" that the user can turn on and off.
// We make this value global so everyone can see it.
//
int g_switch_value = 1;
void switch_off_function() { cout << "Pause\n";}; //YOU COULD DO SOMETHING WITH THESE FUNCTIONS TOO
void switch_on_function() { cout << "Run\n";}; 

//
// This will be the callback that we give to the
// trackbar.
//
void switch_callback( int position, void* ) {
  if( position == 0 ) {
    switch_off_function();
  } else {
    switch_on_function();
  }
}
void help()
{
	cout << "Call: ./ch4_ex4_2 tree.avi" << endl;
	cout << "Shows putting a pause button in a video." << endl;
}

 //OK, OK, I ADDED READING A MOVIE AND USING THE "BUTTON" TO STOP AND GO
int main( int argc, char* argv[] ) {
  Mat frame; //To hold movie images
  VideoCapture g_capture;
  help();
  if(argc < 2 || !g_capture.open( argv[1] )){
   	cout << "Failed to open " << argv[1] << " video file\n" << endl;
   	return -1;
  }

  // Name the main window
  //
  namedWindow( "Example4_2", 1 );
 
  // Create the trackbar.  We give it a name,
  // and tell it the name of the parent window.
  //
  createTrackbar(
    "Switch",
    "Example4_2",
    &g_switch_value,
    1,
    switch_callback
  );
 
  // This will just cause OpenCV to idle until 
  // someone hits the "Escape" key.
  //
  for(;;) {
	  if(g_switch_value){
		  g_capture >> frame;
		  if( frame.empty() ) break;
		  imshow( "Example4_2", frame);
	  }
	  if(waitKey(10)==27 ) break;
  }
  
  return 0;
}
