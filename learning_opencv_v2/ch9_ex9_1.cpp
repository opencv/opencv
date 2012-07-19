//
// Example 9-1. Reading out the RGB values of all pixels in one row of a video and accumulating those
//              values into three separate files
//
// STORE TO DISK A LINE SEGMENT OF BGR PIXELS FROM pt1 to pt2.  
//…
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

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

void help() {
  cout << "\nRead out RGB pixel values and store them to disk\nCall:\n" <<
"./ch9_ex9_1 avi_file\n" <<
"\n This will store to files blines.csv, glines.csv and rlines.csv\n\n" << endl;
}

int main( int argc, char** argv  )
{
    if(argc != 2) {help(); return -1;}
    namedWindow( "Example9_1", CV_WINDOW_AUTOSIZE );

    VideoCapture cap;
    if((argc < 2)|| !cap.open(argv[1]))
    {
    	cerr << "Couldn't open video file" << endl;
    	help();
    	cap.open(0);
    	return -1;
    }

    Point pt1(10,10), pt2(30,30);
    int max_buffer;
    Mat rawImage;
    ofstream b,g,r;
    b.open("blines.csv");
    g.open("glines.csv");
    r.open("rlines.csv");

    //MAIN PROCESSING LOOP:
    for(;;){
    	cap >> rawImage; if(!rawImage.data) break;
    	LineIterator it(rawImage,pt1, pt2, 8);
        for(int j=0; j<it.count; ++j,++it){
        	b << (int)(*it)[0] << ", ";
        	g << (int)(*it)[1] << ", ";
        	r << (int)(*it)[2] << ", ";
            (*it)[2] = 255;  //Mark this sample in red
        }
        imshow( "Example9_1", rawImage );
        int c = waitKey(10);
        b << "\n"; g << "\n"; r << "\n";
    }
    //CLEAN UP:
    b << endl; g << endl; r << endl;
    b.close(); g.close(); r.close();
    cout << "\nData stored to files: blines.csv, glines.csv and rlines.csv\n\n" << endl;
}
