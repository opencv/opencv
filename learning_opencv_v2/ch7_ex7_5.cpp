// 
// Example 7-5. Template matching
// Yes, I switched the order of image and template from the text.  It's now
// 
// Usage: matchTemplate template image
//
// Puts results of all types of matching methods listed i help() below. 
//        Gary Bradski Oct 3, 2008
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
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void help(){
cout << "\n"
"Example of using matchTemplate().  The call is:\n"
"\n"
"ch7_ex7_5 template image_to_be_searched\n"
"\n"
"   This routine will search using all methods:\n"
"         CV_TM_SQDIFF        0\n"
"         CV_TM_SQDIFF_NORMED 1\n"
"         CV_TM_CCORR         2\n"
"         CV_TM_CCORR_NORMED  3\n"
"         CV_TM_CCOEFF        4\n"
"         CV_TM_CCOEFF_NORMED 5\n"
"\n"
"The function prototype is:\n"
"   matchTemplate( Mat image, Mat templ,\n"
"                              Mat result, int method );\n"
"      image\n"
"         Image to be searched. It should be 8-bit or 32-bit floating-point. \n"
"      templ\n"
"         template which should not larger than the above image and is the same type as the image. \n"
"      result\n"
"         A map of comparison results; single-channel 32-bit floating-point. \n"
"      method\n"
"         See the above methods 0-5 starting with CM_TM_SQDIFF\n"
"         \n"
"	If image is W×H and templ is w×h then result will be W-w+1×H-h+1.		\n"
"\n";
}


// Call is
// matchTemplate template image 
//
// Will Display the results of the matchs
// 
int main( int argc, char** argv ) {

    if( argc != 3) {
        help();
        return -1;
    }
        
    Mat src, templ, ftmp[6]; //ftmp is what to display on    
    //Read in the template to be used for matching:
    if((templ=imread(argv[1], 1)).empty()) {
            cout << "Error on reading template " << argv[1] << endl;
            help(); return -1;
    }

    //Read in the source image to be searched:
    if((src=imread(argv[2], 1)).empty()) {
            cout << "Error on reading src image " << argv[2] << endl;
            help(); return -1;
    }

    //DO THE MATCHING OF THE TEMPLATE WITH THE IMAGE
    for(int i=0; i<6; ++i){
        matchTemplate( src, templ, ftmp[i], i); 
        normalize(ftmp[i],ftmp[i],1,0,CV_MINMAX);
    }
    //DISPLAY
    imshow( "Template", templ );
    imshow( "Image", src );
    imshow( "SQDIFF", ftmp[0] );
    imshow( "SQDIFF_NORMED", ftmp[1] );
    imshow( "CCORR", ftmp[2] );
    imshow( "CCORR_NORMED", ftmp[3] );
    imshow( "CCOEFF", ftmp[4] );
    imshow( "CCOEFF_NORMED", ftmp[5] );

    //LET USER VIEW RESULTS:
    waitKey(0);
}
