/* License:
   Oct. 3, 2008
   Right to use this code in any way you want without warrenty, support or any guarentee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008
 
   AVAILABLE AT: 
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    

   OTHER OPENCV SITES:
   * The source code is on sourceforge at:
     http://sourceforge.net/projects/opencvlibrary/
   * The OpenCV wiki page (As of Oct 1, 2008 this is down for changing over servers, but should come back):
     http://opencvlibrary.sourceforge.net/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://pr.willowgarage.com/wiki/OpenCV
*/

#include <cv.h>
#include <highgui.h>
// ch3_ex3_12 image_name x y width height add# 
int main(int argc, char** argv)
{

    IplImage* src;
    cvNamedWindow("Example3_12_pre", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Example3_12_post", CV_WINDOW_AUTOSIZE);  
    if( argc == 7 && ((src=cvLoadImage(argv[1],1)) != 0 ))
    {
		int x = atoi(argv[2]);
		int y = atoi(argv[3]);
		int width = atoi(argv[4]);
		int height = atoi(argv[5]);
		int add = atoi(argv[6]);
		cvShowImage( "Example3_12_pre", src);
		cvSetImageROI(src, cvRect(x,y,width,height));
		cvAddS(src, cvScalar(add),src);
		cvResetImageROI(src);
		cvShowImage( "Example3_12_post",src);
      cvWaitKey();
    }
  cvReleaseImage( &src );
  cvDestroyWindow("Example3_12_pre");
  cvDestroyWindow("Example3_12_post");   
    return 0;
}
