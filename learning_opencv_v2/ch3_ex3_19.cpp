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
#include <stdio.h>

int main(int argc, char** argv)
{
    printf("Example 3_19 Reading in cfg.xml\n");
    CvFileStorage* fs = cvOpenFileStorage(
      "cfg.xml", 
      0, 
      CV_STORAGE_READ
    );

    int frame_count = cvReadIntByName( 
      fs, 
      0, 
      "frame_count", 
      5 /* default value */ 
    );

    CvSeq* s = cvGetFileNodeByName(fs,0,"frame_size")->data.seq;

    int frame_width = cvReadInt( 
      (CvFileNode*)cvGetSeqElem(s,0) 
    );

    int frame_height = cvReadInt( 
      (CvFileNode*)cvGetSeqElem(s,1) 
    );

    CvMat* color_cvt_matrix = (CvMat*) cvRead(
      fs,
      0
    );
    printf("frame_count=%d, frame_width=%d, frame_height=%d\n",frame_count,frame_width,frame_height);

    cvReleaseFileStorage( &fs );
}
