
//--------------------------------------------------------------------------------------------------
//  A demo program of the Extremal Region Filter algorithm described in 
//  Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
//--------------------------------------------------------------------------------------------------

#include  "opencv2/opencv.hpp"
#include  "opencv2/objdetect.hpp"
#include  "opencv2/highgui.hpp"
#include  "opencv2/imgproc.hpp"

#include  <vector>
#include  <iostream>
#include  <iomanip>

using  namespace std;
using  namespace cv;

void  er_draw(Mat &src, Mat &dst, ERStat& er);

void  er_draw(Mat &src, Mat &dst, ERStat& er)
{

    if (er.parent != NULL) // deprecate the root region 
    {
        int newMaskVal = 255;
        int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
        floodFill(src,dst,Point(er.pixel%src.cols,er.pixel/src.cols),Scalar(255),0,Scalar(er.level),Scalar(0),flags);
    }

}
		
int  main(int argc, const char * argv[])
{


    vector<ERStat> regions;

    if (argc < 2) {
        cout << "Demo program of the Extremal Region Filter algorithm described in " << endl;
        cout << "Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012" << endl << endl;
        cout << "    Usage: " << argv[0] << " input_image <optional_groundtruth_image>" << endl;
        cout << "    Default classifier files (trained_classifierNM*.xml) should be in ./" << endl;
        return -1;
    }

    Mat original = imread(argv[1]);
    Mat gt;
    if (argc > 2)
    {
        gt = imread(argv[2]);
        cvtColor(gt, gt, COLOR_RGB2GRAY);
        threshold(gt, gt, 254, 255, THRESH_BINARY);
    }
    Mat grey(original.size(),CV_8UC1);
    cvtColor(original,grey,COLOR_RGB2GRAY);
    
    double t = (double)getTickCount();
    
    // Build ER tree and filter with the 1st stage default classifier
    Ptr<ERFilter> er_filter1 = createERFilterNM1();
    
    er_filter1->run(grey, regions);
    
    t = (double)getTickCount() - t;
    cout << " --------------------------------------------------------------------------------------------------" << endl;
    cout << "\t FIRST STAGE CLASSIFIER done in " << t * 1000. / getTickFrequency() << " ms." << endl;
    cout << " --------------------------------------------------------------------------------------------------" << endl;
    cout << setw(9) << regions.size()+er_filter1->getNumRejected() << "\t Extremal Regions extracted " << endl;
    cout << setw(9) << regions.size() << "\t Extremal Regions selected by the first stage of the sequential classifier." << endl;
    cout << "\t \t (saving into out_second_stage.jpg)" << endl;
    cout << " --------------------------------------------------------------------------------------------------" << endl;

    er_filter1.release();

    // draw regions
    Mat mask = Mat::zeros(grey.rows+2,grey.cols+2,CV_8UC1);
    for (int r=0; r<(int)regions.size(); r++)
        er_draw(grey, mask, regions.at(r));
    mask = 255-mask;
    imwrite("out_first_stage.jpg", mask);

    if (argc > 2)
    {
        Mat tmp_mask = (255-gt) & (255-mask(Rect(Point(1,1),Size(mask.cols-2,mask.rows-2))));
        cout << "Recall for the 1st stage filter = " << (float)countNonZero(tmp_mask) / countNonZero(255-gt) << endl;
    }

    t = (double)getTickCount();
    
    // Default second stage classifier
    Ptr<ERFilter> er_filter2 = createERFilterNM2();
    er_filter2->run(grey, regions);
    
    t = (double)getTickCount() - t;
    cout << " --------------------------------------------------------------------------------------------------" << endl;
    cout << "\t SECOND STAGE CLASSIFIER done in " << t * 1000. / getTickFrequency() << " ms." << endl;
    cout << " --------------------------------------------------------------------------------------------------" << endl;
    cout << setw(9) << regions.size() << "\t Extremal Regions selected by the second stage of the sequential classifier." << endl;
    cout << "\t \t (saving into out_second_stage.jpg)" << endl;
    cout << " --------------------------------------------------------------------------------------------------" << endl;

    er_filter2.release();

    // draw regions
    mask = mask*0;
    for (int r=0; r<(int)regions.size(); r++)
        er_draw(grey, mask, regions.at(r));
    mask = 255-mask;
    imwrite("out_second_stage.jpg", mask);

    if (argc > 2)
    {
        Mat tmp_mask = (255-gt) & (255-mask(Rect(Point(1,1),Size(mask.cols-2,mask.rows-2))));
        cout << "Recall for the 2nd stage filter = " << (float)countNonZero(tmp_mask) / countNonZero(255-gt) << endl;
    }

    regions.clear();

}
