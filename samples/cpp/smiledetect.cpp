#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

static void help()
{
    cout << "\nThis program demonstrates the smile detector.\n"
            "Usage:\n"
            "./smiledetect [--cascade=<cascade_path> this is the frontal face classifier]\n"
            "   [--smile-cascade=[<smile_cascade_path>]]\n"
            "   [--scale=<image scale greater or equal to 1, try 2.0 for example. The larger the faster the processing>]\n"
            "   [--try-flip]\n"
            "   [video_filename|camera_index]\n\n"
            "Example:\n"
            "./smiledetect --cascade=\"../../data/haarcascades/haarcascade_frontalface_alt.xml\" --smile-cascade=\"../../data/haarcascades/haarcascade_smile.xml\" --scale=2.0\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );

string cascadeName;
string nestedCascadeName;

int main( int argc, const char** argv )
{
    VideoCapture capture;
    Mat frame, image;
    string inputName;
    bool tryflip;

    help();

    CascadeClassifier cascade, nestedCascade;
    double scale;
    cv::CommandLineParser parser(argc, argv,
        "{help h||}{scale|1|}"
        "{cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{smile-cascade|../../data/haarcascades/haarcascade_smile.xml|}"
        "{try-flip||}{@input||}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("smile-cascade");
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@input");
    scale = parser.get<int>("scale");
    if (!parser.check())
    {
        help();
        return 1;
    }
    if (scale < 1)
        scale = 1;
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load face cascade" << endl;
        help();
        return -1;
    }
    if( !nestedCascade.load( nestedCascadeName ) )
    {
        cerr << "ERROR: Could not load smile cascade" << endl;
        help();
        return -1;
    }
    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int c = inputName.empty() ? 0 : inputName[0] - '0' ;
        if(!capture.open(c))
            cout << "Capture from camera #" <<  c << " didn't work" << endl;
    }
    else if( inputName.size() )
    {
        if(!capture.open( inputName ))
            cout << "Could not read " << inputName << endl;
    }

    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        cout << endl << "NOTE: Smile intensity will only be valid after a first smile has been detected" << endl;

        for(;;)
        {
            capture >> frame;
            if( frame.empty() )
                break;

            Mat frame1 = frame.clone();
            detectAndDraw( frame1, cascade, nestedCascade, scale, tryflip );

            char c = (char)waitKey(10);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }
    }
    else
    {
        cerr << "ERROR: Could not initiate capture" << endl;
        help();
        return -1;
    }

    return 0;
}

void detectAndDraw( Mat& img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip)
{
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );

    double fx = 1 / scale;
    resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);

        const int half_height=cvRound((float)r.height/2);
        r.y=r.y + half_height;
        r.height = half_height-1;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 0, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );

        // The number of detected neighbors depends on image size (and also illumination, etc.). The
        // following steps use a floating minimum and maximum of neighbors. Intensity thus estimated will be
        //accurate only after a first smile has been displayed by the user.
        const int smile_neighbors = (int)nestedObjects.size();
        static int max_neighbors=-1;
        static int min_neighbors=-1;
        if (min_neighbors == -1) min_neighbors = smile_neighbors;
        max_neighbors = MAX(max_neighbors, smile_neighbors);

        // Draw rectangle on the left side of the image reflecting smile intensity
        float intensityZeroOne = ((float)smile_neighbors - min_neighbors) / (max_neighbors - min_neighbors + 1);
        int rect_height = cvRound((float)img.rows * intensityZeroOne);
        Scalar col = Scalar((float)255 * intensityZeroOne, 0, 0);
        rectangle(img, cvPoint(0, img.rows), cvPoint(img.cols/10, img.rows - rect_height), col, -1);
    }

    imshow( "result", img );
}
