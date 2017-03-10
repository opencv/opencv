#include <iostream>
#include <stdexcept>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace std;


const char* keys =
{
    "{ help h      |                     | print help message }"
    "{ image i     |                     | specify input image}"
    "{ camera c    |                     | enable camera capturing }"
    "{ video v     | ../data/vtest.avi   | use video as input }"
    "{ directory d |                     | images directory}"
};

static void detectAndDraw(const HOGDescriptor &hog, Mat &img)
{
    vector<Rect> found, found_filtered;
    double t = (double) getTickCount();
    // Run the detector with default parameters. to get a higher hit-rate
    // (and more false alarms, respectively), decrease the hitThreshold and
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
    t = (double) getTickCount() - t;
    cout << "detection time = " << (t*1000./cv::getTickFrequency()) << " ms" << endl;

    for(size_t i = 0; i < found.size(); i++ )
    {
        Rect r = found[i];

        size_t j;
        // Do not add small detections inside a bigger detection.
        for ( j = 0; j < found.size(); j++ )
            if ( j != i && (r & found[j]) == r )
                break;

        if ( j == found.size() )
            found_filtered.push_back(r);
    }

    for (size_t i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];

        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
    }
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        cout << "\nThis program demonstrates the use of the HoG descriptor using\n"
            " HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n";
        parser.printMessage();
        cout << "During execution:\n\tHit q or ESC key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n"
            "Note: camera device number must be different from -1.\n" << endl;
        return 0;
    }

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    namedWindow("people detector", 1);

    string pattern_glob = "";
    string video_filename = "../data/vtest.avi";
    int camera_id = -1;
    if (parser.has("directory"))
    {
        pattern_glob = parser.get<string>("directory");
    }
    else if (parser.has("image"))
    {
        pattern_glob = parser.get<string>("image");
    }
    else if (parser.has("camera"))
    {
        camera_id = parser.get<int>("camera");
    }
    else if (parser.has("video"))
    {
        video_filename = parser.get<string>("video");
    }

    if (!pattern_glob.empty() || camera_id != -1 || !video_filename.empty())
    {
        //Read from input image files
        vector<String> filenames;
        //Read from video file
        VideoCapture vc;
        Mat frame;

        if (!pattern_glob.empty())
        {
            String folder(pattern_glob);
            glob(folder, filenames);
        }
        else if (camera_id != -1)
        {
            vc.open(camera_id);
            if (!vc.isOpened())
            {
                stringstream msg;
                msg << "can't open camera: " << camera_id;
                throw runtime_error(msg.str());
            }
        }
        else
        {
            vc.open(video_filename.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + video_filename));
        }

        vector<String>::const_iterator it_image = filenames.begin();

        for (;;)
        {
            if (!pattern_glob.empty())
            {
                bool read_image_ok = false;
                for (; it_image != filenames.end(); ++it_image)
                {
                    cout << "\nRead: " << *it_image << endl;
                    // Read current image
                    frame = imread(*it_image);

                    if (!frame.empty())
                    {
                        ++it_image;
                        read_image_ok = true;
                        break;
                    }
                }

                //No more valid images
                if (!read_image_ok)
                {
                    //Release the image in order to exit the while loop
                    frame.release();
                }
            }
            else
            {
                vc >> frame;
            }

            if (frame.empty())
                break;

            detectAndDraw(hog, frame);

            imshow("people detector", frame);
            int c = waitKey( vc.isOpened() ? 30 : 0 ) & 255;
            if ( c == 'q' || c == 'Q' || c == 27)
                break;
        }
    }

    return 0;
}
