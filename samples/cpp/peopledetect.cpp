#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


static void help()
{
    cout << "\nThis program demonstrates the use of the HoG descriptor using\n"
            "  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"
            "Usage:\n"
            "./peopledetect --i=<image_filename> | --d=<image_directory>\n\n"
            "During execution:\n\tHit q or ESC key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

const char* keys =
{
    "{help h||}"
    "{image i| ../data/basketball2.png|input image name}"
    "{directory d||images directory}"
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
        help();
        return 0;
    }

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    namedWindow("people detector", 1);

    string pattern_glob = "";
    if (parser.has("directory"))
    {
        pattern_glob = parser.get<string>("directory");
    }
    else if (parser.has("image"))
    {
        pattern_glob = parser.get<string>("image");
    }

    if (!pattern_glob.empty())
    {
        vector<String> filenames;
        String folder(pattern_glob);
        glob(folder, filenames);

        for (vector<String>::const_iterator it = filenames.begin(); it != filenames.end(); ++it)
        {
            cout << "\nRead: " << *it << endl;
            // Read current image
            Mat current_image = imread(*it);

            if (current_image.empty())
                continue;

            detectAndDraw(hog, current_image);

            imshow("people detector", current_image);
            int c = waitKey(0) & 255;
            if ( c == 'q' || c == 'Q' || c == 27)
                break;
        }
    }

    return 0;
}
