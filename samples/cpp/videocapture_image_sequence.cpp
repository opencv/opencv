#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static void help(char** argv)
{
    cout << "\nThis sample shows you how to read a sequence of images using the VideoCapture interface.\n"
         << "Usage: " << argv[0] << " <image_mask> (example mask: example_%02d.jpg)\n"
         << "Image mask defines the name variation for the input images that have to be read as a sequence. \n"
         << "Using the mask example_%02d.jpg will read in images labeled as 'example_00.jpg', 'example_01.jpg', etc."
         << endl;
}

int main(int argc, char** argv)
{
    help(argv);
    cv::CommandLineParser parser(argc, argv, "{@image| ../data/left%02d.jpg |}");
    string first_file = parser.get<string>("@image");

    if(first_file.empty())
    {
        return 1;
    }

    VideoCapture sequence(first_file);

    if (!sequence.isOpened())
    {
        cerr << "Failed to open the image sequence!\n" << endl;
        return 1;
    }

    Mat image;
    namedWindow("Image sequence | press ESC to close", 1);

    for(;;)
    {
        // Read in image from sequence
        sequence >> image;

        // If no image was retrieved -> end of sequence
        if(image.empty())
        {
            cout << "End of Sequence" << endl;
            break;
        }

        imshow("Image sequence | press ESC to close", image);

        if(waitKey(500) == 27)
            break;
    }

    return 0;
}
