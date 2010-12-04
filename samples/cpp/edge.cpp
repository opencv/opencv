#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

void help()
{
	cout <<
			"\nDemonstrate Canny edge detection\n"
			"Call:\n"
			"/.edge [image_name -- Default is fruits.jpg]\n" << endl;

}

int edgeThresh = 1;
Mat image, gray, edge, cedge;

// define a trackbar callback
void onTrackbar(int, void*)
{
    blur(gray, edge, Size(3,3));

    // Run the edge detector on grayscale
    Canny(edge, edge, edgeThresh, edgeThresh*3, 3);
    cedge = Scalar::all(0);
    
    image.copyTo(cedge, edge);
    imshow("Edge map", cedge);
}

int main( int argc, char** argv )
{
    char* filename = argc == 2 ? argv[1] : (char*)"fruits.jpg";

    image = imread(filename, 1);
    if(image.empty())
    {
    	help();
        return -1;
    }
    help();
    cedge.create(image.size(), image.type());
    cvtColor(image, gray, CV_BGR2GRAY);

    // Create a window
    namedWindow("Edge map", 1);

    // create a toolbar
    createTrackbar("Canny threshold", "Edge map", &edgeThresh, 100, onTrackbar);

    // Show the image
    onTrackbar(0, 0);

    // Wait for a key stroke; the same function arranges events processing
    waitKey(0);

    return 0;
}
