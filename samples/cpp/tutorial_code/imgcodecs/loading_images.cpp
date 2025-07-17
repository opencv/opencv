#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main(int argc, const char** argv)
{
    std::string filename = argc > 1 ? argv[1] : "animated_image.webp";

    //! [init_imagecollection]
    ImageCollection collection(filename);

    // Check if initialization succeeded
    if (collection.getLastError() != ImageCollection::OK)
    {
        std::cerr << "Failed to initialize ImageCollection";
        return -1;
    }
    //! [init_imagecollection]

    int width = collection.getWidth();
    int height = collection.getHeight();

    //! [read_frames]
    // Iterate through frames and display them
    for (auto it = collection.begin(); it != collection.end(); ++it)
    {
        Mat frame = *it;
        imshow("Frame", frame);
        waitKey(100);
    }
    //! [read_frames]

    return 0;
}
