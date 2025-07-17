#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main(int argc, const char** argv)
{
    std::string filename = argc > 1 ? argv[1] : "animated_image.webp";

    //! [init_imagecollection]
    ImageCollection collection1(filename);
    ImageCollection collection2(filename,IMREAD_REDUCED_GRAYSCALE_2);
    ImageCollection collection3(filename,IMREAD_REDUCED_COLOR_2);
    ImageCollection collection4(filename, IMREAD_COLOR_RGB);

    // Check if initialization succeeded
    if (collection1.getLastError() != ImageCollection::OK)
    {
        std::cerr << "Failed to initialize ImageCollection";
        return -1;
    }
    //! [init_imagecollection]

    size_t size = collection1.size();
    int width = collection1.getWidth();
    int height = collection1.getHeight();
    int type = collection1.getType();

    std::cout << "size   : " << size << std::endl;
    std::cout << "width  : " << width << std::endl;
    std::cout << "height : " << height << std::endl;
    std::cout << "type   : " << type << std::endl;

    //! [read_frames]
    // Iterate through frames and display them
    for (auto it = collection1.begin(); it != collection1.end(); ++it)
    {
        Mat frame = *it;
        imshow("Frame", frame);
        waitKey(100);
    }
    //! [read_frames]

    int idx1 = 0, idx2 = 0, idx3 = 0;

    std::cout << "Controls:\n"
              << "  a/d: prev/next idx1\n"
              << "  j/l: prev/next idx2\n"
              << "  z/c: prev/next idx3\n"
              << "  ESC: exit\n";

    while (true)
    {
        // Show current images
        cv::imshow("Image 1", collection1[idx1]);
        cv::imshow("Image 2", collection2[idx2]);
        cv::imshow("Image 3", collection3[idx3]);
        cv::imshow("Image 4", collection4[idx1]);

        int key = cv::waitKey(0);

        switch (key)
        {
        case 'a': idx1--; break;
        case 'd': idx1++; break;
        case 'j': idx2--; break;
        case 'l': idx2++; break;
        case 'z': idx3--; break;
        case 'c': idx3++; break;
        case 'q': return 0;
        case  27: return 0;
        }

        idx1 = std::max(0, std::min(idx1, static_cast<int>(collection1.size()) - 1));
        idx2 = std::max(0, std::min(idx2, static_cast<int>(collection2.size()) - 1));
        idx3 = std::max(0, std::min(idx3, static_cast<int>(collection3.size()) - 1));
    }

    return 0;
}
