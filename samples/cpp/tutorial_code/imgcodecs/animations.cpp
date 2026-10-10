#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

int main(int argc, const char** argv)
{
    std::string filename = argc > 1 ? argv[1] : "animated_image.webp";

    //! [write_animation]
    if (argc == 1)
    {
        Animation animation_to_save;
        Mat image(128, 256, CV_8UC4, Scalar(150, 150, 150, 255));
        int duration = 200;

        for (int i = 0; i < 10; ++i) {
            animation_to_save.frames.push_back(image.clone());
            putText(animation_to_save.frames[i], format("Frame %d", i), Point(30, 80), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(255, 100, 0, 255), 2);
            animation_to_save.durations.push_back(duration);
        }
        imwriteanimation("animated_image.webp", animation_to_save, { IMWRITE_WEBP_QUALITY, 100 });
    }
    //! [write_animation]

    //! [init_animation]
    Animation animation;
    //! [init_animation]

    //! [read_animation]
    bool success = imreadanimation(filename, animation);
    if (!success) {
        std::cerr << "Failed to load animation frames\n";
        return -1;
    }
    //! [read_animation]

    //! [show_animation]
    int escape = 0;
    while (escape < 1)
        for (size_t i = 0; i < animation.frames.size(); ++i) {
            imshow("Animation", animation.frames[i]);
            int key_code = waitKey(animation.durations[i]); // Delay between frames
            if (key_code == 27)
            {
                escape = 1;
                break;
            }
        }
    //! [show_animation]

//! [init_imagecollection]
// Initialize multiple ImageCollection instances with different flags
    ImageCollection collection1(filename); // default (IMREAD_UNCHANGED)
    ImageCollection collection2(filename, IMREAD_REDUCED_GRAYSCALE_2); // grayscale, reduced 1/2
    ImageCollection collection3(filename, IMREAD_REDUCED_COLOR_2);     // color, reduced 1/2
    ImageCollection collection4(filename, IMREAD_COLOR_RGB);           // full-size RGB
    ImageCollection collection5 = collection2;                         // copy constructor

    // Check if the first collection was initialized successfully
    if (collection1.getStatus() != DECODER_OK)
    {
        std::cerr << "Failed to initialize ImageCollection\n";
        return -1;
    }
    //! [init_imagecollection]

    //! [query_properties]
    // Query basic properties from the first collection
    size_t size = collection1.size();
    int width = collection1.getWidth();
    int height = collection1.getHeight();
    int type = collection1.getType();

    std::cout << "size   : " << size << std::endl;
    std::cout << "width  : " << width << std::endl;
    std::cout << "height : " << height << std::endl;
    std::cout << "type   : " << type << std::endl;
    //! [query_properties]

    //! [read_frames]
    // Iterate through frames in collection1 and display them
    for (auto it = collection1.begin(); it != collection1.end(); ++it)
    {
        Mat frame = *it;
        imshow("Frame", frame);
        waitKey(100); // Delay between frames
    }
    //! [read_frames]

    //! [interactive_navigation]
    // Interactive frame navigation across multiple ImageCollections
    int idx1 = 0, idx2 = 0, idx3 = 0, idx4 = 0;

    std::cout << "Controls:\n"
        << "  a/d: prev/next idx1 (collection1)\n"
        << "  j/l: prev/next idx2 (collection2)\n"
        << "  z/c: prev/next idx3 (collection3)\n"
        << "  q/e: prev/next idx4 (collection5)\n"
        << "  ESC: exit\n";

    while (true)
    {
        // Display current frames from each collection
        cv::imshow("Image 1", collection1[idx1]);
        cv::imshow("Image 2", collection2[idx2]);
        cv::imshow("Image 3", collection3[idx3]);
        cv::imshow("Image 4", collection4[idx1]);
        cv::imshow("Image 5", collection5[idx4]);

        int key = cv::waitKey(0);
        switch (key)
        {
        case 'a': idx1--; break;
        case 'd': idx1++; break;
        case 'j': idx2--; break;
        case 'l': idx2++; break;
        case 'z': idx3--; break;
        case 'c': idx3++; break;
        case 'q': idx4--; break;
        case 'e': idx4++; break;
        case  27: return 0; // ESC key
        }

        // Clamp indices to valid ranges
        idx1 = std::max(0, std::min(idx1, static_cast<int>(collection1.size()) - 1));
        idx2 = std::max(0, std::min(idx2, static_cast<int>(collection2.size()) - 1));
        idx3 = std::max(0, std::min(idx3, static_cast<int>(collection3.size()) - 1));
        idx4 = std::max(0, std::min(idx4, static_cast<int>(collection5.size()) - 1));
    }
    //! [interactive_navigation]
    return 0;
}
