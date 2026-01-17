/**
 * @file resize_demo.cpp
 * @brief Demonstration of cv::resize() function with different interpolation methods
 * @author OpenCV Documentation
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/**
 * @function main
 */
int main(int argc, char** argv)
{
    // Load source image
    CommandLineParser parser(argc, argv, 
        "{@input | lena.jpg | input image}"
        "{help h | | show help message}");
    
    if (parser.has("help")) {
        cout << "Usage: resize_demo [image_path]" << endl;
        cout << "This program demonstrates different interpolation methods in cv::resize()" << endl;
        return 0;
    }
    
    String imageName = parser.get<String>("@input");
    
    // Load the source image
    Mat src = imread(samples::findFile(imageName), IMREAD_COLOR);
    
    if (src.empty()) {
        cout << "Could not open or find the image: " << imageName << endl;
        cout << "Usage: resize_demo [image_path]" << endl;
        return -1;
    }
    
    cout << "Original image size: " << src.cols << "x" << src.rows << endl;
    
    // Create windows for display
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", src);
    
    // Demonstrate different resize methods
    
    // 1. Resize by specifying output size (upscale by 2x)
    Mat dst_size;
    Size newSize(src.cols * 2, src.rows * 2);
    resize(src, dst_size, newSize, 0, 0, INTER_LINEAR);
    cout << "\n1. Resize by size (2x upscale with INTER_LINEAR):" << endl;
    cout << "   Output size: " << dst_size.cols << "x" << dst_size.rows << endl;
    namedWindow("2x Upscale (INTER_LINEAR)", WINDOW_AUTOSIZE);
    imshow("2x Upscale (INTER_LINEAR)", dst_size);
    
    // 2. Resize by scale factor (downscale by 0.5x)
    Mat dst_scale;
    resize(src, dst_scale, Size(), 0.5, 0.5, INTER_AREA);
    cout << "\n2. Resize by scale factor (0.5x downscale with INTER_AREA):" << endl;
    cout << "   Output size: " << dst_scale.cols << "x" << dst_scale.rows << endl;
    namedWindow("0.5x Downscale (INTER_AREA)", WINDOW_AUTOSIZE);
    imshow("0.5x Downscale (INTER_AREA)", dst_scale);
    
    // 3. Compare different interpolation methods for upscaling
    Mat dst_nearest, dst_linear, dst_cubic, dst_lanczos;
    Size upscaleSize(src.cols * 2, src.rows * 2);
    
    resize(src, dst_nearest, upscaleSize, 0, 0, INTER_NEAREST);
    resize(src, dst_linear, upscaleSize, 0, 0, INTER_LINEAR);
    resize(src, dst_cubic, upscaleSize, 0, 0, INTER_CUBIC);
    resize(src, dst_lanczos, upscaleSize, 0, 0, INTER_LANCZOS4);
    
    cout << "\n3. Comparing interpolation methods (2x upscale):" << endl;
    cout << "   INTER_NEAREST - Fastest, lowest quality" << endl;
    cout << "   INTER_LINEAR  - Good balance (default)" << endl;
    cout << "   INTER_CUBIC   - Better quality, slower" << endl;
    cout << "   INTER_LANCZOS4 - Best quality, slowest" << endl;
    
    namedWindow("INTER_NEAREST", WINDOW_AUTOSIZE);
    namedWindow("INTER_LINEAR", WINDOW_AUTOSIZE);
    namedWindow("INTER_CUBIC", WINDOW_AUTOSIZE);
    namedWindow("INTER_LANCZOS4", WINDOW_AUTOSIZE);
    
    imshow("INTER_NEAREST", dst_nearest);
    imshow("INTER_LINEAR", dst_linear);
    imshow("INTER_CUBIC", dst_cubic);
    imshow("INTER_LANCZOS4", dst_lanczos);
    
    // 4. Compare different interpolation methods for downscaling
    Mat dst_down_nearest, dst_down_linear, dst_down_area;
    Size downscaleSize(src.cols / 2, src.rows / 2);
    
    resize(src, dst_down_nearest, downscaleSize, 0, 0, INTER_NEAREST);
    resize(src, dst_down_linear, downscaleSize, 0, 0, INTER_LINEAR);
    resize(src, dst_down_area, downscaleSize, 0, 0, INTER_AREA);
    
    cout << "\n4. Comparing interpolation methods (0.5x downscale):" << endl;
    cout << "   INTER_NEAREST - Fastest, can cause artifacts" << endl;
    cout << "   INTER_LINEAR  - Good quality" << endl;
    cout << "   INTER_AREA    - Best for downscaling (recommended)" << endl;
    
    namedWindow("Downscale INTER_NEAREST", WINDOW_AUTOSIZE);
    namedWindow("Downscale INTER_LINEAR", WINDOW_AUTOSIZE);
    namedWindow("Downscale INTER_AREA", WINDOW_AUTOSIZE);
    
    imshow("Downscale INTER_NEAREST", dst_down_nearest);
    imshow("Downscale INTER_LINEAR", dst_down_linear);
    imshow("Downscale INTER_AREA", dst_down_area);
    
    // 5. Demonstration of aspect ratio change
    Mat dst_aspect;
    resize(src, dst_aspect, Size(src.cols * 2, src.rows), 0, 0, INTER_LINEAR);
    cout << "\n5. Non-uniform scaling (width 2x, height 1x):" << endl;
    cout << "   Output size: " << dst_aspect.cols << "x" << dst_aspect.rows << endl;
    namedWindow("Non-uniform Scale", WINDOW_AUTOSIZE);
    imshow("Non-uniform Scale", dst_aspect);
    
    cout << "\nPress any key to exit..." << endl;
    waitKey(0);
    
    return 0;
}
