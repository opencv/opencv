#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <output_image_name>\n";
        return 1;
    }

    string outputImageName = argv[1];
    Mat framebuffer(160 * 2, 160 * 5, CV_8UC3, cv::Scalar::all(255));
    Mat img(160, 160, CV_8UC3, cv::Scalar::all(255));

    // Create test image.
    {
        const Point center(img.rows / 2, img.cols / 2);
        for (int radius = 5; radius < img.rows; radius += 3) {
            cv::circle(img, center, radius, Scalar(255, 0, 255));
        }
        cv::rectangle(img, Point(0, 0), Point(img.rows - 1, img.cols - 1), Scalar::all(0), 2);
    }

    // Draw original image(s).
    int top = 0; // Upper images
    {
        for (int left = 0; left < img.rows * 5; left += img.rows) {
            Mat roi = framebuffer(Rect(left, top, img.rows, img.cols));
            img.copyTo(roi);

            // 注释掉图像显示相关的代码
            // cv::putText(roi, "original", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0), 2, 4, false);
        }
    }

    // Draw lossy images
    top += img.cols; // Lower images
    {
        struct test_config {
            string comment;
            uint32_t sampling_factor;
        } config[] = {
            {"411", IMWRITE_JPEG_SAMPLING_FACTOR_411},
            {"420", IMWRITE_JPEG_SAMPLING_FACTOR_420},
            {"422", IMWRITE_JPEG_SAMPLING_FACTOR_422},
            {"440", IMWRITE_JPEG_SAMPLING_FACTOR_440},
            {"444", IMWRITE_JPEG_SAMPLING_FACTOR_444},
        };

        const int config_num = 5;
        int left = 0;

        for (int i = 0; i < config_num; i++) {
            // Compress images with sampling factor parameter.
            vector<int> param;
            param.push_back(IMWRITE_JPEG_SAMPLING_FACTOR);
            param.push_back(config[i].sampling_factor);
            vector<uint8_t> jpeg;
            (void)imencode(".jpg", img, jpeg, param);

            // Decompress it.
            Mat jpegMat(jpeg);
            Mat lossy_img = imdecode(jpegMat, -1);

            // Copy into framebuffer and comment
            Mat roi = framebuffer(Rect(left, top, lossy_img.rows, lossy_img.cols));
            lossy_img.copyTo(roi);
            // 注释掉图像显示相关的代码
            // cv::putText(roi, config[i].comment, Point(5, 155), FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(0), 2, 4, false);

            left += lossy_img.rows;
        }
    }

    // Output framebuffer(as lossless).
    imwrite(outputImageName, framebuffer);

    cout << "Output image saved as: " << outputImageName << endl;

    return 0;
}

