#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
int main()
{
    VideoCapture obsensorCapture(0, CAP_OBSENSOR);
    if(!obsensorCapture.isOpened()){
        std::cerr << "Failed to open obsensor capture! Index out of range or no response from device";
        return -1;
    }

    double fx = obsensorCapture.get(CAP_PROP_OBSENSOR_INTRINSIC_FX);
    double fy = obsensorCapture.get(CAP_PROP_OBSENSOR_INTRINSIC_FY);
    double cx = obsensorCapture.get(CAP_PROP_OBSENSOR_INTRINSIC_CX);
    double cy = obsensorCapture.get(CAP_PROP_OBSENSOR_INTRINSIC_CY);
    std::cout << "obsensor camera intrinsic params: fx=" << fx << ", fy=" << fy << ", cx=" << cx << ", cy=" << cy << std::endl;

    Mat image;
    Mat depthMap;
    Mat adjDepthMap;
    while (true)
    {
        // Grab depth map like this:
        // obsensorCapture >> depthMap;

        // Another way to grab depth map (and bgr image).
        if (obsensorCapture.grab())
        {
            if (obsensorCapture.retrieve(image, CAP_OBSENSOR_BGR_IMAGE))
            {
                imshow("RGB", image);
            }

            if (obsensorCapture.retrieve(depthMap, CAP_OBSENSOR_DEPTH_MAP))
            {
                normalize(depthMap, adjDepthMap, 0, 255, NORM_MINMAX, CV_8UC1);
                applyColorMap(adjDepthMap, adjDepthMap, COLORMAP_JET);
                imshow("DEPTH", adjDepthMap);
            }

            // depth map overlay on bgr image
            static const float alpha = 0.6f;
            if (!image.empty() && !depthMap.empty())
            {
                normalize(depthMap, adjDepthMap, 0, 255, NORM_MINMAX, CV_8UC1);
                cv::resize(adjDepthMap, adjDepthMap, cv::Size(image.cols, image.rows));
                for (int i = 0; i < image.rows; i++)
                {
                    for (int j = 0; j < image.cols; j++)
                    {
                        cv::Vec3b& outRgb = image.at<cv::Vec3b>(i, j);
                        uint8_t depthValue = 255 - adjDepthMap.at<uint8_t>(i, j);
                        if (depthValue != 0 && depthValue != 255)
                        {
                            outRgb[0] = (uint8_t)(outRgb[0] * (1.0f - alpha) + depthValue * alpha);
                            outRgb[1] = (uint8_t)(outRgb[1] * (1.0f - alpha) + depthValue *  alpha);
                            outRgb[2] = (uint8_t)(outRgb[2] * (1.0f - alpha) + depthValue *  alpha);
                        }
                    }
                }
                imshow("DepthToColor", image);
            }
            image.release();
            depthMap.release();
        }

        if (pollKey() >= 0)
            break;
    }
    return 0;
}