#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib> // 包含 system 函数

using namespace std;
using namespace cv;

static void help(char** argv)
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo program shows how perspective transformation applied on an image, \n"
         "Using OpenCV version " << CV_VERSION << endl;

    cout << "\nUsage:\n" << argv[0] << " [image_name -- Default right.jpg]\n" << endl;

    cout << "\nHot keys: \n"
         "\tESC, q - quit the program\n"
         "\tr - change order of points to rotate transformation\n"
         "\tc - delete selected points\n"
         "\ti - change order of points to inverse transformation \n"
         "\nUse your mouse to select a point and move it to see transformation changes" << endl;
}

Mat warping(Mat image, Size warped_image_size, vector<Point2f> srcPoints, vector<Point2f> dstPoints);

String windowTitle = "Perspective Transformation Demo";
String labels[4] = {"TL", "TR", "BR", "BL"};
vector<Point2f> roi_corners;
vector<Point2f> midpoints(4);
vector<Point2f> dst_corners(4);
int roiIndex = 0;
bool dragging;
int selected_corner_index = 0;
bool validation_needed = true;

int main(int argc, char** argv)
{
    help(argv);
    CommandLineParser parser(argc, argv, "{@input| right.jpg |}");

    string filename = samples::findFile(parser.get<string>("@input"));
    Mat original_image = imread(filename);
    if (original_image.empty()) {
        cerr << "ERROR! Unable to open image\n";
        return -1;
    }

    // 创建子目录
    system("mkdir -p warpPerspective_demo");

    float original_image_cols = (float)original_image.cols;
    float original_image_rows = (float)original_image.rows;
    roi_corners.push_back(Point2f((float)(original_image_cols / 1.70), (float)(original_image_rows / 4.20)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.15), (float)(original_image.rows / 3.32)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.33), (float)(original_image.rows / 1.10)));
    roi_corners.push_back(Point2f((float)(original_image.cols / 1.93), (float)(original_image.rows / 1.36)));

    bool endProgram = false;
    int frame_count = 0;
    while (!endProgram)
    {
        Mat image = original_image.clone();

        if (validation_needed & (roi_corners.size() < 4))
        {
            validation_needed = false;

            for (size_t i = 0; i < roi_corners.size(); ++i)
            {
                circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);

                if (i > 0)
                {
                    line(image, roi_corners[i - 1], roi_corners[i], Scalar(0, 0, 255), 2);
                    circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
                    putText(image, labels[i].c_str(), roi_corners[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
                }
            }
        }

        if (validation_needed & (roi_corners.size() == 4))
        {
            for (int i = 0; i < 4; ++i)
            {
                line(image, roi_corners[i], roi_corners[(i + 1) % 4], Scalar(0, 0, 255), 2);
                circle(image, roi_corners[i], 5, Scalar(0, 255, 0), 3);
                putText(image, labels[i].c_str(), roi_corners[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
            }

            midpoints[0] = (roi_corners[0] + roi_corners[1]) / 2;
            midpoints[1] = (roi_corners[1] + roi_corners[2]) / 2;
            midpoints[2] = (roi_corners[2] + roi_corners[3]) / 2;
            midpoints[3] = (roi_corners[3] + roi_corners[0]) / 2;

            dst_corners[0].x = 0;
            dst_corners[0].y = 0;
            dst_corners[1].x = (float)norm(midpoints[1] - midpoints[3]);
            dst_corners[1].y = 0;
            dst_corners[2].x = dst_corners[1].x;
            dst_corners[2].y = (float)norm(midpoints[0] - midpoints[2]);
            dst_corners[3].x = 0;
            dst_corners[3].y = dst_corners[2].y;

            Size warped_image_size = Size(cvRound(dst_corners[2].x), cvRound(dst_corners[2].y));

            Mat M = getPerspectiveTransform(roi_corners, dst_corners);

            Mat warped_image;
            warpPerspective(original_image, warped_image, M, warped_image_size); // do perspective transformation

            stringstream warped_filename;
            warped_filename << "warpPerspective_demo/warped_image_" << frame_count++ << ".jpg";
            imwrite(warped_filename.str(), warped_image);
            cout << "Saved warped image: " << warped_filename.str() << endl;
        }

        // 注释掉等待键盘输入的代码
        // char c = (char)waitKey(10);
        // if ((c == 'q') | (c == 'Q') | (c == 27))
        // {
        //     endProgram = true;
        // }

        // 模拟退出条件
        if (frame_count >= 10)  // 示例：处理10帧后退出
            break;
    }
    return 0;
}

