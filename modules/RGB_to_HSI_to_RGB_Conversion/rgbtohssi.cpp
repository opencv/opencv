/// External Stuff//
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
void hsi_to_RGB(float hue, float saturation, float intensity, float r, float g, float b);
int main()
{

    Mat src = imread("Resources/peppers.jpg", 1);
    imshow("Original RGB image", src);
    waitKey(0);

    if (src.empty())
        cerr << "Error: Loading image" << endl;
    Mat hsi(src.rows, src.cols, src.type());
    Mat RGB(src.rows, src.cols, src.type());
    float r, g, b, h, s, in;

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            b = src.at<Vec3b>(i, j)[0];
            g = src.at<Vec3b>(i, j)[1];
            r = src.at<Vec3b>(i, j)[2];

            in = (b + g + r) / 3;

            int min_val = 0;
            min_val = std::min(r, std::min(b, g));

            s = 1 - 3 * (min_val / (b + g + r));
            if (s < 0.00001)
            {
                s = 0;
            }
            else if (s > 0.99999) {
                s = 1;
            }

            if (s != 0)
            {
                h = 0.5 * ((r - g) + (r - b)) / sqrt(((r - g) * (r - g)) + ((r - b) * (g - b)));
                h = acos(h);
                h = h * 180.0 / 3.14159265; // h now in degree
                if (b <= g)
                {
                    h = h;
                }
                else {
                    h = (360 - h);
                }
            }
            hsi_to_RGB(h, s, i, r, g, b);
            hsi.at<Vec3b>(i, j)[0] = (h / 360) * 255;
            hsi.at<Vec3b>(i, j)[1] = s * 255;
            hsi.at<Vec3b>(i, j)[2] = in;


            RGB.at<Vec3b>(i, j)[0] = b;
            RGB.at<Vec3b>(i, j)[1] = g;
            RGB.at<Vec3b>(i, j)[2] = r;
        }

    }

    //namedWindow("RGB image", CV_WINDOW_AUTOSIZE);
    //namedWindow("HSI image", CV_WINDOW_AUTOSIZE);
    imshow("HSI image", hsi);
    waitKey(0);
    imshow("HSI to RGB image", RGB);
    // imshow("HSI image", hsi);


    Mat newRGB(hsi.rows, hsi.cols, hsi.type());


    // imshow("HSI to RGB image", newRGB);

    waitKey(0);
    return 0;
}


void hsi_to_RGB(float h, float s, float in, float r, float g, float b)
{

    float num, deno;

    if ((h >= 0) && (h < 120)) //h must come in degrees
    {
        b = in * (1 - s);//h has to be in radians
        h = (h * 3.14159265) / 180.0;
        num = s * cos(h);
        deno = cos(((60 * 3.14159265) / 180.0) - h);
        r = in * (1 + (num / deno));
        g = 3 * in - (r + b);
    }
    else if ((h >= 120) && (h < 240))
    {

        h = h - 120;  //h is in degree
        r = in * (1 - s);//h has to be in radians
        h = (h * 3.14159265) / 180.0;
        num = s * cos(h);
        deno = cos(((60 * 3.14159265) / 180.0) - h);
        g = in * (1 + (num / deno));
        b = 3 * in - (r + g);

    }
    else if ((h >= 240) && (h < 360))
    {
        h = h - 240;
        g = in * (1 - s);
        h = (h * 3.14159265) / 180.0; //h has to be in radians
        num = s * cos(h);
        deno = cos(((60 * 3.14159265) / 180.0) - h);
        b = in * (1 + (num / deno));
        r = 3 * in - (g + b);
    }
    else
    {
        cout << "Degree limit exceeded " << endl;
    }







}
