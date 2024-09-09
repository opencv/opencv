#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv,
                             "{input   i|lena.jpg|input image}" // 使用 lena.jpg 作为输入图像
                             "{t |templ.png|template name}"     // 使用 templ.png 作为模板图像
                             "{m |mask.png|mask name}"          // 使用 mask.png 作为掩码图像
                             "{cm|3|comparison method}"         // 默认比较方法
                             "{o |result.jpg|output image name}"// 输出图像名称
                             "{help h|false|show help message}");

    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    parser.printMessage();

    string filename = samples::findFile(parser.get<string>("input"));
    string tmplname = samples::findFile(parser.get<string>("t"));
    string maskname = samples::findFile(parser.get<string>("m"));
    string outputname = parser.get<string>("o");
    
    Mat img = imread(filename);
    Mat tmpl = imread(tmplname);
    Mat mask = imread(maskname);
    Mat res;

    if(img.empty())
    {
        cout << "Cannot open " << filename << endl;
        return -1;
    }

    if(tmpl.empty())
    {
        cout << "Cannot open " << tmplname << endl;
        return -1;
    }

    if(mask.empty())
    {
        cout << "Cannot open " << maskname << endl;
        return -1;
    }

    // Resize mask to match template size if necessary
    if (tmpl.size() != mask.size()) {
        resize(mask, mask, tmpl.size());
    }

    int method = parser.get<int>("cm"); // 默认比较方法
    matchTemplate(img, tmpl, res, method, mask);

    double minVal, maxVal;
    Point minLoc, maxLoc;
    Rect rect;
    minMaxLoc(res, &minVal, &maxVal, &minLoc, &maxLoc);

    if(method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
        rect = Rect(minLoc, tmpl.size());
    else
        rect = Rect(maxLoc, tmpl.size());

    rectangle(img, rect, Scalar(0, 255, 0), 2);

    // Save the result image
    imwrite(outputname, img);
    cout << "Result saved to " << outputname << endl;

    return 0;
}

