#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

struct BlendMode
{
    int id;
    string name;
    static vector<BlendMode> getModes()
    {
        vector<BlendMode> res;
        res.push_back(BLEND_MODEL_DARKEN);
        res.push_back(BLEND_MODEL_MULTIPY);
        res.push_back(BLEND_MODEL_COLOR_BURN);
        res.push_back(BLEND_MODEL_LINEAR_BURN);
        res.push_back(BLEND_MODEL_LIGHTEN);
        res.push_back(BLEND_MODEL_SCREEN);
        res.push_back(BLEND_MODEL_COLOR_DODGE);
        res.push_back(BLEND_MODEL_LINEAR_DODGE);
        res.push_back(BLEND_MODEL_OVERLAY);
        res.push_back(BLEND_MODEL_SOFT_LIGHT);
        res.push_back(BLEND_MODEL_HARD_LIGHT);
        res.push_back(BLEND_MODEL_VIVID_LIGHT);
        res.push_back(BLEND_MODEL_LINEAR_LIGHT);
        res.push_back(BLEND_MODEL_PIN_LIGHT);
        res.push_back(BLEND_MODEL_DIFFERENCE);
        res.push_back(BLEND_MODEL_EXCLUSION);
        res.push_back(BLEND_MODEL_DIVIDE);
        return res;
    }
};

int main()
{
    Mat target = imread(samples::findFile("lena.jpg"));
    Mat blend(target.size(), CV_8UC3, Scalar::all(0));
    for (int i = 0; i < blend.rows; ++i)
        blend.row(i) = Scalar::all((double)i / blend.rows * 255);
    imshow("Gradient", blend);
    Mat result;
    vector<BlendMode> modes = BlendMode::getModes();
    for (vector<BlendMode>::const_iterator i = modes.begin(); i != modes.end(); ++i)
    {
        layerModelBlending(target, blend, result, i->id);
        putText(result, i->name, Point(5, 20), FONT_HERSHEY_PLAIN, 1., Scalar(0, 230, 0));
        imshow("Blend", result);
        const char key = waitKey(0);
        if (key == 'q' || key == 27)
            break;
    }
}
