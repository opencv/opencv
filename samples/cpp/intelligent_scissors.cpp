#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
struct Pix
{
    Point next_point;
    double cost;

    bool operator > (const Pix &b) const
    {
        return cost > b.cost;
    }
};

struct Parameters
{
    Mat img, img_pre_render, img_render;
    Point end;
    std::vector<std::vector<Point> > contours;
    std::vector<Point> tmp_contour;
    Mat zero_crossing, gradient_magnitude, Ix, Iy, hit_map_x, hit_map_y;
};


static float local_cost(const Point& p, const Point& q, const Mat& gradient_magnitude, const Mat& Iy, const Mat& Ix, const Mat& zero_crossing)
{
    float fG = gradient_magnitude.at<float>(q.y, q.x);
    float dp;
    float dq;
    const float WEIGHT_LAP_ZERO_CROSS = 0.43f;
    const float WEIGHT_GRADIENT_MAGNITUDE = 0.14f;
    const float WEIGHT_GRADIENT_DIRECTION = 0.43f;
    bool isDiag = (p.x != q.x) && (p.y != q.y);

    if ((Iy.at<float>(p) * (q.x - p.x) - Ix.at<float>(p) * (q.y - p.y)) >= 0)
    {
        dp = Iy.at<float>(p) * (q.x - p.x) - Ix.at<float>(p) * (q.y - p.y);
        dq = Iy.at<float>(q) * (q.x - p.x) - Ix.at<float>(q) * (q.y - p.y);
    }
    else
    {
        dp = Iy.at<float>(p) * (p.x - q.x) + (-Ix.at<float>(p)) * (p.y - q.y);
        dq = Iy.at<float>(q) * (p.x - q.x) + (-Ix.at<float>(q)) * (p.y - q.y);
    }
    if (isDiag)
    {
        dp /= sqrtf(2);
        dq /= sqrtf(2);
    }
    else
    {
        fG /= sqrtf(2);
    }
    return  WEIGHT_LAP_ZERO_CROSS * zero_crossing.at<uchar>(q) +
            WEIGHT_GRADIENT_DIRECTION * (acosf(dp) + acosf(dq)) / static_cast<float>(CV_PI) +
            WEIGHT_GRADIENT_MAGNITUDE * fG;
}

static void find_min_path(const Point& start, Parameters* param)
{
    Pix begin;
    Mat &img = param->img;
    Mat cost_map(img.size(), CV_32F, Scalar(FLT_MAX));
    Mat expand(img.size(), CV_8UC1, Scalar(0));
    Mat processed(img.size(), CV_8UC1, Scalar(0));
    Mat removed(img.size(), CV_8UC1, Scalar(0));
    std::priority_queue < Pix, std::vector<Pix>, std::greater<Pix> > L;

    cost_map.at<float>(start) = 0;
    processed.at<uchar>(start) = 1;
    begin.cost = 0;
    begin.next_point = start;

    L.push(begin);
    while (!L.empty())
    {
        Pix P = L.top();
        L.pop();
        Point p = P.next_point;
        processed.at<uchar>(p) = 0;
        if (removed.at<uchar>(p) == 0)
        {
            expand.at<uchar>(p) = 1;
            for (int i = -1; i <= 1; i++)
            {
                for(int j = -1; j <= 1; j++)
                {
                    int tx = p.x + i;
                    int ty = p.y + j;
                    if (tx < 0 || tx >= img.cols || ty < 0 || ty >= img.rows)
                        continue;
                    if (expand.at<uchar>(ty, tx) == 0)
                    {
                        Point q = Point(tx, ty);
                        float cost = cost_map.at<float>(p) + local_cost(p, q, param->gradient_magnitude, param->Iy, param->Ix, param->zero_crossing);
                        if (processed.at<uchar>(q) == 1 && cost < cost_map.at<float>(q))
                        {
                            removed.at<uchar>(q) = 1;
                        }
                        if (processed.at<uchar>(q) == 0)
                        {
                            cost_map.at<float>(q) = cost;
                            param->hit_map_x.at<int>(q)= p.x;
                            param->hit_map_y.at<int>(q) = p.y;
                            processed.at<uchar>(q) = 1;
                            Pix val;
                            val.cost = cost_map.at<float>(q);
                            val.next_point = q;
                            L.push(val);
                        }
                    }
                }
            }
        }
    }
}


static void onMouse(int event, int x, int y, int , void* userdata)
{
    Parameters* param = reinterpret_cast<Parameters*>(userdata);
    Point &end = param->end;
    std::vector<std::vector<Point> > &contours = param->contours;
    std::vector<Point> &tmp_contour = param->tmp_contour;
    Mat &img_render = param->img_render;
    Mat &img_pre_render = param->img_pre_render;

    if (event == EVENT_LBUTTONDOWN)
    {
        end = Point(x, y);
        if (!contours.back().empty())
        {
            for (int i = static_cast<int>(tmp_contour.size()) - 1; i >= 0; i--)
            {
                contours.back().push_back(tmp_contour[i]);
            }
            tmp_contour.clear();
        }
        else
        {
            contours.back().push_back(end);
        }
        find_min_path(end, param);

        img_render.copyTo(img_pre_render);
        imshow("lasso", img_render);
    }
    else if (event == EVENT_RBUTTONDOWN)
    {
        img_pre_render.copyTo(img_render);
        drawContours(img_pre_render, contours, static_cast<int>(contours.size()) - 1, Scalar(0,255,0), FILLED);
        addWeighted(img_pre_render, 0.3, img_render, 0.7, 0, img_render);
        contours.resize(contours.size() + 1);
        imshow("lasso", img_render);
    }
    else if (event == EVENT_MOUSEMOVE && !contours.back().empty())
    {
        tmp_contour.clear();
        img_pre_render.copyTo(img_render);
        Point val_point = Point(x, y);
        while (val_point != end)
        {
            tmp_contour.push_back(val_point);
            Point cur = Point(param->hit_map_x.at<int>(val_point), param->hit_map_y.at<int>(val_point));
            line(img_render, val_point, cur, Scalar(255, 0, 0), 2);
            val_point = cur;
        }
        imshow("lasso", img_render);
    }
}

const char* keys =
{
    "{help h | |}"
    "{@image | fruits.jpg| Path to image to process}"
};


int main( int argc, const char** argv )
{
    Parameters param;
    const int EDGE_THRESHOLD_LOW = 50;
    const int EDGE_THRESHOLD_HIGH = 100;
    CommandLineParser parser(argc, argv, keys);
    parser.about("\nThis program demonstrates implementation of 'Intelligent Scissors' algorithm designed\n"
                 "by Eric N. Mortensen and William A. Barrett, and described in article\n"
                 "'Intelligent Scissors for Image Composition':\n"
                 "http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.3811&rep=rep1&type=pdf\n"
                 "To start drawing a new contour select a pixel, click LEFT mouse button.\n"
                 "To fix a path click LEFT mouse button again.\n"
                 "To finish drawing a contour click RIGHT mouse button.\n");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 1;
    }
    std::vector<std::vector<Point> > c(1);
    param.contours = c;
    std::string filename = parser.get<std::string>(0);

    Mat grayscale, img_canny;
    param.img = imread(samples::findFile(filename));
    param.hit_map_x.create(param.img.rows, param.img.cols, CV_32SC1);
    param.hit_map_y.create(param.img.rows, param.img.cols, CV_32SC1);

    cvtColor(param.img, grayscale, COLOR_BGR2GRAY);
    Canny(grayscale, img_canny, EDGE_THRESHOLD_LOW, EDGE_THRESHOLD_HIGH);

    threshold(img_canny, param.zero_crossing, 254, 1, THRESH_BINARY_INV);
    Sobel(grayscale, param.Ix, CV_32FC1, 1, 0, 1);
    Sobel(grayscale, param.Iy, CV_32FC1, 0, 1, 1);
    param.Ix.convertTo(param.Ix, CV_32F, 1.0/255);
    param.Iy.convertTo(param.Iy, CV_32F, 1.0/255);

    // Compute gradients magnitude.
    double max_val = 0.0;
    magnitude(param.Iy, param.Ix, param.gradient_magnitude);
    minMaxLoc(param.gradient_magnitude, 0, &max_val);
    param.gradient_magnitude.convertTo(param.gradient_magnitude, CV_32F, -1/max_val, 1.0);

    param.img.copyTo(param.img_pre_render);
    param.img.copyTo(param.img_render);

    namedWindow("lasso");
    setMouseCallback("lasso", onMouse, &param);
    imshow("lasso", param.img);
    waitKey(0);
}
