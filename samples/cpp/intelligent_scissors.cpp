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

struct Calculated_parameters
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
    return  0.43f * zero_crossing.at<uchar>(q) + 0.43f * (acosf(dp)
            + acosf(dq)) / static_cast<float>(CV_PI) + 0.14f * fG;
}

static void find_min_path(const Point& start, Calculated_parameters* tmp)
{
    Pix begin;
    Mat cost_map(tmp->img.size(), CV_32F, Scalar(FLT_MAX));
    Mat expand(tmp->img.size(), CV_8UC1, Scalar(0));
    Mat processed(tmp->img.size(), CV_8UC1, Scalar(0));
    Mat removed(tmp->img.size(), CV_8UC1, Scalar(0));
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
                    if (tx < 0 || tx >= tmp->img.cols || ty < 0 || ty >= tmp->img.rows)
                        continue;
                    if (expand.at<uchar>(ty, tx) == 0)
                    {
                        Point q = Point(tx, ty);
                        float cost = cost_map.at<float>(p) + local_cost(p, q, tmp->gradient_magnitude, tmp->Iy, tmp->Ix, tmp->zero_crossing);
                        if (processed.at<uchar>(q) == 1 && cost < cost_map.at<float>(q))
                        {
                            removed.at<uchar>(q) = 1;
                        }
                        if (processed.at<uchar>(q) == 0)
                        {
                            cost_map.at<float>(q) = cost;
                            tmp->hit_map_x.at<int>(q)= p.x;
                            tmp->hit_map_y.at<int>(q) = p.y;
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
    Calculated_parameters* tmp = reinterpret_cast<Calculated_parameters*>(userdata);
    if (event == EVENT_LBUTTONDOWN)
    {
        tmp->end = Point(x, y);
        if (!tmp->contours.back().empty())
        {
            for (int i = static_cast<int>(tmp->tmp_contour.size()) - 1; i >= 0; i--)
            {
                tmp->contours.back().push_back(tmp->tmp_contour[i]);
            }
            tmp->tmp_contour.clear();
        }
        else
        {
            tmp->contours.back().push_back(tmp->end);
        }
        find_min_path(tmp->end, tmp);

        tmp->img_render.copyTo(tmp->img_pre_render);
        imshow("lasso", tmp->img_render);
    }
    else if (event == EVENT_RBUTTONDOWN)
    {
        tmp->img_pre_render.copyTo(tmp->img_render);
        drawContours(tmp->img_pre_render, tmp->contours, static_cast<int>(tmp->contours.size()) - 1, Scalar(0,255,0), FILLED);
        addWeighted(tmp->img_pre_render, 0.3, tmp->img_render, 0.7, 0, tmp->img_render);
        tmp->contours.resize(tmp->contours.size() + 1);
        imshow("lasso", tmp->img_render);
    }
    else if (event == EVENT_MOUSEMOVE && !tmp->contours.back().empty())
    {
        tmp->tmp_contour.clear();
        tmp->img_pre_render.copyTo(tmp->img_render);
        Point val_point = Point(x, y);
        while (val_point != tmp->end)
        {
            tmp->tmp_contour.push_back(val_point);
            Point cur = Point(tmp->hit_map_x.at<int>(val_point), tmp->hit_map_y.at<int>(val_point));
            line(tmp->img_render, val_point, cur, Scalar(255, 0, 0), 2);
            val_point = cur;
        }
        imshow("lasso", tmp->img_render);
    }
}

const char* keys =
{
    "{help h | |}{@image|fruits.jpg|}"
};


int main( int argc, const char** argv )
{
    Calculated_parameters tmp;

    CommandLineParser parser(argc, argv, keys);
    parser.about("\nThis program demonstrates implementation of 'intelligent scissors' algorithm\n"
                 "To start drawing a new contour select a pixel, click LEFT mouse button.\n"
                 "To fix a path click LEFT mouse button again.\n"
                 "To finish drawing a contour click RIGHT mouse button.\n");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 1;
    }
    std::vector<std::vector<Point> > c(1);
    tmp.contours=c;
    std::string filename = parser.get<std::string>(0);

    Mat grayscale, img_canny;
    tmp.img = imread(samples::findFile(filename));
    tmp.hit_map_x.create(tmp.img.rows, tmp.img.cols, CV_32SC1);
    tmp.hit_map_y.create(tmp.img.rows, tmp.img.cols, CV_32SC1);

    cvtColor(tmp.img, grayscale, COLOR_BGR2GRAY);
    Canny(grayscale, img_canny, 50, 100);

    threshold(img_canny, tmp.zero_crossing, 254, 1, THRESH_BINARY_INV);
    Sobel(grayscale, tmp.Ix, CV_32FC1, 1, 0, 1);
    Sobel(grayscale, tmp.Iy, CV_32FC1, 0, 1, 1);
    tmp.Ix.convertTo(tmp.Ix, CV_32F, 1.0/255);
    tmp.Iy.convertTo(tmp.Iy, CV_32F, 1.0/255);

    // Compute gradients magnitude.
    double max_val = 0.0;
    magnitude(tmp.Iy, tmp.Ix, tmp.gradient_magnitude);
    minMaxLoc(tmp.gradient_magnitude, 0, &max_val);
    tmp.gradient_magnitude.convertTo(tmp.gradient_magnitude, CV_32F, -1/max_val, 1.0);

    tmp.img.copyTo(tmp.img_pre_render);
    tmp.img.copyTo(tmp.img_render);

    namedWindow("lasso");
    setMouseCallback("lasso", onMouse, &tmp);
    imshow("lasso", tmp.img);
    waitKey(0);
}
