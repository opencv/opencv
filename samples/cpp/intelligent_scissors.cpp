#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Mat img, img_pre_render, img_render;
Point end;
std::vector<std::vector<Point> > contours(1);
std::vector<Point> tmp_contour;
Mat zero_crossing, gradient_magnitude, Ix, Iy, hit_map_x, hit_map_y;

struct Pix
{
    Point next_point;
    double cost;

    bool operator > (const Pix &b) const
    {
        return cost > b.cost;
    }
};

static float local_cost(const Point& p, const Point& q)
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
        dp /= sqrt(2);
        dq /= sqrt(2);
    }
    else
    {
        fG /= sqrt(2);
    }
    return  0.43 * zero_crossing.at<uchar>(q) + 0.43 * (acos(dp)
            + acos(dq)) / M_PI + 0.14 * fG;
}

static void find_min_path(const Point& start)
{
    Pix begin;
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
                        float cost = cost_map.at<float>(p) + local_cost(p, q);
                        if (processed.at<uchar>(q) == 1 && cost < cost_map.at<float>(q))
                        {
                            removed.at<uchar>(q) = 1;
                        }
                        if (processed.at<uchar>(q) == 0)
                        {
                            cost_map.at<float>(q) = cost;
                            hit_map_x.at<int>(q)= p.x;
                            hit_map_y.at<int>(q) = p.y;
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

static void onMouse(int event, int x, int y, int , void *)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        end = Point(x, y);
        if (!contours.back().empty())
        {
            for(int i = tmp_contour.size() - 1; i >= 0; i--)
            {
                contours.back().push_back(tmp_contour[i]);
            }
            tmp_contour.clear();
        }
        else
        {
            contours.back().push_back(end);
        }
        find_min_path(end);

        img_render.copyTo(img_pre_render);
        imshow("lasso", img_render);
    }
    else if (event == EVENT_RBUTTONDOWN)
    {
        img_pre_render.copyTo(img_render);
        drawContours(img_pre_render, contours, contours.size() - 1, Scalar(0,255,0), FILLED);
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
            Point cur = Point(hit_map_x.at<int>(val_point), hit_map_y.at<int>(val_point));
            line(img_render, val_point, cur, Scalar(255, 0, 0), 2);
            val_point = cur;
        }
        imshow("lasso", img_render);
    }
}

const char* keys =
{
    "{help h | |}{@image|fruits.jpg|}"
};

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("\nThis program demonstrates implementation of 'intelligent scissors' algorithm\n"
                 "To start the algorithm select a pixel, press lbm and move a mouse to create a path.\n"
                 "To stop the algorithm click rbm\n");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 1;
    }

    std::string filename = parser.get<std::string>(0);

    Mat grayscale, img_canny;
    img = imread(samples::findFile(filename));
    hit_map_x.create(img.rows, img.cols, CV_32SC1);
    hit_map_y.create(img.rows, img.cols, CV_32SC1);

    cvtColor(img, grayscale, COLOR_BGR2GRAY);
    Canny(grayscale, img_canny, 50, 100);

    threshold(img_canny, zero_crossing, 254, 1, THRESH_BINARY_INV);
    Sobel(grayscale, Ix, CV_32FC1, 1, 0, 1);
    Sobel(grayscale, Iy, CV_32FC1, 0, 1, 1);
    Ix.convertTo(Ix, CV_32F, 1.0/255);
    Iy.convertTo(Iy, CV_32F, 1.0/255);

    // Compute gradients magnitude.
    double max_val = 0.0;
    magnitude(Iy, Ix, gradient_magnitude);
    minMaxLoc(gradient_magnitude, 0, &max_val);
    gradient_magnitude.convertTo(gradient_magnitude, CV_32F, -1/max_val, 1.0);

    img.copyTo(img_pre_render);
    img.copyTo(img_render);

    namedWindow("lasso");
    setMouseCallback("lasso", onMouse, 0);
    imshow("lasso", img);
    waitKey(0);
}