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
    double value;

    bool operator > (const Pix &b) const
    {
        return value > b.value;
    }
};

double local_cost(const Point& p, const Point& q, bool diag)
{
    double  fG = 0.0;
    fG = gradient_magnitude.at<double>(q.y, q.x);
    double dp;
    double dq;

    if ((Iy.at<double>(p) * (q.x - p.x) - Ix.at<double>(p) * (q.y - p.y)) >= 0)
    {
        dp = Iy.at<double>(p) * (q.x - p.x) - Ix.at<double>(p) * (q.y - p.y);
        dq = Iy.at<double>(q) * (q.x - p.x) - Ix.at<double>(q) * (q.y - p.y);
    }
    else
    {
        dp = Iy.at<double>(p) * (p.x - q.x) + (-Ix.at<double>(p)) * (p.y - q.y);
        dq = Iy.at<double>(q) * (p.x - q.x) + (-Ix.at<double>(q)) * (p.y - q.y);
    }
    if (!diag)
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



void find_min_path(const Point& start)
{
    Mat processed;
    Mat removed;
    Mat expand;
    Mat cost_map;
    Pix begin;
    cost_map.create(img.rows, img.cols, CV_64FC1);
    expand.create(img.rows, img.cols, CV_8UC1);
    processed.create(img.rows, img.cols, CV_8UC1);
    removed.create(img.rows, img.cols, CV_8UC1);
    expand.setTo(0);
    processed.setTo(0);
    cost_map.setTo(INT_MAX);
    cost_map.at<double>(start) = 0;
    processed.at<uchar>(start) = 1;
    std :: priority_queue < Pix, std :: vector<Pix>, std:: greater<Pix> > L;
    begin.value=0;
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
                 if ((tx >= 0 && tx < img.cols && ty >= 0 && ty < img.rows && expand.at<uchar>(ty, tx) == 0) && ((i!=0)||(j!=0)))
                 { 
                    Point q = Point(tx, ty);
                    double tmp = cost_map.at<double>(p) + local_cost(p, q, ((p.x == q.x) || (p.y == q.y)));
                    if (processed.at<uchar>(q) == 1 && tmp < cost_map.at<double>(q))
                    {
                       removed.at<uchar>(q) = 1;
                    }
                    if (processed.at<uchar>(q) == 0)
                    {
                       cost_map.at<double>(q) = tmp;
                       hit_map_x.at<int>(q)= p.x;
                       hit_map_y.at<int>(q) = p.y;
                       processed.at<uchar>(q) = 1;
                       Pix val;
                       val.value = cost_map.at<double>(q);
                       val.next_point = q;
                       L.push(val);
                    }
                 }
              }
           }
        }
    }
}

void onMouse(int event, int x, int y, int flags, void *param)
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
    "{help h||}{@image |fruits.jpg|input image name}"
};


int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    std::string filename = parser.get<std::string>(0);

    Mat grayscale, img_canny;
    img = imread(samples::findFile(filename));
    hit_map_x.create(img.rows, img.cols, CV_32SC1);
    hit_map_y.create(img.rows, img.cols, CV_32SC1);

    cvtColor(img, grayscale, COLOR_BGR2GRAY);
    Canny(grayscale, img_canny, 50, 100);

    threshold(img_canny, zero_crossing, 254, 1, THRESH_BINARY_INV);
    Sobel(grayscale, Ix, CV_64FC1, 1, 0, 1);
    Sobel(grayscale, Iy, CV_64FC1, 0, 1, 1);
    Ix = Ix / 255.0;
    Iy = Iy / 255.0;

    // Compute gradients magnitude.
    double max_val = 0.0;
    magnitude(Iy, Ix, gradient_magnitude); 
    minMaxLoc(gradient_magnitude, 0, &max_val);
    gradient_magnitude = 1.0 - gradient_magnitude / max_val;

    img.copyTo(img_pre_render);
    img.copyTo(img_render);

    namedWindow("lasso");
    setMouseCallback("lasso", onMouse, 0);
    imshow("lasso", img);
    waitKey(0);
}