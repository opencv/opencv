#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <queue>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

bool cont;
Mat img, img_pre_draw;
Point end;

Scalar colored_line(255,0,0);

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

void grad_mag()
{
    double max_val = 0.0;
    gradient_magnitude.create(img.rows, img.cols, CV_64FC1);
    magnitude(Iy, Ix, gradient_magnitude);
    minMaxLoc(gradient_magnitude, 0, &max_val);
    gradient_magnitude = 1.0 - gradient_magnitude / max_val;
}

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
        cont = true;
        find_min_path(end);
        img.copyTo(img_pre_draw);
        imshow("lasso", img);
    }
    else
     if (event == EVENT_RBUTTONDOWN)
       {
        cont = false;
        img_pre_draw.copyTo(img);
        imshow("lasso", img);
       }
    else
       if (event == EVENT_MOUSEMOVE && cont)
       {
          img_pre_draw.copyTo(img);
          Point val_point = Point(x, y);
          while (val_point != end)
          {
              Point  cur = Point(hit_map_x.at<int>(val_point), hit_map_y.at<int>(val_point));
              line(img, val_point, cur, colored_line, 2);
              val_point = cur;
          }
          imshow("lasso", img);
       }
}

const char* keys =
{
    "{help h||}{@image |fruits.jpg|input image name}"
};


int main( int argc, const char** argv )
{
    Mat grayscale;
    Mat img_canny;
    Mat test_sobel;
    CommandLineParser parser(argc, argv, keys);
    std ::  string filename = parser.get<std :: string>(0);
    img = imread(samples::findFile(filename), IMREAD_COLOR);
    namedWindow("lasso");
    hit_map_x.create(img.rows, img.cols, CV_32SC1);
    hit_map_y.create(img.rows, img.cols, CV_32SC1);
    cvtColor(img, grayscale, COLOR_BGR2GRAY);
    Canny(grayscale, img_canny, 50, 100);
    threshold(img_canny, zero_crossing, 254, 1, THRESH_BINARY_INV);
    Sobel(grayscale, Ix, CV_64FC1, 1, 0, 1);
    Sobel(grayscale, Iy, CV_64FC1, 0, 1, 1);
    Ix = Ix / 255.0;
    Iy = Iy / 255.0;
    grad_mag();
    img.copyTo(img_pre_draw);
    setMouseCallback("lasso", onMouse, 0);
    imshow("lasso", img);
    waitKey(0);
}