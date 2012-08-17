#include "perf_precomp.hpp"

using namespace std;
using namespace testing;

namespace {

DEF_PARAM_TEST_1(Image, string);

struct GreedyLabeling
{
    struct dot
    {
        int x;
        int y;

        static dot make(int i, int j)
        {
            dot d; d.x = i; d.y = j;
            return d;
        }
    };

    struct InInterval
    {
        InInterval(const int& _lo, const int& _hi) : lo(-_lo), hi(_hi) {};
        const int lo, hi;

        bool operator() (const unsigned char a, const unsigned char b) const
        {
            int d = a - b;
            return lo <= d && d <= hi;
        }
    };

    GreedyLabeling(cv::Mat img)
    : image(img), _labels(image.size(), CV_32SC1, cv::Scalar::all(-1)) {stack = new dot[image.cols * image.rows];}

    ~GreedyLabeling(){delete[] stack;}

    void operator() (cv::Mat labels) const
    {
        labels.setTo(cv::Scalar::all(-1));
        InInterval inInt(0, 2);
        int cc = -1;

        int* dist_labels = (int*)labels.data;
        int pitch = labels.step1();

        unsigned char* source = (unsigned char*)image.data;
        int width = image.cols;
        int height = image.rows;

        for (int j = 0; j < image.rows; ++j)
            for (int i = 0; i < image.cols; ++i)
            {
                if (dist_labels[j * pitch + i] != -1) continue;

                dot* top = stack;
                dot p = dot::make(i, j);
                cc++;

                dist_labels[j * pitch + i] = cc;

                while (top >= stack)
                {
                    int*  dl = &dist_labels[p.y * pitch + p.x];
                    unsigned char* sp = &source[p.y * image.step1() + p.x];

                    dl[0] = cc;

                    //right
                    if( p.x < (width - 1) && dl[ +1] == -1 && inInt(sp[0], sp[+1]))
                        *top++ = dot::make(p.x + 1, p.y);

                    //left
                    if( p.x > 0 && dl[-1] == -1 && inInt(sp[0], sp[-1]))
                        *top++ = dot::make(p.x - 1, p.y);

                    //bottom
                    if( p.y < (height - 1) && dl[+pitch] == -1 && inInt(sp[0], sp[+image.step1()]))
                        *top++ = dot::make(p.x, p.y + 1);

                    //top
                    if( p.y > 0 && dl[-pitch] == -1 && inInt(sp[0], sp[-image.step1()]))
                        *top++ = dot::make(p.x, p.y - 1);

                    p = *--top;
                }
            }
    }

    cv::Mat image;
    cv::Mat _labels;
    dot* stack;
};

PERF_TEST_P(Image, Labeling_ConnectedComponents, Values<string>("gpu/labeling/aloe-disp.png"))
{
    declare.time(1.0);

    cv::Mat image = readImage(GetParam(), cv::IMREAD_GRAYSCALE);

    if (runOnGpu)
    {
        cv::gpu::GpuMat mask;
        mask.create(image.rows, image.cols, CV_8UC1);

        cv::gpu::GpuMat components;
        components.create(image.rows, image.cols, CV_32SC1);

        cv::gpu::connectivityMask(cv::gpu::GpuMat(image), mask, cv::Scalar::all(0), cv::Scalar::all(2));

        ASSERT_NO_THROW(cv::gpu::labelComponents(mask, components));

        TEST_CYCLE()
        {
            cv::gpu::labelComponents(mask, components);
        }
    }
    else
    {
        GreedyLabeling host(image);

        host(host._labels);

        declare.time(1.0);

        TEST_CYCLE()
        {
            host(host._labels);
        }
    }
}

} // namespace
