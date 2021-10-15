#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

namespace
{
//! [convolution-sequential]
void conv_seq(Mat src, Mat &dst, Mat kernel)
{
    //![convolution-make-borders]
    int rows = src.rows, cols = src.cols;
    dst = Mat(rows, cols, src.type());

    // Taking care of edge values
    // Make border = kernel.rows / 2;

    int sz = kernel.rows / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);
    //![convolution-make-borders]

    //! [convolution-kernel-loop]
    for (int i = 0; i < rows; i++)
    {
        uchar *dptr = dst.ptr(i);
        for (int j = 0; j < cols; j++)
        {
            double value = 0;

            for (int k = -sz; k <= sz; k++)
            {
                // slightly faster results when we create a ptr due to more efficient memory access.
                uchar *sptr = src.ptr(i + sz + k);
                for (int l = -sz; l <= sz; l++)
                {
                    value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                }
            }
            dptr[j] = saturate_cast<uchar>(value);
        }
    }
    //! [convolution-kernel-loop]
}
//! [convolution-sequential]

#ifdef CV_CXX11
void conv_parallel(Mat src, Mat &dst, Mat kernel)
{
    int rows = src.rows, cols = src.cols;

    dst = Mat(rows, cols, CV_8UC1, Scalar(0));

    // Taking care of edge values
    // Make border = kernel.rows / 2;

    int sz = kernel.rows / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);

    //! [convolution-parallel-cxx11]
    parallel_for_(Range(0, rows * cols), [&](const Range &range)
                    {
                        for (int r = range.start; r < range.end; r++)
                        {
                            int i = r / cols, j = r % cols;

                            double value = 0;
                            for (int k = -sz; k <= sz; k++)
                            {
                                uchar *sptr = src.ptr(i + sz + k);
                                for (int l = -sz; l <= sz; l++)
                                {
                                    value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                                }
                            }
                            dst.ptr(i)[j] = saturate_cast<uchar>(value);
                        }
                    });
    //! [convolution-parallel-cxx11]
}

void conv_parallel_row_split(Mat src, Mat &dst, Mat kernel)
{
    int rows = src.rows, cols = src.cols;

    dst = Mat(rows, cols, CV_8UC1, Scalar(0));

    // Taking care of edge values
    // Make border = kernel.rows / 2;

    int sz = kernel.rows / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);

    //! [convolution-parallel-cxx11-row-split]
    parallel_for_(Range(0, rows), [&](const Range &range)
                    {
                        for (int i = range.start; i < range.end; i++)
                        {

                            uchar *dptr = dst.ptr(i);
                            for (int j = 0; j < cols; j++)
                            {
                                double value = 0;
                                for (int k = -sz; k <= sz; k++)
                                {
                                    uchar *sptr = src.ptr(i + sz + k);
                                    for (int l = -sz; l <= sz; l++)
                                    {
                                        value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                                    }
                                }
                                dptr[j] = saturate_cast<uchar>(value);
                            }
                        }
                    });
    //! [convolution-parallel-cxx11-row-split]
}
#else

//! [convolution-parallel]
class parallelConvolution : public ParallelLoopBody
{
private:
    Mat m_src, &m_dst;
    Mat m_kernel;
    int sz;

public:
    parallelConvolution(Mat src, Mat &dst, Mat kernel)
        : m_src(src), m_dst(dst), m_kernel(kernel)
    {
        sz = kernel.rows / 2;
    }

    //! [overload-full]
    virtual void operator()(const Range &range) const CV_OVERRIDE
    {
        for (int r = range.start; r < range.end; r++)
        {
            int i = r / m_src.cols, j = r % m_src.cols;

            double value = 0;
            for (int k = -sz; k <= sz; k++)
            {
                uchar *sptr = m_src.ptr(i + sz + k);
                for (int l = -sz; l <= sz; l++)
                {
                    value += m_kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                }
            }
            m_dst.ptr(i)[j] = saturate_cast<uchar>(value);
        }
    }
    //! [overload-full]
};
//! [convolution-parallel]

void conv_parallel(Mat src, Mat &dst, Mat kernel)
{
    int rows = src.rows, cols = src.cols;

    dst = Mat(rows, cols, CV_8UC1, Scalar(0));

    // Taking care of edge values
    // Make border = kernel.rows / 2;

    int sz = kernel.rows / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);

    //! [convolution-parallel-function]
    parallelConvolution obj(src, dst, kernel);
    parallel_for_(Range(0, rows * cols), obj);
    //! [convolution-parallel-function]
}

//! [conv-parallel-row-split]
class parallelConvolutionRowSplit : public ParallelLoopBody
{
private:
    Mat m_src, &m_dst;
    Mat m_kernel;
    int sz;

public:
    parallelConvolutionRowSplit(Mat src, Mat &dst, Mat kernel)
        : m_src(src), m_dst(dst), m_kernel(kernel)
    {
        sz = kernel.rows / 2;
    }

    //! [overload-row-split]
    virtual void operator()(const Range &range) const CV_OVERRIDE
    {
        for (int i = range.start; i < range.end; i++)
        {

            uchar *dptr = dst.ptr(i);
            for (int j = 0; j < cols; j++)
            {
                double value = 0;
                for (int k = -sz; k <= sz; k++)
                {
                    uchar *sptr = src.ptr(i + sz + k);
                    for (int l = -sz; l <= sz; l++)
                    {
                        value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                    }
                }
                dptr[j] = saturate_cast<uchar>(value);
            }
        }
    }
    //! [overload-row-split]
};
//! [conv-parallel-row-split]

void conv_parallel_row_split(Mat src, Mat &dst, Mat kernel)
{
    int rows = src.rows, cols = src.cols;

    dst = Mat(rows, cols, CV_8UC1, Scalar(0));

    // Taking care of edge values
    // Make border = kernel.rows / 2;

    int sz = kernel.rows / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);

    //! [convolution-parallel-function-row]
    parallelConvolutionRowSplit obj(src, dst, kernel);
    parallel_for_(Range(0, rows), obj);
    //! [convolution-parallel-function-row]
}

#endif

static void help(char *progName)
{
    cout << endl
            << " This program shows how to use the OpenCV parallel_for_ function and \n"
            << " compares the performance of the sequential and parallel implementations for a \n"
            << " convolution operation\n"
            << " Usage:\n "
            << progName << " [image_path -- default lena.jpg] " << endl
            << endl;
}
}

int main(int argc, char *argv[])
{

    help(argv[0]);
    const char *filepath = argc >= 2 ? argv[1] : "../../../../data/lena.jpg";

    Mat src, dst, kernel;
    src = imread(filepath, IMREAD_GRAYSCALE);

    if (src.empty())
    {
        cerr << "Can't open [" << filepath << "]" << endl;
        return EXIT_FAILURE;
    }
    namedWindow("Input", 1);
    namedWindow("Output1", 1);
    namedWindow("Output2", 1);
    namedWindow("Output3", 1);
    imshow("Input", src);

    kernel = (Mat_<double>(3, 3) << 1, 0, -1,
              1, 0, -1,
              1, 0, -1);

    /*
    Uncomment the kernels you want to use or write your own kernels to test out
    performance.
    */

    /*
    kernel = (Mat_<double>(5, 5) <<   1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1,
                                      1, 1, 1, 1, 1);
    kernel /= 100;
    */

    /*
    kernel = (Mat_<double>(3, 3) <<  1,  1,  1,
                                     0,  0,  0,
                                    -1, -1, -1);

    */

    double t = (double)getTickCount();

    conv_seq(src, dst, kernel);

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Sequential implementation: " << t << "s" << endl;

    imshow("Output1", dst);
    waitKey(0);

    t = (double)getTickCount();

    conv_parallel(src, dst, kernel);

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Parallel Implementation: " << t << "s" << endl;

    imshow("Output2", dst);
    waitKey(0);

    t = (double)getTickCount();

    conv_parallel_row_split(src, dst, kernel);

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Parallel Implementation(Row Split): " << t << "s" << endl
         << endl;

    imshow("Output3", dst);
    waitKey(0);

    // imwrite("src.png", src);
    // imwrite("dst.png", dst);

    return 0;
}