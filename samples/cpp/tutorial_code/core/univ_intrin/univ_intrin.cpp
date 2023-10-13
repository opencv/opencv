#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/simd_intrinsics.hpp>

using namespace std;
using namespace cv;

const int N = 100005, K = 2000;

namespace
{

void conv_seq(Mat src, Mat &dst, Mat kernel)
{
    int rows = src.rows, cols = src.cols;
    dst = Mat(rows, cols, CV_8UC1);

    int sz = kernel.rows / 2;
    copyMakeBorder(src, src, sz, sz, sz, sz, BORDER_REPLICATE);
    for (int i = 0; i < rows; i++)
    {
        uchar *dptr = dst.ptr<uchar>(i);
        for (int j = 0; j < cols; j++)
        {
            float value = 0;

            for (int k = -sz; k <= sz; k++)
            {
                // slightly faster results when we create a ptr due to more efficient memory access.
                uchar *sptr = src.ptr<uchar>(i + sz + k);
                for (int l = -sz; l <= sz; l++)
                {
                    value += kernel.ptr<float>(k + sz)[l + sz] * sptr[j + sz + l];
                }
            }
            dptr[j] = saturate_cast<uchar>(value);
        }
    }
}

//! [convolution-1D-scalar]
void conv1d(Mat src, Mat &dst, Mat kernel)
{

    //! [convolution-1D-border]
    int len = src.cols;
    dst = Mat(1, len, CV_8UC1);

    int sz = kernel.cols / 2;
    copyMakeBorder(src, src, 0, 0, sz, sz, BORDER_REPLICATE);
    //! [convolution-1D-border]

    //! [convolution-1D-scalar-main]
    for (int i = 0; i < len; i++)
    {
        double value = 0;
        for (int k = -sz; k <= sz; k++)
            value += src.ptr<uchar>(0)[i + k + sz] * kernel.ptr<float>(0)[k + sz];

        dst.ptr<uchar>(0)[i] = saturate_cast<uchar>(value);
    }
    //! [convolution-1D-scalar-main]
}
//! [convolution-1D-scalar]

//! [convolution-1D-vector]
void conv1dsimd(Mat src, Mat kernel, float *ans, int row = 0, int rowk = 0, int len = -1)
{
    if (len == -1)
        len = src.cols;

    //! [convolution-1D-convert]
    Mat src_32, kernel_32;

    const int alpha = 1;
    src.convertTo(src_32, CV_32FC1, alpha);

    int ksize = kernel.cols, sz = kernel.cols / 2;
    copyMakeBorder(src_32, src_32, 0, 0, sz, sz, BORDER_REPLICATE);
    //! [convolution-1D-convert]


    //! [convolution-1D-main]
    //! [convolution-1D-main-h1]
    int step = VTraits<v_float32x4>::vlanes();
    float *sptr = src_32.ptr<float>(row), *kptr = kernel.ptr<float>(rowk);
    for (int k = 0; k < ksize; k++)
    {
    //! [convolution-1D-main-h1]
    //! [convolution-1D-main-h2]
        v_float32 kernel_wide = vx_setall_f32(kptr[k]);
        int i;
        for (i = 0; i + step < len; i += step)
        {
            v_float32 window = vx_load(sptr + i + k);
            v_float32 sum = v_add(vx_load(ans + i), v_mul(kernel_wide, window));
            v_store(ans + i, sum);
        }
    //! [convolution-1D-main-h2]

    //! [convolution-1D-main-h3]
        for (; i < len; i++)
        {
            *(ans + i) += sptr[i + k]*kptr[k];
        }
    //! [convolution-1D-main-h3]
    }
    //! [convolution-1D-main]
}
//! [convolution-1D-vector]

//! [convolution-2D]
void convolute_simd(Mat src, Mat &dst, Mat kernel)
{
    //! [convolution-2D-init]
    int rows = src.rows, cols = src.cols;
    int ksize = kernel.rows, sz = ksize / 2;
    dst = Mat(rows, cols, CV_32FC1);

    copyMakeBorder(src, src, sz, sz, 0, 0, BORDER_REPLICATE);

    int step = VTraits<v_float32x4>::vlanes();
    //! [convolution-2D-init]

    //! [convolution-2D-main]
    for (int i = 0; i < rows; i++)
    {
        for (int k = 0; k < ksize; k++)
        {
            float ans[N] = {0};
            conv1dsimd(src, kernel, ans, i + k, k, cols);
            int j;
            for (j = 0; j + step < cols; j += step)
            {
                v_float32 sum = v_add(vx_load(&dst.ptr<float>(i)[j]), vx_load(&ans[j]));
                v_store(&dst.ptr<float>(i)[j], sum);
            }

            for (; j < cols; j++)
                dst.ptr<float>(i)[j] += ans[j];
        }
    }
    //! [convolution-2D-main]

    //! [convolution-2D-conv]
    const int alpha = 1;
    dst.convertTo(dst, CV_8UC1, alpha);
    //! [convolution-2D-conv]
}
//! [convolution-2D]

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

    //  1-D Convolution  //
    Mat vsrc(1, N, CV_8UC1), k(1, K, CV_32FC1), vdst;
    RNG rng(time(0));
    rng.RNG::fill(vsrc, RNG::UNIFORM, Scalar(0), Scalar(255));
    rng.RNG::fill(k, RNG::UNIFORM, Scalar(-50), Scalar(50));

    double t = (double)getTickCount();
    conv1d(vsrc, vdst, k);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Sequential 1-D convolution implementation: " << t << "s" << endl;

    t = (double)getTickCount();
    float ans[N] = {0};
    conv1dsimd(vsrc, k, ans);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Vectorized 1-D convolution implementation: " << t << "s" << endl;

    //  2-D Convolution  //
    help(argv[0]);

    const char *filepath = argc >= 2 ? argv[1] : "../../../../data/lena.jpg";

    Mat src, dst1, dst2, kernel;
    src = imread(filepath, IMREAD_GRAYSCALE);

    if (src.empty())
    {
        cerr << "Can't open [" << filepath << "]" << endl;
        return EXIT_FAILURE;
    }
    namedWindow("Input", 1);
    namedWindow("Output", 1);
    imshow("Input", src);

    kernel = (Mat_<float>(3, 3) << 1, 0, -1,
                                   2, 0, -2,
                                   1, 0, -1);

    t = (double)getTickCount();

    conv_seq(src, dst1, kernel);

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Sequential 2-D convolution implementation: " << t << "s" << endl;

    imshow("Output", dst1);
    waitKey(0);

    t = (double)getTickCount();

    convolute_simd(src, dst2, kernel);

    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Vectorized 2-D convolution implementation: " << t << "s" << endl
         << endl;

    imshow("Output", dst2);
    waitKey(0);

    return 0;
}