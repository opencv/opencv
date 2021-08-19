#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/simd_intrinsics.hpp>

using namespace std;
using namespace cv;

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

            dst.ptr<uchar>(0)[i] = value;
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

        //! [convolution-1D-kernel]
        v_float32 kernel_wide[ksize];
        int step = kernel_wide[0].nlanes;
        for (int i = 0; i < ksize; i++)
            kernel_wide[i] = vx_setall_f32(kernel.ptr<float>(rowk)[i]);

        // For example:

        // kernel = {1, 0, -1} ksize = 3
        // kernel_wide: [ksize x nlanes]
        //     | 1| 1|...| 1| 1|
        //     | 0| 0|...| 0| 0|
        //     |-1|-1|...|-1|-1|

        //! [convolution-1D-kernel]

        //! [convolution-1D-main]
        //! [convolution-1D-main-h1]
        float *sptr = src_32.ptr<float>(row);
        int i;
        for (i = 0; i + step < len; i += step)
        {
            //! [convolution-1D-main-h1]
            //! [convolution-1D-main-h2]
            v_float32 sum = vx_setzero_f32();
            for (int k = -sz; k <= sz; k++)
            {
                v_float32 window = vx_load(sptr + i + sz + k);
                sum += kernel_wide[k + sz] * window;
            }

            v_store(ans + i, sum);
        }

        // For example:
        // kernel: {k1, k2, k3}
        //             idx:   i
        // src:           ...|a1|a2|a3|a4|...

        // kernel_wide:      |k1|k1|k1|k1|
        //                 |k2|k2|k2|k2|
        //                 |k3|k3|k3|k3|

        // iter1:
        // sum =  |0|0|0|0| + |a0|a1|a2|a3| * |k1|k1|k1|k1|
        //     =  |a0 * k1|a1 * k1|a2 * k1|a3 * k1|
        // iter2:
        // sum =  sum + |a1|a2|a3|a4| * |k2|k2|k2|k2|
        //     =  |a0 * k1 + a1 * k2|a1 * k1 + a2 * k2|a2 * k1 + a3 * k2|a3 * k1 + a4 * k2|
        // iter3:
        // sum =  sum + |a2|a3|a4|a5| * |k3|k3|k3|k3|
        //     =  |a0*k1 + a1*k2 + a2*k3|a1*k1 + a2*k2 + a3*k3|a2*k1 + a3*k2 + a4*k3|a3*k1 + a4*k2 + a5*k3|

        // We can see that sum holds the results for convolution for 4 consecutive values, starting from i

        //! [convolution-1D-main-h2]

        //! [convolution-1D-main-h3]
        for (; i < len; i++)
        {
            float value = 0;
            for (int k = -sz; k <= sz; k++)
                value += src_32.at<float>(row, i + k + sz) * kernel.at<float>(rowk, k + sz);

            *(ans + i) = value;
        }
        //! [convolution-1D-main-h3]
        //! [convolution-1D-main]
    }
    //! [convolution-1D-vector]

    void convolute_simd(Mat src, Mat &dst, Mat kernel)
    {
        int rows = src.rows, cols = src.cols;
        int ksize = kernel.rows, sz = ksize / 2;
        dst = Mat(rows, cols, CV_32FC1);

        copyMakeBorder(src, src, sz, sz, 0, 0, BORDER_REPLICATE);

        int step = v_float32().nlanes;
        for (int i = 0; i < rows; i++)
        {
            float ans[ksize][cols];
            int j;
            for (j = -sz; j <= sz; j++)
                conv1dsimd(src, kernel, ans[j + sz], i + j + sz, j + sz, cols);

            for (j = 0; j + step < cols; j += step)
            {
                v_float32 sum = vx_setzero_f32();
                for (int k = 0; k < ksize; k++)
                {
                    sum += vx_load(&ans[k][j]);
                }

                v_store(&dst.ptr<float>(i)[j], sum);
            }

            for (; j < cols; j++)
            {
                float val = 0;
                for (int k = 0; k < ksize; k++)
                {
                    val += ans[k][j];
                }

                dst.ptr<float>(i)[j] = val;
            }
        }

        const int alpha = 1;
        dst.convertTo(dst, CV_8UC1, alpha);
    }

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
    const int N = 1e5, K = 2e3;
    Mat vsrc(1, N, CV_8UC1), k(1, K, CV_32FC1), vdst;
    RNG rng(time(0));
    rng.RNG::fill(vsrc, RNG::UNIFORM, Scalar(0), Scalar(255));
    rng.RNG::fill(k, RNG::UNIFORM, Scalar(-50), Scalar(50));

    double t = (double)getTickCount();
    conv1d(vsrc, vdst, k);
    t = ((double)getTickCount() - t) / getTickFrequency();
    cout << " Sequential 1-D convolution implementation: " << t << "s" << endl;

    t = (double)getTickCount();
    float ans[vsrc.cols] = {0};
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

    kernel = (Mat_<float>(3, 3) << 1, 2, 1,
              2, 3, 2,
              1, 2, 1);

    kernel /= 15;

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