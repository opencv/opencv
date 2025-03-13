#include <iostream> // Console I/O
#include <sstream>  // String to number conversion

#include <opencv2/core.hpp> // Basic OpenCV structures
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>   // Image processing methods for the CPU
#include <opencv2/imgcodecs.hpp> // Read images

// CUDA structures and methods
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

using namespace std;
using namespace cv;

double getPSNR(const Mat &I1, const Mat &I2); // CPU versions
Scalar getMSSIM(const Mat &I1, const Mat &I2);

double getPSNR_CUDA(const Mat &I1, const Mat &I2); // Basic CUDA versions
Scalar getMSSIM_CUDA(const Mat &I1, const Mat &I2);

//! [psnr]
struct BufferPSNR // Optimized CUDA versions
{                 // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
    cuda::GpuMat gI1, gI2, gs, t1, t2;

    cuda::GpuMat buf;
};
//! [psnr]
double getPSNR_CUDA_optimized(const Mat &I1, const Mat &I2, BufferPSNR &b);

//! [ssim]
struct BufferMSSIM // Optimized CUDA versions
{                  // Data allocations are very expensive on CUDA. Use a buffer to solve: allocate once reuse later.
    cuda::GpuMat gI1, gI2, gs, t1, t2;

    cuda::GpuMat I1_2, I2_2, I1_I2;
    vector<cuda::GpuMat> vI1, vI2;

    cuda::GpuMat mu1, mu2;
    cuda::GpuMat mu1_2, mu2_2, mu1_mu2;

    cuda::GpuMat sigma1_2, sigma2_2, sigma12;
    cuda::GpuMat t3;

    cuda::GpuMat ssim_map;

    cuda::GpuMat buf;
};
//! [ssim]
Scalar getMSSIM_CUDA_optimized(const Mat &i1, const Mat &i2, BufferMSSIM &b);

static void help()
{
    cout
        << "\n--------------------------------------------------------------------------" << endl
        << "This program shows how to port your CPU code to CUDA or write that from scratch." << endl
        << "You can see the performance improvement for the similarity check methods (PSNR and SSIM)." << endl
        << "Usage:" << endl
        << "./gpu-basics-similarity referenceImage comparedImage numberOfTimesToRunTest(like 10)." << endl
        << "--------------------------------------------------------------------------" << endl
        << endl;
}

int main(int, char *argv[])
{
    help();
    Mat I1 = imread(argv[1]); // Read the two images
    Mat I2 = imread(argv[2]);

    if (!I1.data || !I2.data) // Check for success
    {
        cout << "Couldn't read the image";
        return 0;
    }

    BufferPSNR bufferPSNR;
    BufferMSSIM bufferMSSIM;

    int TIMES = 10;
    stringstream sstr(argv[3]);
    sstr >> TIMES;
    double time, result = 0;

    //------------------------------- PSNR CPU ----------------------------------------------------
    time = (double)getTickCount();

    for (int i = 0; i < TIMES; ++i)
        result = getPSNR(I1, I2);

    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    time /= TIMES;

    cout << "Time of PSNR CPU (averaged for " << TIMES << " runs): " << time << " milliseconds."
         << " With result of: " << result << endl;

    //------------------------------- PSNR CUDA ----------------------------------------------------
    time = (double)getTickCount();

    for (int i = 0; i < TIMES; ++i)
        result = getPSNR_CUDA(I1, I2);

    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    time /= TIMES;

    cout << "Time of PSNR CUDA (averaged for " << TIMES << " runs): " << time << " milliseconds."
         << " With result of: " << result << endl;

    //------------------------------- PSNR CUDA Optimized--------------------------------------------
    time = (double)getTickCount(); // Initial call
    result = getPSNR_CUDA_optimized(I1, I2, bufferPSNR);
    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    cout << "Initial call CUDA optimized:              " << time << " milliseconds."
         << " With result of: " << result << endl;

    time = (double)getTickCount();
    for (int i = 0; i < TIMES; ++i)
        result = getPSNR_CUDA_optimized(I1, I2, bufferPSNR);

    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    time /= TIMES;

    cout << "Time of PSNR CUDA OPTIMIZED ( / " << TIMES << " runs): " << time
         << " milliseconds." << " With result of: " << result << endl
         << endl;

    //------------------------------- SSIM CPU -----------------------------------------------------
    Scalar x;
    time = (double)getTickCount();

    for (int i = 0; i < TIMES; ++i)
        x = getMSSIM(I1, I2);

    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    time /= TIMES;

    cout << "Time of MSSIM CPU (averaged for " << TIMES << " runs): " << time << " milliseconds."
         << " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl;

    //------------------------------- SSIM CUDA -----------------------------------------------------
    time = (double)getTickCount();

    for (int i = 0; i < TIMES; ++i)
        x = getMSSIM_CUDA(I1, I2);

    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    time /= TIMES;

    cout << "Time of MSSIM CUDA (averaged for " << TIMES << " runs): " << time << " milliseconds."
         << " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl;

    //------------------------------- SSIM CUDA Optimized--------------------------------------------
    time = (double)getTickCount();
    x = getMSSIM_CUDA_optimized(I1, I2, bufferMSSIM);
    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    cout << "Time of MSSIM CUDA Initial Call            " << time << " milliseconds."
         << " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl;

    time = (double)getTickCount();

    for (int i = 0; i < TIMES; ++i)
        x = getMSSIM_CUDA_optimized(I1, I2, bufferMSSIM);

    time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
    time /= TIMES;

    cout << "Time of MSSIM CUDA OPTIMIZED ( / " << TIMES << " runs): " << time << " milliseconds."
         << " With result of B" << x.val[0] << " G" << x.val[1] << " R" << x.val[2] << endl<< endl;
    return 0;
}

//! [getpsnr]
double getPSNR(const Mat &I1, const Mat &I2)
{
    Mat s1;
    absdiff(I1, I2, s1);      // |I1 - I2|
    s1.convertTo(s1, CV_32F); // cannot make a square on 8 bits
    s1 = s1.mul(s1);          // |I1 - I2|^2

    Scalar s = sum(s1); // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
//! [getpsnr]

//! [getpsnropt]
double getPSNR_CUDA_optimized(const Mat &I1, const Mat &I2, BufferPSNR &b)
{
    b.gI1.upload(I1);
    b.gI2.upload(I2);

    b.gI1.convertTo(b.t1, CV_32F);
    b.gI2.convertTo(b.t2, CV_32F);

    cuda::absdiff(b.t1.reshape(1), b.t2.reshape(1), b.gs);
    cuda::multiply(b.gs, b.gs, b.gs);

    double sse = cuda::sum(b.gs, b.buf)[0];

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
//! [getpsnropt]

//! [getpsnrcuda]
double getPSNR_CUDA(const Mat &I1, const Mat &I2)
{
    cuda::GpuMat gI1, gI2, gs, t1, t2;

    gI1.upload(I1);
    gI2.upload(I2);

    gI1.convertTo(t1, CV_32F);
    gI2.convertTo(t2, CV_32F);

    cuda::absdiff(t1.reshape(1), t2.reshape(1), gs);
    cuda::multiply(gs, gs, gs);

    Scalar s = cuda::sum(gs);
    double sse = s.val[0] + s.val[1] + s.val[2];

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)(gI1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
//! [getpsnrcuda]

//! [getssim]
Scalar getMSSIM(const Mat &i1, const Mat &i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d); // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);  // I2^2
    Mat I1_2 = I1.mul(I1);  // I1^2
    Mat I1_I2 = I1.mul(I2); // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2; // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2); // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2); // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map); // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim;
}
//! [getssim]

//! [getssimcuda]
Scalar getMSSIM_CUDA(const Mat &i1, const Mat &i2)
{
    const float C1 = 6.5025f, C2 = 58.5225f;
    /***************************** INITS **********************************/
    cuda::GpuMat gI1, gI2, gs1, tmp1, tmp2;

    gI1.upload(i1);
    gI2.upload(i2);

    gI1.convertTo(tmp1, CV_MAKE_TYPE(CV_32F, gI1.channels()));
    gI2.convertTo(tmp2, CV_MAKE_TYPE(CV_32F, gI2.channels()));

    vector<cuda::GpuMat> vI1, vI2;
    cuda::split(tmp1, vI1);
    cuda::split(tmp2, vI2);
    Scalar mssim;

    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(vI2[0].type(), -1, Size(11, 11), 1.5);

    for (int i = 0; i < gI1.channels(); ++i)
    {
        cuda::GpuMat I2_2, I1_2, I1_I2;

        cuda::multiply(vI2[i], vI2[i], I2_2);  // I2^2
        cuda::multiply(vI1[i], vI1[i], I1_2);  // I1^2
        cuda::multiply(vI1[i], vI2[i], I1_I2); // I1 * I2

        /*************************** END INITS **********************************/
        cuda::GpuMat mu1, mu2; // PRELIMINARY COMPUTING
        gauss->apply(vI1[i], mu1);
        gauss->apply(vI2[i], mu2);

        cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
        cuda::multiply(mu1, mu1, mu1_2);
        cuda::multiply(mu2, mu2, mu2_2);
        cuda::multiply(mu1, mu2, mu1_mu2);

        cuda::GpuMat sigma1_2, sigma2_2, sigma12;

        gauss->apply(I1_2, sigma1_2);
        cuda::subtract(sigma1_2, mu1_2, sigma1_2); // sigma1_2 -= mu1_2;

        gauss->apply(I2_2, sigma2_2);
        cuda::subtract(sigma2_2, mu2_2, sigma2_2); // sigma2_2 -= mu2_2;

        gauss->apply(I1_I2, sigma12);
        cuda::subtract(sigma12, mu1_mu2, sigma12); // sigma12 -= mu1_mu2;

        ///////////////////////////////// FORMULA ////////////////////////////////
        cuda::GpuMat t1, t2, t3;

        mu1_mu2.convertTo(t1, -1, 2, C1); // t1 = 2 * mu1_mu2 + C1;
        sigma12.convertTo(t2, -1, 2, C2); // t2 = 2 * sigma12 + C2;
        cuda::multiply(t1, t2, t3);       // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

        cuda::addWeighted(mu1_2, 1.0, mu2_2, 1.0, C1, t1);       // t1 = mu1_2 + mu2_2 + C1;
        cuda::addWeighted(sigma1_2, 1.0, sigma2_2, 1.0, C2, t2); // t2 = sigma1_2 + sigma2_2 + C2;
        cuda::multiply(t1, t2, t1);                              // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

        cuda::GpuMat ssim_map;
        cuda::divide(t3, t1, ssim_map); // ssim_map =  t3./t1;

        Scalar s = cuda::sum(ssim_map);
        mssim.val[i] = s.val[0] / (ssim_map.rows * ssim_map.cols);
    }
    return mssim;
}
//! [getssimcuda]

//! [getssimopt]
Scalar getMSSIM_CUDA_optimized(const Mat &i1, const Mat &i2, BufferMSSIM &b)
{
    const float C1 = 6.5025f, C2 = 58.5225f;
    /***************************** INITS **********************************/

    b.gI1.upload(i1);
    b.gI2.upload(i2);

    cuda::Stream stream;

    b.gI1.convertTo(b.t1, CV_32F, stream);
    b.gI2.convertTo(b.t2, CV_32F, stream);

    cuda::split(b.t1, b.vI1, stream);
    cuda::split(b.t2, b.vI2, stream);
    Scalar mssim;

    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(b.vI1[0].type(), -1, Size(11, 11), 1.5);

    for (int i = 0; i < b.gI1.channels(); ++i)
    {
        cuda::multiply(b.vI2[i], b.vI2[i], b.I2_2, 1, -1, stream);  // I2^2
        cuda::multiply(b.vI1[i], b.vI1[i], b.I1_2, 1, -1, stream);  // I1^2
        cuda::multiply(b.vI1[i], b.vI2[i], b.I1_I2, 1, -1, stream); // I1 * I2

        gauss->apply(b.vI1[i], b.mu1, stream);
        gauss->apply(b.vI2[i], b.mu2, stream);

        cuda::multiply(b.mu1, b.mu1, b.mu1_2, 1, -1, stream);
        cuda::multiply(b.mu2, b.mu2, b.mu2_2, 1, -1, stream);
        cuda::multiply(b.mu1, b.mu2, b.mu1_mu2, 1, -1, stream);

        gauss->apply(b.I1_2, b.sigma1_2, stream);
        cuda::subtract(b.sigma1_2, b.mu1_2, b.sigma1_2, cuda::GpuMat(), -1, stream);
        // b.sigma1_2 -= b.mu1_2;  - This would result in an extra data transfer operation

        gauss->apply(b.I2_2, b.sigma2_2, stream);
        cuda::subtract(b.sigma2_2, b.mu2_2, b.sigma2_2, cuda::GpuMat(), -1, stream);
        // b.sigma2_2 -= b.mu2_2;

        gauss->apply(b.I1_I2, b.sigma12, stream);
        cuda::subtract(b.sigma12, b.mu1_mu2, b.sigma12, cuda::GpuMat(), -1, stream);
        // b.sigma12 -= b.mu1_mu2;

        // here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
        cuda::multiply(b.mu1_mu2, 2, b.t1, 1, -1, stream); // b.t1 = 2 * b.mu1_mu2 + C1;
        cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);
        cuda::multiply(b.sigma12, 2, b.t2, 1, -1, stream); // b.t2 = 2 * b.sigma12 + C2;
        cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -12, stream);

        cuda::multiply(b.t1, b.t2, b.t3, 1, -1, stream); // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

        cuda::add(b.mu1_2, b.mu2_2, b.t1, cuda::GpuMat(), -1, stream);
        cuda::add(b.t1, C1, b.t1, cuda::GpuMat(), -1, stream);

        cuda::add(b.sigma1_2, b.sigma2_2, b.t2, cuda::GpuMat(), -1, stream);
        cuda::add(b.t2, C2, b.t2, cuda::GpuMat(), -1, stream);

        cuda::multiply(b.t1, b.t2, b.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        cuda::divide(b.t3, b.t1, b.ssim_map, 1, -1, stream); // ssim_map =  t3./t1;

        stream.waitForCompletion();

        Scalar s = cuda::sum(b.ssim_map, b.buf);
        mssim.val[i] = s.val[0] / (b.ssim_map.rows * b.ssim_map.cols);
    }
    return mssim;
}
//! [getssimopt]
