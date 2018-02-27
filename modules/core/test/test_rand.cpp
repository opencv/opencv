// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

class Core_RandTest : public cvtest::BaseTest
{
public:
    Core_RandTest();
protected:
    void run(int);
    bool check_pdf(const Mat& hist, double scale, int dist_type,
                   double& refval, double& realval);
};


Core_RandTest::Core_RandTest()
{
}

static double chi2_p95(int n)
{
    static float chi2_tab95[] = {
        3.841f, 5.991f, 7.815f, 9.488f, 11.07f, 12.59f, 14.07f, 15.51f,
        16.92f, 18.31f, 19.68f, 21.03f, 21.03f, 22.36f, 23.69f, 25.00f,
        26.30f, 27.59f, 28.87f, 30.14f, 31.41f, 32.67f, 33.92f, 35.17f,
        36.42f, 37.65f, 38.89f, 40.11f, 41.34f, 42.56f, 43.77f };
    static const double xp = 1.64;
    CV_Assert(n >= 1);

    if( n <= 30 )
        return chi2_tab95[n-1];
    return n + sqrt((double)2*n)*xp + 0.6666666666666*(xp*xp - 1);
}

bool Core_RandTest::check_pdf(const Mat& hist, double scale,
                            int dist_type, double& refval, double& realval)
{
    Mat hist0(hist.size(), CV_32F);
    const int* H = hist.ptr<int>();
    float* H0 = hist0.ptr<float>();
    int i, hsz = hist.cols;

    double sum = 0;
    for( i = 0; i < hsz; i++ )
        sum += H[i];
    CV_Assert( fabs(1./sum - scale) < FLT_EPSILON );

    if( dist_type == CV_RAND_UNI )
    {
        float scale0 = (float)(1./hsz);
        for( i = 0; i < hsz; i++ )
            H0[i] = scale0;
    }
    else
    {
        double sum2 = 0, r = (hsz-1.)/2;
        double alpha = 2*sqrt(2.)/r, beta = -alpha*r;
        for( i = 0; i < hsz; i++ )
        {
            double x = i*alpha + beta;
            H0[i] = (float)exp(-x*x);
            sum2 += H0[i];
        }
        sum2 = 1./sum2;
        for( i = 0; i < hsz; i++ )
            H0[i] = (float)(H0[i]*sum2);
    }

    double chi2 = 0;
    for( i = 0; i < hsz; i++ )
    {
        double a = H0[i];
        double b = H[i]*scale;
        if( a > DBL_EPSILON )
            chi2 += (a - b)*(a - b)/(a + b);
    }
    realval = chi2;

    double chi2_pval = chi2_p95(hsz - 1 - (dist_type == CV_RAND_NORMAL ? 2 : 0));
    refval = chi2_pval*0.01;
    return realval <= refval;
}

void Core_RandTest::run( int )
{
    static int _ranges[][2] =
    {{ 0, 256 }, { -128, 128 }, { 0, 65536 }, { -32768, 32768 },
        { -1000000, 1000000 }, { -1000, 1000 }, { -1000, 1000 }};

    const int MAX_SDIM = 10;
    const int N = 2000000;
    const int maxSlice = 1000;
    const int MAX_HIST_SIZE = 1000;
    int progress = 0;

    RNG& rng = ts->get_rng();
    RNG tested_rng = theRNG();
    test_case_count = 200;

    for( int idx = 0; idx < test_case_count; idx++ )
    {
        progress = update_progress( progress, idx, test_case_count, 0 );
        ts->update_context( this, idx, false );

        int depth = cvtest::randInt(rng) % (CV_64F+1);
        int c, cn = (cvtest::randInt(rng) % 4) + 1;
        int type = CV_MAKETYPE(depth, cn);
        int dist_type = cvtest::randInt(rng) % (CV_RAND_NORMAL+1);
        int i, k, SZ = N/cn;
        Scalar A, B;

        double eps = 1.e-4;
        if (depth == CV_64F)
            eps = 1.e-7;

        bool do_sphere_test = dist_type == CV_RAND_UNI;
        Mat arr[2], hist[4];
        int W[] = {0,0,0,0};

        arr[0].create(1, SZ, type);
        arr[1].create(1, SZ, type);
        bool fast_algo = dist_type == CV_RAND_UNI && depth < CV_32F;

        for( c = 0; c < cn; c++ )
        {
            int a, b, hsz;
            if( dist_type == CV_RAND_UNI )
            {
                a = (int)(cvtest::randInt(rng) % (_ranges[depth][1] -
                                              _ranges[depth][0])) + _ranges[depth][0];
                do
                {
                    b = (int)(cvtest::randInt(rng) % (_ranges[depth][1] -
                                                  _ranges[depth][0])) + _ranges[depth][0];
                }
                while( abs(a-b) <= 1 );
                if( a > b )
                    std::swap(a, b);

                unsigned r = (unsigned)(b - a);
                fast_algo = fast_algo && r <= 256 && (r & (r-1)) == 0;
                hsz = min((unsigned)(b - a), (unsigned)MAX_HIST_SIZE);
                do_sphere_test = do_sphere_test && b - a >= 100;
            }
            else
            {
                int vrange = _ranges[depth][1] - _ranges[depth][0];
                int meanrange = vrange/16;
                int mindiv = MAX(vrange/20, 5);
                int maxdiv = MIN(vrange/8, 10000);

                a = cvtest::randInt(rng) % meanrange - meanrange/2 +
                (_ranges[depth][0] + _ranges[depth][1])/2;
                b = cvtest::randInt(rng) % (maxdiv - mindiv) + mindiv;
                hsz = min((unsigned)b*9, (unsigned)MAX_HIST_SIZE);
            }
            A[c] = a;
            B[c] = b;
            hist[c].create(1, hsz, CV_32S);
        }

        cv::RNG saved_rng = tested_rng;
        int maxk = fast_algo ? 0 : 1;
        for( k = 0; k <= maxk; k++ )
        {
            tested_rng = saved_rng;
            int sz = 0, dsz = 0, slice;
            for( slice = 0; slice < maxSlice; slice++, sz += dsz )
            {
                dsz = slice+1 < maxSlice ? (int)(cvtest::randInt(rng) % (SZ - sz + 1)) : SZ - sz;
                Mat aslice = arr[k].colRange(sz, sz + dsz);
                tested_rng.fill(aslice, dist_type, A, B);
            }
        }

        if( maxk >= 1 && cvtest::norm(arr[0], arr[1], NORM_INF) > eps)
        {
            ts->printf( cvtest::TS::LOG, "RNG output depends on the array lengths (some generated numbers get lost?)" );
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
            return;
        }

        for( c = 0; c < cn; c++ )
        {
            const uchar* data = arr[0].ptr();
            int* H = hist[c].ptr<int>();
            int HSZ = hist[c].cols;
            double minVal = dist_type == CV_RAND_UNI ? A[c] : A[c] - B[c]*4;
            double maxVal = dist_type == CV_RAND_UNI ? B[c] : A[c] + B[c]*4;
            double scale = HSZ/(maxVal - minVal);
            double delta = -minVal*scale;

            hist[c] = Scalar::all(0);

            for( i = c; i < SZ*cn; i += cn )
            {
                double val = depth == CV_8U ? ((const uchar*)data)[i] :
                depth == CV_8S ? ((const schar*)data)[i] :
                depth == CV_16U ? ((const ushort*)data)[i] :
                depth == CV_16S ? ((const short*)data)[i] :
                depth == CV_32S ? ((const int*)data)[i] :
                depth == CV_32F ? ((const float*)data)[i] :
                ((const double*)data)[i];
                int ival = cvFloor(val*scale + delta);
                if( (unsigned)ival < (unsigned)HSZ )
                {
                    H[ival]++;
                    W[c]++;
                }
                else if( dist_type == CV_RAND_UNI )
                {
                    if( (minVal <= val && val < maxVal) || (depth >= CV_32F && val == maxVal) )
                    {
                        H[ival < 0 ? 0 : HSZ-1]++;
                        W[c]++;
                    }
                    else
                    {
                        putchar('^');
                    }
                }
            }

            if( dist_type == CV_RAND_UNI && W[c] != SZ )
            {
                ts->printf( cvtest::TS::LOG, "Uniform RNG gave values out of the range [%g,%g) on channel %d/%d\n",
                           A[c], B[c], c, cn);
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            if( dist_type == CV_RAND_NORMAL && W[c] < SZ*.90)
            {
                ts->printf( cvtest::TS::LOG, "Normal RNG gave too many values out of the range (%g+4*%g,%g+4*%g) on channel %d/%d\n",
                           A[c], B[c], A[c], B[c], c, cn);
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
            double refval = 0, realval = 0;

            if( !check_pdf(hist[c], 1./W[c], dist_type, refval, realval) )
            {
                ts->printf( cvtest::TS::LOG, "RNG failed Chi-square test "
                           "(got %g vs probable maximum %g) on channel %d/%d\n",
                           realval, refval, c, cn);
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
        }

        // Monte-Carlo test. Compute volume of SDIM-dimensional sphere
        // inscribed in [-1,1]^SDIM cube.
        if( do_sphere_test )
        {
            int SDIM = cvtest::randInt(rng) % (MAX_SDIM-1) + 2;
            int N0 = (SZ*cn/SDIM), n = 0;
            double r2 = 0;
            const uchar* data = arr[0].ptr();
            double scale[4], delta[4];
            for( c = 0; c < cn; c++ )
            {
                scale[c] = 2./(B[c] - A[c]);
                delta[c] = -A[c]*scale[c] - 1;
            }

            for( i = k = c = 0; i <= SZ*cn - SDIM; i++, k++, c++ )
            {
                double val = depth == CV_8U ? ((const uchar*)data)[i] :
                depth == CV_8S ? ((const schar*)data)[i] :
                depth == CV_16U ? ((const ushort*)data)[i] :
                depth == CV_16S ? ((const short*)data)[i] :
                depth == CV_32S ? ((const int*)data)[i] :
                depth == CV_32F ? ((const float*)data)[i] : ((const double*)data)[i];
                c &= c < cn ? -1 : 0;
                val = val*scale[c] + delta[c];
                r2 += val*val;
                if( k == SDIM-1 )
                {
                    n += r2 <= 1;
                    r2 = 0;
                    k = -1;
                }
            }

            double V = ((double)n/N0)*(1 << SDIM);

            // the theoretically computed volume
            int sdim = SDIM % 2;
            double V0 = sdim + 1;
            for( sdim += 2; sdim <= SDIM; sdim += 2 )
                V0 *= 2*CV_PI/sdim;

            if( fabs(V - V0) > 0.3*fabs(V0) )
            {
                ts->printf( cvtest::TS::LOG, "RNG failed %d-dim sphere volume test (got %g instead of %g)\n",
                           SDIM, V, V0);
                ts->printf( cvtest::TS::LOG, "depth = %d, N0 = %d\n", depth, N0);
                ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
                return;
            }
        }
    }
}

TEST(Core_Rand, quality) { Core_RandTest test; test.safe_run(); }


class Core_RandRangeTest : public cvtest::BaseTest
{
public:
    Core_RandRangeTest() {}
    ~Core_RandRangeTest() {}
protected:
    void run(int)
    {
        Mat a(Size(1280, 720), CV_8U, Scalar(20));
        Mat af(Size(1280, 720), CV_32F, Scalar(20));
        theRNG().fill(a, RNG::UNIFORM, -DBL_MAX, DBL_MAX);
        theRNG().fill(af, RNG::UNIFORM, -DBL_MAX, DBL_MAX);
        int n0 = 0, n255 = 0, nx = 0;
        int nfmin = 0, nfmax = 0, nfx = 0;

        for( int i = 0; i < a.rows; i++ )
            for( int j = 0; j < a.cols; j++ )
            {
                int v = a.at<uchar>(i,j);
                double vf = af.at<float>(i,j);
                if( v == 0 ) n0++;
                else if( v == 255 ) n255++;
                else nx++;
                if( vf < FLT_MAX*-0.999f ) nfmin++;
                else if( vf > FLT_MAX*0.999f ) nfmax++;
                else nfx++;
            }
        CV_Assert( n0 > nx*2 && n255 > nx*2 );
        CV_Assert( nfmin > nfx*2 && nfmax > nfx*2 );
    }
};

TEST(Core_Rand, range) { Core_RandRangeTest test; test.safe_run(); }


TEST(Core_RNG_MT19937, regression)
{
    cv::RNG_MT19937 rng;
    int actual[61] = {0, };
    const size_t length = (sizeof(actual) / sizeof(actual[0]));
    for (int i = 0; i < 10000; ++i )
    {
        actual[(unsigned)(rng.next() ^ i) % length]++;
    }

    int expected[length] = {
        177, 158, 180, 177,  160, 179, 143, 162,
        177, 144, 170, 174,  165, 168, 168, 156,
        177, 157, 159, 169,  177, 182, 166, 154,
        144, 180, 168, 152,  170, 187, 160, 145,
        139, 164, 157, 179,  148, 183, 159, 160,
        196, 184, 149, 142,  162, 148, 163, 152,
        168, 173, 160, 181,  172, 181, 155, 153,
        158, 171, 138, 150,  150 };

    for (size_t i = 0; i < length; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]);
    }
}


TEST(Core_Rand, Regression_Stack_Corruption)
{
    int bufsz = 128; //enough for 14 doubles
    AutoBuffer<uchar> buffer(bufsz);
    size_t offset = 0;
    cv::Mat_<cv::Point2d> x(2, 3, (cv::Point2d*)(buffer+offset)); offset += x.total()*x.elemSize();
    double& param1 = *(double*)(buffer+offset); offset += sizeof(double);
    double& param2 = *(double*)(buffer+offset); offset += sizeof(double);
    param1 = -9; param2 = 2;

    cv::theRNG().fill(x, cv::RNG::NORMAL, param1, param2);

    ASSERT_EQ(param1, -9);
    ASSERT_EQ(param2,  2);
}


class RandRowFillParallelLoopBody : public cv::ParallelLoopBody
{
public:
    RandRowFillParallelLoopBody(Mat& dst) : dst_(dst) {}
    ~RandRowFillParallelLoopBody() {}
    void operator()(const cv::Range& r) const
    {
        cv::RNG rng = cv::theRNG(); // copy state
        for (int y = r.start; y < r.end; y++)
        {
            cv::theRNG() = cv::RNG(rng.state + y); // seed is based on processed row
            cv::randu(dst_.row(y), Scalar(-100), Scalar(100));
        }
        // theRNG() state is changed here (but state collision has low probability, so we don't check this)
    }
protected:
    Mat& dst_;
};

TEST(Core_Rand, parallel_for_stable_results)
{
    cv::RNG rng = cv::theRNG(); // save rng state
    Mat dst1(1000, 100, CV_8SC1);
    parallel_for_(cv::Range(0, dst1.rows), RandRowFillParallelLoopBody(dst1));

    cv::theRNG() = rng; // restore rng state
    Mat dst2(1000, 100, CV_8SC1);
    parallel_for_(cv::Range(0, dst2.rows), RandRowFillParallelLoopBody(dst2));

    ASSERT_EQ(0, countNonZero(dst1 != dst2));
}

}} // namespace
