// 2008-05-14, Xavier Delacour <xavier.delacour@gmail.com>

#include "cxcoretest.h"

#include <algorithm>
#include <complex>
#include <vector>
#include <iostream>

typedef std::complex<double> complex_type;

struct pred_complex
{
    bool operator() (const complex_type& lhs, const complex_type& rhs) const
    {
        return fabs(lhs.real() - rhs.real()) > fabs(rhs.real())*FLT_EPSILON ? lhs.real() < rhs.real() : lhs.imag() < rhs.imag();
    }
};

struct pred_double
{
    bool operator() (const double& lhs, const double& rhs) const
    {
        return lhs < rhs;
    }
};

class CV_SolvePolyTest : public CvTest
{
public:
    CV_SolvePolyTest();
    ~CV_SolvePolyTest();
protected:
    virtual void run( int start_from );
};

CV_SolvePolyTest::CV_SolvePolyTest() : CvTest( "solve-poly", "cvSolvePoly" ) {}

CV_SolvePolyTest::~CV_SolvePolyTest() {}

void CV_SolvePolyTest::run( int )
{
    CvRNG rng = cvRNG();
    int fig = 100;
    double range = 50;
    double err_eps = 1e-4;

    for (int idx = 0, max_idx = 1000, progress = 0; idx < max_idx; ++idx)
    {
        progress = update_progress(progress, idx-1, max_idx, 0);
        int n = cvRandInt(&rng) % 13 + 1;
        std::vector<complex_type> r(n), ar(n), c(n + 1, 0);
        std::vector<double> a(n + 1), u(n * 2), ar1(n), ar2(n);

        int rr_odds = 3; // odds that we get a real root
        for (int j = 0; j < n;)
        {
            if (cvRandInt(&rng) % rr_odds == 0 || j == n - 1)
	            r[j++] = cvRandReal(&rng) * range;
            else
            {
	            r[j] = complex_type(cvRandReal(&rng) * range,
			    cvRandReal(&rng) * range + 1);
	            r[j + 1] = std::conj(r[j]);
	            j += 2;
            }
        }

        for (int j = 0, k = 1 << n, jj, kk; j < k; ++j)
        {
            int p = 0;
            complex_type v(1);
            for (jj = 0, kk = 1; jj < n && !(j & kk); ++jj, ++p, kk <<= 1)
                ;
            for (; jj < n; ++jj, kk <<= 1)
            {
	            if (j & kk)
	                v *= -r[jj];
	            else
	                ++p;
            }
            c[p] += v;
        }

        bool pass = false;
        double div = 0, s = 0;
        int cubic_case = idx & 1;
        for (int maxiter = 100; !pass && maxiter < 10000; maxiter *= 2, cubic_case = (cubic_case + 1) % 2)
        {
            for (int j = 0; j < n + 1; ++j)
	            a[j] = c[j].real();

            CvMat amat, umat;
            cvInitMatHeader(&amat, n + 1, 1, CV_64FC1, &a[0]);
            cvInitMatHeader(&umat, n, 1, CV_64FC2, &u[0]);
            cvSolvePoly(&amat, &umat, maxiter, fig);

            for (int j = 0; j < n; ++j)
	            ar[j] = complex_type(u[j * 2], u[j * 2 + 1]);

            std::sort(r.begin(), r.end(), pred_complex());
            std::sort(ar.begin(), ar.end(), pred_complex());

            pass = true;
            if( n == 3 )
            {
                ar2.resize(n);
                cv::Mat _umat2(3, 1, CV_64F, &ar2[0]), umat2 = _umat2;
                cvFlip(&amat, &amat, 0);
                int nr2;
                if( cubic_case == 0 )
                    nr2 = cv::solveCubic(cv::Mat(&amat),umat2);
                else
                    nr2 = cv::solveCubic(cv::Mat_<float>(cv::Mat(&amat)), umat2);
                cvFlip(&amat, &amat, 0);
                if(nr2 > 0)
                    std::sort(ar2.begin(), ar2.begin()+nr2, pred_double());
                ar2.resize(nr2);

                int nr1 = 0;
                for(int j = 0; j < n; j++)
                    if( fabs(r[j].imag()) < DBL_EPSILON )
                        ar1[nr1++] = r[j].real();

                pass = pass && nr1 == nr2;
                if( nr2 > 0 )
                {
                    div = s = 0;
                    for(int j = 0; j < nr1; j++)
                    {
                        s += fabs(ar1[j]);
                        div += fabs(ar1[j] - ar2[j]);
                    }
                    div /= s;
                    pass = pass && div < err_eps;
                }
            }

            div = s = 0;
            for (int j = 0; j < n; ++j)
            {
                s += fabs(r[j].real()) + fabs(r[j].imag());
                div += sqrt(pow(r[j].real() - ar[j].real(), 2) + pow(r[j].imag() - ar[j].imag(), 2));
            }
            div /= s;
            pass = pass && div < err_eps;
        }

        if (!pass)
        {
            ts->set_failed_test_info(CvTS::FAIL_INVALID_OUTPUT);
            ts->printf( CvTS::LOG, "too big diff = %g\n", div );

            for (size_t j=0;j<ar2.size();++j)
                ts->printf( CvTS::LOG, "ar2[%d]=%g\n", j, ar2[j]);
            ts->printf(CvTS::LOG, "\n");

            for (size_t j=0;j<r.size();++j)
	            ts->printf( CvTS::LOG, "r[%d]=(%g, %g)\n", j, r[j].real(), r[j].imag());
            ts->printf( CvTS::LOG, "\n" );
            for (size_t j=0;j<ar.size();++j)
	            ts->printf( CvTS::LOG, "ar[%d]=(%g, %g)\n", j, ar[j].real(), ar[j].imag());
            break;
        }
    }
}

CV_SolvePolyTest solve_poly_test;
