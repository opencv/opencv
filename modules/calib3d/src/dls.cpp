#include "precomp.hpp"
#include "dls.h"

#include <iostream>

#ifdef HAVE_EIGEN
#  if defined __GNUC__ && defined __APPLE__
#    pragma GCC diagnostic ignored "-Wshadow"
#  endif
#  if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable:4701)  // potentially uninitialized local variable
#    pragma warning(disable:4702)  // unreachable code
#    pragma warning(disable:4714)  // const marked as __forceinline not inlined
#  endif
#  include <Eigen/Core>
#  include <Eigen/Eigenvalues>
#  if defined(_MSC_VER)
#    pragma warning(pop)
#  endif
#  include "opencv2/core/eigen.hpp"
#endif

using namespace std;

dls::dls(const cv::Mat& opoints, const cv::Mat& ipoints)
{

    N =  std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));
    p = cv::Mat(3, N, CV_64F);
    z = cv::Mat(3, N, CV_64F);
    mn = cv::Mat::zeros(3, 1, CV_64F);

    cost__ = 9999;

    f1coeff.resize(21);
    f2coeff.resize(21);
    f3coeff.resize(21);

    if (opoints.depth() == ipoints.depth())
    {
        if (opoints.depth() == CV_32F)
            init_points<cv::Point3f, cv::Point2f>(opoints, ipoints);
        else
            init_points<cv::Point3d, cv::Point2d>(opoints, ipoints);
    }
    else if (opoints.depth() == CV_32F)
        init_points<cv::Point3f, cv::Point2d>(opoints, ipoints);
    else
        init_points<cv::Point3d, cv::Point2f>(opoints, ipoints);
}

dls::~dls()
{
  // TODO Auto-generated destructor stub
}

bool dls::compute_pose(cv::Mat& R, cv::Mat& t)
{

    std::vector<cv::Mat> R_;
    R_.push_back(rotx(CV_PI/2));
    R_.push_back(roty(CV_PI/2));
    R_.push_back(rotz(CV_PI/2));

    // version that calls dls 3 times, to avoid Cayley singularity
    for (int i = 0; i < 3; ++i)
    {
        // Make a random rotation
        cv::Mat pp = R_[i] * ( p - cv::repeat(mn, 1, p.cols) );

        // clear for new data
        C_est_.clear();
        t_est_.clear();
        cost_.clear();

        this->run_kernel(pp); // run dls_pnp()

        // find global minimum
        for (unsigned int j = 0; j < cost_.size(); ++j)
        {
            if( cost_[j] < cost__ )
            {
                t_est__ = t_est_[j] - C_est_[j] * R_[i] * mn;
                C_est__ = C_est_[j] * R_[i];
                cost__ = cost_[j];
            }
        }

    }

    if(C_est__.cols > 0 && C_est__.rows > 0)
    {
        C_est__.copyTo(R);
        t_est__.copyTo(t);
        return true;
    }

    return false;
}

void dls::run_kernel(const cv::Mat& pp)
{
    cv::Mat Mtilde(27, 27, CV_64F);
    cv::Mat D = cv::Mat::zeros(9, 9, CV_64F);
    build_coeff_matrix(pp, Mtilde, D);

    cv::Mat eigenval_r, eigenval_i, eigenvec_r, eigenvec_i;
    compute_eigenvec(Mtilde, eigenval_r, eigenval_i, eigenvec_r, eigenvec_i);

    /*
     *  Now check the solutions
     */

    // extract the optimal solutions from the eigen decomposition of the
    // Multiplication matrix

    cv::Mat sols = cv::Mat::zeros(3, 27, CV_64F);
    std::vector<double> cost;
    int count = 0;
    for (int k = 0; k < 27; ++k)
    {
        //  V(:,k) = V(:,k)/V(1,k);
        cv::Mat V_kA = eigenvec_r.col(k); // 27x1
        cv::Mat V_kB = cv::Mat(1, 1, z.depth(), V_kA.at<double>(0)); // 1x1
        cv::Mat V_k; cv::solve(V_kB.t(), V_kA.t(), V_k); // A/B = B'\A'
        cv::Mat( V_k.t()).copyTo( eigenvec_r.col(k) );

        //if (imag(V(2,k)) == 0)
#ifdef HAVE_EIGEN
        const double epsilon = 1e-4;
        if( eigenval_i.at<double>(k,0) >= -epsilon && eigenval_i.at<double>(k,0) <= epsilon )
#endif
        {

            double stmp[3];
            stmp[0] = eigenvec_r.at<double>(9, k);
            stmp[1] = eigenvec_r.at<double>(3, k);
            stmp[2] = eigenvec_r.at<double>(1, k);

            cv::Mat H = Hessian(stmp);

            cv::Mat eigenvalues, eigenvectors;
            cv::eigen(H, eigenvalues, eigenvectors);

            if(positive_eigenvalues(&eigenvalues))
            {

                // sols(:,i) = stmp;
                cv::Mat stmp_mat(3, 1, CV_64F, &stmp);

                stmp_mat.copyTo( sols.col(count) );

                cv::Mat Cbar = cayley2rotbar(stmp_mat);
                cv::Mat Cbarvec = Cbar.reshape(1,1).t();

                // cost(i) = CbarVec' * D * CbarVec;
                cv::Mat cost_mat =  Cbarvec.t() * D * Cbarvec;
                cost.push_back( cost_mat.at<double>(0) );

                count++;
            }
        }
    }

    // extract solutions
    sols = sols.clone().colRange(0, count);

    std::vector<cv::Mat> C_est, t_est;
    for (int j = 0; j < sols.cols; ++j)
    {
        // recover the optimal orientation
        // C_est(:,:,j) = 1/(1 + sols(:,j)' * sols(:,j)) * cayley2rotbar(sols(:,j));

        cv::Mat sols_j = sols.col(j);
        double sols_mult = 1./(1.+cv::Mat( sols_j.t() * sols_j ).at<double>(0));
        cv::Mat C_est_j = cayley2rotbar(sols_j).mul(sols_mult);
        C_est.push_back( C_est_j );

        cv::Mat A2 = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat b2 = cv::Mat::zeros(3, 1, CV_64F);
        for (int i = 0; i < N; ++i)
        {
            cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat z_mul = z.col(i)*z.col(i).t();

            A2 += eye - z_mul;
            b2 += (z_mul - eye) * C_est_j * pp.col(i);
        }

        // recover the optimal translation
        cv::Mat X2; cv::solve(A2, b2, X2); // A\B
        t_est.push_back(X2);

    }

    // check that the points are infront of the center of perspectivity
    for (int k = 0; k < sols.cols; ++k)
    {
        cv::Mat cam_points = C_est[k] * pp + cv::repeat(t_est[k], 1, pp.cols);
        cv::Mat cam_points_k = cam_points.row(2);

        if(is_empty(&cam_points_k))
        {
            cv::Mat C_valid = C_est[k], t_valid = t_est[k];
            double cost_valid = cost[k];

            C_est_.push_back(C_valid);
            t_est_.push_back(t_valid);
            cost_.push_back(cost_valid);
        }
    }

}

void dls::build_coeff_matrix(const cv::Mat& pp, cv::Mat& Mtilde, cv::Mat& D)
{
    CV_Assert(!pp.empty() && N > 0);
    cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);

    // build coeff matrix
    // An intermediate matrix, the inverse of what is called "H" in the paper
    // (see eq. 25)

    cv::Mat H = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat A = cv::Mat::zeros(3, 9, CV_64F);
    cv::Mat pp_i(3, 1, CV_64F);

    cv::Mat z_i(3, 1, CV_64F);
    for (int i = 0; i < N; ++i)
    {
        z.col(i).copyTo(z_i);
        A += ( z_i*z_i.t() - eye ) * LeftMultVec(pp.col(i));
    }

    H = eye.mul(N) - z * z.t();

    // A\B
    cv::solve(H, A, A, cv::DECOMP_NORMAL);
    H.release();

    cv::Mat ppi_A(3, 1, CV_64F);
    for (int i = 0; i < N; ++i)
    {
        z.col(i).copyTo(z_i);
        ppi_A = LeftMultVec(pp.col(i)) + A;
        D += ppi_A.t() * ( eye - z_i*z_i.t() ) * ppi_A;
    }
    A.release();

    // fill the coefficients
    fill_coeff(&D);

    // generate random samples
    std::vector<double> u(5);
    cv::randn(u, 0, 200);

    cv::Mat M2 = cayley_LS_M(f1coeff, f2coeff, f3coeff, u);

    cv::Mat M2_1 = M2(cv::Range(0,27), cv::Range(0,27));
    cv::Mat M2_2 = M2(cv::Range(0,27), cv::Range(27,120));
    cv::Mat M2_3 = M2(cv::Range(27,120), cv::Range(27,120));
    cv::Mat M2_4 = M2(cv::Range(27,120), cv::Range(0,27));
    M2.release();

    // A/B = B'\A'
    cv::Mat M2_5; cv::solve(M2_3.t(), M2_2.t(), M2_5);
    M2_2.release(); M2_3.release();

    // construct the multiplication matrix via schur compliment of the Macaulay
    // matrix
    Mtilde = M2_1 - M2_5.t()*M2_4;

}

void dls::compute_eigenvec(const cv::Mat& Mtilde, cv::Mat& eigenval_real, cv::Mat& eigenval_imag,
                                                  cv::Mat& eigenvec_real, cv::Mat& eigenvec_imag)
{
#ifdef HAVE_EIGEN
    Eigen::MatrixXd Mtilde_eig, zeros_eig;
    cv::cv2eigen(Mtilde, Mtilde_eig);
    cv::cv2eigen(cv::Mat::zeros(27, 27, CV_64F), zeros_eig);

    Eigen::MatrixXcd Mtilde_eig_cmplx(27, 27);
    Mtilde_eig_cmplx.real() = Mtilde_eig;
    Mtilde_eig_cmplx.imag() = zeros_eig;

    Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
    ces.compute(Mtilde_eig_cmplx);

    Eigen::MatrixXd eigval_real = ces.eigenvalues().real();
    Eigen::MatrixXd eigval_imag = ces.eigenvalues().imag();
    Eigen::MatrixXd eigvec_real = ces.eigenvectors().real();
    Eigen::MatrixXd eigvec_imag = ces.eigenvectors().imag();

    cv::eigen2cv(eigval_real, eigenval_real);
    cv::eigen2cv(eigval_imag, eigenval_imag);
    cv::eigen2cv(eigvec_real, eigenvec_real);
    cv::eigen2cv(eigvec_imag, eigenvec_imag);
#else
    EigenvalueDecomposition es(Mtilde);
    eigenval_real = es.eigenvalues();
    eigenvec_real = es.eigenvectors();
    eigenval_imag = eigenvec_imag = cv::Mat();
#endif

}

void dls::fill_coeff(const cv::Mat * D_mat)
{
    // TODO: shift D and coefficients one position to left

    double D[10][10]; // put D_mat into array

    for (int i = 0; i < D_mat->rows; ++i)
    {
        const double* Di = D_mat->ptr<double>(i);
        for (int j = 0; j < D_mat->cols; ++j)
        {
            D[i+1][j+1] = Di[j];
        }
    }

    // F1 COEFFICIENT

    f1coeff[1] = 2*D[1][6] - 2*D[1][8] + 2*D[5][6] - 2*D[5][8] + 2*D[6][1] + 2*D[6][5] + 2*D[6][9] - 2*D[8][1] - 2*D[8][5] - 2*D[8][9] + 2*D[9][6] - 2*D[9][8]; // constant term
    f1coeff[2] = 6*D[1][2] + 6*D[1][4] + 6*D[2][1] - 6*D[2][5] - 6*D[2][9] + 6*D[4][1] - 6*D[4][5] - 6*D[4][9] - 6*D[5][2] - 6*D[5][4] - 6*D[9][2] - 6*D[9][4]; // s1^2  * s2
    f1coeff[3] = 4*D[1][7] - 4*D[1][3] + 8*D[2][6] - 8*D[2][8] - 4*D[3][1] + 4*D[3][5] + 4*D[3][9] + 8*D[4][6] - 8*D[4][8] + 4*D[5][3] - 4*D[5][7] + 8*D[6][2] + 8*D[6][4] + 4*D[7][1] - 4*D[7][5] - 4*D[7][9] - 8*D[8][2] - 8*D[8][4] + 4*D[9][3] - 4*D[9][7]; // s1 * s2
    f1coeff[4] = 4*D[1][2] - 4*D[1][4] + 4*D[2][1] - 4*D[2][5] - 4*D[2][9] + 8*D[3][6] - 8*D[3][8] - 4*D[4][1] + 4*D[4][5] + 4*D[4][9] - 4*D[5][2] + 4*D[5][4] + 8*D[6][3] + 8*D[6][7] + 8*D[7][6] - 8*D[7][8] - 8*D[8][3] - 8*D[8][7] - 4*D[9][2] + 4*D[9][4]; //s1 * s3
    f1coeff[5] = 8*D[2][2] - 8*D[3][3] - 8*D[4][4] + 8*D[6][6] + 8*D[7][7] - 8*D[8][8]; // s2 * s3
    f1coeff[6] = 4*D[2][6] - 2*D[1][7] - 2*D[1][3] + 4*D[2][8] - 2*D[3][1] + 2*D[3][5] - 2*D[3][9] + 4*D[4][6] + 4*D[4][8] + 2*D[5][3] + 2*D[5][7] + 4*D[6][2] + 4*D[6][4] - 2*D[7][1] + 2*D[7][5] - 2*D[7][9] + 4*D[8][2] + 4*D[8][4] - 2*D[9][3] - 2*D[9][7]; // s2^2 * s3
    f1coeff[7] = 2*D[2][5] - 2*D[1][4] - 2*D[2][1] - 2*D[1][2] - 2*D[2][9] - 2*D[4][1] + 2*D[4][5] - 2*D[4][9] + 2*D[5][2] + 2*D[5][4] - 2*D[9][2] - 2*D[9][4]; //s2^3
    f1coeff[8] = 4*D[1][9] - 4*D[1][1] + 8*D[3][3] + 8*D[3][7] + 4*D[5][5] + 8*D[7][3] + 8*D[7][7] + 4*D[9][1] - 4*D[9][9]; // s1 * s3^2
    f1coeff[9] = 4*D[1][1] - 4*D[5][5] - 4*D[5][9] + 8*D[6][6] - 8*D[6][8] - 8*D[8][6] + 8*D[8][8] - 4*D[9][5] - 4*D[9][9]; // s1
    f1coeff[10] = 2*D[1][3] + 2*D[1][7] + 4*D[2][6] - 4*D[2][8] + 2*D[3][1] + 2*D[3][5] + 2*D[3][9] - 4*D[4][6] + 4*D[4][8] + 2*D[5][3] + 2*D[5][7] + 4*D[6][2] - 4*D[6][4] + 2*D[7][1] + 2*D[7][5] + 2*D[7][9] - 4*D[8][2] + 4*D[8][4] + 2*D[9][3] + 2*D[9][7]; // s3
    f1coeff[11] = 2*D[1][2] + 2*D[1][4] + 2*D[2][1] + 2*D[2][5] + 2*D[2][9] - 4*D[3][6] + 4*D[3][8] + 2*D[4][1] + 2*D[4][5] + 2*D[4][9] + 2*D[5][2] + 2*D[5][4] - 4*D[6][3] + 4*D[6][7] + 4*D[7][6] - 4*D[7][8] + 4*D[8][3] - 4*D[8][7] + 2*D[9][2] + 2*D[9][4]; // s2
    f1coeff[12] = 2*D[2][9] - 2*D[1][4] - 2*D[2][1] - 2*D[2][5] - 2*D[1][2] + 4*D[3][6] + 4*D[3][8] - 2*D[4][1] - 2*D[4][5] + 2*D[4][9] - 2*D[5][2] - 2*D[5][4] + 4*D[6][3] + 4*D[6][7] + 4*D[7][6] + 4*D[7][8] + 4*D[8][3] + 4*D[8][7] + 2*D[9][2] + 2*D[9][4]; // s2 * s3^2
    f1coeff[13] = 6*D[1][6] - 6*D[1][8] - 6*D[5][6] + 6*D[5][8] + 6*D[6][1] - 6*D[6][5] - 6*D[6][9] - 6*D[8][1] + 6*D[8][5] + 6*D[8][9] - 6*D[9][6] + 6*D[9][8]; // s1^2
    f1coeff[14] = 2*D[1][8] - 2*D[1][6] + 4*D[2][3] + 4*D[2][7] + 4*D[3][2] - 4*D[3][4] - 4*D[4][3] - 4*D[4][7] - 2*D[5][6] + 2*D[5][8] - 2*D[6][1] - 2*D[6][5] + 2*D[6][9] + 4*D[7][2] - 4*D[7][4] + 2*D[8][1] + 2*D[8][5] - 2*D[8][9] + 2*D[9][6] - 2*D[9][8]; // s3^2
    f1coeff[15] = 2*D[1][8] - 2*D[1][6] - 4*D[2][3] + 4*D[2][7] - 4*D[3][2] - 4*D[3][4] - 4*D[4][3] + 4*D[4][7] + 2*D[5][6] - 2*D[5][8] - 2*D[6][1] + 2*D[6][5] - 2*D[6][9] + 4*D[7][2] + 4*D[7][4] + 2*D[8][1] - 2*D[8][5] + 2*D[8][9] - 2*D[9][6] + 2*D[9][8]; // s2^2
    f1coeff[16] = 2*D[3][9] - 2*D[1][7] - 2*D[3][1] - 2*D[3][5] - 2*D[1][3] - 2*D[5][3] - 2*D[5][7] - 2*D[7][1] - 2*D[7][5] + 2*D[7][9] + 2*D[9][3] + 2*D[9][7]; // s3^3
    f1coeff[17] = 4*D[1][6] + 4*D[1][8] + 8*D[2][3] + 8*D[2][7] + 8*D[3][2] + 8*D[3][4] + 8*D[4][3] + 8*D[4][7] - 4*D[5][6] - 4*D[5][8] + 4*D[6][1] - 4*D[6][5] - 4*D[6][9] + 8*D[7][2] + 8*D[7][4] + 4*D[8][1] - 4*D[8][5] - 4*D[8][9] - 4*D[9][6] - 4*D[9][8]; // s1 * s2 * s3
    f1coeff[18] = 4*D[1][5] - 4*D[1][1] + 8*D[2][2] + 8*D[2][4] + 8*D[4][2] + 8*D[4][4] + 4*D[5][1] - 4*D[5][5] + 4*D[9][9]; // s1 * s2^2
    f1coeff[19] = 6*D[1][3] + 6*D[1][7] + 6*D[3][1] - 6*D[3][5] - 6*D[3][9] - 6*D[5][3] - 6*D[5][7] + 6*D[7][1] - 6*D[7][5] - 6*D[7][9] - 6*D[9][3] - 6*D[9][7]; // s1^2 * s3
    f1coeff[20] = 4*D[1][1] - 4*D[1][5] - 4*D[1][9] - 4*D[5][1] + 4*D[5][5] + 4*D[5][9] - 4*D[9][1] + 4*D[9][5] + 4*D[9][9]; // s1^3


    // F2 COEFFICIENT

    f2coeff[1] = - 2*D[1][3] + 2*D[1][7] - 2*D[3][1] - 2*D[3][5] - 2*D[3][9] - 2*D[5][3] + 2*D[5][7] + 2*D[7][1] + 2*D[7][5] + 2*D[7][9] - 2*D[9][3] + 2*D[9][7]; // constant term
    f2coeff[2] = 4*D[1][5] - 4*D[1][1] + 8*D[2][2] + 8*D[2][4] + 8*D[4][2] + 8*D[4][4] + 4*D[5][1] - 4*D[5][5] + 4*D[9][9]; // s1^2  * s2
    f2coeff[3] = 4*D[1][8] - 4*D[1][6] - 8*D[2][3] + 8*D[2][7] - 8*D[3][2] - 8*D[3][4] - 8*D[4][3] + 8*D[4][7] + 4*D[5][6] - 4*D[5][8] - 4*D[6][1] + 4*D[6][5] - 4*D[6][9] + 8*D[7][2] + 8*D[7][4] + 4*D[8][1] - 4*D[8][5] + 4*D[8][9] - 4*D[9][6] + 4*D[9][8]; // s1 * s2
    f2coeff[4] = 8*D[2][2] - 8*D[3][3] - 8*D[4][4] + 8*D[6][6] + 8*D[7][7] - 8*D[8][8]; // s1 * s3
    f2coeff[5] = 4*D[1][4] - 4*D[1][2] - 4*D[2][1] + 4*D[2][5] - 4*D[2][9] - 8*D[3][6] - 8*D[3][8] + 4*D[4][1] - 4*D[4][5] + 4*D[4][9] + 4*D[5][2] - 4*D[5][4] - 8*D[6][3] + 8*D[6][7] + 8*D[7][6] + 8*D[7][8] - 8*D[8][3] + 8*D[8][7] - 4*D[9][2] + 4*D[9][4]; // s2 * s3
    f2coeff[6] = 6*D[5][6] - 6*D[1][8] - 6*D[1][6] + 6*D[5][8] - 6*D[6][1] + 6*D[6][5] - 6*D[6][9] - 6*D[8][1] + 6*D[8][5] - 6*D[8][9] - 6*D[9][6] - 6*D[9][8]; // s2^2 * s3
    f2coeff[7] = 4*D[1][1] - 4*D[1][5] + 4*D[1][9] - 4*D[5][1] + 4*D[5][5] - 4*D[5][9] + 4*D[9][1] - 4*D[9][5] + 4*D[9][9]; // s2^3
    f2coeff[8] = 2*D[2][9] - 2*D[1][4] - 2*D[2][1] - 2*D[2][5] - 2*D[1][2] + 4*D[3][6] + 4*D[3][8] - 2*D[4][1] - 2*D[4][5] + 2*D[4][9] - 2*D[5][2] - 2*D[5][4] + 4*D[6][3] + 4*D[6][7] + 4*D[7][6] + 4*D[7][8] + 4*D[8][3] + 4*D[8][7] + 2*D[9][2] + 2*D[9][4]; // s1 * s3^2
    f2coeff[9] = 2*D[1][2] + 2*D[1][4] + 2*D[2][1] + 2*D[2][5] + 2*D[2][9] - 4*D[3][6] + 4*D[3][8] + 2*D[4][1] + 2*D[4][5] + 2*D[4][9] + 2*D[5][2] + 2*D[5][4] - 4*D[6][3] + 4*D[6][7] + 4*D[7][6] - 4*D[7][8] + 4*D[8][3] - 4*D[8][7] + 2*D[9][2] + 2*D[9][4]; // s1
    f2coeff[10] = 2*D[1][6] + 2*D[1][8] - 4*D[2][3] + 4*D[2][7] - 4*D[3][2] + 4*D[3][4] + 4*D[4][3] - 4*D[4][7] + 2*D[5][6] + 2*D[5][8] + 2*D[6][1] + 2*D[6][5] + 2*D[6][9] + 4*D[7][2] - 4*D[7][4] + 2*D[8][1] + 2*D[8][5] + 2*D[8][9] + 2*D[9][6] + 2*D[9][8]; // s3
    f2coeff[11] = 8*D[3][3] - 4*D[1][9] - 4*D[1][1] - 8*D[3][7] + 4*D[5][5] - 8*D[7][3] + 8*D[7][7] - 4*D[9][1] - 4*D[9][9]; // s2
    f2coeff[12] = 4*D[1][1] - 4*D[5][5] + 4*D[5][9] + 8*D[6][6] + 8*D[6][8] + 8*D[8][6] + 8*D[8][8] + 4*D[9][5] - 4*D[9][9]; // s2 * s3^2
    f2coeff[13] = 2*D[1][7] - 2*D[1][3] + 4*D[2][6] - 4*D[2][8] - 2*D[3][1] + 2*D[3][5] + 2*D[3][9] + 4*D[4][6] - 4*D[4][8] + 2*D[5][3] - 2*D[5][7] + 4*D[6][2] + 4*D[6][4] + 2*D[7][1] - 2*D[7][5] - 2*D[7][9] - 4*D[8][2] - 4*D[8][4] + 2*D[9][3] - 2*D[9][7]; // s1^2
    f2coeff[14] = 2*D[1][3] - 2*D[1][7] + 4*D[2][6] + 4*D[2][8] + 2*D[3][1] + 2*D[3][5] - 2*D[3][9] - 4*D[4][6] - 4*D[4][8] + 2*D[5][3] - 2*D[5][7] + 4*D[6][2] - 4*D[6][4] - 2*D[7][1] - 2*D[7][5] + 2*D[7][9] + 4*D[8][2] - 4*D[8][4] - 2*D[9][3] + 2*D[9][7]; // s3^2
    f2coeff[15] = 6*D[1][3] - 6*D[1][7] + 6*D[3][1] - 6*D[3][5] + 6*D[3][9] - 6*D[5][3] + 6*D[5][7] - 6*D[7][1] + 6*D[7][5] - 6*D[7][9] + 6*D[9][3] - 6*D[9][7]; // s2^2
    f2coeff[16] = 2*D[6][9] - 2*D[1][8] - 2*D[5][6] - 2*D[5][8] - 2*D[6][1] - 2*D[6][5] - 2*D[1][6] - 2*D[8][1] - 2*D[8][5] + 2*D[8][9] + 2*D[9][6] + 2*D[9][8]; // s3^3
    f2coeff[17] = 8*D[2][6] - 4*D[1][7] - 4*D[1][3] + 8*D[2][8] - 4*D[3][1] + 4*D[3][5] - 4*D[3][9] + 8*D[4][6] + 8*D[4][8] + 4*D[5][3] + 4*D[5][7] + 8*D[6][2] + 8*D[6][4] - 4*D[7][1] + 4*D[7][5] - 4*D[7][9] + 8*D[8][2] + 8*D[8][4] - 4*D[9][3] - 4*D[9][7]; // s1 * s2 * s3
    f2coeff[18] = 6*D[2][5] - 6*D[1][4] - 6*D[2][1] - 6*D[1][2] - 6*D[2][9] - 6*D[4][1] + 6*D[4][5] - 6*D[4][9] + 6*D[5][2] + 6*D[5][4] - 6*D[9][2] - 6*D[9][4]; // s1 * s2^2
    f2coeff[19] = 2*D[1][6] + 2*D[1][8] + 4*D[2][3] + 4*D[2][7] + 4*D[3][2] + 4*D[3][4] + 4*D[4][3] + 4*D[4][7] - 2*D[5][6] - 2*D[5][8] + 2*D[6][1] - 2*D[6][5] - 2*D[6][9] + 4*D[7][2] + 4*D[7][4] + 2*D[8][1] - 2*D[8][5] - 2*D[8][9] - 2*D[9][6] - 2*D[9][8]; // s1^2 * s3
    f2coeff[20] = 2*D[1][2] + 2*D[1][4] + 2*D[2][1] - 2*D[2][5] - 2*D[2][9] + 2*D[4][1] - 2*D[4][5] - 2*D[4][9] - 2*D[5][2] - 2*D[5][4] - 2*D[9][2] - 2*D[9][4]; // s1^3


    // F3 COEFFICIENT

    f3coeff[1] = 2*D[1][2] - 2*D[1][4] + 2*D[2][1] + 2*D[2][5] + 2*D[2][9] - 2*D[4][1] - 2*D[4][5] - 2*D[4][9] + 2*D[5][2] - 2*D[5][4] + 2*D[9][2] - 2*D[9][4]; // constant term
    f3coeff[2] = 2*D[1][6] + 2*D[1][8] + 4*D[2][3] + 4*D[2][7] + 4*D[3][2] + 4*D[3][4] + 4*D[4][3] + 4*D[4][7] - 2*D[5][6] - 2*D[5][8] + 2*D[6][1] - 2*D[6][5] - 2*D[6][9] + 4*D[7][2] + 4*D[7][4] + 2*D[8][1] - 2*D[8][5] - 2*D[8][9] - 2*D[9][6] - 2*D[9][8]; // s1^2  * s2
    f3coeff[3] = 8*D[2][2] - 8*D[3][3] - 8*D[4][4] + 8*D[6][6] + 8*D[7][7] - 8*D[8][8]; // s1 * s2
    f3coeff[4] = 4*D[1][8] - 4*D[1][6] + 8*D[2][3] + 8*D[2][7] + 8*D[3][2] - 8*D[3][4] - 8*D[4][3] - 8*D[4][7] - 4*D[5][6] + 4*D[5][8] - 4*D[6][1] - 4*D[6][5] + 4*D[6][9] + 8*D[7][2] - 8*D[7][4] + 4*D[8][1] + 4*D[8][5] - 4*D[8][9] + 4*D[9][6] - 4*D[9][8]; // s1 * s3
    f3coeff[5] = 4*D[1][3] - 4*D[1][7] + 8*D[2][6] + 8*D[2][8] + 4*D[3][1] + 4*D[3][5] - 4*D[3][9] - 8*D[4][6] - 8*D[4][8] + 4*D[5][3] - 4*D[5][7] + 8*D[6][2] - 8*D[6][4] - 4*D[7][1] - 4*D[7][5] + 4*D[7][9] + 8*D[8][2] - 8*D[8][4] - 4*D[9][3] + 4*D[9][7]; // s2 * s3
    f3coeff[6] = 4*D[1][1] - 4*D[5][5] + 4*D[5][9] + 8*D[6][6] + 8*D[6][8] + 8*D[8][6] + 8*D[8][8] + 4*D[9][5] - 4*D[9][9]; // s2^2 * s3
    f3coeff[7] = 2*D[5][6] - 2*D[1][8] - 2*D[1][6] + 2*D[5][8] - 2*D[6][1] + 2*D[6][5] - 2*D[6][9] - 2*D[8][1] + 2*D[8][5] - 2*D[8][9] - 2*D[9][6] - 2*D[9][8]; // s2^3
    f3coeff[8] = 6*D[3][9] - 6*D[1][7] - 6*D[3][1] - 6*D[3][5] - 6*D[1][3] - 6*D[5][3] - 6*D[5][7] - 6*D[7][1] - 6*D[7][5] + 6*D[7][9] + 6*D[9][3] + 6*D[9][7]; // s1 * s3^2
    f3coeff[9] = 2*D[1][3] + 2*D[1][7] + 4*D[2][6] - 4*D[2][8] + 2*D[3][1] + 2*D[3][5] + 2*D[3][9] - 4*D[4][6] + 4*D[4][8] + 2*D[5][3] + 2*D[5][7] + 4*D[6][2] - 4*D[6][4] + 2*D[7][1] + 2*D[7][5] + 2*D[7][9] - 4*D[8][2] + 4*D[8][4] + 2*D[9][3] + 2*D[9][7]; // s1
    f3coeff[10] = 8*D[2][2] - 4*D[1][5] - 4*D[1][1] - 8*D[2][4] - 8*D[4][2] + 8*D[4][4] - 4*D[5][1] - 4*D[5][5] + 4*D[9][9]; // s3
    f3coeff[11] = 2*D[1][6] + 2*D[1][8] - 4*D[2][3] + 4*D[2][7] - 4*D[3][2] + 4*D[3][4] + 4*D[4][3] - 4*D[4][7] + 2*D[5][6] + 2*D[5][8] + 2*D[6][1] + 2*D[6][5] + 2*D[6][9] + 4*D[7][2] - 4*D[7][4] + 2*D[8][1] + 2*D[8][5] + 2*D[8][9] + 2*D[9][6] + 2*D[9][8]; // s2
    f3coeff[12] = 6*D[6][9] - 6*D[1][8] - 6*D[5][6] - 6*D[5][8] - 6*D[6][1] - 6*D[6][5] - 6*D[1][6] - 6*D[8][1] - 6*D[8][5] + 6*D[8][9] + 6*D[9][6] + 6*D[9][8]; // s2 * s3^2
    f3coeff[13] = 2*D[1][2] - 2*D[1][4] + 2*D[2][1] - 2*D[2][5] - 2*D[2][9] + 4*D[3][6] - 4*D[3][8] - 2*D[4][1] + 2*D[4][5] + 2*D[4][9] - 2*D[5][2] + 2*D[5][4] + 4*D[6][3] + 4*D[6][7] + 4*D[7][6] - 4*D[7][8] - 4*D[8][3] - 4*D[8][7] - 2*D[9][2] + 2*D[9][4]; // s1^2
    f3coeff[14] = 6*D[1][4] - 6*D[1][2] - 6*D[2][1] - 6*D[2][5] + 6*D[2][9] + 6*D[4][1] + 6*D[4][5] - 6*D[4][9] - 6*D[5][2] + 6*D[5][4] + 6*D[9][2] - 6*D[9][4]; // s3^2
    f3coeff[15] = 2*D[1][4] - 2*D[1][2] - 2*D[2][1] + 2*D[2][5] - 2*D[2][9] - 4*D[3][6] - 4*D[3][8] + 2*D[4][1] - 2*D[4][5] + 2*D[4][9] + 2*D[5][2] - 2*D[5][4] - 4*D[6][3] + 4*D[6][7] + 4*D[7][6] + 4*D[7][8] - 4*D[8][3] + 4*D[8][7] - 2*D[9][2] + 2*D[9][4]; // s2^2
    f3coeff[16] = 4*D[1][1] + 4*D[1][5] - 4*D[1][9] + 4*D[5][1] + 4*D[5][5] - 4*D[5][9] - 4*D[9][1] - 4*D[9][5] + 4*D[9][9]; // s3^3
    f3coeff[17] = 4*D[2][9] - 4*D[1][4] - 4*D[2][1] - 4*D[2][5] - 4*D[1][2] + 8*D[3][6] + 8*D[3][8] - 4*D[4][1] - 4*D[4][5] + 4*D[4][9] - 4*D[5][2] - 4*D[5][4] + 8*D[6][3] + 8*D[6][7] + 8*D[7][6] + 8*D[7][8] + 8*D[8][3] + 8*D[8][7] + 4*D[9][2] + 4*D[9][4]; // s1 * s2 * s3
    f3coeff[18] = 4*D[2][6] - 2*D[1][7] - 2*D[1][3] + 4*D[2][8] - 2*D[3][1] + 2*D[3][5] - 2*D[3][9] + 4*D[4][6] + 4*D[4][8] + 2*D[5][3] + 2*D[5][7] + 4*D[6][2] + 4*D[6][4] - 2*D[7][1] + 2*D[7][5] - 2*D[7][9] + 4*D[8][2] + 4*D[8][4] - 2*D[9][3] - 2*D[9][7]; // s1 * s2^2
    f3coeff[19] = 4*D[1][9] - 4*D[1][1] + 8*D[3][3] + 8*D[3][7] + 4*D[5][5] + 8*D[7][3] + 8*D[7][7] + 4*D[9][1] - 4*D[9][9]; // s1^2 * s3
    f3coeff[20] = 2*D[1][3] + 2*D[1][7] + 2*D[3][1] - 2*D[3][5] - 2*D[3][9] - 2*D[5][3] - 2*D[5][7] + 2*D[7][1] - 2*D[7][5] - 2*D[7][9] - 2*D[9][3] - 2*D[9][7]; // s1^3

}

cv::Mat dls::LeftMultVec(const cv::Mat& v)
{
    cv::Mat mat_ = cv::Mat::zeros(3, 9, CV_64F);

    for (int i = 0; i < 3; ++i)
    {
        mat_.at<double>(i, 3*i + 0) = v.at<double>(0);
        mat_.at<double>(i, 3*i + 1) = v.at<double>(1);
        mat_.at<double>(i, 3*i + 2) = v.at<double>(2);
    }
    return mat_;
}

cv::Mat dls::cayley_LS_M(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, const std::vector<double>& u)
{
    // TODO: input matrix pointer
    // TODO: shift coefficients one position to left

    cv::Mat M = cv::Mat::zeros(120, 120, CV_64F);

    M.at<double>(0,0)=u[1]; M.at<double>(0,35)=a[1]; M.at<double>(0,83)=b[1]; M.at<double>(0,118)=c[1];
    M.at<double>(1,0)=u[4]; M.at<double>(1,1)=u[1]; M.at<double>(1,34)=a[1]; M.at<double>(1,35)=a[10]; M.at<double>(1,54)=b[1]; M.at<double>(1,83)=b[10]; M.at<double>(1,99)=c[1]; M.at<double>(1,118)=c[10];
    M.at<double>(2,1)=u[4]; M.at<double>(2,2)=u[1]; M.at<double>(2,34)=a[10]; M.at<double>(2,35)=a[14]; M.at<double>(2,51)=a[1]; M.at<double>(2,54)=b[10]; M.at<double>(2,65)=b[1]; M.at<double>(2,83)=b[14]; M.at<double>(2,89)=c[1]; M.at<double>(2,99)=c[10]; M.at<double>(2,118)=c[14];
    M.at<double>(3,0)=u[3]; M.at<double>(3,3)=u[1]; M.at<double>(3,35)=a[11]; M.at<double>(3,49)=a[1]; M.at<double>(3,76)=b[1]; M.at<double>(3,83)=b[11]; M.at<double>(3,118)=c[11]; M.at<double>(3,119)=c[1];
    M.at<double>(4,1)=u[3]; M.at<double>(4,3)=u[4]; M.at<double>(4,4)=u[1]; M.at<double>(4,34)=a[11]; M.at<double>(4,35)=a[5]; M.at<double>(4,43)=a[1]; M.at<double>(4,49)=a[10]; M.at<double>(4,54)=b[11]; M.at<double>(4,71)=b[1]; M.at<double>(4,76)=b[10]; M.at<double>(4,83)=b[5]; M.at<double>(4,99)=c[11]; M.at<double>(4,100)=c[1]; M.at<double>(4,118)=c[5]; M.at<double>(4,119)=c[10];
    M.at<double>(5,2)=u[3]; M.at<double>(5,4)=u[4]; M.at<double>(5,5)=u[1]; M.at<double>(5,34)=a[5]; M.at<double>(5,35)=a[12]; M.at<double>(5,41)=a[1]; M.at<double>(5,43)=a[10]; M.at<double>(5,49)=a[14]; M.at<double>(5,51)=a[11]; M.at<double>(5,54)=b[5]; M.at<double>(5,62)=b[1]; M.at<double>(5,65)=b[11]; M.at<double>(5,71)=b[10]; M.at<double>(5,76)=b[14]; M.at<double>(5,83)=b[12]; M.at<double>(5,89)=c[11]; M.at<double>(5,99)=c[5]; M.at<double>(5,100)=c[10]; M.at<double>(5,111)=c[1]; M.at<double>(5,118)=c[12]; M.at<double>(5,119)=c[14];
    M.at<double>(6,3)=u[3]; M.at<double>(6,6)=u[1]; M.at<double>(6,30)=a[1]; M.at<double>(6,35)=a[15]; M.at<double>(6,49)=a[11]; M.at<double>(6,75)=b[1]; M.at<double>(6,76)=b[11]; M.at<double>(6,83)=b[15]; M.at<double>(6,107)=c[1]; M.at<double>(6,118)=c[15]; M.at<double>(6,119)=c[11];
    M.at<double>(7,4)=u[3]; M.at<double>(7,6)=u[4]; M.at<double>(7,7)=u[1]; M.at<double>(7,30)=a[10]; M.at<double>(7,34)=a[15]; M.at<double>(7,35)=a[6]; M.at<double>(7,43)=a[11]; M.at<double>(7,45)=a[1]; M.at<double>(7,49)=a[5]; M.at<double>(7,54)=b[15]; M.at<double>(7,63)=b[1]; M.at<double>(7,71)=b[11]; M.at<double>(7,75)=b[10]; M.at<double>(7,76)=b[5]; M.at<double>(7,83)=b[6]; M.at<double>(7,99)=c[15]; M.at<double>(7,100)=c[11]; M.at<double>(7,107)=c[10]; M.at<double>(7,112)=c[1]; M.at<double>(7,118)=c[6]; M.at<double>(7,119)=c[5];
    M.at<double>(8,5)=u[3]; M.at<double>(8,7)=u[4]; M.at<double>(8,8)=u[1]; M.at<double>(8,30)=a[14]; M.at<double>(8,34)=a[6]; M.at<double>(8,41)=a[11]; M.at<double>(8,43)=a[5]; M.at<double>(8,45)=a[10]; M.at<double>(8,46)=a[1]; M.at<double>(8,49)=a[12]; M.at<double>(8,51)=a[15]; M.at<double>(8,54)=b[6]; M.at<double>(8,62)=b[11]; M.at<double>(8,63)=b[10]; M.at<double>(8,65)=b[15]; M.at<double>(8,66)=b[1]; M.at<double>(8,71)=b[5]; M.at<double>(8,75)=b[14]; M.at<double>(8,76)=b[12]; M.at<double>(8,89)=c[15]; M.at<double>(8,99)=c[6]; M.at<double>(8,100)=c[5]; M.at<double>(8,102)=c[1]; M.at<double>(8,107)=c[14]; M.at<double>(8,111)=c[11]; M.at<double>(8,112)=c[10]; M.at<double>(8,119)=c[12];
    M.at<double>(9,0)=u[2]; M.at<double>(9,9)=u[1]; M.at<double>(9,35)=a[9]; M.at<double>(9,36)=a[1]; M.at<double>(9,83)=b[9]; M.at<double>(9,84)=b[1]; M.at<double>(9,88)=c[1]; M.at<double>(9,118)=c[9];
    M.at<double>(10,1)=u[2]; M.at<double>(10,9)=u[4]; M.at<double>(10,10)=u[1]; M.at<double>(10,33)=a[1]; M.at<double>(10,34)=a[9]; M.at<double>(10,35)=a[4]; M.at<double>(10,36)=a[10]; M.at<double>(10,54)=b[9]; M.at<double>(10,59)=b[1]; M.at<double>(10,83)=b[4]; M.at<double>(10,84)=b[10]; M.at<double>(10,88)=c[10]; M.at<double>(10,99)=c[9]; M.at<double>(10,117)=c[1]; M.at<double>(10,118)=c[4];
    M.at<double>(11,2)=u[2]; M.at<double>(11,10)=u[4]; M.at<double>(11,11)=u[1]; M.at<double>(11,28)=a[1]; M.at<double>(11,33)=a[10]; M.at<double>(11,34)=a[4]; M.at<double>(11,35)=a[8]; M.at<double>(11,36)=a[14]; M.at<double>(11,51)=a[9]; M.at<double>(11,54)=b[4]; M.at<double>(11,57)=b[1]; M.at<double>(11,59)=b[10]; M.at<double>(11,65)=b[9]; M.at<double>(11,83)=b[8]; M.at<double>(11,84)=b[14]; M.at<double>(11,88)=c[14]; M.at<double>(11,89)=c[9]; M.at<double>(11,99)=c[4]; M.at<double>(11,114)=c[1]; M.at<double>(11,117)=c[10]; M.at<double>(11,118)=c[8];
    M.at<double>(12,3)=u[2]; M.at<double>(12,9)=u[3]; M.at<double>(12,12)=u[1]; M.at<double>(12,35)=a[3]; M.at<double>(12,36)=a[11]; M.at<double>(12,39)=a[1]; M.at<double>(12,49)=a[9]; M.at<double>(12,76)=b[9]; M.at<double>(12,79)=b[1]; M.at<double>(12,83)=b[3]; M.at<double>(12,84)=b[11]; M.at<double>(12,88)=c[11]; M.at<double>(12,96)=c[1]; M.at<double>(12,118)=c[3]; M.at<double>(12,119)=c[9];
    M.at<double>(13,4)=u[2]; M.at<double>(13,10)=u[3]; M.at<double>(13,12)=u[4]; M.at<double>(13,13)=u[1]; M.at<double>(13,33)=a[11]; M.at<double>(13,34)=a[3]; M.at<double>(13,35)=a[17]; M.at<double>(13,36)=a[5]; M.at<double>(13,39)=a[10]; M.at<double>(13,43)=a[9]; M.at<double>(13,47)=a[1]; M.at<double>(13,49)=a[4]; M.at<double>(13,54)=b[3]; M.at<double>(13,59)=b[11]; M.at<double>(13,60)=b[1]; M.at<double>(13,71)=b[9]; M.at<double>(13,76)=b[4]; M.at<double>(13,79)=b[10]; M.at<double>(13,83)=b[17]; M.at<double>(13,84)=b[5]; M.at<double>(13,88)=c[5]; M.at<double>(13,90)=c[1]; M.at<double>(13,96)=c[10]; M.at<double>(13,99)=c[3]; M.at<double>(13,100)=c[9]; M.at<double>(13,117)=c[11]; M.at<double>(13,118)=c[17]; M.at<double>(13,119)=c[4];
    M.at<double>(14,5)=u[2]; M.at<double>(14,11)=u[3]; M.at<double>(14,13)=u[4]; M.at<double>(14,14)=u[1]; M.at<double>(14,28)=a[11]; M.at<double>(14,33)=a[5]; M.at<double>(14,34)=a[17]; M.at<double>(14,36)=a[12]; M.at<double>(14,39)=a[14]; M.at<double>(14,41)=a[9]; M.at<double>(14,42)=a[1]; M.at<double>(14,43)=a[4]; M.at<double>(14,47)=a[10]; M.at<double>(14,49)=a[8]; M.at<double>(14,51)=a[3]; M.at<double>(14,54)=b[17]; M.at<double>(14,56)=b[1]; M.at<double>(14,57)=b[11]; M.at<double>(14,59)=b[5]; M.at<double>(14,60)=b[10]; M.at<double>(14,62)=b[9]; M.at<double>(14,65)=b[3]; M.at<double>(14,71)=b[4]; M.at<double>(14,76)=b[8]; M.at<double>(14,79)=b[14]; M.at<double>(14,84)=b[12]; M.at<double>(14,88)=c[12]; M.at<double>(14,89)=c[3]; M.at<double>(14,90)=c[10]; M.at<double>(14,96)=c[14]; M.at<double>(14,99)=c[17]; M.at<double>(14,100)=c[4]; M.at<double>(14,106)=c[1]; M.at<double>(14,111)=c[9]; M.at<double>(14,114)=c[11]; M.at<double>(14,117)=c[5]; M.at<double>(14,119)=c[8];
    M.at<double>(15,6)=u[2]; M.at<double>(15,12)=u[3]; M.at<double>(15,15)=u[1]; M.at<double>(15,29)=a[1]; M.at<double>(15,30)=a[9]; M.at<double>(15,35)=a[18]; M.at<double>(15,36)=a[15]; M.at<double>(15,39)=a[11]; M.at<double>(15,49)=a[3]; M.at<double>(15,74)=b[1]; M.at<double>(15,75)=b[9]; M.at<double>(15,76)=b[3]; M.at<double>(15,79)=b[11]; M.at<double>(15,83)=b[18]; M.at<double>(15,84)=b[15]; M.at<double>(15,88)=c[15]; M.at<double>(15,94)=c[1]; M.at<double>(15,96)=c[11]; M.at<double>(15,107)=c[9]; M.at<double>(15,118)=c[18]; M.at<double>(15,119)=c[3];
    M.at<double>(16,7)=u[2]; M.at<double>(16,13)=u[3]; M.at<double>(16,15)=u[4]; M.at<double>(16,16)=u[1]; M.at<double>(16,29)=a[10]; M.at<double>(16,30)=a[4]; M.at<double>(16,33)=a[15]; M.at<double>(16,34)=a[18]; M.at<double>(16,36)=a[6]; M.at<double>(16,39)=a[5]; M.at<double>(16,43)=a[3]; M.at<double>(16,44)=a[1]; M.at<double>(16,45)=a[9]; M.at<double>(16,47)=a[11]; M.at<double>(16,49)=a[17]; M.at<double>(16,54)=b[18]; M.at<double>(16,59)=b[15]; M.at<double>(16,60)=b[11]; M.at<double>(16,63)=b[9]; M.at<double>(16,68)=b[1]; M.at<double>(16,71)=b[3]; M.at<double>(16,74)=b[10]; M.at<double>(16,75)=b[4]; M.at<double>(16,76)=b[17]; M.at<double>(16,79)=b[5]; M.at<double>(16,84)=b[6]; M.at<double>(16,88)=c[6]; M.at<double>(16,90)=c[11]; M.at<double>(16,94)=c[10]; M.at<double>(16,96)=c[5]; M.at<double>(16,97)=c[1]; M.at<double>(16,99)=c[18]; M.at<double>(16,100)=c[3]; M.at<double>(16,107)=c[4]; M.at<double>(16,112)=c[9]; M.at<double>(16,117)=c[15]; M.at<double>(16,119)=c[17];
    M.at<double>(17,8)=u[2]; M.at<double>(17,14)=u[3]; M.at<double>(17,16)=u[4]; M.at<double>(17,17)=u[1]; M.at<double>(17,28)=a[15]; M.at<double>(17,29)=a[14]; M.at<double>(17,30)=a[8]; M.at<double>(17,33)=a[6]; M.at<double>(17,39)=a[12]; M.at<double>(17,41)=a[3]; M.at<double>(17,42)=a[11]; M.at<double>(17,43)=a[17]; M.at<double>(17,44)=a[10]; M.at<double>(17,45)=a[4]; M.at<double>(17,46)=a[9]; M.at<double>(17,47)=a[5]; M.at<double>(17,51)=a[18]; M.at<double>(17,56)=b[11]; M.at<double>(17,57)=b[15]; M.at<double>(17,59)=b[6]; M.at<double>(17,60)=b[5]; M.at<double>(17,62)=b[3]; M.at<double>(17,63)=b[4]; M.at<double>(17,65)=b[18]; M.at<double>(17,66)=b[9]; M.at<double>(17,68)=b[10]; M.at<double>(17,71)=b[17]; M.at<double>(17,74)=b[14]; M.at<double>(17,75)=b[8]; M.at<double>(17,79)=b[12]; M.at<double>(17,89)=c[18]; M.at<double>(17,90)=c[5]; M.at<double>(17,94)=c[14]; M.at<double>(17,96)=c[12]; M.at<double>(17,97)=c[10]; M.at<double>(17,100)=c[17]; M.at<double>(17,102)=c[9]; M.at<double>(17,106)=c[11]; M.at<double>(17,107)=c[8]; M.at<double>(17,111)=c[3]; M.at<double>(17,112)=c[4]; M.at<double>(17,114)=c[15]; M.at<double>(17,117)=c[6];
    M.at<double>(18,9)=u[2]; M.at<double>(18,18)=u[1]; M.at<double>(18,35)=a[13]; M.at<double>(18,36)=a[9]; M.at<double>(18,53)=a[1]; M.at<double>(18,82)=b[1]; M.at<double>(18,83)=b[13]; M.at<double>(18,84)=b[9]; M.at<double>(18,87)=c[1]; M.at<double>(18,88)=c[9]; M.at<double>(18,118)=c[13];
    M.at<double>(19,10)=u[2]; M.at<double>(19,18)=u[4]; M.at<double>(19,19)=u[1]; M.at<double>(19,32)=a[1]; M.at<double>(19,33)=a[9]; M.at<double>(19,34)=a[13]; M.at<double>(19,35)=a[19]; M.at<double>(19,36)=a[4]; M.at<double>(19,53)=a[10]; M.at<double>(19,54)=b[13]; M.at<double>(19,59)=b[9]; M.at<double>(19,61)=b[1]; M.at<double>(19,82)=b[10]; M.at<double>(19,83)=b[19]; M.at<double>(19,84)=b[4]; M.at<double>(19,87)=c[10]; M.at<double>(19,88)=c[4]; M.at<double>(19,99)=c[13]; M.at<double>(19,116)=c[1]; M.at<double>(19,117)=c[9]; M.at<double>(19,118)=c[19];
    M.at<double>(20,11)=u[2]; M.at<double>(20,19)=u[4]; M.at<double>(20,20)=u[1]; M.at<double>(20,27)=a[1]; M.at<double>(20,28)=a[9]; M.at<double>(20,32)=a[10]; M.at<double>(20,33)=a[4]; M.at<double>(20,34)=a[19]; M.at<double>(20,36)=a[8]; M.at<double>(20,51)=a[13]; M.at<double>(20,53)=a[14]; M.at<double>(20,54)=b[19]; M.at<double>(20,55)=b[1]; M.at<double>(20,57)=b[9]; M.at<double>(20,59)=b[4]; M.at<double>(20,61)=b[10]; M.at<double>(20,65)=b[13]; M.at<double>(20,82)=b[14]; M.at<double>(20,84)=b[8]; M.at<double>(20,87)=c[14]; M.at<double>(20,88)=c[8]; M.at<double>(20,89)=c[13]; M.at<double>(20,99)=c[19]; M.at<double>(20,113)=c[1]; M.at<double>(20,114)=c[9]; M.at<double>(20,116)=c[10]; M.at<double>(20,117)=c[4];
    M.at<double>(21,12)=u[2]; M.at<double>(21,18)=u[3]; M.at<double>(21,21)=u[1]; M.at<double>(21,35)=a[2]; M.at<double>(21,36)=a[3]; M.at<double>(21,38)=a[1]; M.at<double>(21,39)=a[9]; M.at<double>(21,49)=a[13]; M.at<double>(21,53)=a[11]; M.at<double>(21,76)=b[13]; M.at<double>(21,78)=b[1]; M.at<double>(21,79)=b[9]; M.at<double>(21,82)=b[11]; M.at<double>(21,83)=b[2]; M.at<double>(21,84)=b[3]; M.at<double>(21,87)=c[11]; M.at<double>(21,88)=c[3]; M.at<double>(21,92)=c[1]; M.at<double>(21,96)=c[9]; M.at<double>(21,118)=c[2]; M.at<double>(21,119)=c[13];
    M.at<double>(22,13)=u[2]; M.at<double>(22,19)=u[3]; M.at<double>(22,21)=u[4]; M.at<double>(22,22)=u[1]; M.at<double>(22,32)=a[11]; M.at<double>(22,33)=a[3]; M.at<double>(22,34)=a[2]; M.at<double>(22,36)=a[17]; M.at<double>(22,38)=a[10]; M.at<double>(22,39)=a[4]; M.at<double>(22,40)=a[1]; M.at<double>(22,43)=a[13]; M.at<double>(22,47)=a[9]; M.at<double>(22,49)=a[19]; M.at<double>(22,53)=a[5]; M.at<double>(22,54)=b[2]; M.at<double>(22,59)=b[3]; M.at<double>(22,60)=b[9]; M.at<double>(22,61)=b[11]; M.at<double>(22,71)=b[13]; M.at<double>(22,72)=b[1]; M.at<double>(22,76)=b[19]; M.at<double>(22,78)=b[10]; M.at<double>(22,79)=b[4]; M.at<double>(22,82)=b[5]; M.at<double>(22,84)=b[17]; M.at<double>(22,87)=c[5]; M.at<double>(22,88)=c[17]; M.at<double>(22,90)=c[9]; M.at<double>(22,92)=c[10]; M.at<double>(22,95)=c[1]; M.at<double>(22,96)=c[4]; M.at<double>(22,99)=c[2]; M.at<double>(22,100)=c[13]; M.at<double>(22,116)=c[11]; M.at<double>(22,117)=c[3]; M.at<double>(22,119)=c[19];
    M.at<double>(23,14)=u[2]; M.at<double>(23,20)=u[3]; M.at<double>(23,22)=u[4]; M.at<double>(23,23)=u[1]; M.at<double>(23,27)=a[11]; M.at<double>(23,28)=a[3]; M.at<double>(23,32)=a[5]; M.at<double>(23,33)=a[17]; M.at<double>(23,38)=a[14]; M.at<double>(23,39)=a[8]; M.at<double>(23,40)=a[10]; M.at<double>(23,41)=a[13]; M.at<double>(23,42)=a[9]; M.at<double>(23,43)=a[19]; M.at<double>(23,47)=a[4]; M.at<double>(23,51)=a[2]; M.at<double>(23,53)=a[12]; M.at<double>(23,55)=b[11]; M.at<double>(23,56)=b[9]; M.at<double>(23,57)=b[3]; M.at<double>(23,59)=b[17]; M.at<double>(23,60)=b[4]; M.at<double>(23,61)=b[5]; M.at<double>(23,62)=b[13]; M.at<double>(23,65)=b[2]; M.at<double>(23,71)=b[19]; M.at<double>(23,72)=b[10]; M.at<double>(23,78)=b[14]; M.at<double>(23,79)=b[8]; M.at<double>(23,82)=b[12]; M.at<double>(23,87)=c[12]; M.at<double>(23,89)=c[2]; M.at<double>(23,90)=c[4]; M.at<double>(23,92)=c[14]; M.at<double>(23,95)=c[10]; M.at<double>(23,96)=c[8]; M.at<double>(23,100)=c[19]; M.at<double>(23,106)=c[9]; M.at<double>(23,111)=c[13]; M.at<double>(23,113)=c[11]; M.at<double>(23,114)=c[3]; M.at<double>(23,116)=c[5]; M.at<double>(23,117)=c[17];
    M.at<double>(24,15)=u[2]; M.at<double>(24,21)=u[3]; M.at<double>(24,24)=u[1]; M.at<double>(24,29)=a[9]; M.at<double>(24,30)=a[13]; M.at<double>(24,36)=a[18]; M.at<double>(24,38)=a[11]; M.at<double>(24,39)=a[3]; M.at<double>(24,49)=a[2]; M.at<double>(24,52)=a[1]; M.at<double>(24,53)=a[15]; M.at<double>(24,73)=b[1]; M.at<double>(24,74)=b[9]; M.at<double>(24,75)=b[13]; M.at<double>(24,76)=b[2]; M.at<double>(24,78)=b[11]; M.at<double>(24,79)=b[3]; M.at<double>(24,82)=b[15]; M.at<double>(24,84)=b[18]; M.at<double>(24,87)=c[15]; M.at<double>(24,88)=c[18]; M.at<double>(24,92)=c[11]; M.at<double>(24,93)=c[1]; M.at<double>(24,94)=c[9]; M.at<double>(24,96)=c[3]; M.at<double>(24,107)=c[13]; M.at<double>(24,119)=c[2];
    M.at<double>(25,16)=u[2]; M.at<double>(25,22)=u[3]; M.at<double>(25,24)=u[4]; M.at<double>(25,25)=u[1]; M.at<double>(25,29)=a[4]; M.at<double>(25,30)=a[19]; M.at<double>(25,32)=a[15]; M.at<double>(25,33)=a[18]; M.at<double>(25,38)=a[5]; M.at<double>(25,39)=a[17]; M.at<double>(25,40)=a[11]; M.at<double>(25,43)=a[2]; M.at<double>(25,44)=a[9]; M.at<double>(25,45)=a[13]; M.at<double>(25,47)=a[3]; M.at<double>(25,52)=a[10]; M.at<double>(25,53)=a[6]; M.at<double>(25,59)=b[18]; M.at<double>(25,60)=b[3]; M.at<double>(25,61)=b[15]; M.at<double>(25,63)=b[13]; M.at<double>(25,68)=b[9]; M.at<double>(25,71)=b[2]; M.at<double>(25,72)=b[11]; M.at<double>(25,73)=b[10]; M.at<double>(25,74)=b[4]; M.at<double>(25,75)=b[19]; M.at<double>(25,78)=b[5]; M.at<double>(25,79)=b[17]; M.at<double>(25,82)=b[6]; M.at<double>(25,87)=c[6]; M.at<double>(25,90)=c[3]; M.at<double>(25,92)=c[5]; M.at<double>(25,93)=c[10]; M.at<double>(25,94)=c[4]; M.at<double>(25,95)=c[11]; M.at<double>(25,96)=c[17]; M.at<double>(25,97)=c[9]; M.at<double>(25,100)=c[2]; M.at<double>(25,107)=c[19]; M.at<double>(25,112)=c[13]; M.at<double>(25,116)=c[15]; M.at<double>(25,117)=c[18];
    M.at<double>(26,17)=u[2]; M.at<double>(26,23)=u[3]; M.at<double>(26,25)=u[4]; M.at<double>(26,26)=u[1]; M.at<double>(26,27)=a[15]; M.at<double>(26,28)=a[18]; M.at<double>(26,29)=a[8]; M.at<double>(26,32)=a[6]; M.at<double>(26,38)=a[12]; M.at<double>(26,40)=a[5]; M.at<double>(26,41)=a[2]; M.at<double>(26,42)=a[3]; M.at<double>(26,44)=a[4]; M.at<double>(26,45)=a[19]; M.at<double>(26,46)=a[13]; M.at<double>(26,47)=a[17]; M.at<double>(26,52)=a[14]; M.at<double>(26,55)=b[15]; M.at<double>(26,56)=b[3]; M.at<double>(26,57)=b[18]; M.at<double>(26,60)=b[17]; M.at<double>(26,61)=b[6]; M.at<double>(26,62)=b[2]; M.at<double>(26,63)=b[19]; M.at<double>(26,66)=b[13]; M.at<double>(26,68)=b[4]; M.at<double>(26,72)=b[5]; M.at<double>(26,73)=b[14]; M.at<double>(26,74)=b[8]; M.at<double>(26,78)=b[12]; M.at<double>(26,90)=c[17]; M.at<double>(26,92)=c[12]; M.at<double>(26,93)=c[14]; M.at<double>(26,94)=c[8]; M.at<double>(26,95)=c[5]; M.at<double>(26,97)=c[4]; M.at<double>(26,102)=c[13]; M.at<double>(26,106)=c[3]; M.at<double>(26,111)=c[2]; M.at<double>(26,112)=c[19]; M.at<double>(26,113)=c[15]; M.at<double>(26,114)=c[18]; M.at<double>(26,116)=c[6];
    M.at<double>(27,15)=u[3]; M.at<double>(27,29)=a[11]; M.at<double>(27,30)=a[3]; M.at<double>(27,36)=a[7]; M.at<double>(27,39)=a[15]; M.at<double>(27,49)=a[18]; M.at<double>(27,69)=b[9]; M.at<double>(27,70)=b[1]; M.at<double>(27,74)=b[11]; M.at<double>(27,75)=b[3]; M.at<double>(27,76)=b[18]; M.at<double>(27,79)=b[15]; M.at<double>(27,84)=b[7]; M.at<double>(27,88)=c[7]; M.at<double>(27,91)=c[1]; M.at<double>(27,94)=c[11]; M.at<double>(27,96)=c[15]; M.at<double>(27,107)=c[3]; M.at<double>(27,110)=c[9]; M.at<double>(27,119)=c[18];
    M.at<double>(28,6)=u[3]; M.at<double>(28,30)=a[11]; M.at<double>(28,35)=a[7]; M.at<double>(28,49)=a[15]; M.at<double>(28,69)=b[1]; M.at<double>(28,75)=b[11]; M.at<double>(28,76)=b[15]; M.at<double>(28,83)=b[7]; M.at<double>(28,107)=c[11]; M.at<double>(28,110)=c[1]; M.at<double>(28,118)=c[7]; M.at<double>(28,119)=c[15];
    M.at<double>(29,24)=u[3]; M.at<double>(29,29)=a[3]; M.at<double>(29,30)=a[2]; M.at<double>(29,38)=a[15]; M.at<double>(29,39)=a[18]; M.at<double>(29,52)=a[11]; M.at<double>(29,53)=a[7]; M.at<double>(29,69)=b[13]; M.at<double>(29,70)=b[9]; M.at<double>(29,73)=b[11]; M.at<double>(29,74)=b[3]; M.at<double>(29,75)=b[2]; M.at<double>(29,78)=b[15]; M.at<double>(29,79)=b[18]; M.at<double>(29,82)=b[7]; M.at<double>(29,87)=c[7]; M.at<double>(29,91)=c[9]; M.at<double>(29,92)=c[15]; M.at<double>(29,93)=c[11]; M.at<double>(29,94)=c[3]; M.at<double>(29,96)=c[18]; M.at<double>(29,107)=c[2]; M.at<double>(29,110)=c[13];
    M.at<double>(30,37)=a[18]; M.at<double>(30,48)=a[7]; M.at<double>(30,52)=a[2]; M.at<double>(30,70)=b[20]; M.at<double>(30,73)=b[2]; M.at<double>(30,77)=b[18]; M.at<double>(30,81)=b[7]; M.at<double>(30,85)=c[7]; M.at<double>(30,91)=c[20]; M.at<double>(30,93)=c[2]; M.at<double>(30,98)=c[18];
    M.at<double>(31,29)=a[2]; M.at<double>(31,37)=a[15]; M.at<double>(31,38)=a[18]; M.at<double>(31,50)=a[7]; M.at<double>(31,52)=a[3]; M.at<double>(31,69)=b[20]; M.at<double>(31,70)=b[13]; M.at<double>(31,73)=b[3]; M.at<double>(31,74)=b[2]; M.at<double>(31,77)=b[15]; M.at<double>(31,78)=b[18]; M.at<double>(31,80)=b[7]; M.at<double>(31,86)=c[7]; M.at<double>(31,91)=c[13]; M.at<double>(31,92)=c[18]; M.at<double>(31,93)=c[3]; M.at<double>(31,94)=c[2]; M.at<double>(31,98)=c[15]; M.at<double>(31,110)=c[20];
    M.at<double>(32,48)=a[9]; M.at<double>(32,50)=a[13]; M.at<double>(32,53)=a[20]; M.at<double>(32,80)=b[13]; M.at<double>(32,81)=b[9]; M.at<double>(32,82)=b[20]; M.at<double>(32,85)=c[9]; M.at<double>(32,86)=c[13]; M.at<double>(32,87)=c[20];
    M.at<double>(33,29)=a[15]; M.at<double>(33,30)=a[18]; M.at<double>(33,39)=a[7]; M.at<double>(33,64)=b[9]; M.at<double>(33,69)=b[3]; M.at<double>(33,70)=b[11]; M.at<double>(33,74)=b[15]; M.at<double>(33,75)=b[18]; M.at<double>(33,79)=b[7]; M.at<double>(33,91)=c[11]; M.at<double>(33,94)=c[15]; M.at<double>(33,96)=c[7]; M.at<double>(33,103)=c[9]; M.at<double>(33,107)=c[18]; M.at<double>(33,110)=c[3];
    M.at<double>(34,29)=a[18]; M.at<double>(34,38)=a[7]; M.at<double>(34,52)=a[15]; M.at<double>(34,64)=b[13]; M.at<double>(34,69)=b[2]; M.at<double>(34,70)=b[3]; M.at<double>(34,73)=b[15]; M.at<double>(34,74)=b[18]; M.at<double>(34,78)=b[7]; M.at<double>(34,91)=c[3]; M.at<double>(34,92)=c[7]; M.at<double>(34,93)=c[15]; M.at<double>(34,94)=c[18]; M.at<double>(34,103)=c[13]; M.at<double>(34,110)=c[2];
    M.at<double>(35,37)=a[7]; M.at<double>(35,52)=a[18]; M.at<double>(35,64)=b[20]; M.at<double>(35,70)=b[2]; M.at<double>(35,73)=b[18]; M.at<double>(35,77)=b[7]; M.at<double>(35,91)=c[2]; M.at<double>(35,93)=c[18]; M.at<double>(35,98)=c[7]; M.at<double>(35,103)=c[20];
    M.at<double>(36,5)=u[4]; M.at<double>(36,34)=a[12]; M.at<double>(36,41)=a[10]; M.at<double>(36,43)=a[14]; M.at<double>(36,49)=a[16]; M.at<double>(36,51)=a[5]; M.at<double>(36,54)=b[12]; M.at<double>(36,62)=b[10]; M.at<double>(36,65)=b[5]; M.at<double>(36,71)=b[14]; M.at<double>(36,76)=b[16]; M.at<double>(36,89)=c[5]; M.at<double>(36,99)=c[12]; M.at<double>(36,100)=c[14]; M.at<double>(36,101)=c[1]; M.at<double>(36,109)=c[11]; M.at<double>(36,111)=c[10]; M.at<double>(36,119)=c[16];
    M.at<double>(37,2)=u[4]; M.at<double>(37,34)=a[14]; M.at<double>(37,35)=a[16]; M.at<double>(37,51)=a[10]; M.at<double>(37,54)=b[14]; M.at<double>(37,65)=b[10]; M.at<double>(37,83)=b[16]; M.at<double>(37,89)=c[10]; M.at<double>(37,99)=c[14]; M.at<double>(37,109)=c[1]; M.at<double>(37,118)=c[16];
    M.at<double>(38,30)=a[15]; M.at<double>(38,49)=a[7]; M.at<double>(38,64)=b[1]; M.at<double>(38,69)=b[11]; M.at<double>(38,75)=b[15]; M.at<double>(38,76)=b[7]; M.at<double>(38,103)=c[1]; M.at<double>(38,107)=c[15]; M.at<double>(38,110)=c[11]; M.at<double>(38,119)=c[7];
    M.at<double>(39,28)=a[14]; M.at<double>(39,33)=a[16]; M.at<double>(39,51)=a[8]; M.at<double>(39,57)=b[14]; M.at<double>(39,59)=b[16]; M.at<double>(39,65)=b[8]; M.at<double>(39,89)=c[8]; M.at<double>(39,105)=c[9]; M.at<double>(39,108)=c[10]; M.at<double>(39,109)=c[4]; M.at<double>(39,114)=c[14]; M.at<double>(39,117)=c[16];
    M.at<double>(40,27)=a[14]; M.at<double>(40,28)=a[8]; M.at<double>(40,32)=a[16]; M.at<double>(40,55)=b[14]; M.at<double>(40,57)=b[8]; M.at<double>(40,61)=b[16]; M.at<double>(40,105)=c[13]; M.at<double>(40,108)=c[4]; M.at<double>(40,109)=c[19]; M.at<double>(40,113)=c[14]; M.at<double>(40,114)=c[8]; M.at<double>(40,116)=c[16];
    M.at<double>(41,30)=a[7]; M.at<double>(41,64)=b[11]; M.at<double>(41,69)=b[15]; M.at<double>(41,75)=b[7]; M.at<double>(41,103)=c[11]; M.at<double>(41,107)=c[7]; M.at<double>(41,110)=c[15];
    M.at<double>(42,27)=a[8]; M.at<double>(42,31)=a[16]; M.at<double>(42,55)=b[8]; M.at<double>(42,58)=b[16]; M.at<double>(42,105)=c[20]; M.at<double>(42,108)=c[19]; M.at<double>(42,113)=c[8]; M.at<double>(42,115)=c[16];
    M.at<double>(43,29)=a[7]; M.at<double>(43,64)=b[3]; M.at<double>(43,69)=b[18]; M.at<double>(43,70)=b[15]; M.at<double>(43,74)=b[7]; M.at<double>(43,91)=c[15]; M.at<double>(43,94)=c[7]; M.at<double>(43,103)=c[3]; M.at<double>(43,110)=c[18];
    M.at<double>(44,28)=a[16]; M.at<double>(44,57)=b[16]; M.at<double>(44,105)=c[4]; M.at<double>(44,108)=c[14]; M.at<double>(44,109)=c[8]; M.at<double>(44,114)=c[16];
    M.at<double>(45,27)=a[16]; M.at<double>(45,55)=b[16]; M.at<double>(45,105)=c[19]; M.at<double>(45,108)=c[8]; M.at<double>(45,113)=c[16];
    M.at<double>(46,52)=a[7]; M.at<double>(46,64)=b[2]; M.at<double>(46,70)=b[18]; M.at<double>(46,73)=b[7]; M.at<double>(46,91)=c[18]; M.at<double>(46,93)=c[7]; M.at<double>(46,103)=c[2];
    M.at<double>(47,40)=a[7]; M.at<double>(47,44)=a[18]; M.at<double>(47,52)=a[6]; M.at<double>(47,64)=b[19]; M.at<double>(47,67)=b[2]; M.at<double>(47,68)=b[18]; M.at<double>(47,70)=b[17]; M.at<double>(47,72)=b[7]; M.at<double>(47,73)=b[6]; M.at<double>(47,91)=c[17]; M.at<double>(47,93)=c[6]; M.at<double>(47,95)=c[7]; M.at<double>(47,97)=c[18]; M.at<double>(47,103)=c[19]; M.at<double>(47,104)=c[2];
    M.at<double>(48,30)=a[6]; M.at<double>(48,43)=a[7]; M.at<double>(48,45)=a[15]; M.at<double>(48,63)=b[15]; M.at<double>(48,64)=b[10]; M.at<double>(48,67)=b[11]; M.at<double>(48,69)=b[5]; M.at<double>(48,71)=b[7]; M.at<double>(48,75)=b[6]; M.at<double>(48,100)=c[7]; M.at<double>(48,103)=c[10]; M.at<double>(48,104)=c[11]; M.at<double>(48,107)=c[6]; M.at<double>(48,110)=c[5]; M.at<double>(48,112)=c[15];
    M.at<double>(49,41)=a[12]; M.at<double>(49,45)=a[16]; M.at<double>(49,46)=a[14]; M.at<double>(49,62)=b[12]; M.at<double>(49,63)=b[16]; M.at<double>(49,66)=b[14]; M.at<double>(49,101)=c[5]; M.at<double>(49,102)=c[14]; M.at<double>(49,105)=c[15]; M.at<double>(49,109)=c[6]; M.at<double>(49,111)=c[12]; M.at<double>(49,112)=c[16];
    M.at<double>(50,41)=a[16]; M.at<double>(50,62)=b[16]; M.at<double>(50,101)=c[14]; M.at<double>(50,105)=c[5]; M.at<double>(50,109)=c[12]; M.at<double>(50,111)=c[16];
    M.at<double>(51,64)=b[18]; M.at<double>(51,70)=b[7]; M.at<double>(51,91)=c[7]; M.at<double>(51,103)=c[18];
    M.at<double>(52,41)=a[6]; M.at<double>(52,45)=a[12]; M.at<double>(52,46)=a[5]; M.at<double>(52,62)=b[6]; M.at<double>(52,63)=b[12]; M.at<double>(52,66)=b[5]; M.at<double>(52,67)=b[14]; M.at<double>(52,69)=b[16]; M.at<double>(52,101)=c[15]; M.at<double>(52,102)=c[5]; M.at<double>(52,104)=c[14]; M.at<double>(52,109)=c[7]; M.at<double>(52,110)=c[16]; M.at<double>(52,111)=c[6]; M.at<double>(52,112)=c[12];
    M.at<double>(53,64)=b[15]; M.at<double>(53,69)=b[7]; M.at<double>(53,103)=c[15]; M.at<double>(53,110)=c[7];
    M.at<double>(54,105)=c[14]; M.at<double>(54,109)=c[16];
    M.at<double>(55,44)=a[7]; M.at<double>(55,64)=b[17]; M.at<double>(55,67)=b[18]; M.at<double>(55,68)=b[7]; M.at<double>(55,70)=b[6]; M.at<double>(55,91)=c[6]; M.at<double>(55,97)=c[7]; M.at<double>(55,103)=c[17]; M.at<double>(55,104)=c[18];
    M.at<double>(56,105)=c[8]; M.at<double>(56,108)=c[16];
    M.at<double>(57,64)=b[6]; M.at<double>(57,67)=b[7]; M.at<double>(57,103)=c[6]; M.at<double>(57,104)=c[7];
    M.at<double>(58,46)=a[7]; M.at<double>(58,64)=b[12]; M.at<double>(58,66)=b[7]; M.at<double>(58,67)=b[6]; M.at<double>(58,102)=c[7]; M.at<double>(58,103)=c[12]; M.at<double>(58,104)=c[6];
    M.at<double>(59,8)=u[4]; M.at<double>(59,30)=a[16]; M.at<double>(59,41)=a[5]; M.at<double>(59,43)=a[12]; M.at<double>(59,45)=a[14]; M.at<double>(59,46)=a[10]; M.at<double>(59,51)=a[6]; M.at<double>(59,62)=b[5]; M.at<double>(59,63)=b[14]; M.at<double>(59,65)=b[6]; M.at<double>(59,66)=b[10]; M.at<double>(59,71)=b[12]; M.at<double>(59,75)=b[16]; M.at<double>(59,89)=c[6]; M.at<double>(59,100)=c[12]; M.at<double>(59,101)=c[11]; M.at<double>(59,102)=c[10]; M.at<double>(59,107)=c[16]; M.at<double>(59,109)=c[15]; M.at<double>(59,111)=c[5]; M.at<double>(59,112)=c[14];
    M.at<double>(60,8)=u[3]; M.at<double>(60,30)=a[12]; M.at<double>(60,41)=a[15]; M.at<double>(60,43)=a[6]; M.at<double>(60,45)=a[5]; M.at<double>(60,46)=a[11]; M.at<double>(60,51)=a[7]; M.at<double>(60,62)=b[15]; M.at<double>(60,63)=b[5]; M.at<double>(60,65)=b[7]; M.at<double>(60,66)=b[11]; M.at<double>(60,67)=b[10]; M.at<double>(60,69)=b[14]; M.at<double>(60,71)=b[6]; M.at<double>(60,75)=b[12]; M.at<double>(60,89)=c[7]; M.at<double>(60,100)=c[6]; M.at<double>(60,102)=c[11]; M.at<double>(60,104)=c[10]; M.at<double>(60,107)=c[12]; M.at<double>(60,110)=c[14]; M.at<double>(60,111)=c[15]; M.at<double>(60,112)=c[5];
    M.at<double>(61,42)=a[16]; M.at<double>(61,56)=b[16]; M.at<double>(61,101)=c[8]; M.at<double>(61,105)=c[17]; M.at<double>(61,106)=c[16]; M.at<double>(61,108)=c[12];
    M.at<double>(62,64)=b[7]; M.at<double>(62,103)=c[7];
    M.at<double>(63,105)=c[16];
    M.at<double>(64,46)=a[12]; M.at<double>(64,66)=b[12]; M.at<double>(64,67)=b[16]; M.at<double>(64,101)=c[6]; M.at<double>(64,102)=c[12]; M.at<double>(64,104)=c[16]; M.at<double>(64,105)=c[7];
    M.at<double>(65,46)=a[6]; M.at<double>(65,64)=b[16]; M.at<double>(65,66)=b[6]; M.at<double>(65,67)=b[12]; M.at<double>(65,101)=c[7]; M.at<double>(65,102)=c[6]; M.at<double>(65,103)=c[16]; M.at<double>(65,104)=c[12];
    M.at<double>(66,46)=a[16]; M.at<double>(66,66)=b[16]; M.at<double>(66,101)=c[12]; M.at<double>(66,102)=c[16]; M.at<double>(66,105)=c[6];
    M.at<double>(67,101)=c[16]; M.at<double>(67,105)=c[12];
    M.at<double>(68,41)=a[14]; M.at<double>(68,43)=a[16]; M.at<double>(68,51)=a[12]; M.at<double>(68,62)=b[14]; M.at<double>(68,65)=b[12]; M.at<double>(68,71)=b[16]; M.at<double>(68,89)=c[12]; M.at<double>(68,100)=c[16]; M.at<double>(68,101)=c[10]; M.at<double>(68,105)=c[11]; M.at<double>(68,109)=c[5]; M.at<double>(68,111)=c[14];
    M.at<double>(69,37)=a[2]; M.at<double>(69,48)=a[18]; M.at<double>(69,52)=a[20]; M.at<double>(69,73)=b[20]; M.at<double>(69,77)=b[2]; M.at<double>(69,81)=b[18]; M.at<double>(69,85)=c[18]; M.at<double>(69,93)=c[20]; M.at<double>(69,98)=c[2];
    M.at<double>(70,20)=u[2]; M.at<double>(70,27)=a[9]; M.at<double>(70,28)=a[13]; M.at<double>(70,31)=a[10]; M.at<double>(70,32)=a[4]; M.at<double>(70,33)=a[19]; M.at<double>(70,50)=a[14]; M.at<double>(70,51)=a[20]; M.at<double>(70,53)=a[8]; M.at<double>(70,55)=b[9]; M.at<double>(70,57)=b[13]; M.at<double>(70,58)=b[10]; M.at<double>(70,59)=b[19]; M.at<double>(70,61)=b[4]; M.at<double>(70,65)=b[20]; M.at<double>(70,80)=b[14]; M.at<double>(70,82)=b[8]; M.at<double>(70,86)=c[14]; M.at<double>(70,87)=c[8]; M.at<double>(70,89)=c[20]; M.at<double>(70,113)=c[9]; M.at<double>(70,114)=c[13]; M.at<double>(70,115)=c[10]; M.at<double>(70,116)=c[4]; M.at<double>(70,117)=c[19];
    M.at<double>(71,45)=a[7]; M.at<double>(71,63)=b[7]; M.at<double>(71,64)=b[5]; M.at<double>(71,67)=b[15]; M.at<double>(71,69)=b[6]; M.at<double>(71,103)=c[5]; M.at<double>(71,104)=c[15]; M.at<double>(71,110)=c[6]; M.at<double>(71,112)=c[7];
    M.at<double>(72,41)=a[7]; M.at<double>(72,45)=a[6]; M.at<double>(72,46)=a[15]; M.at<double>(72,62)=b[7]; M.at<double>(72,63)=b[6]; M.at<double>(72,64)=b[14]; M.at<double>(72,66)=b[15]; M.at<double>(72,67)=b[5]; M.at<double>(72,69)=b[12]; M.at<double>(72,102)=c[15]; M.at<double>(72,103)=c[14]; M.at<double>(72,104)=c[5]; M.at<double>(72,110)=c[12]; M.at<double>(72,111)=c[7]; M.at<double>(72,112)=c[6];
    M.at<double>(73,48)=a[13]; M.at<double>(73,50)=a[20]; M.at<double>(73,80)=b[20]; M.at<double>(73,81)=b[13]; M.at<double>(73,85)=c[13]; M.at<double>(73,86)=c[20];
    M.at<double>(74,25)=u[3]; M.at<double>(74,29)=a[17]; M.at<double>(74,32)=a[7]; M.at<double>(74,38)=a[6]; M.at<double>(74,40)=a[15]; M.at<double>(74,44)=a[3]; M.at<double>(74,45)=a[2]; M.at<double>(74,47)=a[18]; M.at<double>(74,52)=a[5]; M.at<double>(74,60)=b[18]; M.at<double>(74,61)=b[7]; M.at<double>(74,63)=b[2]; M.at<double>(74,67)=b[13]; M.at<double>(74,68)=b[3]; M.at<double>(74,69)=b[19]; M.at<double>(74,70)=b[4]; M.at<double>(74,72)=b[15]; M.at<double>(74,73)=b[5]; M.at<double>(74,74)=b[17]; M.at<double>(74,78)=b[6]; M.at<double>(74,90)=c[18]; M.at<double>(74,91)=c[4]; M.at<double>(74,92)=c[6]; M.at<double>(74,93)=c[5]; M.at<double>(74,94)=c[17]; M.at<double>(74,95)=c[15]; M.at<double>(74,97)=c[3]; M.at<double>(74,104)=c[13]; M.at<double>(74,110)=c[19]; M.at<double>(74,112)=c[2]; M.at<double>(74,116)=c[7];
    M.at<double>(75,21)=u[2]; M.at<double>(75,36)=a[2]; M.at<double>(75,37)=a[1]; M.at<double>(75,38)=a[9]; M.at<double>(75,39)=a[13]; M.at<double>(75,49)=a[20]; M.at<double>(75,50)=a[11]; M.at<double>(75,53)=a[3]; M.at<double>(75,76)=b[20]; M.at<double>(75,77)=b[1]; M.at<double>(75,78)=b[9]; M.at<double>(75,79)=b[13]; M.at<double>(75,80)=b[11]; M.at<double>(75,82)=b[3]; M.at<double>(75,84)=b[2]; M.at<double>(75,86)=c[11]; M.at<double>(75,87)=c[3]; M.at<double>(75,88)=c[2]; M.at<double>(75,92)=c[9]; M.at<double>(75,96)=c[13]; M.at<double>(75,98)=c[1]; M.at<double>(75,119)=c[20];
    M.at<double>(76,48)=a[20]; M.at<double>(76,81)=b[20]; M.at<double>(76,85)=c[20];
    M.at<double>(77,34)=a[16]; M.at<double>(77,51)=a[14]; M.at<double>(77,54)=b[16]; M.at<double>(77,65)=b[14]; M.at<double>(77,89)=c[14]; M.at<double>(77,99)=c[16]; M.at<double>(77,105)=c[1]; M.at<double>(77,109)=c[10];
    M.at<double>(78,27)=a[17]; M.at<double>(78,31)=a[12]; M.at<double>(78,37)=a[16]; M.at<double>(78,40)=a[8]; M.at<double>(78,42)=a[19]; M.at<double>(78,55)=b[17]; M.at<double>(78,56)=b[19]; M.at<double>(78,58)=b[12]; M.at<double>(78,72)=b[8]; M.at<double>(78,77)=b[16]; M.at<double>(78,95)=c[8]; M.at<double>(78,98)=c[16]; M.at<double>(78,101)=c[20]; M.at<double>(78,106)=c[19]; M.at<double>(78,108)=c[2]; M.at<double>(78,113)=c[17]; M.at<double>(78,115)=c[12];
    M.at<double>(79,42)=a[12]; M.at<double>(79,44)=a[16]; M.at<double>(79,46)=a[8]; M.at<double>(79,56)=b[12]; M.at<double>(79,66)=b[8]; M.at<double>(79,68)=b[16]; M.at<double>(79,97)=c[16]; M.at<double>(79,101)=c[17]; M.at<double>(79,102)=c[8]; M.at<double>(79,105)=c[18]; M.at<double>(79,106)=c[12]; M.at<double>(79,108)=c[6];
    M.at<double>(80,14)=u[4]; M.at<double>(80,28)=a[5]; M.at<double>(80,33)=a[12]; M.at<double>(80,39)=a[16]; M.at<double>(80,41)=a[4]; M.at<double>(80,42)=a[10]; M.at<double>(80,43)=a[8]; M.at<double>(80,47)=a[14]; M.at<double>(80,51)=a[17]; M.at<double>(80,56)=b[10]; M.at<double>(80,57)=b[5]; M.at<double>(80,59)=b[12]; M.at<double>(80,60)=b[14]; M.at<double>(80,62)=b[4]; M.at<double>(80,65)=b[17]; M.at<double>(80,71)=b[8]; M.at<double>(80,79)=b[16]; M.at<double>(80,89)=c[17]; M.at<double>(80,90)=c[14]; M.at<double>(80,96)=c[16]; M.at<double>(80,100)=c[8]; M.at<double>(80,101)=c[9]; M.at<double>(80,106)=c[10]; M.at<double>(80,108)=c[11]; M.at<double>(80,109)=c[3]; M.at<double>(80,111)=c[4]; M.at<double>(80,114)=c[5]; M.at<double>(80,117)=c[12];
    M.at<double>(81,31)=a[3]; M.at<double>(81,32)=a[2]; M.at<double>(81,37)=a[4]; M.at<double>(81,38)=a[19]; M.at<double>(81,40)=a[13]; M.at<double>(81,47)=a[20]; M.at<double>(81,48)=a[5]; M.at<double>(81,50)=a[17]; M.at<double>(81,58)=b[3]; M.at<double>(81,60)=b[20]; M.at<double>(81,61)=b[2]; M.at<double>(81,72)=b[13]; M.at<double>(81,77)=b[4]; M.at<double>(81,78)=b[19]; M.at<double>(81,80)=b[17]; M.at<double>(81,81)=b[5]; M.at<double>(81,85)=c[5]; M.at<double>(81,86)=c[17]; M.at<double>(81,90)=c[20]; M.at<double>(81,92)=c[19]; M.at<double>(81,95)=c[13]; M.at<double>(81,98)=c[4]; M.at<double>(81,115)=c[3]; M.at<double>(81,116)=c[2];
    M.at<double>(82,29)=a[6]; M.at<double>(82,44)=a[15]; M.at<double>(82,45)=a[18]; M.at<double>(82,47)=a[7]; M.at<double>(82,60)=b[7]; M.at<double>(82,63)=b[18]; M.at<double>(82,64)=b[4]; M.at<double>(82,67)=b[3]; M.at<double>(82,68)=b[15]; M.at<double>(82,69)=b[17]; M.at<double>(82,70)=b[5]; M.at<double>(82,74)=b[6]; M.at<double>(82,90)=c[7]; M.at<double>(82,91)=c[5]; M.at<double>(82,94)=c[6]; M.at<double>(82,97)=c[15]; M.at<double>(82,103)=c[4]; M.at<double>(82,104)=c[3]; M.at<double>(82,110)=c[17]; M.at<double>(82,112)=c[18];
    M.at<double>(83,26)=u[2]; M.at<double>(83,27)=a[18]; M.at<double>(83,31)=a[6]; M.at<double>(83,37)=a[12]; M.at<double>(83,40)=a[17]; M.at<double>(83,42)=a[2]; M.at<double>(83,44)=a[19]; M.at<double>(83,46)=a[20]; M.at<double>(83,52)=a[8]; M.at<double>(83,55)=b[18]; M.at<double>(83,56)=b[2]; M.at<double>(83,58)=b[6]; M.at<double>(83,66)=b[20]; M.at<double>(83,68)=b[19]; M.at<double>(83,72)=b[17]; M.at<double>(83,73)=b[8]; M.at<double>(83,77)=b[12]; M.at<double>(83,93)=c[8]; M.at<double>(83,95)=c[17]; M.at<double>(83,97)=c[19]; M.at<double>(83,98)=c[12]; M.at<double>(83,102)=c[20]; M.at<double>(83,106)=c[2]; M.at<double>(83,113)=c[18]; M.at<double>(83,115)=c[6];
    M.at<double>(84,16)=u[3]; M.at<double>(84,29)=a[5]; M.at<double>(84,30)=a[17]; M.at<double>(84,33)=a[7]; M.at<double>(84,39)=a[6]; M.at<double>(84,43)=a[18]; M.at<double>(84,44)=a[11]; M.at<double>(84,45)=a[3]; M.at<double>(84,47)=a[15]; M.at<double>(84,59)=b[7]; M.at<double>(84,60)=b[15]; M.at<double>(84,63)=b[3]; M.at<double>(84,67)=b[9]; M.at<double>(84,68)=b[11]; M.at<double>(84,69)=b[4]; M.at<double>(84,70)=b[10]; M.at<double>(84,71)=b[18]; M.at<double>(84,74)=b[5]; M.at<double>(84,75)=b[17]; M.at<double>(84,79)=b[6]; M.at<double>(84,90)=c[15]; M.at<double>(84,91)=c[10]; M.at<double>(84,94)=c[5]; M.at<double>(84,96)=c[6]; M.at<double>(84,97)=c[11]; M.at<double>(84,100)=c[18]; M.at<double>(84,104)=c[9]; M.at<double>(84,107)=c[17]; M.at<double>(84,110)=c[4]; M.at<double>(84,112)=c[3]; M.at<double>(84,117)=c[7];
    M.at<double>(85,25)=u[2]; M.at<double>(85,29)=a[19]; M.at<double>(85,31)=a[15]; M.at<double>(85,32)=a[18]; M.at<double>(85,37)=a[5]; M.at<double>(85,38)=a[17]; M.at<double>(85,40)=a[3]; M.at<double>(85,44)=a[13]; M.at<double>(85,45)=a[20]; M.at<double>(85,47)=a[2]; M.at<double>(85,50)=a[6]; M.at<double>(85,52)=a[4]; M.at<double>(85,58)=b[15]; M.at<double>(85,60)=b[2]; M.at<double>(85,61)=b[18]; M.at<double>(85,63)=b[20]; M.at<double>(85,68)=b[13]; M.at<double>(85,72)=b[3]; M.at<double>(85,73)=b[4]; M.at<double>(85,74)=b[19]; M.at<double>(85,77)=b[5]; M.at<double>(85,78)=b[17]; M.at<double>(85,80)=b[6]; M.at<double>(85,86)=c[6]; M.at<double>(85,90)=c[2]; M.at<double>(85,92)=c[17]; M.at<double>(85,93)=c[4]; M.at<double>(85,94)=c[19]; M.at<double>(85,95)=c[3]; M.at<double>(85,97)=c[13]; M.at<double>(85,98)=c[5]; M.at<double>(85,112)=c[20]; M.at<double>(85,115)=c[15]; M.at<double>(85,116)=c[18];
    M.at<double>(86,31)=a[18]; M.at<double>(86,37)=a[17]; M.at<double>(86,40)=a[2]; M.at<double>(86,44)=a[20]; M.at<double>(86,48)=a[6]; M.at<double>(86,52)=a[19]; M.at<double>(86,58)=b[18]; M.at<double>(86,68)=b[20]; M.at<double>(86,72)=b[2]; M.at<double>(86,73)=b[19]; M.at<double>(86,77)=b[17]; M.at<double>(86,81)=b[6]; M.at<double>(86,85)=c[6]; M.at<double>(86,93)=c[19]; M.at<double>(86,95)=c[2]; M.at<double>(86,97)=c[20]; M.at<double>(86,98)=c[17]; M.at<double>(86,115)=c[18];
    M.at<double>(87,22)=u[2]; M.at<double>(87,31)=a[11]; M.at<double>(87,32)=a[3]; M.at<double>(87,33)=a[2]; M.at<double>(87,37)=a[10]; M.at<double>(87,38)=a[4]; M.at<double>(87,39)=a[19]; M.at<double>(87,40)=a[9]; M.at<double>(87,43)=a[20]; M.at<double>(87,47)=a[13]; M.at<double>(87,50)=a[5]; M.at<double>(87,53)=a[17]; M.at<double>(87,58)=b[11]; M.at<double>(87,59)=b[2]; M.at<double>(87,60)=b[13]; M.at<double>(87,61)=b[3]; M.at<double>(87,71)=b[20]; M.at<double>(87,72)=b[9]; M.at<double>(87,77)=b[10]; M.at<double>(87,78)=b[4]; M.at<double>(87,79)=b[19]; M.at<double>(87,80)=b[5]; M.at<double>(87,82)=b[17]; M.at<double>(87,86)=c[5]; M.at<double>(87,87)=c[17]; M.at<double>(87,90)=c[13]; M.at<double>(87,92)=c[4]; M.at<double>(87,95)=c[9]; M.at<double>(87,96)=c[19]; M.at<double>(87,98)=c[10]; M.at<double>(87,100)=c[20]; M.at<double>(87,115)=c[11]; M.at<double>(87,116)=c[3]; M.at<double>(87,117)=c[2];
    M.at<double>(88,27)=a[2]; M.at<double>(88,31)=a[17]; M.at<double>(88,37)=a[8]; M.at<double>(88,40)=a[19]; M.at<double>(88,42)=a[20]; M.at<double>(88,48)=a[12]; M.at<double>(88,55)=b[2]; M.at<double>(88,56)=b[20]; M.at<double>(88,58)=b[17]; M.at<double>(88,72)=b[19]; M.at<double>(88,77)=b[8]; M.at<double>(88,81)=b[12]; M.at<double>(88,85)=c[12]; M.at<double>(88,95)=c[19]; M.at<double>(88,98)=c[8]; M.at<double>(88,106)=c[20]; M.at<double>(88,113)=c[2]; M.at<double>(88,115)=c[17];
    M.at<double>(89,31)=a[7]; M.at<double>(89,37)=a[6]; M.at<double>(89,40)=a[18]; M.at<double>(89,44)=a[2]; M.at<double>(89,52)=a[17]; M.at<double>(89,58)=b[7]; M.at<double>(89,67)=b[20]; M.at<double>(89,68)=b[2]; M.at<double>(89,70)=b[19]; M.at<double>(89,72)=b[18]; M.at<double>(89,73)=b[17]; M.at<double>(89,77)=b[6]; M.at<double>(89,91)=c[19]; M.at<double>(89,93)=c[17]; M.at<double>(89,95)=c[18]; M.at<double>(89,97)=c[2]; M.at<double>(89,98)=c[6]; M.at<double>(89,104)=c[20]; M.at<double>(89,115)=c[7];
    M.at<double>(90,27)=a[12]; M.at<double>(90,40)=a[16]; M.at<double>(90,42)=a[8]; M.at<double>(90,55)=b[12]; M.at<double>(90,56)=b[8]; M.at<double>(90,72)=b[16]; M.at<double>(90,95)=c[16]; M.at<double>(90,101)=c[19]; M.at<double>(90,105)=c[2]; M.at<double>(90,106)=c[8]; M.at<double>(90,108)=c[17]; M.at<double>(90,113)=c[12];
    M.at<double>(91,23)=u[2]; M.at<double>(91,27)=a[3]; M.at<double>(91,28)=a[2]; M.at<double>(91,31)=a[5]; M.at<double>(91,32)=a[17]; M.at<double>(91,37)=a[14]; M.at<double>(91,38)=a[8]; M.at<double>(91,40)=a[4]; M.at<double>(91,41)=a[20]; M.at<double>(91,42)=a[13]; M.at<double>(91,47)=a[19]; M.at<double>(91,50)=a[12]; M.at<double>(91,55)=b[3]; M.at<double>(91,56)=b[13]; M.at<double>(91,57)=b[2]; M.at<double>(91,58)=b[5]; M.at<double>(91,60)=b[19]; M.at<double>(91,61)=b[17]; M.at<double>(91,62)=b[20]; M.at<double>(91,72)=b[4]; M.at<double>(91,77)=b[14]; M.at<double>(91,78)=b[8]; M.at<double>(91,80)=b[12]; M.at<double>(91,86)=c[12]; M.at<double>(91,90)=c[19]; M.at<double>(91,92)=c[8]; M.at<double>(91,95)=c[4]; M.at<double>(91,98)=c[14]; M.at<double>(91,106)=c[13]; M.at<double>(91,111)=c[20]; M.at<double>(91,113)=c[3]; M.at<double>(91,114)=c[2]; M.at<double>(91,115)=c[5]; M.at<double>(91,116)=c[17];
    M.at<double>(92,17)=u[4]; M.at<double>(92,28)=a[6]; M.at<double>(92,29)=a[16]; M.at<double>(92,41)=a[17]; M.at<double>(92,42)=a[5]; M.at<double>(92,44)=a[14]; M.at<double>(92,45)=a[8]; M.at<double>(92,46)=a[4]; M.at<double>(92,47)=a[12]; M.at<double>(92,56)=b[5]; M.at<double>(92,57)=b[6]; M.at<double>(92,60)=b[12]; M.at<double>(92,62)=b[17]; M.at<double>(92,63)=b[8]; M.at<double>(92,66)=b[4]; M.at<double>(92,68)=b[14]; M.at<double>(92,74)=b[16]; M.at<double>(92,90)=c[12]; M.at<double>(92,94)=c[16]; M.at<double>(92,97)=c[14]; M.at<double>(92,101)=c[3]; M.at<double>(92,102)=c[4]; M.at<double>(92,106)=c[5]; M.at<double>(92,108)=c[15]; M.at<double>(92,109)=c[18]; M.at<double>(92,111)=c[17]; M.at<double>(92,112)=c[8]; M.at<double>(92,114)=c[6];
    M.at<double>(93,17)=u[3]; M.at<double>(93,28)=a[7]; M.at<double>(93,29)=a[12]; M.at<double>(93,41)=a[18]; M.at<double>(93,42)=a[15]; M.at<double>(93,44)=a[5]; M.at<double>(93,45)=a[17]; M.at<double>(93,46)=a[3]; M.at<double>(93,47)=a[6]; M.at<double>(93,56)=b[15]; M.at<double>(93,57)=b[7]; M.at<double>(93,60)=b[6]; M.at<double>(93,62)=b[18]; M.at<double>(93,63)=b[17]; M.at<double>(93,66)=b[3]; M.at<double>(93,67)=b[4]; M.at<double>(93,68)=b[5]; M.at<double>(93,69)=b[8]; M.at<double>(93,70)=b[14]; M.at<double>(93,74)=b[12]; M.at<double>(93,90)=c[6]; M.at<double>(93,91)=c[14]; M.at<double>(93,94)=c[12]; M.at<double>(93,97)=c[5]; M.at<double>(93,102)=c[3]; M.at<double>(93,104)=c[4]; M.at<double>(93,106)=c[15]; M.at<double>(93,110)=c[8]; M.at<double>(93,111)=c[18]; M.at<double>(93,112)=c[17]; M.at<double>(93,114)=c[7];
    M.at<double>(94,31)=a[2]; M.at<double>(94,37)=a[19]; M.at<double>(94,40)=a[20]; M.at<double>(94,48)=a[17]; M.at<double>(94,58)=b[2]; M.at<double>(94,72)=b[20]; M.at<double>(94,77)=b[19]; M.at<double>(94,81)=b[17]; M.at<double>(94,85)=c[17]; M.at<double>(94,95)=c[20]; M.at<double>(94,98)=c[19]; M.at<double>(94,115)=c[2];
    M.at<double>(95,26)=u[4]; M.at<double>(95,27)=a[6]; M.at<double>(95,40)=a[12]; M.at<double>(95,42)=a[17]; M.at<double>(95,44)=a[8]; M.at<double>(95,46)=a[19]; M.at<double>(95,52)=a[16]; M.at<double>(95,55)=b[6]; M.at<double>(95,56)=b[17]; M.at<double>(95,66)=b[19]; M.at<double>(95,68)=b[8]; M.at<double>(95,72)=b[12]; M.at<double>(95,73)=b[16]; M.at<double>(95,93)=c[16]; M.at<double>(95,95)=c[12]; M.at<double>(95,97)=c[8]; M.at<double>(95,101)=c[2]; M.at<double>(95,102)=c[19]; M.at<double>(95,106)=c[17]; M.at<double>(95,108)=c[18]; M.at<double>(95,113)=c[6];
    M.at<double>(96,23)=u[4]; M.at<double>(96,27)=a[5]; M.at<double>(96,28)=a[17]; M.at<double>(96,32)=a[12]; M.at<double>(96,38)=a[16]; M.at<double>(96,40)=a[14]; M.at<double>(96,41)=a[19]; M.at<double>(96,42)=a[4]; M.at<double>(96,47)=a[8]; M.at<double>(96,55)=b[5]; M.at<double>(96,56)=b[4]; M.at<double>(96,57)=b[17]; M.at<double>(96,60)=b[8]; M.at<double>(96,61)=b[12]; M.at<double>(96,62)=b[19]; M.at<double>(96,72)=b[14]; M.at<double>(96,78)=b[16]; M.at<double>(96,90)=c[8]; M.at<double>(96,92)=c[16]; M.at<double>(96,95)=c[14]; M.at<double>(96,101)=c[13]; M.at<double>(96,106)=c[4]; M.at<double>(96,108)=c[3]; M.at<double>(96,109)=c[2]; M.at<double>(96,111)=c[19]; M.at<double>(96,113)=c[5]; M.at<double>(96,114)=c[17]; M.at<double>(96,116)=c[12];
    M.at<double>(97,42)=a[6]; M.at<double>(97,44)=a[12]; M.at<double>(97,46)=a[17]; M.at<double>(97,56)=b[6]; M.at<double>(97,66)=b[17]; M.at<double>(97,67)=b[8]; M.at<double>(97,68)=b[12]; M.at<double>(97,70)=b[16]; M.at<double>(97,91)=c[16]; M.at<double>(97,97)=c[12]; M.at<double>(97,101)=c[18]; M.at<double>(97,102)=c[17]; M.at<double>(97,104)=c[8]; M.at<double>(97,106)=c[6]; M.at<double>(97,108)=c[7];
    M.at<double>(98,28)=a[12]; M.at<double>(98,41)=a[8]; M.at<double>(98,42)=a[14]; M.at<double>(98,47)=a[16]; M.at<double>(98,56)=b[14]; M.at<double>(98,57)=b[12]; M.at<double>(98,60)=b[16]; M.at<double>(98,62)=b[8]; M.at<double>(98,90)=c[16]; M.at<double>(98,101)=c[4]; M.at<double>(98,105)=c[3]; M.at<double>(98,106)=c[14]; M.at<double>(98,108)=c[5]; M.at<double>(98,109)=c[17]; M.at<double>(98,111)=c[8]; M.at<double>(98,114)=c[12];
    M.at<double>(99,42)=a[7]; M.at<double>(99,44)=a[6]; M.at<double>(99,46)=a[18]; M.at<double>(99,56)=b[7]; M.at<double>(99,64)=b[8]; M.at<double>(99,66)=b[18]; M.at<double>(99,67)=b[17]; M.at<double>(99,68)=b[6]; M.at<double>(99,70)=b[12]; M.at<double>(99,91)=c[12]; M.at<double>(99,97)=c[6]; M.at<double>(99,102)=c[18]; M.at<double>(99,103)=c[8]; M.at<double>(99,104)=c[17]; M.at<double>(99,106)=c[7];
    M.at<double>(100,51)=a[16]; M.at<double>(100,65)=b[16]; M.at<double>(100,89)=c[16]; M.at<double>(100,105)=c[10]; M.at<double>(100,109)=c[14];
    M.at<double>(101,37)=a[9]; M.at<double>(101,38)=a[13]; M.at<double>(101,39)=a[20]; M.at<double>(101,48)=a[11]; M.at<double>(101,50)=a[3]; M.at<double>(101,53)=a[2]; M.at<double>(101,77)=b[9]; M.at<double>(101,78)=b[13]; M.at<double>(101,79)=b[20]; M.at<double>(101,80)=b[3]; M.at<double>(101,81)=b[11]; M.at<double>(101,82)=b[2]; M.at<double>(101,85)=c[11]; M.at<double>(101,86)=c[3]; M.at<double>(101,87)=c[2]; M.at<double>(101,92)=c[13]; M.at<double>(101,96)=c[20]; M.at<double>(101,98)=c[9];
    M.at<double>(102,37)=a[13]; M.at<double>(102,38)=a[20]; M.at<double>(102,48)=a[3]; M.at<double>(102,50)=a[2]; M.at<double>(102,77)=b[13]; M.at<double>(102,78)=b[20]; M.at<double>(102,80)=b[2]; M.at<double>(102,81)=b[3]; M.at<double>(102,85)=c[3]; M.at<double>(102,86)=c[2]; M.at<double>(102,92)=c[20]; M.at<double>(102,98)=c[13];
    M.at<double>(103,37)=a[20]; M.at<double>(103,48)=a[2]; M.at<double>(103,77)=b[20]; M.at<double>(103,81)=b[2]; M.at<double>(103,85)=c[2]; M.at<double>(103,98)=c[20];
    M.at<double>(104,11)=u[4]; M.at<double>(104,28)=a[10]; M.at<double>(104,33)=a[14]; M.at<double>(104,34)=a[8]; M.at<double>(104,36)=a[16]; M.at<double>(104,51)=a[4]; M.at<double>(104,54)=b[8]; M.at<double>(104,57)=b[10]; M.at<double>(104,59)=b[14]; M.at<double>(104,65)=b[4]; M.at<double>(104,84)=b[16]; M.at<double>(104,88)=c[16]; M.at<double>(104,89)=c[4]; M.at<double>(104,99)=c[8]; M.at<double>(104,108)=c[1]; M.at<double>(104,109)=c[9]; M.at<double>(104,114)=c[10]; M.at<double>(104,117)=c[14];
    M.at<double>(105,20)=u[4]; M.at<double>(105,27)=a[10]; M.at<double>(105,28)=a[4]; M.at<double>(105,32)=a[14]; M.at<double>(105,33)=a[8]; M.at<double>(105,51)=a[19]; M.at<double>(105,53)=a[16]; M.at<double>(105,55)=b[10]; M.at<double>(105,57)=b[4]; M.at<double>(105,59)=b[8]; M.at<double>(105,61)=b[14]; M.at<double>(105,65)=b[19]; M.at<double>(105,82)=b[16]; M.at<double>(105,87)=c[16]; M.at<double>(105,89)=c[19]; M.at<double>(105,108)=c[9]; M.at<double>(105,109)=c[13]; M.at<double>(105,113)=c[10]; M.at<double>(105,114)=c[4]; M.at<double>(105,116)=c[14]; M.at<double>(105,117)=c[8];
    M.at<double>(106,27)=a[4]; M.at<double>(106,28)=a[19]; M.at<double>(106,31)=a[14]; M.at<double>(106,32)=a[8]; M.at<double>(106,50)=a[16]; M.at<double>(106,55)=b[4]; M.at<double>(106,57)=b[19]; M.at<double>(106,58)=b[14]; M.at<double>(106,61)=b[8]; M.at<double>(106,80)=b[16]; M.at<double>(106,86)=c[16]; M.at<double>(106,108)=c[13]; M.at<double>(106,109)=c[20]; M.at<double>(106,113)=c[4]; M.at<double>(106,114)=c[19]; M.at<double>(106,115)=c[14]; M.at<double>(106,116)=c[8];
    M.at<double>(107,27)=a[19]; M.at<double>(107,31)=a[8]; M.at<double>(107,48)=a[16]; M.at<double>(107,55)=b[19]; M.at<double>(107,58)=b[8]; M.at<double>(107,81)=b[16]; M.at<double>(107,85)=c[16]; M.at<double>(107,108)=c[20]; M.at<double>(107,113)=c[19]; M.at<double>(107,115)=c[8];
    M.at<double>(108,36)=a[20]; M.at<double>(108,48)=a[1]; M.at<double>(108,50)=a[9]; M.at<double>(108,53)=a[13]; M.at<double>(108,80)=b[9]; M.at<double>(108,81)=b[1]; M.at<double>(108,82)=b[13]; M.at<double>(108,84)=b[20]; M.at<double>(108,85)=c[1]; M.at<double>(108,86)=c[9]; M.at<double>(108,87)=c[13]; M.at<double>(108,88)=c[20];
    M.at<double>(109,26)=u[3]; M.at<double>(109,27)=a[7]; M.at<double>(109,40)=a[6]; M.at<double>(109,42)=a[18]; M.at<double>(109,44)=a[17]; M.at<double>(109,46)=a[2]; M.at<double>(109,52)=a[12]; M.at<double>(109,55)=b[7]; M.at<double>(109,56)=b[18]; M.at<double>(109,66)=b[2]; M.at<double>(109,67)=b[19]; M.at<double>(109,68)=b[17]; M.at<double>(109,70)=b[8]; M.at<double>(109,72)=b[6]; M.at<double>(109,73)=b[12]; M.at<double>(109,91)=c[8]; M.at<double>(109,93)=c[12]; M.at<double>(109,95)=c[6]; M.at<double>(109,97)=c[17]; M.at<double>(109,102)=c[2]; M.at<double>(109,104)=c[19]; M.at<double>(109,106)=c[18]; M.at<double>(109,113)=c[7];
    M.at<double>(110,7)=u[3]; M.at<double>(110,30)=a[5]; M.at<double>(110,34)=a[7]; M.at<double>(110,43)=a[15]; M.at<double>(110,45)=a[11]; M.at<double>(110,49)=a[6]; M.at<double>(110,54)=b[7]; M.at<double>(110,63)=b[11]; M.at<double>(110,67)=b[1]; M.at<double>(110,69)=b[10]; M.at<double>(110,71)=b[15]; M.at<double>(110,75)=b[5]; M.at<double>(110,76)=b[6]; M.at<double>(110,99)=c[7]; M.at<double>(110,100)=c[15]; M.at<double>(110,104)=c[1]; M.at<double>(110,107)=c[5]; M.at<double>(110,110)=c[10]; M.at<double>(110,112)=c[11]; M.at<double>(110,119)=c[6];
    M.at<double>(111,18)=u[2]; M.at<double>(111,35)=a[20]; M.at<double>(111,36)=a[13]; M.at<double>(111,50)=a[1]; M.at<double>(111,53)=a[9]; M.at<double>(111,80)=b[1]; M.at<double>(111,82)=b[9]; M.at<double>(111,83)=b[20]; M.at<double>(111,84)=b[13]; M.at<double>(111,86)=c[1]; M.at<double>(111,87)=c[9]; M.at<double>(111,88)=c[13]; M.at<double>(111,118)=c[20];
    M.at<double>(112,19)=u[2]; M.at<double>(112,31)=a[1]; M.at<double>(112,32)=a[9]; M.at<double>(112,33)=a[13]; M.at<double>(112,34)=a[20]; M.at<double>(112,36)=a[19]; M.at<double>(112,50)=a[10]; M.at<double>(112,53)=a[4]; M.at<double>(112,54)=b[20]; M.at<double>(112,58)=b[1]; M.at<double>(112,59)=b[13]; M.at<double>(112,61)=b[9]; M.at<double>(112,80)=b[10]; M.at<double>(112,82)=b[4]; M.at<double>(112,84)=b[19]; M.at<double>(112,86)=c[10]; M.at<double>(112,87)=c[4]; M.at<double>(112,88)=c[19]; M.at<double>(112,99)=c[20]; M.at<double>(112,115)=c[1]; M.at<double>(112,116)=c[9]; M.at<double>(112,117)=c[13];
    M.at<double>(113,31)=a[9]; M.at<double>(113,32)=a[13]; M.at<double>(113,33)=a[20]; M.at<double>(113,48)=a[10]; M.at<double>(113,50)=a[4]; M.at<double>(113,53)=a[19]; M.at<double>(113,58)=b[9]; M.at<double>(113,59)=b[20]; M.at<double>(113,61)=b[13]; M.at<double>(113,80)=b[4]; M.at<double>(113,81)=b[10]; M.at<double>(113,82)=b[19]; M.at<double>(113,85)=c[10]; M.at<double>(113,86)=c[4]; M.at<double>(113,87)=c[19]; M.at<double>(113,115)=c[9]; M.at<double>(113,116)=c[13]; M.at<double>(113,117)=c[20];
    M.at<double>(114,31)=a[13]; M.at<double>(114,32)=a[20]; M.at<double>(114,48)=a[4]; M.at<double>(114,50)=a[19]; M.at<double>(114,58)=b[13]; M.at<double>(114,61)=b[20]; M.at<double>(114,80)=b[19]; M.at<double>(114,81)=b[4]; M.at<double>(114,85)=c[4]; M.at<double>(114,86)=c[19]; M.at<double>(114,115)=c[13]; M.at<double>(114,116)=c[20];
    M.at<double>(115,31)=a[20]; M.at<double>(115,48)=a[19]; M.at<double>(115,58)=b[20]; M.at<double>(115,81)=b[19]; M.at<double>(115,85)=c[19]; M.at<double>(115,115)=c[20];
    M.at<double>(116,24)=u[2]; M.at<double>(116,29)=a[13]; M.at<double>(116,30)=a[20]; M.at<double>(116,37)=a[11]; M.at<double>(116,38)=a[3]; M.at<double>(116,39)=a[2]; M.at<double>(116,50)=a[15]; M.at<double>(116,52)=a[9]; M.at<double>(116,53)=a[18]; M.at<double>(116,73)=b[9]; M.at<double>(116,74)=b[13]; M.at<double>(116,75)=b[20]; M.at<double>(116,77)=b[11]; M.at<double>(116,78)=b[3]; M.at<double>(116,79)=b[2]; M.at<double>(116,80)=b[15]; M.at<double>(116,82)=b[18]; M.at<double>(116,86)=c[15]; M.at<double>(116,87)=c[18]; M.at<double>(116,92)=c[3]; M.at<double>(116,93)=c[9]; M.at<double>(116,94)=c[13]; M.at<double>(116,96)=c[2]; M.at<double>(116,98)=c[11]; M.at<double>(116,107)=c[20];
    M.at<double>(117,29)=a[20]; M.at<double>(117,37)=a[3]; M.at<double>(117,38)=a[2]; M.at<double>(117,48)=a[15]; M.at<double>(117,50)=a[18]; M.at<double>(117,52)=a[13]; M.at<double>(117,73)=b[13]; M.at<double>(117,74)=b[20]; M.at<double>(117,77)=b[3]; M.at<double>(117,78)=b[2]; M.at<double>(117,80)=b[18]; M.at<double>(117,81)=b[15]; M.at<double>(117,85)=c[15]; M.at<double>(117,86)=c[18]; M.at<double>(117,92)=c[2]; M.at<double>(117,93)=c[13]; M.at<double>(117,94)=c[20]; M.at<double>(117,98)=c[3];
    M.at<double>(118,27)=a[13]; M.at<double>(118,28)=a[20]; M.at<double>(118,31)=a[4]; M.at<double>(118,32)=a[19]; M.at<double>(118,48)=a[14]; M.at<double>(118,50)=a[8]; M.at<double>(118,55)=b[13]; M.at<double>(118,57)=b[20]; M.at<double>(118,58)=b[4]; M.at<double>(118,61)=b[19]; M.at<double>(118,80)=b[8]; M.at<double>(118,81)=b[14]; M.at<double>(118,85)=c[14]; M.at<double>(118,86)=c[8]; M.at<double>(118,113)=c[13]; M.at<double>(118,114)=c[20]; M.at<double>(118,115)=c[4]; M.at<double>(118,116)=c[19];
    M.at<double>(119,27)=a[20]; M.at<double>(119,31)=a[19]; M.at<double>(119,48)=a[8]; M.at<double>(119,55)=b[20]; M.at<double>(119,58)=b[19]; M.at<double>(119,81)=b[8]; M.at<double>(119,85)=c[8]; M.at<double>(119,113)=c[20]; M.at<double>(119,115)=c[19];

    return M.t();
}

cv::Mat dls::Hessian(const double s[])
{
    // the vector of monomials is
    // m = [ const ; s1^2 * s2 ; s1 * s2 ; s1 * s3 ; s2 * s3 ; s2^2 * s3 ; s2^3 ; ...
    //       s1 * s3^2 ; s1 ; s3 ; s2 ; s2 * s3^2 ; s1^2 ; s3^2 ; s2^2 ; s3^3 ;   ...
    //       s1 * s2 * s3 ; s1 * s2^2 ; s1^2 * s3 ; s1^3]
    //

    // deriv of m w.r.t. s1
    //Hs3 = [0 ; 2 * s(1) * s(2) ; s(2) ; s(3) ; 0 ; 0 ; 0 ; ...
    //         s(3)^2 ; 1 ; 0 ; 0 ; 0 ; 2 * s(1) ; 0 ; 0 ; 0 ; ...
    //         s(2) * s(3) ; s(2)^2 ; 2*s(1)*s(3); 3 * s(1)^2];

    double Hs1[20];
    Hs1[0]=0; Hs1[1]=2*s[0]*s[1]; Hs1[2]=s[1]; Hs1[3]=s[2]; Hs1[4]=0; Hs1[5]=0; Hs1[6]=0;
    Hs1[7]=s[2]*s[2]; Hs1[8]=1; Hs1[9]=0; Hs1[10]=0; Hs1[11]=0; Hs1[12]=2*s[0]; Hs1[13]=0;
    Hs1[14]=0; Hs1[15]=0; Hs1[16]=s[1]*s[2]; Hs1[17]=s[1]*s[1]; Hs1[18]=2*s[0]*s[2]; Hs1[19]=3*s[0]*s[0];

    // deriv of m w.r.t. s2
    //Hs2 = [0 ; s(1)^2 ; s(1) ; 0 ; s(3) ; 2 * s(2) * s(3) ; 3 * s(2)^2 ; ...
    //         0 ; 0 ; 0 ; 1 ; s(3)^2 ; 0 ; 0 ; 2 * s(2) ; 0 ; ...
    //         s(1) * s(3) ; s(1) * 2 * s(2) ; 0 ; 0];

    double Hs2[20];
    Hs2[0]=0; Hs2[1]=s[0]*s[0]; Hs2[2]=s[0]; Hs2[3]=0; Hs2[4]=s[2]; Hs2[5]=2*s[1]*s[2]; Hs2[6]=3*s[1]*s[1];
    Hs2[7]=0; Hs2[8]=0; Hs2[9]=0; Hs2[10]=1; Hs2[11]=s[2]*s[2]; Hs2[12]=0; Hs2[13]=0;
    Hs2[14]=2*s[1]; Hs2[15]=0; Hs2[16]=s[0]*s[2]; Hs2[17]=2*s[0]*s[1]; Hs2[18]=0; Hs2[19]=0;

    // deriv of m w.r.t. s3
    //Hs3 = [0 ; 0 ; 0 ; s(1) ; s(2) ; s(2)^2 ; 0 ; ...
    //         s(1) * 2 * s(3) ; 0 ; 1 ; 0 ; s(2) * 2 * s(3) ; 0 ; 2 * s(3) ; 0 ; 3 * s(3)^2 ; ...
    //         s(1) * s(2) ; 0 ; s(1)^2 ; 0];

    double Hs3[20];
    Hs3[0]=0; Hs3[1]=0; Hs3[2]=0; Hs3[3]=s[0]; Hs3[4]=s[1]; Hs3[5]=s[1]*s[1]; Hs3[6]=0;
    Hs3[7]=2*s[0]*s[2]; Hs3[8]=0; Hs3[9]=1; Hs3[10]=0; Hs3[11]=2*s[1]*s[2]; Hs3[12]=0; Hs3[13]=2*s[2];
    Hs3[14]=0; Hs3[15]=3*s[2]*s[2]; Hs3[16]=s[0]*s[1]; Hs3[17]=0; Hs3[18]=s[0]*s[0]; Hs3[19]=0;

    // fill Hessian matrix
    cv::Mat H(3, 3, CV_64F);
    H.at<double>(0,0) = cv::Mat(cv::Mat(f1coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs1)).at<double>(0,0);
    H.at<double>(0,1) = cv::Mat(cv::Mat(f1coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs2)).at<double>(0,0);
    H.at<double>(0,2) = cv::Mat(cv::Mat(f1coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs3)).at<double>(0,0);

    H.at<double>(1,0) = cv::Mat(cv::Mat(f2coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs1)).at<double>(0,0);
    H.at<double>(1,1) = cv::Mat(cv::Mat(f2coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs2)).at<double>(0,0);
    H.at<double>(1,2) = cv::Mat(cv::Mat(f2coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs3)).at<double>(0,0);

    H.at<double>(2,0) = cv::Mat(cv::Mat(f3coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs1)).at<double>(0,0);
    H.at<double>(2,1) = cv::Mat(cv::Mat(f3coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs2)).at<double>(0,0);
    H.at<double>(2,2) = cv::Mat(cv::Mat(f3coeff).rowRange(1,21).t()*cv::Mat(20, 1, CV_64F, &Hs3)).at<double>(0,0);

    return H;
}

cv::Mat dls::cayley2rotbar(const cv::Mat& s)
{
    double s_mul1 = cv::Mat(s.t()*s).at<double>(0,0);
    cv::Mat s_mul2 = s*s.t();
    cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);

    return cv::Mat( eye.mul(1.-s_mul1) + skewsymm(&s).mul(2.) + s_mul2.mul(2.) ).t();
}

cv::Mat dls::skewsymm(const cv::Mat * X1)
{
    cv::MatConstIterator_<double> it = X1->begin<double>();
    return (cv::Mat_<double>(3,3) <<        0, -*(it+2),  *(it+1),
                                      *(it+2),        0, -*(it+0),
                                     -*(it+1),  *(it+0),       0);
}

cv::Mat dls::rotx(const double t)
{
    // rotx: rotation about y-axis
    double ct = cos(t);
    double st = sin(t);
    return (cv::Mat_<double>(3,3) << 1, 0, 0, 0, ct, -st, 0, st, ct);
}

cv::Mat dls::roty(const double t)
{
    // roty: rotation about y-axis
    double ct = cos(t);
    double st = sin(t);
    return (cv::Mat_<double>(3,3) << ct, 0, st, 0, 1, 0, -st, 0, ct);
}

cv::Mat dls::rotz(const double t)
{
    // rotz: rotation about y-axis
    double ct = cos(t);
    double st = sin(t);
    return (cv::Mat_<double>(3,3) << ct, -st, 0, st, ct, 0, 0, 0, 1);
}

cv::Mat dls::mean(const cv::Mat& M)
{
    cv::Mat m = cv::Mat::zeros(3, 1, CV_64F);
    for (int i = 0; i < M.cols; ++i) m += M.col(i);
    return m.mul(1./(double)M.cols);
}

bool dls::is_empty(const cv::Mat * M)
{
    cv::MatConstIterator_<double> it = M->begin<double>(), it_end = M->end<double>();
    for(; it != it_end; ++it)
    {
        if(*it < 0) return false;
    }
    return true;
}

bool dls::positive_eigenvalues(const cv::Mat * eigenvalues)
{
    CV_Assert(eigenvalues && !eigenvalues->empty());
    cv::MatConstIterator_<double> it = eigenvalues->begin<double>();
    return *(it) > 0 && *(it+1) > 0 && *(it+2) > 0;
}
