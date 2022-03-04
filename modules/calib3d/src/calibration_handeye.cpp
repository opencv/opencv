// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/calib3d.hpp"

namespace cv {

static Mat homogeneousInverse(const Mat& T)
{
    CV_Assert(T.rows == 4 && T.cols == 4);

    Mat R = T(Rect(0, 0, 3, 3));
    Mat t = T(Rect(3, 0, 1, 3));
    Mat Rt = R.t();
    Mat tinv = -Rt * t;
    Mat Tinv = Mat::eye(4, 4, T.type());
    Rt.copyTo(Tinv(Rect(0, 0, 3, 3)));
    tinv.copyTo(Tinv(Rect(3, 0, 1, 3)));

    return Tinv;
}

// q = rot2quatMinimal(R)
//
// R - 3x3 rotation matrix, or 4x4 homogeneous matrix
// q - 3x1 unit quaternion <qx, qy, qz>
// q = sin(theta/2) * v
// theta - rotation angle
// v     - unit rotation axis, |v| = 1
// Reference: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
static Mat rot2quatMinimal(const Mat& R)
{
    CV_Assert(R.type() == CV_64FC1 && R.rows >= 3 && R.cols >= 3);

    double m00 = R.at<double>(0,0), m01 = R.at<double>(0,1), m02 = R.at<double>(0,2);
    double m10 = R.at<double>(1,0), m11 = R.at<double>(1,1), m12 = R.at<double>(1,2);
    double m20 = R.at<double>(2,0), m21 = R.at<double>(2,1), m22 = R.at<double>(2,2);
    double trace = m00 + m11 + m22;

    double qx, qy, qz;
    if (trace > 0) {
        double S = sqrt(trace + 1.0) * 2; // S=4*qw
        qx = (m21 - m12) / S;
        qy = (m02 - m20) / S;
        qz = (m10 - m01) / S;
    } else if (m00 > m11 && m00 > m22) {
        double S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
        qx = 0.25 * S;
        qy = (m01 + m10) / S;
        qz = (m02 + m20) / S;
    } else if (m11 > m22) {
        double S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
        qx = (m01 + m10) / S;
        qy = 0.25 * S;
        qz = (m12 + m21) / S;
    } else {
        double S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
        qx = (m02 + m20) / S;
        qy = (m12 + m21) / S;
        qz = 0.25 * S;
    }

    return (Mat_<double>(3,1) << qx, qy, qz);
}

static Mat skew(const Mat& v)
{
    CV_Assert(v.type() == CV_64FC1 && v.rows == 3 && v.cols == 1);

    double vx = v.at<double>(0,0);
    double vy = v.at<double>(1,0);
    double vz = v.at<double>(2,0);
    return (Mat_<double>(3,3) << 0, -vz, vy,
                                vz, 0, -vx,
                                -vy, vx, 0);
}

// R = quatMinimal2rot(q)
//
// q - 3x1 unit quaternion <qx, qy, qz>
// R - 3x3 rotation matrix
// q = sin(theta/2) * v
// theta - rotation angle
// v     - unit rotation axis, |v| = 1
static Mat quatMinimal2rot(const Mat& q)
{
    CV_Assert(q.type() == CV_64FC1 && q.rows == 3 && q.cols == 1);

    Mat p = q.t()*q;
    double w = sqrt(1 - p.at<double>(0,0));

    Mat diag_p = Mat::eye(3,3,CV_64FC1)*p.at<double>(0,0);
    return 2*q*q.t() + 2*w*skew(q) + Mat::eye(3,3,CV_64FC1) - 2*diag_p;
}

// q = rot2quat(R)
//
// q - 4x1 unit quaternion <qw, qx, qy, qz>
// R - 3x3 rotation matrix
// Reference: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
static Mat rot2quat(const Mat& R)
{
    CV_Assert(R.type() == CV_64FC1 && R.rows >= 3 && R.cols >= 3);

    double m00 = R.at<double>(0,0), m01 = R.at<double>(0,1), m02 = R.at<double>(0,2);
    double m10 = R.at<double>(1,0), m11 = R.at<double>(1,1), m12 = R.at<double>(1,2);
    double m20 = R.at<double>(2,0), m21 = R.at<double>(2,1), m22 = R.at<double>(2,2);
    double trace = m00 + m11 + m22;

    double qw, qx, qy, qz;
    if (trace > 0) {
        double S = sqrt(trace + 1.0) * 2; // S=4*qw
        qw = 0.25 * S;
        qx = (m21 - m12) / S;
        qy = (m02 - m20) / S;
        qz = (m10 - m01) / S;
    } else if (m00 > m11 && m00 > m22) {
        double S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
        qw = (m21 - m12) / S;
        qx = 0.25 * S;
        qy = (m01 + m10) / S;
        qz = (m02 + m20) / S;
    } else if (m11 > m22) {
        double S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
        qw = (m02 - m20) / S;
        qx = (m01 + m10) / S;
        qy = 0.25 * S;
        qz = (m12 + m21) / S;
    } else {
        double S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
        qw = (m10 - m01) / S;
        qx = (m02 + m20) / S;
        qy = (m12 + m21) / S;
        qz = 0.25 * S;
    }

    return (Mat_<double>(4,1) << qw, qx, qy, qz);
}

// R = quat2rot(q)
//
// q - 4x1 unit quaternion <qw, qx, qy, qz>
// R - 3x3 rotation matrix
static Mat quat2rot(const Mat& q)
{
    CV_Assert(q.type() == CV_64FC1 && q.rows == 4 && q.cols == 1);

    double qw = q.at<double>(0,0);
    double qx = q.at<double>(1,0);
    double qy = q.at<double>(2,0);
    double qz = q.at<double>(3,0);

    Mat R(3, 3, CV_64FC1);
    R.at<double>(0, 0) = 1 - 2*qy*qy - 2*qz*qz;
    R.at<double>(0, 1) = 2*qx*qy - 2*qz*qw;
    R.at<double>(0, 2) = 2*qx*qz + 2*qy*qw;

    R.at<double>(1, 0) = 2*qx*qy + 2*qz*qw;
    R.at<double>(1, 1) = 1 - 2*qx*qx - 2*qz*qz;
    R.at<double>(1, 2) = 2*qy*qz - 2*qx*qw;

    R.at<double>(2, 0) = 2*qx*qz - 2*qy*qw;
    R.at<double>(2, 1) = 2*qy*qz + 2*qx*qw;
    R.at<double>(2, 2) = 1 - 2*qx*qx - 2*qy*qy;

    return R;
}

// Kronecker product or tensor product
// https://stackoverflow.com/a/36552682
static Mat kron(const Mat& A, const Mat& B)
{
    CV_Assert(A.channels() == 1 && B.channels() == 1);

    Mat1d Ad, Bd;
    A.convertTo(Ad, CV_64F);
    B.convertTo(Bd, CV_64F);

    Mat1d Kd(Ad.rows * Bd.rows, Ad.cols * Bd.cols, 0.0);
    for (int ra = 0; ra < Ad.rows; ra++)
    {
        for (int ca = 0; ca < Ad.cols; ca++)
        {
            Kd(Range(ra*Bd.rows, (ra + 1)*Bd.rows), Range(ca*Bd.cols, (ca + 1)*Bd.cols)) = Bd.mul(Ad(ra, ca));
        }
    }

    Mat K;
    Kd.convertTo(K, A.type());
    return K;
}

// quaternion multiplication
static Mat qmult(const Mat& s, const Mat& t)
{
    CV_Assert(s.type() == CV_64FC1 && t.type() == CV_64FC1);
    CV_Assert(s.rows == 4 && s.cols == 1);
    CV_Assert(t.rows == 4 && t.cols == 1);

    double s0 = s.at<double>(0,0);
    double s1 = s.at<double>(1,0);
    double s2 = s.at<double>(2,0);
    double s3 = s.at<double>(3,0);

    double t0 = t.at<double>(0,0);
    double t1 = t.at<double>(1,0);
    double t2 = t.at<double>(2,0);
    double t3 = t.at<double>(3,0);

    Mat q(4, 1, CV_64FC1);
    q.at<double>(0,0) = s0*t0 - s1*t1 - s2*t2 - s3*t3;
    q.at<double>(1,0) = s0*t1 + s1*t0 + s2*t3 - s3*t2;
    q.at<double>(2,0) = s0*t2 - s1*t3 + s2*t0 + s3*t1;
    q.at<double>(3,0) = s0*t3 + s1*t2 - s2*t1 + s3*t0;

    return q;
}

// dq = homogeneous2dualQuaternion(H)
//
// H  - 4x4 homogeneous transformation: [R | t; 0 0 0 | 1]
// dq - 8x1 dual quaternion: <q (rotation part), qprime (translation part)>
static Mat homogeneous2dualQuaternion(const Mat& H)
{
    CV_Assert(H.type() == CV_64FC1 && H.rows == 4 && H.cols == 4);

    Mat dualq(8, 1, CV_64FC1);
    Mat R = H(Rect(0, 0, 3, 3));
    Mat t = H(Rect(3, 0, 1, 3));

    Mat q = rot2quat(R);
    Mat qt = Mat::zeros(4, 1, CV_64FC1);
    t.copyTo(qt(Rect(0, 1, 1, 3)));
    Mat qprime = 0.5 * qmult(qt, q);

    q.copyTo(dualq(Rect(0, 0, 1, 4)));
    qprime.copyTo(dualq(Rect(0, 4, 1, 4)));

    return dualq;
}

// H = dualQuaternion2homogeneous(dq)
//
// H  - 4x4 homogeneous transformation: [R | t; 0 0 0 | 1]
// dq - 8x1 dual quaternion: <q (rotation part), qprime (translation part)>
static Mat dualQuaternion2homogeneous(const Mat& dualq)
{
    CV_Assert(dualq.type() == CV_64FC1 && dualq.rows == 8 && dualq.cols == 1);

    Mat q = dualq(Rect(0, 0, 1, 4));
    Mat qprime = dualq(Rect(0, 4, 1, 4));

    Mat R = quat2rot(q);
    q.at<double>(1,0) = -q.at<double>(1,0);
    q.at<double>(2,0) = -q.at<double>(2,0);
    q.at<double>(3,0) = -q.at<double>(3,0);

    Mat qt = 2*qmult(qprime, q);
    Mat t = qt(Rect(0, 1, 1, 3));

    Mat H = Mat::eye(4, 4, CV_64FC1);
    R.copyTo(H(Rect(0, 0, 3, 3)));
    t.copyTo(H(Rect(3, 0, 1, 3)));

    return H;
}

//Reference:
//R. Y. Tsai and R. K. Lenz, "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration."
//In IEEE Transactions on Robotics and Automation, vol. 5, no. 3, pp. 345-358, June 1989.
//C++ code converted from Zoran Lazarevic's Matlab code:
//http://lazax.com/www.cs.columbia.edu/~laza/html/Stewart/matlab/handEye.m
static void calibrateHandEyeTsai(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
                                 Mat& R_cam2gripper, Mat& t_cam2gripper)
{
    //Number of unique camera position pairs
    int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
    //Will store: skew(Pgij+Pcij)
    Mat A(3*K, 3, CV_64FC1);
    //Will store: Pcij - Pgij
    Mat B(3*K, 1, CV_64FC1);

    std::vector<Mat> vec_Hgij, vec_Hcij;
    vec_Hgij.reserve(static_cast<size_t>(K));
    vec_Hcij.reserve(static_cast<size_t>(K));

    int idx = 0;
    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++, idx++)
        {
            //Defines coordinate transformation from Gi to Gj
            //Hgi is from Gi (gripper) to RW (robot base)
            //Hgj is from Gj (gripper) to RW (robot base)
            Mat Hgij = homogeneousInverse(Hg[j]) * Hg[i]; //eq 6
            vec_Hgij.push_back(Hgij);
            //Rotation axis for Rgij which is the 3D rotation from gripper coordinate frame Gi to Gj
            Mat Pgij = 2*rot2quatMinimal(Hgij);

            //Defines coordinate transformation from Ci to Cj
            //Hci is from CW (calibration target) to Ci (camera)
            //Hcj is from CW (calibration target) to Cj (camera)
            Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]); //eq 7
            vec_Hcij.push_back(Hcij);
            //Rotation axis for Rcij
            Mat Pcij = 2*rot2quatMinimal(Hcij);

            //Left-hand side: skew(Pgij+Pcij)
            skew(Pgij+Pcij).copyTo(A(Rect(0, idx*3, 3, 3)));
            //Right-hand side: Pcij - Pgij
            Mat diff = Pcij - Pgij;
            diff.copyTo(B(Rect(0, idx*3, 1, 3)));
        }
    }

    Mat Pcg_;
    //Rotation from camera to gripper is obtained from the set of equations:
    //    skew(Pgij+Pcij) * Pcg_ = Pcij - Pgij    (eq 12)
    solve(A, B, Pcg_, DECOMP_SVD);

    Mat Pcg_norm = Pcg_.t() * Pcg_;
    //Obtained non-unit quaternion is scaled back to unit value that
    //designates camera-gripper rotation
    Mat Pcg = 2 * Pcg_ / sqrt(1 + Pcg_norm.at<double>(0,0)); //eq 14

    Mat Rcg = quatMinimal2rot(Pcg/2.0);

    idx = 0;
    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++, idx++)
        {
            //Defines coordinate transformation from Gi to Gj
            //Hgi is from Gi (gripper) to RW (robot base)
            //Hgj is from Gj (gripper) to RW (robot base)
            Mat Hgij = vec_Hgij[static_cast<size_t>(idx)];
            //Defines coordinate transformation from Ci to Cj
            //Hci is from CW (calibration target) to Ci (camera)
            //Hcj is from CW (calibration target) to Cj (camera)
            Mat Hcij = vec_Hcij[static_cast<size_t>(idx)];

            //Left-hand side: (Rgij - I)
            Mat diff = Hgij(Rect(0,0,3,3)) - Mat::eye(3,3,CV_64FC1);
            diff.copyTo(A(Rect(0, idx*3, 3, 3)));

            //Right-hand side: Rcg*Tcij - Tgij
            diff = Rcg*Hcij(Rect(3, 0, 1, 3)) - Hgij(Rect(3, 0, 1, 3));
            diff.copyTo(B(Rect(0, idx*3, 1, 3)));
        }
    }

    Mat Tcg;
    //Translation from camera to gripper is obtained from the set of equations:
    //    (Rgij - I) * Tcg = Rcg*Tcij - Tgij    (eq 15)
    solve(A, B, Tcg, DECOMP_SVD);

    R_cam2gripper = Rcg;
    t_cam2gripper = Tcg;
}

//Reference:
//F. Park, B. Martin, "Robot Sensor Calibration: Solving AX = XB on the Euclidean Group."
//In IEEE Transactions on Robotics and Automation, 10(5): 717-721, 1994.
//Matlab code: http://math.loyola.edu/~mili/Calibration/
static void calibrateHandEyePark(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
                                 Mat& R_cam2gripper, Mat& t_cam2gripper)
{
    Mat M = Mat::zeros(3, 3, CV_64FC1);

    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++)
        {
            Mat Hgij = homogeneousInverse(Hg[j]) * Hg[i];
            Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

            Mat Rgij = Hgij(Rect(0, 0, 3, 3));
            Mat Rcij = Hcij(Rect(0, 0, 3, 3));

            Mat a, b;
            Rodrigues(Rgij, a);
            Rodrigues(Rcij, b);

            M += b * a.t();
        }
    }

    Mat eigenvalues, eigenvectors;
    eigen(M.t()*M, eigenvalues, eigenvectors);

    Mat v = Mat::zeros(3, 3, CV_64FC1);
    for (int i = 0; i < 3; i++) {
        v.at<double>(i,i) = 1.0 / sqrt(eigenvalues.at<double>(i,0));
    }

    Mat R = eigenvectors.t() * v * eigenvectors * M.t();
    R_cam2gripper = R;

    int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
    Mat C(3*K, 3, CV_64FC1);
    Mat d(3*K, 1, CV_64FC1);
    Mat I3 = Mat::eye(3, 3, CV_64FC1);

    int idx = 0;
    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++, idx++)
        {
            Mat Hgij = homogeneousInverse(Hg[j]) * Hg[i];
            Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

            Mat Rgij = Hgij(Rect(0, 0, 3, 3));

            Mat tgij = Hgij(Rect(3, 0, 1, 3));
            Mat tcij = Hcij(Rect(3, 0, 1, 3));

            Mat I_tgij = I3 - Rgij;
            I_tgij.copyTo(C(Rect(0, 3*idx, 3, 3)));

            Mat A_RB = tgij - R*tcij;
            A_RB.copyTo(d(Rect(0, 3*idx, 1, 3)));
        }
    }

    Mat t;
    solve(C, d, t, DECOMP_SVD);
    t_cam2gripper = t;
}

//Reference:
//R. Horaud, F. Dornaika, "Hand-Eye Calibration"
//In International Journal of Robotics Research, 14(3): 195-210, 1995.
//Matlab code: http://math.loyola.edu/~mili/Calibration/
static void calibrateHandEyeHoraud(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
                                   Mat& R_cam2gripper, Mat& t_cam2gripper)
{
    Mat A = Mat::zeros(4, 4, CV_64FC1);

    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++)
        {
            Mat Hgij = homogeneousInverse(Hg[j]) * Hg[i];
            Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

            Mat Rgij = Hgij(Rect(0, 0, 3, 3));
            Mat Rcij = Hcij(Rect(0, 0, 3, 3));

            Mat qgij = rot2quat(Rgij);
            double r0 = qgij.at<double>(0,0);
            double rx = qgij.at<double>(1,0);
            double ry = qgij.at<double>(2,0);
            double rz = qgij.at<double>(3,0);

            // Q(r) Appendix A
            Matx44d Qvi(r0, -rx, -ry, -rz,
                        rx,  r0, -rz,  ry,
                        ry,  rz,  r0, -rx,
                        rz, -ry,  rx,  r0);

            Mat qcij = rot2quat(Rcij);
            r0 = qcij.at<double>(0,0);
            rx = qcij.at<double>(1,0);
            ry = qcij.at<double>(2,0);
            rz = qcij.at<double>(3,0);

            // W(r) Appendix A
            Matx44d Wvi(r0, -rx, -ry, -rz,
                        rx,  r0,  rz, -ry,
                        ry, -rz,  r0,  rx,
                        rz,  ry, -rx,  r0);

            // Ai = (Q(vi') - W(vi))^T (Q(vi') - W(vi))
            A += (Qvi - Wvi).t() * (Qvi - Wvi);
        }
    }

    Mat eigenvalues, eigenvectors;
    eigen(A, eigenvalues, eigenvectors);

    Mat R = quat2rot(eigenvectors.row(3).t());
    R_cam2gripper = R;

    int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
    Mat C(3*K, 3, CV_64FC1);
    Mat d(3*K, 1, CV_64FC1);
    Mat I3 = Mat::eye(3, 3, CV_64FC1);

    int idx = 0;
    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++, idx++)
        {
            Mat Hgij = homogeneousInverse(Hg[j]) * Hg[i];
            Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

            Mat Rgij = Hgij(Rect(0, 0, 3, 3));

            Mat tgij = Hgij(Rect(3, 0, 1, 3));
            Mat tcij = Hcij(Rect(3, 0, 1, 3));

            Mat I_tgij = I3 - Rgij;
            I_tgij.copyTo(C(Rect(0, 3*idx, 3, 3)));

            Mat A_RB = tgij - R*tcij;
            A_RB.copyTo(d(Rect(0, 3*idx, 1, 3)));
        }
    }

    Mat t;
    solve(C, d, t, DECOMP_SVD);
    t_cam2gripper = t;
}

static Mat_<double> normalizeRotation(const Mat_<double>& R_)
{
    // Make R unit determinant
    Mat_<double> R = R_.clone();
    double det = determinant(R);
    if (std::fabs(det) < FLT_EPSILON)
    {
        CV_Error(Error::StsNoConv, "Rotation normalization issue: determinant(R) is null");
    }
    R = std::cbrt(std::copysign(1, det) / std::fabs(det)) * R;

    // Make R orthogonal
    Mat w, u, vt;
    SVDecomp(R, w, u, vt);
    R = u*vt;

    // Handle reflection case
    if (determinant(R) < 0)
    {
        Matx33d diag(1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, -1.0);
        R = u*diag*vt;
    }

    return R;
}

//Reference:
//N. Andreff, R. Horaud, B. Espiau, "On-line Hand-Eye Calibration."
//In Second International Conference on 3-D Digital Imaging and Modeling (3DIM'99), pages 430-436, 1999.
//Matlab code: http://math.loyola.edu/~mili/Calibration/
static void calibrateHandEyeAndreff(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
                                    Mat& R_cam2gripper, Mat& t_cam2gripper)
{
    int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
    Mat A(12*K, 12, CV_64FC1);
    Mat B(12*K, 1, CV_64FC1);

    Mat I9 = Mat::eye(9, 9, CV_64FC1);
    Mat I3 = Mat::eye(3, 3, CV_64FC1);
    Mat O9x3 = Mat::zeros(9, 3, CV_64FC1);
    Mat O9x1 = Mat::zeros(9, 1, CV_64FC1);

    int idx = 0;
    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++, idx++)
        {
            Mat Hgij = homogeneousInverse(Hg[j]) * Hg[i];
            Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

            Mat Rgij = Hgij(Rect(0, 0, 3, 3));
            Mat Rcij = Hcij(Rect(0, 0, 3, 3));

            Mat tgij = Hgij(Rect(3, 0, 1, 3));
            Mat tcij = Hcij(Rect(3, 0, 1, 3));

            //Eq 10
            Mat a00 = I9 - kron(Rgij, Rcij);
            Mat a01 = O9x3;
            Mat a10 = kron(I3, tcij.t());
            Mat a11 = I3 - Rgij;

            a00.copyTo(A(Rect(0, idx*12, 9, 9)));
            a01.copyTo(A(Rect(9, idx*12, 3, 9)));
            a10.copyTo(A(Rect(0, idx*12 + 9, 9, 3)));
            a11.copyTo(A(Rect(9, idx*12 + 9, 3, 3)));

            O9x1.copyTo(B(Rect(0, idx*12, 1, 9)));
            tgij.copyTo(B(Rect(0, idx*12 + 9, 1, 3)));
        }
    }

    Mat X;
    solve(A, B, X, DECOMP_SVD);

    Mat R = X(Rect(0, 0, 1, 9));
    int newSize[] = {3, 3};
    R = R.reshape(1, 2, newSize);
    //Eq 15
    R_cam2gripper = normalizeRotation(R);
    t_cam2gripper = X(Rect(0, 9, 1, 3));
}

//Reference:
//K. Daniilidis, "Hand-Eye Calibration Using Dual Quaternions."
//In The International Journal of Robotics Research,18(3): 286-298, 1998.
//Matlab code: http://math.loyola.edu/~mili/Calibration/
static void calibrateHandEyeDaniilidis(const std::vector<Mat>& Hg, const std::vector<Mat>& Hc,
                                       Mat& R_cam2gripper, Mat& t_cam2gripper)
{
    int K = static_cast<int>((Hg.size()*Hg.size() - Hg.size()) / 2.0);
    Mat T = Mat::zeros(6*K, 8, CV_64FC1);

    int idx = 0;
    for (size_t i = 0; i < Hg.size(); i++)
    {
        for (size_t j = i+1; j < Hg.size(); j++, idx++)
        {
            Mat Hgij = homogeneousInverse(Hg[j]) * Hg[i];
            Mat Hcij = Hc[j] * homogeneousInverse(Hc[i]);

            Mat dualqa = homogeneous2dualQuaternion(Hgij);
            Mat dualqb = homogeneous2dualQuaternion(Hcij);

            Mat a = dualqa(Rect(0, 1, 1, 3));
            Mat b = dualqb(Rect(0, 1, 1, 3));

            Mat aprime = dualqa(Rect(0, 5, 1, 3));
            Mat bprime = dualqb(Rect(0, 5, 1, 3));

            //Eq 31
            Mat s00 = a - b;
            Mat s01 = skew(a + b);
            Mat s10 = aprime - bprime;
            Mat s11 = skew(aprime + bprime);
            Mat s12 = a - b;
            Mat s13 = skew(a + b);

            s00.copyTo(T(Rect(0, idx*6, 1, 3)));
            s01.copyTo(T(Rect(1, idx*6, 3, 3)));
            s10.copyTo(T(Rect(0, idx*6 + 3, 1, 3)));
            s11.copyTo(T(Rect(1, idx*6 + 3, 3, 3)));
            s12.copyTo(T(Rect(4, idx*6 + 3, 1, 3)));
            s13.copyTo(T(Rect(5, idx*6 + 3, 3, 3)));
        }
    }

    Mat w, u, vt;
    SVDecomp(T, w, u, vt);
    Mat v = vt.t();

    Mat u1 = v(Rect(6, 0, 1, 4));
    Mat v1 = v(Rect(6, 4, 1, 4));
    Mat u2 = v(Rect(7, 0, 1, 4));
    Mat v2 = v(Rect(7, 4, 1, 4));

    //Solves Eq 34, Eq 35
    Mat ma = u1.t()*v1;
    Mat mb = u1.t()*v2 + u2.t()*v1;
    Mat mc = u2.t()*v2;

    double a = ma.at<double>(0,0);
    double b = mb.at<double>(0,0);
    double c = mc.at<double>(0,0);

    double s1 = (-b + sqrt(b*b - 4*a*c)) / (2*a);
    double s2 = (-b - sqrt(b*b - 4*a*c)) / (2*a);

    Mat sol1 = s1*s1*u1.t()*u1 + 2*s1*u1.t()*u2 + u2.t()*u2;
    Mat sol2 = s2*s2*u1.t()*u1 + 2*s2*u1.t()*u2 + u2.t()*u2;
    double s, val;
    if (sol1.at<double>(0,0) > sol2.at<double>(0,0))
    {
        s = s1;
        val = sol1.at<double>(0,0);
    }
    else
    {
        s = s2;
        val = sol2.at<double>(0,0);
    }

    double lambda2 = sqrt(1.0 / val);
    double lambda1 = s * lambda2;

    Mat dualq = lambda1 * v(Rect(6, 0, 1, 8)) + lambda2*v(Rect(7, 0, 1, 8));
    Mat X = dualQuaternion2homogeneous(dualq);

    Mat R = X(Rect(0, 0, 3, 3));
    Mat t = X(Rect(3, 0, 1, 3));
    R_cam2gripper = R;
    t_cam2gripper = t;
}

void calibrateHandEye(InputArrayOfArrays R_gripper2base, InputArrayOfArrays t_gripper2base,
                      InputArrayOfArrays R_target2cam, InputArrayOfArrays t_target2cam,
                      OutputArray R_cam2gripper, OutputArray t_cam2gripper,
                      HandEyeCalibrationMethod method)
{
    CV_Assert(R_gripper2base.isMatVector() && t_gripper2base.isMatVector() &&
              R_target2cam.isMatVector() && t_target2cam.isMatVector());

    std::vector<Mat> R_gripper2base_, t_gripper2base_;
    R_gripper2base.getMatVector(R_gripper2base_);
    t_gripper2base.getMatVector(t_gripper2base_);

    std::vector<Mat> R_target2cam_, t_target2cam_;
    R_target2cam.getMatVector(R_target2cam_);
    t_target2cam.getMatVector(t_target2cam_);

    CV_Assert(R_gripper2base_.size() == t_gripper2base_.size() &&
              R_target2cam_.size() == t_target2cam_.size() &&
              R_gripper2base_.size() == R_target2cam_.size());
    CV_Check(R_gripper2base_.size(), R_gripper2base_.size() >= 3, "At least 3 measurements are needed");

    //Notation used in Tsai paper
    //Defines coordinate transformation from G (gripper) to RW (robot base)
    std::vector<Mat> Hg;
    Hg.reserve(R_gripper2base_.size());
    for (size_t i = 0; i < R_gripper2base_.size(); i++)
    {
        Mat m = Mat::eye(4, 4, CV_64FC1);
        Mat R = m(Rect(0, 0, 3, 3));
        if(R_gripper2base_[i].size() == Size(3, 3))
            R_gripper2base_[i].convertTo(R, CV_64F);
        else
            Rodrigues(R_gripper2base_[i], R);

        Mat t = m(Rect(3, 0, 1, 3));
        t_gripper2base_[i].convertTo(t, CV_64F);

        Hg.push_back(m);
    }

    //Defines coordinate transformation from CW (calibration target) to C (camera)
    std::vector<Mat> Hc;
    Hc.reserve(R_target2cam_.size());
    for (size_t i = 0; i < R_target2cam_.size(); i++)
    {
        Mat m = Mat::eye(4, 4, CV_64FC1);
        Mat R = m(Rect(0, 0, 3, 3));
        if(R_target2cam_[i].size() == Size(3, 3))
            R_target2cam_[i].convertTo(R, CV_64F);
        else
            Rodrigues(R_target2cam_[i], R);

        Mat t = m(Rect(3, 0, 1, 3));
        t_target2cam_[i].convertTo(t, CV_64F);

        Hc.push_back(m);
    }

    Mat Rcg = Mat::eye(3, 3, CV_64FC1);
    Mat Tcg = Mat::zeros(3, 1, CV_64FC1);

    switch (method)
    {
    case CALIB_HAND_EYE_TSAI:
        calibrateHandEyeTsai(Hg, Hc, Rcg, Tcg);
        break;

    case CALIB_HAND_EYE_PARK:
        calibrateHandEyePark(Hg, Hc, Rcg, Tcg);
        break;

    case CALIB_HAND_EYE_HORAUD:
        calibrateHandEyeHoraud(Hg, Hc, Rcg, Tcg);
        break;

    case CALIB_HAND_EYE_ANDREFF:
        calibrateHandEyeAndreff(Hg, Hc, Rcg, Tcg);
        break;

    case CALIB_HAND_EYE_DANIILIDIS:
        calibrateHandEyeDaniilidis(Hg, Hc, Rcg, Tcg);
        break;

    default:
        break;
    }

    Rcg.copyTo(R_cam2gripper);
    Tcg.copyTo(t_cam2gripper);
}

//Reference:
//M. Shah, "Solving the robot-world/hand-eye calibration problem using the kronecker product"
//Journal of Mechanisms and Robotics, vol. 5, p. 031007, 2013.
//Matlab code: http://math.loyola.edu/~mili/Calibration/
static void calibrateRobotWorldHandEyeShah(const std::vector<Mat_<double>>& cRw, const std::vector<Mat_<double>>& ctw,
                                           const std::vector<Mat_<double>>& gRb, const std::vector<Mat_<double>>& gtb,
                                           Matx33d& wRb, Matx31d& wtb, Matx33d& cRg, Matx31d& ctg)
{
    Mat_<double> T = Mat_<double>::zeros(9, 9);
    for (size_t i = 0; i < cRw.size(); i++)
    {
        T += kron(gRb[i], cRw[i]);
    }

    Mat_<double> w, u, vt;
    SVDecomp(T, w, u, vt);

    Mat_<double> RX(3,3), RZ(3,3);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            RX(j,i) = vt(0, i*3+j);
            RZ(j,i) = u(i*3+j, 0);
        }
    }

    wRb = normalizeRotation(RX);
    cRg = normalizeRotation(RZ);
    Mat_<double> Z = Mat(cRg.t()).reshape(1, 9);

    const int n = static_cast<int>(cRw.size());
    Mat_<double> A = Mat_<double>::zeros(3*n, 6);
    Mat_<double> b = Mat_<double>::zeros(3*n, 1);
    Mat_<double> I3 = Mat_<double>::eye(3,3);

    for (int i = 0; i < n; i++)
    {
        Mat cRw_ = -cRw[i];
        cRw_.copyTo(A(Range(i*3, (i+1)*3), Range(0,3)));
        I3.copyTo(A(Range(i*3, (i+1)*3), Range(3,6)));

        Mat ctw_ = ctw[i] - kron(gtb[i].t(), I3) * Z;
        ctw_.copyTo(b(Range(i*3, (i+1)*3), Range::all()));
    }

    Mat_<double> t;
    solve(A, b, t, DECOMP_SVD);

    for (int i = 0; i < 3; i++)
    {
        wtb(i) = t(i);
        ctg(i) = t(i+3);
    }
}

//Reference:
//A. Li, L. Wang, and D. Wu, "Simultaneous robot-world and hand-eye calibration using dual-quaternions and kronecker product"
//International Journal of Physical Sciences, vol. 5, pp. 1530–1536, 2010.
//Matlab code: http://math.loyola.edu/~mili/Calibration/
static void calibrateRobotWorldHandEyeLi(const std::vector<Mat_<double>>& cRw, const std::vector<Mat_<double>>& ctw,
                                         const std::vector<Mat_<double>>& gRb, const std::vector<Mat_<double>>& gtb,
                                         Matx33d& wRb, Matx31d& wtb, Matx33d& cRg, Matx31d& ctg)
{
    const int n = static_cast<int>(cRw.size());
    Mat_<double> A = Mat_<double>::zeros(12*n, 24);
    Mat_<double> b = Mat_<double>::zeros(12*n, 1);
    Mat_<double> I3 = Mat_<double>::eye(3,3);

    for (int i = 0; i < n; i++)
    {
        //Eq 19
        kron(cRw[i], I3).copyTo(A(Range(i*12, i*12 + 9), Range(0, 9)));
        kron(-I3, gRb[i].t()).copyTo(A(Range(i*12, i*12 + 9), Range(9, 18)));

        kron(I3, gtb[i].t()).copyTo(A(Range(i*12 + 9, (i+1)*12), Range(9, 18)));
        Mat cRw_ = -cRw[i];
        cRw_.copyTo(A(Range(i*12 + 9, (i+1)*12), Range(18, 21)));
        I3.copyTo(A(Range(i*12 + 9, (i+1)*12), Range(21, 24)));

        ctw[i].copyTo(b(Range(i*12 + 9, i*12+12), Range::all()));
    }

    Mat_<double> x;
    solve(A, b, x, DECOMP_SVD);

    Mat_<double> RX = x(Range(0,9), Range::all()).reshape(3, 3);
    wRb = normalizeRotation(RX);
    x(Range(18,21), Range::all()).copyTo(wtb);

    Mat_<double> RZ = x(Range(9,18), Range::all()).reshape(3, 3);
    cRg = normalizeRotation(RZ);
    x(Range(21,24), Range::all()).copyTo(ctg);
}

void calibrateRobotWorldHandEye(InputArrayOfArrays R_world2cam, InputArrayOfArrays t_world2cam,
                                InputArrayOfArrays R_base2gripper, InputArrayOfArrays t_base2gripper,
                                OutputArray R_base2world, OutputArray t_base2world,
                                OutputArray R_gripper2cam, OutputArray t_gripper2cam,
                                RobotWorldHandEyeCalibrationMethod method)
{
    CV_Assert(R_base2gripper.isMatVector() && t_base2gripper.isMatVector() &&
              R_world2cam.isMatVector() && t_world2cam.isMatVector());

    std::vector<Mat> R_base2gripper_tmp, t_base2gripper_tmp;
    R_base2gripper.getMatVector(R_base2gripper_tmp);
    t_base2gripper.getMatVector(t_base2gripper_tmp);

    std::vector<Mat> R_world2cam_tmp, t_world2cam_tmp;
    R_world2cam.getMatVector(R_world2cam_tmp);
    t_world2cam.getMatVector(t_world2cam_tmp);

    CV_Assert(R_base2gripper_tmp.size() == t_base2gripper_tmp.size() &&
              R_world2cam_tmp.size() == t_world2cam_tmp.size() &&
              R_base2gripper_tmp.size() == R_world2cam_tmp.size());
    CV_Check(R_base2gripper_tmp.size(), R_base2gripper_tmp.size() >= 3, "At least 3 measurements are needed");

    // Convert to double
    std::vector<Mat_<double>> R_base2gripper_, t_base2gripper_;
    std::vector<Mat_<double>> R_world2cam_, t_world2cam_;

    R_base2gripper_.reserve(R_base2gripper_tmp.size());
    t_base2gripper_.reserve(R_base2gripper_tmp.size());
    R_world2cam_.reserve(R_world2cam_tmp.size());
    t_world2cam_.reserve(R_base2gripper_tmp.size());

    // Convert to rotation matrix if needed
    for (size_t i = 0; i < R_base2gripper_tmp.size(); i++)
    {
        {
            Mat rot = R_base2gripper_tmp[i];
            Mat R(3, 3, CV_64FC1);
            if (rot.size() == Size(3,3))
            {
                rot.convertTo(R, CV_64F);
                R_base2gripper_.push_back(R);
            }
            else
            {
                Rodrigues(rot, R);
                R_base2gripper_.push_back(R);
            }
            Mat tvec = t_base2gripper_tmp[i];
            Mat t;
            tvec.convertTo(t, CV_64F);
            t_base2gripper_.push_back(t);
        }
        {
            Mat rot  = R_world2cam_tmp[i];
            Mat R(3, 3, CV_64FC1);
            if (rot.size() == Size(3,3))
            {
                rot.convertTo(R, CV_64F);
                R_world2cam_.push_back(R);
            }
            else
            {
                Rodrigues(rot, R);
                R_world2cam_.push_back(R);
            }
            Mat tvec = t_world2cam_tmp[i];
            Mat t;
            tvec.convertTo(t, CV_64F);
            t_world2cam_.push_back(t);
        }
    }

    CV_Assert(R_world2cam_.size() == t_world2cam_.size() &&
              R_base2gripper_.size() == t_base2gripper_.size() &&
              R_world2cam_.size() == R_base2gripper_.size());

    Matx33d wRb, cRg;
    Matx31d wtb, ctg;
    switch (method)
    {
    case CALIB_ROBOT_WORLD_HAND_EYE_SHAH:
        calibrateRobotWorldHandEyeShah(R_world2cam_, t_world2cam_, R_base2gripper_, t_base2gripper_, wRb, wtb, cRg, ctg);
        break;

    case CALIB_ROBOT_WORLD_HAND_EYE_LI:
        calibrateRobotWorldHandEyeLi(R_world2cam_, t_world2cam_, R_base2gripper_, t_base2gripper_, wRb, wtb, cRg, ctg);
        break;
    }

    Mat(wRb).copyTo(R_base2world);
    Mat(wtb).copyTo(t_base2world);

    Mat(cRg).copyTo(R_gripper2cam);
    Mat(ctg).copyTo(t_gripper2cam);
}
}
