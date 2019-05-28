// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on file issued with the following license:

/*============================================================================

Copyright 2017 Toby Collins
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "precomp.hpp"
#include "ippe.hpp"

namespace cv {
namespace IPPE {
PoseSolver::PoseSolver() : IPPE_SMALL(1e-3)
{
}

void PoseSolver::solveGeneric(InputArray _objectPoints, InputArray _imagePoints, OutputArray _rvec1, OutputArray _tvec1,
                              float& err1, OutputArray _rvec2, OutputArray _tvec2, float& err2)
{
    Mat normalizedImagePoints;
    if (_imagePoints.getMat().type() == CV_32FC2)
    {
        _imagePoints.getMat().convertTo(normalizedImagePoints, CV_64F);
    }
    else
    {
        normalizedImagePoints = _imagePoints.getMat();
    }

    //solve:
    Mat Ma, Mb;
    solveGeneric(_objectPoints, normalizedImagePoints, Ma, Mb);

    //the two poses computed by IPPE (sorted):
    Mat M1, M2;

    //sort poses by reprojection error:
    sortPosesByReprojError(_objectPoints, normalizedImagePoints, Ma, Mb, M1, M2, err1, err2);

    //fill outputs
    rot2vec(M1.colRange(0, 3).rowRange(0, 3), _rvec1);
    rot2vec(M2.colRange(0, 3).rowRange(0, 3), _rvec2);

    M1.colRange(3, 4).rowRange(0, 3).copyTo(_tvec1);
    M2.colRange(3, 4).rowRange(0, 3).copyTo(_tvec2);
}

void PoseSolver::solveGeneric(InputArray _objectPoints, InputArray _normalizedInputPoints,
                              OutputArray _Ma, OutputArray _Mb)
{
    //argument checking:
    size_t n = static_cast<size_t>(_objectPoints.rows() * _objectPoints.cols()); //number of points
    int objType = _objectPoints.type();
    int type_input = _normalizedInputPoints.type();

    CV_CheckType(objType, objType == CV_32FC3 || objType == CV_64FC3,
                 "Type of _objectPoints must be CV_32FC3 or CV_64FC3" );
    CV_CheckType(type_input, type_input == CV_32FC2 || type_input == CV_64FC2,
                 "Type of _normalizedInputPoints must be CV_32FC3 or CV_64FC3" );
    CV_Assert(_objectPoints.rows() == 1 || _objectPoints.cols() == 1);
    CV_Assert(_objectPoints.rows() >= 4 || _objectPoints.cols() >= 4);
    CV_Assert(_normalizedInputPoints.rows() == 1 || _normalizedInputPoints.cols() == 1);
    CV_Assert(static_cast<size_t>(_objectPoints.rows() * _objectPoints.cols()) == n);

    Mat normalizedInputPoints;
    if (type_input == CV_32FC2)
    {
        _normalizedInputPoints.getMat().convertTo(normalizedInputPoints, CV_64F);
    }
    else
    {
        normalizedInputPoints = _normalizedInputPoints.getMat();
    }

    Mat objectInputPoints;
    if (type_input == CV_32FC3)
    {
        _objectPoints.getMat().convertTo(objectInputPoints, CV_64F);
    }
    else
    {
        objectInputPoints = _objectPoints.getMat();
    }

    Mat canonicalObjPoints;
    Mat MmodelPoints2Canonical;

    //transform object points to the canonical position (zero centred and on the plane z=0):
    makeCanonicalObjectPoints(objectInputPoints, canonicalObjPoints, MmodelPoints2Canonical);

    //compute the homography mapping the model's points to normalizedInputPoints
    Matx33d H;
    HomographyHO::homographyHO(canonicalObjPoints, _normalizedInputPoints, H);

    //now solve
    Mat MaCanon, MbCanon;
    solveCanonicalForm(canonicalObjPoints, normalizedInputPoints, H, MaCanon, MbCanon);

    //transform computed poses to account for canonical transform:
    Mat Ma = MaCanon * MmodelPoints2Canonical;
    Mat Mb = MbCanon * MmodelPoints2Canonical;

    //output poses:
    Ma.copyTo(_Ma);
    Mb.copyTo(_Mb);
}

void PoseSolver::solveCanonicalForm(InputArray _canonicalObjPoints, InputArray _normalizedInputPoints, const Matx33d& H,
                                    OutputArray _Ma, OutputArray _Mb)
{
    _Ma.create(4, 4, CV_64FC1);
    _Mb.create(4, 4, CV_64FC1);

    Mat Ma = _Ma.getMat();
    Mat Mb = _Mb.getMat();

    //initialise poses:
    Ma.setTo(0);
    Ma.at<double>(3, 3) = 1;
    Mb.setTo(0);
    Mb.at<double>(3, 3) = 1;

    //Compute the Jacobian J of the homography at (0,0):
    double j00 = H(0, 0) - H(2, 0) * H(0, 2);
    double j01 = H(0, 1) - H(2, 1) * H(0, 2);
    double j10 = H(1, 0) - H(2, 0) * H(1, 2);
    double j11 = H(1, 1) - H(2, 1) * H(1, 2);

    //Compute the transformation of (0,0) into the image:
    double v0 = H(0, 2);
    double v1 = H(1, 2);

    //compute the two rotation solutions:
    Mat Ra = Ma.colRange(0, 3).rowRange(0, 3);
    Mat Rb = Mb.colRange(0, 3).rowRange(0, 3);
    computeRotations(j00, j01, j10, j11, v0, v1, Ra, Rb);

    //for each rotation solution, compute the corresponding translation solution:
    Mat ta = Ma.colRange(3, 4).rowRange(0, 3);
    Mat tb = Mb.colRange(3, 4).rowRange(0, 3);
    computeTranslation(_canonicalObjPoints, _normalizedInputPoints, Ra, ta);
    computeTranslation(_canonicalObjPoints, _normalizedInputPoints, Rb, tb);
}

void PoseSolver::solveSquare(InputArray _objectPoints, InputArray _imagePoints, OutputArray _rvec1, OutputArray _tvec1,
                             float& err1, OutputArray _rvec2, OutputArray _tvec2, float& err2)
{
    //allocate outputs:
    _rvec1.create(3, 1, CV_64FC1);
    _tvec1.create(3, 1, CV_64FC1);
    _rvec2.create(3, 1, CV_64FC1);
    _tvec2.create(3, 1, CV_64FC1);

    Mat objectPoints2D;

    //generate the object points:
    objectPoints2D.create(1, 4, CV_64FC2);
    Mat objectPoints = _objectPoints.getMat();
    double squareLength;
    if (objectPoints.depth() == CV_32F)
    {
        objectPoints2D.ptr<Vec2d>(0)[0] = Vec2d(objectPoints.ptr<Vec3f>(0)[0](0), objectPoints.ptr<Vec3f>(0)[0](1));
        objectPoints2D.ptr<Vec2d>(0)[1] = Vec2d(objectPoints.ptr<Vec3f>(0)[1](0), objectPoints.ptr<Vec3f>(0)[1](1));
        objectPoints2D.ptr<Vec2d>(0)[2] = Vec2d(objectPoints.ptr<Vec3f>(0)[2](0), objectPoints.ptr<Vec3f>(0)[2](1));
        objectPoints2D.ptr<Vec2d>(0)[3] = Vec2d(objectPoints.ptr<Vec3f>(0)[3](0), objectPoints.ptr<Vec3f>(0)[3](1));

        squareLength = sqrt( (objectPoints.ptr<Vec3f>(0)[1](0) - objectPoints.ptr<Vec3f>(0)[0](0))*
                             (objectPoints.ptr<Vec3f>(0)[1](0) - objectPoints.ptr<Vec3f>(0)[0](0)) +
                             (objectPoints.ptr<Vec3f>(0)[1](1) - objectPoints.ptr<Vec3f>(0)[0](1))*
                             (objectPoints.ptr<Vec3f>(0)[1](1) - objectPoints.ptr<Vec3f>(0)[0](1)) );
    }
    else
    {
        objectPoints2D.ptr<Vec2d>(0)[0] = Vec2d(objectPoints.ptr<Vec3d>(0)[0](0), objectPoints.ptr<Vec3d>(0)[0](1));
        objectPoints2D.ptr<Vec2d>(0)[1] = Vec2d(objectPoints.ptr<Vec3d>(0)[1](0), objectPoints.ptr<Vec3d>(0)[1](1));
        objectPoints2D.ptr<Vec2d>(0)[2] = Vec2d(objectPoints.ptr<Vec3d>(0)[2](0), objectPoints.ptr<Vec3d>(0)[2](1));
        objectPoints2D.ptr<Vec2d>(0)[3] = Vec2d(objectPoints.ptr<Vec3d>(0)[3](0), objectPoints.ptr<Vec3d>(0)[3](1));

        squareLength = sqrt( (objectPoints.ptr<Vec3d>(0)[1](0) - objectPoints.ptr<Vec3d>(0)[0](0))*
                             (objectPoints.ptr<Vec3d>(0)[1](0) - objectPoints.ptr<Vec3d>(0)[0](0)) +
                             (objectPoints.ptr<Vec3d>(0)[1](1) - objectPoints.ptr<Vec3d>(0)[0](1))*
                             (objectPoints.ptr<Vec3d>(0)[1](1) - objectPoints.ptr<Vec3d>(0)[0](1)) );
    }

    Mat H; //homography from canonical object points to normalized pixels

    Mat normalizedInputPoints;
    if (_imagePoints.getMat().type() == CV_32FC2)
    {
        _imagePoints.getMat().convertTo(normalizedInputPoints, CV_64F);
    }
    else
    {
        normalizedInputPoints = _imagePoints.getMat();
    }

    //compute H
    homographyFromSquarePoints(normalizedInputPoints, squareLength / 2.0, H);

    //now solve
    Mat Ma, Mb;
    solveCanonicalForm(objectPoints2D, normalizedInputPoints, H, Ma, Mb);

    //sort poses according to reprojection error:
    Mat M1, M2;
    sortPosesByReprojError(_objectPoints, normalizedInputPoints, Ma, Mb, M1, M2, err1, err2);

    //fill outputs
    rot2vec(M1.colRange(0, 3).rowRange(0, 3), _rvec1);
    rot2vec(M2.colRange(0, 3).rowRange(0, 3), _rvec2);

    M1.colRange(3, 4).rowRange(0, 3).copyTo(_tvec1);
    M2.colRange(3, 4).rowRange(0, 3).copyTo(_tvec2);
}

void PoseSolver::generateSquareObjectCorners3D(double squareLength, OutputArray _objectPoints)
{
    _objectPoints.create(1, 4, CV_64FC3);
    Mat objectPoints = _objectPoints.getMat();
    objectPoints.ptr<Vec3d>(0)[0] = Vec3d(-squareLength / 2.0, squareLength / 2.0, 0.0);
    objectPoints.ptr<Vec3d>(0)[1] = Vec3d(squareLength / 2.0, squareLength / 2.0, 0.0);
    objectPoints.ptr<Vec3d>(0)[2] = Vec3d(squareLength / 2.0, -squareLength / 2.0, 0.0);
    objectPoints.ptr<Vec3d>(0)[3] = Vec3d(-squareLength / 2.0, -squareLength / 2.0, 0.0);
}

void PoseSolver::generateSquareObjectCorners2D(double squareLength, OutputArray _objectPoints)
{
    _objectPoints.create(1, 4, CV_64FC2);
    Mat objectPoints = _objectPoints.getMat();
    objectPoints.ptr<Vec2d>(0)[0] = Vec2d(-squareLength / 2.0, squareLength / 2.0);
    objectPoints.ptr<Vec2d>(0)[1] = Vec2d(squareLength / 2.0, squareLength / 2.0);
    objectPoints.ptr<Vec2d>(0)[2] = Vec2d(squareLength / 2.0, -squareLength / 2.0);
    objectPoints.ptr<Vec2d>(0)[3] = Vec2d(-squareLength / 2.0, -squareLength / 2.0);
}

double PoseSolver::meanSceneDepth(InputArray _objectPoints, InputArray _rvec, InputArray _tvec)
{
    CV_CheckType(_objectPoints.type(), _objectPoints.type() == CV_64FC3,
                 "Type of _objectPoints must be CV_64FC3" );

    size_t n = static_cast<size_t>(_objectPoints.rows() * _objectPoints.cols());
    Mat R;
    Mat q;
    Rodrigues(_rvec, R);
    double zBar = 0;

    for (size_t i = 0; i < n; i++)
    {
        Mat p(_objectPoints.getMat().at<Point3d>(static_cast<int>(i)));
        q = R * p + _tvec.getMat();
        double z;
        if (q.depth() == CV_64F)
        {
            z = q.at<double>(2);
        }
        else
        {
            z = static_cast<double>(q.at<float>(2));
        }
        zBar += z;
    }
    return zBar / static_cast<double>(n);
}

void PoseSolver::rot2vec(InputArray _R, OutputArray _r)
{
    CV_CheckType(_R.type(), _R.type() == CV_64FC1,
                 "Type of _R must be CV_64FC1" );
    CV_Assert(_R.rows() == 3);
    CV_Assert(_R.cols() == 3);

    _r.create(3, 1, CV_64FC1);

    Mat R = _R.getMat();
    Mat rvec = _r.getMat();

    double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
    double w_norm = acos((trace - 1.0) / 2.0);
    double eps = std::numeric_limits<float>::epsilon();
    double d = 1 / (2 * sin(w_norm)) * w_norm;
    if (w_norm < eps) //rotation is the identity
    {
        rvec.setTo(0);
    }
    else
    {
        double c0 = R.at<double>(2, 1) - R.at<double>(1, 2);
        double c1 = R.at<double>(0, 2) - R.at<double>(2, 0);
        double c2 = R.at<double>(1, 0) - R.at<double>(0, 1);
        rvec.at<double>(0) = d * c0;
        rvec.at<double>(1) = d * c1;
        rvec.at<double>(2) = d * c2;
    }
}

void PoseSolver::computeTranslation(InputArray _objectPoints, InputArray _normalizedImgPoints, InputArray _R, OutputArray _t)
{
    //This is solved by building the linear system At = b, where t corresponds to the (unknown) translation.
    //This is then inverted with the associated normal equations to give t = inv(transpose(A)*A)*transpose(A)*b
    //For efficiency we only store the coefficients of (transpose(A)*A) and (transpose(A)*b)

    CV_CheckType(_objectPoints.type(), _objectPoints.type() == CV_64FC2,
                 "Type of _objectPoints must be CV_64FC2" );
    CV_CheckType(_normalizedImgPoints.type(), _normalizedImgPoints.type() == CV_64FC2,
                 "Type of _normalizedImgPoints must be CV_64FC2" );
    CV_CheckType(_R.type(), _R.type() == CV_64FC1,
                 "Type of _R must be CV_64FC1" );
    CV_Assert(_R.rows() == 3 && _R.cols() == 3);
    CV_Assert(_objectPoints.rows() == 1 || _objectPoints.cols() == 1);
    CV_Assert(_normalizedImgPoints.rows() == 1 || _normalizedImgPoints.cols() == 1);

    size_t n = static_cast<size_t>(_normalizedImgPoints.rows() * _normalizedImgPoints.cols());
    CV_Assert(n == static_cast<size_t>(_objectPoints.rows() * _objectPoints.cols()));

    Mat objectPoints = _objectPoints.getMat();
    Mat imgPoints = _normalizedImgPoints.getMat();

    _t.create(3, 1, CV_64FC1);

    Mat R = _R.getMat();

    //coefficients of (transpose(A)*A)
    double ATA00 = static_cast<double>(n);
    double ATA02 = 0;
    double ATA11 = static_cast<double>(n);
    double ATA12 = 0;
    double ATA20 = 0;
    double ATA21 = 0;
    double ATA22 = 0;

    //coefficients of (transpose(A)*b)
    double ATb0 = 0;
    double ATb1 = 0;
    double ATb2 = 0;

    //now loop through each point and increment the coefficients:
    for (int i = 0; i < static_cast<int>(n); i++)
    {
        const Vec2d& objPt = objectPoints.at<Vec2d>(i);
        double rx = R.at<double>(0, 0) * objPt(0) + R.at<double>(0, 1) * objPt(1);
        double ry = R.at<double>(1, 0) * objPt(0) + R.at<double>(1, 1) * objPt(1);
        double rz = R.at<double>(2, 0) * objPt(0) + R.at<double>(2, 1) * objPt(1);

        const Vec2d& imgPt = imgPoints.at<Vec2d>(i);
        double a2 = -imgPt(0);
        double b2 = -imgPt(1);

        ATA02 = ATA02 + a2;
        ATA12 = ATA12 + b2;
        ATA20 = ATA20 + a2;
        ATA21 = ATA21 + b2;
        ATA22 = ATA22 + a2 * a2 + b2 * b2;

        double bx = -a2 * rz - rx;
        double by = -b2 * rz - ry;

        ATb0 = ATb0 + bx;
        ATb1 = ATb1 + by;
        ATb2 = ATb2 + a2 * bx + b2 * by;
    }

    double detAInv = 1.0 / (ATA00 * ATA11 * ATA22 - ATA00 * ATA12 * ATA21 - ATA02 * ATA11 * ATA20);

    //S gives inv(transpose(A)*A)/det(A)^2
    //construct S:
    double S00 = ATA11 * ATA22 - ATA12 * ATA21;
    double S01 = ATA02 * ATA21;
    double S02 = -ATA02 * ATA11;
    double S10 = ATA12 * ATA20;
    double S11 = ATA00 * ATA22 - ATA02 * ATA20;
    double S12 = -ATA00 * ATA12;
    double S20 = -ATA11 * ATA20;
    double S21 = -ATA00 * ATA21;
    double S22 = ATA00 * ATA11;

    //solve t:
    Mat t = _t.getMat();
    t.at<double>(0) = detAInv * (S00 * ATb0 + S01 * ATb1 + S02 * ATb2);
    t.at<double>(1) = detAInv * (S10 * ATb0 + S11 * ATb1 + S12 * ATb2);
    t.at<double>(2) = detAInv * (S20 * ATb0 + S21 * ATb1 + S22 * ATb2);
}

void PoseSolver::computeRotations(double j00, double j01, double j10, double j11, double p, double q, OutputArray _R1, OutputArray _R2)
{
    //This is fairly optimized code which makes it hard to understand. The matlab code is certainly easier to read.
    _R1.create(3, 3, CV_64FC1);
    _R2.create(3, 3, CV_64FC1);

    Matx33d Rv;
    Matx31d v(p, q, 1);
    rotateVec2ZAxis(v,Rv);
    Rv = Rv.t();

    //setup the 2x2 SVD decomposition:
    double rv00 = Rv(0,0);
    double rv01 = Rv(0,1);
    double rv02 = Rv(0,2);

    double rv10 = Rv(1,0);
    double rv11 = Rv(1,1);
    double rv12 = Rv(1,2);

    double rv20 = Rv(2,0);
    double rv21 = Rv(2,1);
    double rv22 = Rv(2,2);

    double b00 = rv00 - p * rv20;
    double b01 = rv01 - p * rv21;
    double b10 = rv10 - q * rv20;
    double b11 = rv11 - q * rv21;

    double dtinv = 1.0 / ((b00 * b11 - b01 * b10));

    double binv00 = dtinv * b11;
    double binv01 = -dtinv * b01;
    double binv10 = -dtinv * b10;
    double binv11 = dtinv * b00;

    double a00 = binv00 * j00 + binv01 * j10;
    double a01 = binv00 * j01 + binv01 * j11;
    double a10 = binv10 * j00 + binv11 * j10;
    double a11 = binv10 * j01 + binv11 * j11;

    //compute the largest singular value of A:
    double ata00 = a00 * a00 + a01 * a01;
    double ata01 = a00 * a10 + a01 * a11;
    double ata11 = a10 * a10 + a11 * a11;

    double gamma2 = 0.5 * (ata00 + ata11 + sqrt((ata00 - ata11) * (ata00 - ata11) + 4.0 * ata01 * ata01));
    if (gamma2 < 0)
        CV_Error(Error::StsNoConv, "gamma2 is negative.");

    double gamma = sqrt(gamma2);

    if (std::fabs(gamma) < std::numeric_limits<float>::epsilon())
        CV_Error(Error::StsNoConv, "gamma is zero.");

    //reconstruct the full rotation matrices:
    double rtilde00 = a00 / gamma;
    double rtilde01 = a01 / gamma;
    double rtilde10 = a10 / gamma;
    double rtilde11 = a11 / gamma;

    double rtilde00_2 = rtilde00 * rtilde00;
    double rtilde01_2 = rtilde01 * rtilde01;
    double rtilde10_2 = rtilde10 * rtilde10;
    double rtilde11_2 = rtilde11 * rtilde11;

    double b0 = sqrt(-rtilde00_2 - rtilde10_2 + 1);
    double b1 = sqrt(-rtilde01_2 - rtilde11_2 + 1);
    double sp = (-rtilde00 * rtilde01 - rtilde10 * rtilde11);

    if (sp < 0)
    {
        b1 = -b1;
    }

    //store results:
    Mat R1 = _R1.getMat();
    Mat R2 = _R2.getMat();

    R1.at<double>(0, 0) = (rtilde00)*rv00 + (rtilde10)*rv01 + (b0)*rv02;
    R1.at<double>(0, 1) = (rtilde01)*rv00 + (rtilde11)*rv01 + (b1)*rv02;
    R1.at<double>(0, 2) = (b1 * rtilde10 - b0 * rtilde11) * rv00 + (b0 * rtilde01 - b1 * rtilde00) * rv01 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv02;
    R1.at<double>(1, 0) = (rtilde00)*rv10 + (rtilde10)*rv11 + (b0)*rv12;
    R1.at<double>(1, 1) = (rtilde01)*rv10 + (rtilde11)*rv11 + (b1)*rv12;
    R1.at<double>(1, 2) = (b1 * rtilde10 - b0 * rtilde11) * rv10 + (b0 * rtilde01 - b1 * rtilde00) * rv11 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv12;
    R1.at<double>(2, 0) = (rtilde00)*rv20 + (rtilde10)*rv21 + (b0)*rv22;
    R1.at<double>(2, 1) = (rtilde01)*rv20 + (rtilde11)*rv21 + (b1)*rv22;
    R1.at<double>(2, 2) = (b1 * rtilde10 - b0 * rtilde11) * rv20 + (b0 * rtilde01 - b1 * rtilde00) * rv21 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv22;

    R2.at<double>(0, 0) = (rtilde00)*rv00 + (rtilde10)*rv01 + (-b0) * rv02;
    R2.at<double>(0, 1) = (rtilde01)*rv00 + (rtilde11)*rv01 + (-b1) * rv02;
    R2.at<double>(0, 2) = (b0 * rtilde11 - b1 * rtilde10) * rv00 + (b1 * rtilde00 - b0 * rtilde01) * rv01 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv02;
    R2.at<double>(1, 0) = (rtilde00)*rv10 + (rtilde10)*rv11 + (-b0) * rv12;
    R2.at<double>(1, 1) = (rtilde01)*rv10 + (rtilde11)*rv11 + (-b1) * rv12;
    R2.at<double>(1, 2) = (b0 * rtilde11 - b1 * rtilde10) * rv10 + (b1 * rtilde00 - b0 * rtilde01) * rv11 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv12;
    R2.at<double>(2, 0) = (rtilde00)*rv20 + (rtilde10)*rv21 + (-b0) * rv22;
    R2.at<double>(2, 1) = (rtilde01)*rv20 + (rtilde11)*rv21 + (-b1) * rv22;
    R2.at<double>(2, 2) = (b0 * rtilde11 - b1 * rtilde10) * rv20 + (b1 * rtilde00 - b0 * rtilde01) * rv21 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv22;
}

void PoseSolver::homographyFromSquarePoints(InputArray _targetPoints, double halfLength, OutputArray H_)
{
    CV_CheckType(_targetPoints.type(), _targetPoints.type() == CV_32FC2 || _targetPoints.type() == CV_64FC2,
                 "Type of _targetPoints must be CV_32FC2 or CV_64FC2" );

    Mat pts = _targetPoints.getMat();

    double p1x, p1y;
    double p2x, p2y;
    double p3x, p3y;
    double p4x, p4y;

    if (_targetPoints.type() == CV_32FC2)
    {
        p1x = -pts.at<Vec2f>(0)(0);
        p1y = -pts.at<Vec2f>(0)(1);

        p2x = -pts.at<Vec2f>(1)(0);
        p2y = -pts.at<Vec2f>(1)(1);

        p3x = -pts.at<Vec2f>(2)(0);
        p3y = -pts.at<Vec2f>(2)(1);

        p4x = -pts.at<Vec2f>(3)(0);
        p4y = -pts.at<Vec2f>(3)(1);
    }
    else
    {
        p1x = -pts.at<Vec2d>(0)(0);
        p1y = -pts.at<Vec2d>(0)(1);

        p2x = -pts.at<Vec2d>(1)(0);
        p2y = -pts.at<Vec2d>(1)(1);

        p3x = -pts.at<Vec2d>(2)(0);
        p3y = -pts.at<Vec2d>(2)(1);

        p4x = -pts.at<Vec2d>(3)(0);
        p4y = -pts.at<Vec2d>(3)(1);
    }

    //analytic solution:
    double det = (halfLength * (p1x * p2y - p2x * p1y - p1x * p4y + p2x * p3y - p3x * p2y + p4x * p1y + p3x * p4y - p4x * p3y));
    if (abs(det) < 1e-9)
        CV_Error(Error::StsNoConv, "Determinant is zero!");
    double detsInv = -1 / det;

    Matx33d H;
    H(0, 0) = detsInv * (p1x * p3x * p2y - p2x * p3x * p1y - p1x * p4x * p2y + p2x * p4x * p1y - p1x * p3x * p4y + p1x * p4x * p3y + p2x * p3x * p4y - p2x * p4x * p3y);
    H(0, 1) = detsInv * (p1x * p2x * p3y - p1x * p3x * p2y - p1x * p2x * p4y + p2x * p4x * p1y + p1x * p3x * p4y - p3x * p4x * p1y - p2x * p4x * p3y + p3x * p4x * p2y);
    H(0, 2) = detsInv * halfLength * (p1x * p2x * p3y - p2x * p3x * p1y - p1x * p2x * p4y + p1x * p4x * p2y - p1x * p4x * p3y + p3x * p4x * p1y + p2x * p3x * p4y - p3x * p4x * p2y);
    H(1, 0) = detsInv * (p1x * p2y * p3y - p2x * p1y * p3y - p1x * p2y * p4y + p2x * p1y * p4y - p3x * p1y * p4y + p4x * p1y * p3y + p3x * p2y * p4y - p4x * p2y * p3y);
    H(1, 1) = detsInv * (p2x * p1y * p3y - p3x * p1y * p2y - p1x * p2y * p4y + p4x * p1y * p2y + p1x * p3y * p4y - p4x * p1y * p3y - p2x * p3y * p4y + p3x * p2y * p4y);
    H(1, 2) = detsInv * halfLength * (p1x * p2y * p3y - p3x * p1y * p2y - p2x * p1y * p4y + p4x * p1y * p2y - p1x * p3y * p4y + p3x * p1y * p4y + p2x * p3y * p4y - p4x * p2y * p3y);
    H(2, 0) = -detsInv * (p1x * p3y - p3x * p1y - p1x * p4y - p2x * p3y + p3x * p2y + p4x * p1y + p2x * p4y - p4x * p2y);
    H(2, 1) = detsInv * (p1x * p2y - p2x * p1y - p1x * p3y + p3x * p1y + p2x * p4y - p4x * p2y - p3x * p4y + p4x * p3y);
    H(2, 2) = 1.0;

    Mat(H, false).copyTo(H_);
}

void PoseSolver::makeCanonicalObjectPoints(InputArray _objectPoints, OutputArray _canonicalObjPoints, OutputArray _MmodelPoints2Canonical)
{
    int objType = _objectPoints.type();
    CV_CheckType(objType, objType == CV_32FC3 || objType == CV_64FC3,
                 "Type of _objectPoints must be CV_32FC3 or CV_64FC3" );

    int n = _objectPoints.rows() * _objectPoints.cols();

    _canonicalObjPoints.create(1, n, CV_64FC2);

    Mat objectPoints = _objectPoints.getMat();
    Mat canonicalObjPoints = _canonicalObjPoints.getMat();

    Mat UZero(3, n, CV_64FC1);

    double xBar = 0;
    double yBar = 0;
    double zBar = 0;
    bool isOnZPlane = true;
    for (int i = 0; i < n; i++)
    {
        double x, y, z;
        if (objType == CV_32FC3)
        {
            x = static_cast<double>(objectPoints.at<Vec3f>(i)[0]);
            y = static_cast<double>(objectPoints.at<Vec3f>(i)[1]);
            z = static_cast<double>(objectPoints.at<Vec3f>(i)[2]);
        }
        else
        {
            x = objectPoints.at<Vec3d>(i)[0];
            y = objectPoints.at<Vec3d>(i)[1];
            z = objectPoints.at<Vec3d>(i)[2];
        }

        if (abs(z) > IPPE_SMALL)
        {
            isOnZPlane = false;
        }

        xBar += x;
        yBar += y;
        zBar += z;

        UZero.at<double>(0, i) = x;
        UZero.at<double>(1, i) = y;
        UZero.at<double>(2, i) = z;
    }
    xBar = xBar / static_cast<double>(n);
    yBar = yBar / static_cast<double>(n);
    zBar = zBar / static_cast<double>(n);

    for (int i = 0; i < n; i++)
    {
        UZero.at<double>(0, i) -= xBar;
        UZero.at<double>(1, i) -= yBar;
        UZero.at<double>(2, i) -= zBar;
    }

    Matx44d MCenter = Matx44d::eye();
    MCenter(0, 3) = -xBar;
    MCenter(1, 3) = -yBar;
    MCenter(2, 3) = -zBar;

    if (isOnZPlane)
    {
        //MmodelPoints2Canonical is given by MCenter
        Mat(MCenter, false).copyTo(_MmodelPoints2Canonical);
        for (int i = 0; i < n; i++)
        {
            canonicalObjPoints.at<Vec2d>(i)[0] = UZero.at<double>(0, i);
            canonicalObjPoints.at<Vec2d>(i)[1] = UZero.at<double>(1, i);
        }
    }
    else
    {
        Mat UZeroAligned(3, n, CV_64FC1);
        Matx33d R; //rotation that rotates objectPoints to the plane z=0

        if (!computeObjextSpaceR3Pts(objectPoints,R))
        {
            //we could not compute R, problably because there is a duplicate point in {objectPoints(0),objectPoints(1),objectPoints(2)}.
            //So we compute it with the SVD (which is slower):
            computeObjextSpaceRSvD(UZero,R);
        }

        UZeroAligned = R * UZero;

        for (int i = 0; i < n; i++)
        {
            canonicalObjPoints.at<Vec2d>(i)[0] = UZeroAligned.at<double>(0, i);
            canonicalObjPoints.at<Vec2d>(i)[1] = UZeroAligned.at<double>(1, i);
            if (abs(UZeroAligned.at<double>(2, i)) > IPPE_SMALL)
                CV_Error(Error::StsNoConv, "Cannot transform object points to the plane z=0!");
        }

        Matx44d MRot = Matx44d::zeros();
        MRot(3, 3) = 1;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                MRot(i,j) = R(i,j);
            }
        }
        Matx44d Mb = MRot * MCenter;
        Mat(Mb, false).copyTo(_MmodelPoints2Canonical);
    }
}

void PoseSolver::evalReprojError(InputArray _objectPoints, InputArray _imagePoints, InputArray _M, float& err)
{
    Mat projectedPoints;
    Mat imagePoints = _imagePoints.getMat();
    Mat r;
    rot2vec(_M.getMat().colRange(0, 3).rowRange(0, 3), r);

    Mat K = Mat::eye(3, 3, CV_64FC1);
    Mat dist;
    projectPoints(_objectPoints, r, _M.getMat().colRange(3, 4).rowRange(0, 3), K, dist, projectedPoints);

    err = 0;
    int n = _objectPoints.rows() * _objectPoints.cols();

    float dx, dy;
    const int projPtsDepth = projectedPoints.depth();
    for (int i = 0; i < n; i++)
    {
        if (projPtsDepth == CV_32F)
        {
            dx = projectedPoints.at<Vec2f>(i)[0] - static_cast<float>(imagePoints.at<Vec2d>(i)[0]);
            dy = projectedPoints.at<Vec2f>(i)[1] - static_cast<float>(imagePoints.at<Vec2d>(i)[1]);
        }
        else
        {
            dx = static_cast<float>(projectedPoints.at<Vec2d>(i)[0] - imagePoints.at<Vec2d>(i)[0]);
            dy = static_cast<float>(projectedPoints.at<Vec2d>(i)[1] - imagePoints.at<Vec2d>(i)[1]);
        }

        err += dx * dx + dy * dy;
    }
    err = sqrt(err / (2.0f * n));
}

void PoseSolver::sortPosesByReprojError(InputArray _objectPoints, InputArray _imagePoints, InputArray _Ma, InputArray _Mb,
                                        OutputArray _M1, OutputArray _M2, float& err1, float& err2)
{
    float erra, errb;
    evalReprojError(_objectPoints, _imagePoints, _Ma, erra);
    evalReprojError(_objectPoints, _imagePoints, _Mb, errb);
    if (erra < errb)
    {
        err1 = erra;
        _Ma.copyTo(_M1);

        err2 = errb;
        _Mb.copyTo(_M2);
    }
    else
    {
        err1 = errb;
        _Mb.copyTo(_M1);

        err2 = erra;
        _Ma.copyTo(_M2);
    }
}

void PoseSolver::rotateVec2ZAxis(const Matx31d& a, Matx33d& Ra)
{
    double ax = a(0);
    double ay = a(1);
    double az = a(2);

    double nrm = sqrt(ax*ax + ay*ay + az*az);
    ax = ax/nrm;
    ay = ay/nrm;
    az = az/nrm;

    double c = az;

    if (abs(1.0+c) < std::numeric_limits<float>::epsilon())
    {
        Ra = Matx33d::zeros();
        Ra(0,0) = 1.0;
        Ra(1,1) = 1.0;
        Ra(2,2) = -1.0;
    }
    else
    {
        double d = 1.0/(1.0+c);
        double ax2 = ax*ax;
        double ay2 = ay*ay;
        double axay = ax*ay;

        Ra(0,0) = -ax2*d + 1.0;
        Ra(0,1) = -axay*d;
        Ra(0,2) = -ax;

        Ra(1,0) = -axay*d;
        Ra(1,1) = -ay2*d + 1.0;
        Ra(1,2) = -ay;

        Ra(2,0) = ax;
        Ra(2,1) = ay;
        Ra(2,2) = 1.0 - (ax2 + ay2)*d;
    }
}

bool PoseSolver::computeObjextSpaceR3Pts(InputArray _objectPoints, Matx33d& R)
{
    bool ret; //return argument
    double p1x,p1y,p1z;
    double p2x,p2y,p2z;
    double p3x,p3y,p3z;

    Mat objectPoints = _objectPoints.getMat();
    if (objectPoints.type() == CV_32FC3)
    {
        p1x = objectPoints.at<Vec3f>(0)[0];
        p1y = objectPoints.at<Vec3f>(0)[1];
        p1z = objectPoints.at<Vec3f>(0)[2];

        p2x = objectPoints.at<Vec3f>(1)[0];
        p2y = objectPoints.at<Vec3f>(1)[1];
        p2z = objectPoints.at<Vec3f>(1)[2];

        p3x = objectPoints.at<Vec3f>(2)[0];
        p3y = objectPoints.at<Vec3f>(2)[1];
        p3z = objectPoints.at<Vec3f>(2)[2];
    }
    else
    {
        p1x = objectPoints.at<Vec3d>(0)[0];
        p1y = objectPoints.at<Vec3d>(0)[1];
        p1z = objectPoints.at<Vec3d>(0)[2];

        p2x = objectPoints.at<Vec3d>(1)[0];
        p2y = objectPoints.at<Vec3d>(1)[1];
        p2z = objectPoints.at<Vec3d>(1)[2];

        p3x = objectPoints.at<Vec3d>(2)[0];
        p3y = objectPoints.at<Vec3d>(2)[1];
        p3z = objectPoints.at<Vec3d>(2)[2];
    }

    double nx = (p1y - p2y)*(p1z - p3z) - (p1y - p3y)*(p1z - p2z);
    double ny = (p1x - p3x)*(p1z - p2z) - (p1x - p2x)*(p1z - p3z);
    double nz = (p1x - p2x)*(p1y - p3y) - (p1x - p3x)*(p1y - p2y);

    double nrm = sqrt(nx*nx+ ny*ny + nz*nz);
    if (nrm > IPPE_SMALL)
    {
        nx = nx/nrm;
        ny = ny/nrm;
        nz = nz/nrm;
        Matx31d v(nx, ny, nz);
        rotateVec2ZAxis(v,R);
        ret = true;
    }
    else
    {
        ret = false;
    }
    return ret;
}

void PoseSolver::computeObjextSpaceRSvD(InputArray _objectPointsZeroMean, OutputArray _R)
{
    _R.create(3, 3, CV_64FC1);
    Mat R = _R.getMat();

    //we could not compute R with the first three points, so lets use the SVD
    SVD s;
    Mat W, U, VT;
    s.compute(_objectPointsZeroMean.getMat() * _objectPointsZeroMean.getMat().t(), W, U, VT);
    double s3 = W.at<double>(2);
    double s2 = W.at<double>(1);

    //check if points are coplanar:
    CV_Assert(s3 / s2 < IPPE_SMALL);

    R = U.t();
    if (determinant(R) < 0)
    {
        //this ensures R is a rotation matrix and not a general unitary matrix:
        R.at<double>(2, 0) = -R.at<double>(2, 0);
        R.at<double>(2, 1) = -R.at<double>(2, 1);
        R.at<double>(2, 2) = -R.at<double>(2, 2);
    }
}
} //namespace IPPE

namespace HomographyHO {
void normalizeDataIsotropic(InputArray _Data, OutputArray _DataN, OutputArray _T, OutputArray _Ti)
{
    Mat Data = _Data.getMat();
    int numPoints = Data.rows * Data.cols;
    CV_Assert(Data.rows == 1 || Data.cols == 1);
    CV_Assert(Data.channels() == 2 || Data.channels() == 3);
    CV_Assert(numPoints >= 4);

    int dataType = _Data.type();
    CV_CheckType(dataType, dataType == CV_32FC2 || dataType == CV_32FC3 || dataType == CV_64FC2 || dataType == CV_64FC3,
                 "Type of _Data must be one of CV_32FC2, CV_32FC3, CV_64FC2, CV_64FC3");

    _DataN.create(2, numPoints, CV_64FC1);

    _T.create(3, 3, CV_64FC1);
    _Ti.create(3, 3, CV_64FC1);

    Mat DataN = _DataN.getMat();
    Mat T = _T.getMat();
    Mat Ti = _Ti.getMat();

    _T.setTo(0);
    _Ti.setTo(0);

    int numChannels = Data.channels();
    double xm = 0;
    double ym = 0;
    for (int i = 0; i < numPoints; i++)
    {
        if (numChannels == 2)
        {
            if (dataType == CV_32FC2)
            {
                xm = xm + Data.at<Vec2f>(i)[0];
                ym = ym + Data.at<Vec2f>(i)[1];
            }
            else
            {
                xm = xm + Data.at<Vec2d>(i)[0];
                ym = ym + Data.at<Vec2d>(i)[1];
            }
        }
        else
        {
            if (dataType == CV_32FC3)
            {
                xm = xm + Data.at<Vec3f>(i)[0];
                ym = ym + Data.at<Vec3f>(i)[1];
            }
            else
            {
                xm = xm + Data.at<Vec3d>(i)[0];
                ym = ym + Data.at<Vec3d>(i)[1];
            }
        }
    }
    xm = xm / static_cast<double>(numPoints);
    ym = ym / static_cast<double>(numPoints);

    double kappa = 0;
    double xh, yh;

    for (int i = 0; i < numPoints; i++)
    {

        if (numChannels == 2)
        {
            if (dataType == CV_32FC2)
            {
                xh = Data.at<Vec2f>(i)[0] - xm;
                yh = Data.at<Vec2f>(i)[1] - ym;
            }
            else
            {
                xh = Data.at<Vec2d>(i)[0] - xm;
                yh = Data.at<Vec2d>(i)[1] - ym;
            }
        }
        else
        {
            if (dataType == CV_32FC3)
            {
                xh = Data.at<Vec3f>(i)[0] - xm;
                yh = Data.at<Vec3f>(i)[1] - ym;
            }
            else
            {
                xh = Data.at<Vec3d>(i)[0] - xm;
                yh = Data.at<Vec3d>(i)[1] - ym;
            }
        }

        DataN.at<double>(0, i) = xh;
        DataN.at<double>(1, i) = yh;
        kappa = kappa + xh * xh + yh * yh;
    }
    double beta = sqrt(2 * numPoints / kappa);
    DataN = DataN * beta;

    T.at<double>(0, 0) = 1.0 / beta;
    T.at<double>(1, 1) = 1.0 / beta;

    T.at<double>(0, 2) = xm;
    T.at<double>(1, 2) = ym;

    T.at<double>(2, 2) = 1;

    Ti.at<double>(0, 0) = beta;
    Ti.at<double>(1, 1) = beta;

    Ti.at<double>(0, 2) = -beta * xm;
    Ti.at<double>(1, 2) = -beta * ym;

    Ti.at<double>(2, 2) = 1;
}

void homographyHO(InputArray _srcPoints, InputArray _targPoints, Matx33d& H)
{
    Mat DataA, DataB, TA, TAi, TB, TBi;

    HomographyHO::normalizeDataIsotropic(_srcPoints, DataA, TA, TAi);
    HomographyHO::normalizeDataIsotropic(_targPoints, DataB, TB, TBi);

    int n = DataA.cols;
    CV_Assert(n == DataB.cols);

    Mat C1(1, n, CV_64FC1);
    Mat C2(1, n, CV_64FC1);
    Mat C3(1, n, CV_64FC1);
    Mat C4(1, n, CV_64FC1);

    double mC1 = 0, mC2 = 0, mC3 = 0, mC4 = 0;

    for (int i = 0; i < n; i++)
    {
        C1.at<double>(0, i) = -DataB.at<double>(0, i) * DataA.at<double>(0, i);
        C2.at<double>(0, i) = -DataB.at<double>(0, i) * DataA.at<double>(1, i);
        C3.at<double>(0, i) = -DataB.at<double>(1, i) * DataA.at<double>(0, i);
        C4.at<double>(0, i) = -DataB.at<double>(1, i) * DataA.at<double>(1, i);

        mC1 += C1.at<double>(0, i);
        mC2 += C2.at<double>(0, i);
        mC3 += C3.at<double>(0, i);
        mC4 += C4.at<double>(0, i);
    }

    mC1 /= n;
    mC2 /= n;
    mC3 /= n;
    mC4 /= n;

    Mat Mx(n, 3, CV_64FC1);
    Mat My(n, 3, CV_64FC1);

    for (int i = 0; i < n; i++)
    {
        Mx.at<double>(i, 0) = C1.at<double>(0, i) - mC1;
        Mx.at<double>(i, 1) = C2.at<double>(0, i) - mC2;
        Mx.at<double>(i, 2) = -DataB.at<double>(0, i);

        My.at<double>(i, 0) = C3.at<double>(0, i) - mC3;
        My.at<double>(i, 1) = C4.at<double>(0, i) - mC4;
        My.at<double>(i, 2) = -DataB.at<double>(1, i);
    }

    Mat DataAT, DataADataAT;

    transpose(DataA, DataAT);
    DataADataAT = DataA * DataAT;
    double dt = DataADataAT.at<double>(0, 0) * DataADataAT.at<double>(1, 1) - DataADataAT.at<double>(0, 1) * DataADataAT.at<double>(1, 0);

    Mat DataADataATi(2, 2, CV_64FC1);
    DataADataATi.at<double>(0, 0) = DataADataAT.at<double>(1, 1) / dt;
    DataADataATi.at<double>(0, 1) = -DataADataAT.at<double>(0, 1) / dt;
    DataADataATi.at<double>(1, 0) = -DataADataAT.at<double>(1, 0) / dt;
    DataADataATi.at<double>(1, 1) = DataADataAT.at<double>(0, 0) / dt;

    Mat Pp = DataADataATi * DataA;

    Mat Bx = Pp * Mx;
    Mat By = Pp * My;

    Mat Ex = DataAT * Bx;
    Mat Ey = DataAT * By;

    Mat D(2 * n, 3, CV_64FC1);

    for (int i = 0; i < n; i++)
    {
        D.at<double>(i, 0) = Mx.at<double>(i, 0) - Ex.at<double>(i, 0);
        D.at<double>(i, 1) = Mx.at<double>(i, 1) - Ex.at<double>(i, 1);
        D.at<double>(i, 2) = Mx.at<double>(i, 2) - Ex.at<double>(i, 2);

        D.at<double>(i + n, 0) = My.at<double>(i, 0) - Ey.at<double>(i, 0);
        D.at<double>(i + n, 1) = My.at<double>(i, 1) - Ey.at<double>(i, 1);
        D.at<double>(i + n, 2) = My.at<double>(i, 2) - Ey.at<double>(i, 2);
    }

    Mat DT, DDT;
    transpose(D, DT);
    DDT = DT * D;

    Mat S, U;
    eigen(DDT, S, U);

    Mat h789(3, 1, CV_64FC1);
    h789.at<double>(0, 0) = U.at<double>(2, 0);
    h789.at<double>(1, 0) = U.at<double>(2, 1);
    h789.at<double>(2, 0) = U.at<double>(2, 2);

    Mat h12 = -Bx * h789;
    Mat h45 = -By * h789;

    double h3 = -(mC1 * h789.at<double>(0, 0) + mC2 * h789.at<double>(1, 0));
    double h6 = -(mC3 * h789.at<double>(0, 0) + mC4 * h789.at<double>(1, 0));

    H(0, 0) = h12.at<double>(0, 0);
    H(0, 1) = h12.at<double>(1, 0);
    H(0, 2) = h3;

    H(1, 0) = h45.at<double>(0, 0);
    H(1, 1) = h45.at<double>(1, 0);
    H(1, 2) = h6;

    H(2, 0) = h789.at<double>(0, 0);
    H(2, 1) = h789.at<double>(1, 0);
    H(2, 2) = h789.at<double>(2, 0);

    H = Mat(TB * H * TAi);
    double h22_inv = 1 / H(2, 2);
    H = H * h22_inv;
}
}
} //namespace cv
