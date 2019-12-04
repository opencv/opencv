// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

#include "upnp.h"
#include "dls.h"
#include "epnp.h"
#include "p3p.h"
#include "ap3p.h"
#include "ippe.hpp"
#include "opencv2/core/core_c.h"
#include "calib3d_c_api.h"

using namespace cv;

static bool approxEqual(double a, double b, double eps)
{
    return std::fabs(a-b) < eps;
}

PnPSolver::PnPSolver(bool _withGeometricTests): withGeometricTests(_withGeometricTests)
{

}

PnPSolver::~PnPSolver()
{

}

std::vector<double> PnPSolver::solveProblem(InputArray _opoints, InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs, bool sortOnReprojectionError) const
{
    std::vector<double> reprojErrs;

#if defined _DEBUG || defined CV_STATIC_ANALYSIS
    checkArgTypes(_opoints,_ipoints);
    checkNumberOfPoints(_opoints);
#endif
    rVecs.clear();
    tVecs.clear();
    if (!checkNumberOfPoints(_opoints))
    {
        return reprojErrs;
    }
    if (withGeometricTests)
    {
        if (validProblem(_opoints,_ipoints))
        {
            solve(_opoints,_ipoints,rVecs,tVecs);
        }
    }
    else
    {
        solve(_opoints,_ipoints,rVecs,tVecs);
    }
    if (sortOnReprojectionError)
    {
        sortPosesOnReprojectionError(_opoints, _ipoints, makeIdentityIntrinsic(), cv::noArray(),  rVecs, tVecs, reprojErrs);
    }
    return reprojErrs;
}


cv::Mat PnPSolver::makeIdentityIntrinsic() const
{

    cv::Mat K(3,3,CV_64FC1);
    K.setTo(0.0);
    K.at<double>(0,0) = 1.0;
    K.at<double>(1,1) = 1.0;
    K.at<double>(2,2) = 1.0;
    return K;
}

size_t PnPSolver::getNumberOfPoints(cv::InputArray _opoints) const
{
    Mat opoints = _opoints.getMat();
    return opoints.checkVector(3, CV_64F);
}

void PnPSolver::checkArgTypes(InputArray _opoints, InputArray _ipoints) const
{
    const Mat opoints = _opoints.getMat();
    const Mat ipoints = _ipoints.getMat();
    CV_DbgAssert(opoints.depth() == CV_64F);
    CV_DbgAssert(ipoints.depth() == CV_64F);
}

bool PnPSolver::checkNumberOfPoints(InputArray _opoints) const
{
    bool pmin,pmax;
    const size_t n = getNumberOfPoints(_opoints);
    pmin = n>=static_cast<size_t>(minPointNumber());
    if (maxPointNumber()!=-1)
    {
        pmax = n<=static_cast<size_t>(maxPointNumber());
    }
    else
    {
        pmax = true;
    }
    return pmin & pmax;
}

bool PnPSolver::validProblem(InputArray _opoints,InputArray _ipoints) const
{
    Mat opoints_ = _opoints.getMat(), ipoints_ = _ipoints.getMat();
    Mat opoints, ipoints;

    opoints_.convertTo(opoints, CV_64F);
    if (!_ipoints.empty())
    {
        ipoints_.convertTo(ipoints, CV_64F);
        if (ipoints.channels()==1)
        {
            ipoints = ipoints.reshape(2);
        }
    }
    if (opoints.channels()==1)
    {
        opoints = opoints.reshape(3);
    }


    if (!checkNumberOfPoints(opoints))
    {
        return false;
    }
    else
    {
        double coplaneThresh = 1e-5;
        double colinThresh = 1e-5;

        cv::Mat singValues;
        get3DPointsetShape(_opoints, singValues);
        bool isCoplanar = singValues.at<double>(2) < singValues.at<double>(1) * coplaneThresh;
        bool isColinear = singValues.at<double>(1) < singValues.at<double>(0) * colinThresh;

        if (isColinear)
        {
            return false;
        }
        else {
            bool t = true;
            if (this->requires3DObject())
            {
                t = t && !isCoplanar;
            }
            if (this->requiresPlanarObject())
            {
                t = t && isCoplanar;
            }
            if (this->requiresPlanarTagObject())
            {
                t = t && isPlanarTag(opoints);
            }
            return t & noArtificialDegeneracy(opoints,ipoints);
        }

    }
}

bool PnPSolver::noArtificialDegeneracy(InputArray opoints,InputArray ipoints) const
{
    (void)opoints;
    (void)ipoints;
    return true;
}



bool PnPSolver::isPlanarTag(InputArray _opoints) const
{
    const Mat opoints = _opoints.getMat();
    double Xs[4][3];
    if (opoints.depth() == CV_32F)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Xs[i][j] = static_cast<double>(opoints.ptr<Vec3f>(0)[i](j));
            }
        }
    }
    else
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Xs[i][j] = opoints.ptr<Vec3d>(0)[i](j);
            }
        }
    }

    const double equalThreshold = 1e-9;
    bool pass = true;
    //Z must be zero
    bool t;
    for (int i = 0; i < 4; i++)
    {
        t = approxEqual(Xs[i][2], 0, equalThreshold);
        pass = pass & t;

    }
    t = approxEqual(Xs[0][1], Xs[1][1], equalThreshold);
    pass = pass &  t;

    t =  approxEqual(Xs[2][1], Xs[3][1], equalThreshold);
    pass = pass &  t;

    t =  approxEqual(Xs[0][0], Xs[3][0], equalThreshold);
    pass = pass & t;

    t = approxEqual(Xs[1][0], Xs[2][0], equalThreshold);
    pass = pass &  t;

    t = approxEqual(Xs[1][0], Xs[1][1], equalThreshold);
    pass = pass & t;

    t = approxEqual(Xs[3][0], Xs[3][1], equalThreshold);
    pass = pass & t;
    return pass;
}



PnPSolverP3PComplete::PnPSolverP3PComplete(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

Ptr<PnPSolverP3PComplete> PnPSolverP3PComplete::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverP3PComplete>(_withGeometricTests);
}

void PnPSolverP3PComplete::solve(cv::InputArray _opoints, cv::InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    std::vector<Mat> rMats, tvecs;
    const auto K = makeIdentityIntrinsic();
    auto solver = p3p(K);
    solver.solve(rMats,tvecs,_opoints.getMat(),_ipoints.getMat());
    for (size_t i = 0; i < rMats.size(); i++)
    {
        Mat rvec;
        Rodrigues(rMats[i],rvec);
        rVecs.push_back(rvec);
        tVecs.push_back(tvecs[i]);
    }
}


int PnPSolverP3PComplete::minPointNumber() const
{
    return 3;
}

int PnPSolverP3PComplete::maxPointNumber() const
{
    return 4;
}

bool PnPSolverP3PComplete::requires3DObject() const
{
    return false;
}

bool PnPSolverP3PComplete::requiresPlanarObject() const
{
    return false; //this is true only when the number of points is 4
}

bool PnPSolverP3PComplete::requiresPlanarTagObject() const
{
    return false; //this is true only when the number of points is 4
}




PnPSolverAP3P::PnPSolverAP3P(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

Ptr<PnPSolverAP3P> PnPSolverAP3P::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverAP3P>(_withGeometricTests);
}

void PnPSolverAP3P::solve(cv::InputArray _opoints, cv::InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    std::vector<Mat> rMats, tvecs;
    const auto K = makeIdentityIntrinsic();
    auto solver = ap3p(K);
    solver.solve(rMats,tvecs,_opoints.getMat(),_ipoints.getMat());
    for (size_t i = 0; i < rMats.size(); i++)
    {
        Mat rvec;
        Rodrigues(rMats[i],rvec);
        rVecs.push_back(rvec);
        tVecs.push_back(tvecs[i]);
    }
}

int PnPSolverAP3P::minPointNumber() const
{
    return 3;
}
int PnPSolverAP3P::maxPointNumber() const
{
    return 4;
}

bool PnPSolverAP3P::requires3DObject() const
{
    return false;

}

bool PnPSolverAP3P::requiresPlanarObject() const
{
    return false; //this is true only when the number of points is 4
}

bool PnPSolverAP3P::requiresPlanarTagObject() const
{
    return false; //this is true only when the number of points is 4
}

PnPSolverIPPE::PnPSolverIPPE(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

Ptr<PnPSolverIPPE> PnPSolverIPPE::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverIPPE>(_withGeometricTests);
}

void PnPSolverIPPE::solve(cv::InputArray _opoints, cv::InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    IPPE::PoseSolver poseSolver;
    Mat rvec1, tvec1, rvec2, tvec2;
    float reprojErr1, reprojErr2;
    poseSolver.solveGeneric(_opoints, _ipoints, rvec1, tvec1, reprojErr1, rvec2, tvec2, reprojErr2);
    rVecs.push_back(rvec1);
    rVecs.push_back(rvec2);
    tVecs.push_back(tvec1);
    tVecs.push_back(tvec2);
}

int PnPSolverIPPE::minPointNumber() const
{
    return 4;
}

int PnPSolverIPPE::maxPointNumber() const
{
    return -1;
}

bool PnPSolverIPPE::requires3DObject() const
{
    return false;

}

bool PnPSolverIPPE::requiresPlanarObject() const
{
    return true;
}

bool PnPSolverIPPE::requiresPlanarTagObject() const
{
    return false;
}

PnPSolverIPPESquare::PnPSolverIPPESquare(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

Ptr<PnPSolverIPPESquare> PnPSolverIPPESquare::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverIPPESquare>(_withGeometricTests);
}

void PnPSolverIPPESquare::solve(cv::InputArray _opoints, cv::InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    IPPE::PoseSolver poseSolver;
    Mat rvec1, tvec1, rvec2, tvec2;
    float reprojErr1, reprojErr2;
    poseSolver.solveSquare(_opoints, _ipoints, rvec1, tvec1, reprojErr1, rvec2, tvec2, reprojErr2);
    rVecs.push_back(rvec1);
    rVecs.push_back(rvec2);
    tVecs.push_back(tvec1);
    tVecs.push_back(tvec2);
}


int PnPSolverIPPESquare::minPointNumber() const
{
    return 4;
}

int PnPSolverIPPESquare::maxPointNumber() const
{
    return 4;
}

bool PnPSolverIPPESquare::requires3DObject() const
{
    return false;

}

bool PnPSolverIPPESquare::requiresPlanarObject() const
{
    return true;
}

bool PnPSolverIPPESquare::requiresPlanarTagObject() const
{
    return true;
}

PnPSolverZhang::PnPSolverZhang(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

Ptr<PnPSolverZhang> PnPSolverZhang::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverZhang>(_withGeometricTests);
}

bool PnPSolverZhang::solveCImpl(InputArray _objectPoints,
                                InputArray _imagePoints, InputOutputArray _rvec, InputOutputArray _tvec) const
{
    const CvMat c_objectPoints = cvMat(_objectPoints.getMat()), c_imagePoints = cvMat(_imagePoints.getMat());
    CvMat c_rvec = cvMat(_rvec.getMat()), c_tvec = cvMat(_tvec.getMat());

    const CvMat * objectPoints = &c_objectPoints;
    const CvMat * imagePoints = &c_imagePoints;
    CvMat * rvec = &c_rvec;
    CvMat * tvec = &c_tvec;


    //taken from cvFindExtrinsicCameraParams2
    bool succ = false;
    Ptr<CvMat> matM, _Mxy, _m;

    int i, count;
    double R[9];
    double MM[9], V[9], W[3];
    cv::Scalar Mc;
    double param[6];
    CvMat matR = cvMat( 3, 3, CV_64F, R );
    CvMat _r = cvMat( 3, 1, CV_64F, param );
    CvMat _t = cvMat( 3, 1, CV_64F, param + 3 );
    CvMat _Mc = cvMat( 1, 3, CV_64F, Mc.val );
    CvMat _MM = cvMat( 3, 3, CV_64F, MM );
    CvMat matV = cvMat( 3, 3, CV_64F, V );
    CvMat matW = cvMat( 3, 1, CV_64F, W );

    CV_Assert( CV_IS_MAT(objectPoints) && CV_IS_MAT(imagePoints) && CV_IS_MAT(rvec) && CV_IS_MAT(tvec) );

    count = MAX(objectPoints->cols, objectPoints->rows);
    matM.reset(cvCreateMat( 1, count, CV_64FC3 ));
    _m.reset(cvCreateMat( 1, count, CV_64FC2 ));

    cvConvertPointsHomogeneous( objectPoints, matM );
    cvConvertPointsHomogeneous( imagePoints, _m );

    CV_Assert( (CV_MAT_DEPTH(rvec->type) == CV_64F || CV_MAT_DEPTH(rvec->type) == CV_32F) &&
               (rvec->rows == 1 || rvec->cols == 1) && rvec->rows*rvec->cols*CV_MAT_CN(rvec->type) == 3 );

    CV_Assert( (CV_MAT_DEPTH(tvec->type) == CV_64F || CV_MAT_DEPTH(tvec->type) == CV_32F) &&
               (tvec->rows == 1 || tvec->cols == 1) && tvec->rows*tvec->cols*CV_MAT_CN(tvec->type) == 3 );

    CV_Assert((count >= 4)); // it is unsafe to call LM optimisation without an extrinsic guess in the case of 3 points. This is because there is no guarantee that it will converge on the correct solution.

    //_mn.reset(cvCreateMat( 1, count, CV_64FC2 ));
    _Mxy.reset(cvCreateMat( 1, count, CV_64FC2 ));

    Mc = cvAvg(matM);
    cvReshape( matM, matM, 1, count );
    cvMulTransposed( matM, &_MM, 1, &_Mc );
    cvSVD( &_MM, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );

    double tt[3], h[9], h1_norm, h2_norm;
    CvMat* R_transform = &matV;
    CvMat T_transform = cvMat( 3, 1, CV_64F, tt );
    CvMat matH = cvMat( 3, 3, CV_64F, h );
    CvMat _h1, _h2, _h3;

    if( V[2]*V[2] + V[5]*V[5] < 1e-10 )
    {
        return false;
    }

    if( cvDet(R_transform) < 0 )
        cvScale( R_transform, R_transform, -1 );

    cvGEMM( R_transform, &_Mc, -1, 0, 0, &T_transform, CV_GEMM_B_T );

    for( i = 0; i < count; i++ )
    {
        const double* Rp = R_transform->data.db;
        const double* Tp = T_transform.data.db;
        const double* src = matM->data.db + i*3;
        double* dst = _Mxy->data.db + i*2;

        dst[0] = Rp[0]*src[0] + Rp[1]*src[1] + Rp[2]*src[2] + Tp[0];
        dst[1] = Rp[3]*src[0] + Rp[4]*src[1] + Rp[5]*src[2] + Tp[1];
    }

    cvFindHomography( _Mxy, _m, &matH );

    if( cvCheckArr(&matH, CV_CHECK_QUIET) )
    {
        cvGetCol( &matH, &_h1, 0 );
        _h2 = _h1; _h2.data.db++;
        _h3 = _h2; _h3.data.db++;
        h1_norm = std::sqrt(h[0]*h[0] + h[3]*h[3] + h[6]*h[6]);
        h2_norm = std::sqrt(h[1]*h[1] + h[4]*h[4] + h[7]*h[7]);

        cvScale( &_h1, &_h1, 1./MAX(h1_norm, DBL_EPSILON) );
        cvScale( &_h2, &_h2, 1./MAX(h2_norm, DBL_EPSILON) );
        cvScale( &_h3, &_t, 2./MAX(h1_norm + h2_norm, DBL_EPSILON));
        cvCrossProduct( &_h1, &_h2, &_h3 );

        cvRodrigues2( &matH, &_r );
        cvRodrigues2( &_r, &matH );
        cvMatMulAdd( &matH, &T_transform, &_t, &_t );
        cvMatMul( &matH, R_transform, &matR );

        cvRodrigues2( &matR, &_r );
        // }

        cvReshape( matM, matM, 3, 1 );
        //cvReshape( _mn, _mn, 2, 1 );

        _r = cvMat( rvec->rows, rvec->cols,
                    CV_MAKETYPE(CV_64F,CV_MAT_CN(rvec->type)), param );
        _t = cvMat( tvec->rows, tvec->cols,
                    CV_MAKETYPE(CV_64F,CV_MAT_CN(tvec->type)), param + 3 );

        cvConvert( &_r, rvec );
        cvConvert( &_t, tvec );

        succ = true;

    }
    else
    {
        succ = false;
    }

    return succ;
}


void PnPSolverZhang::solve(cv::InputArray _opoints, cv::InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    cv::Mat rVec(3,1,CV_64FC1);
    cv::Mat tVec(3,1,CV_64FC1);


    if (solveCImpl(_opoints, _ipoints,
                   rVec, tVec))
    {
        rVecs.push_back(rVec);
        tVecs.push_back(tVec);
    }
}

int PnPSolverZhang::minPointNumber() const
{
    return 4;
}

int PnPSolverZhang::maxPointNumber() const
{
    return -1;
}

bool PnPSolverZhang::requires3DObject() const
{
    return false;
}

bool PnPSolverZhang::requiresPlanarObject() const
{
    return true; //this is true only when the number of points is 4
}

bool PnPSolverZhang::requiresPlanarTagObject() const
{
    return false; //this is true only when the number of points is 4
}

bool PnPSolverZhang::noArtificialDegeneracy(InputArray opoints,InputArray ipoints) const
{
    (void)ipoints;
    return(!isPlanarTag(opoints)); //this is an artificially degenerate configuration
}

PnPSolverDLT::PnPSolverDLT(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

bool PnPSolverDLT::solveCImpl(InputArray _objectPoints,
                              InputArray _imagePoints, InputOutputArray _rvec, InputOutputArray _tvec) const
{
    const CvMat c_objectPoints = cvMat(_objectPoints.getMat()), c_imagePoints = cvMat(_imagePoints.getMat());
    CvMat c_rvec = cvMat(_rvec.getMat()), c_tvec = cvMat(_tvec.getMat());

    const CvMat * objectPoints = &c_objectPoints;
    const CvMat * imagePoints = &c_imagePoints;
    CvMat * rvec = &c_rvec;
    CvMat * tvec = &c_tvec;

    //adapted from cvFindExtrinsicCameraParams2
    bool succ = false;
    Ptr<CvMat> matM, _Mxy, _m,matL;

    int i, count;
    double R[9];
    double U[9], V[9], W[3];
    cv::Scalar Mc;
    double param[6];
    CvMat matR = cvMat( 3, 3, CV_64F, R );
    CvMat _r = cvMat( 3, 1, CV_64F, param );
    CvMat _t = cvMat( 3, 1, CV_64F, param + 3 );
    CvMat matU = cvMat( 3, 3, CV_64F, U );
    CvMat matV = cvMat( 3, 3, CV_64F, V );
    CvMat matW = cvMat( 3, 1, CV_64F, W );

    CV_Assert( CV_IS_MAT(objectPoints) && CV_IS_MAT(imagePoints) && CV_IS_MAT(rvec) && CV_IS_MAT(tvec) );

    count = MAX(objectPoints->cols, objectPoints->rows);
    matM.reset(cvCreateMat( 1, count, CV_64FC3 ));
    _m.reset(cvCreateMat( 1, count, CV_64FC2 ));

    cvConvertPointsHomogeneous( objectPoints, matM );
    cvConvertPointsHomogeneous( imagePoints, _m );

    CV_Assert( (CV_MAT_DEPTH(rvec->type) == CV_64F || CV_MAT_DEPTH(rvec->type) == CV_32F) &&
               (rvec->rows == 1 || rvec->cols == 1) && rvec->rows*rvec->cols*CV_MAT_CN(rvec->type) == 3 );

    CV_Assert( (CV_MAT_DEPTH(tvec->type) == CV_64F || CV_MAT_DEPTH(tvec->type) == CV_32F) &&
               (tvec->rows == 1 || tvec->cols == 1) && tvec->rows*tvec->cols*CV_MAT_CN(tvec->type) == 3 );

    CV_Assert((count >= 4)); // it is unsafe to call LM optimisation without an extrinsic guess in the case of 3 points. This is because there is no guarantee that it will converge on the correct solution.

    //_mn.reset(cvCreateMat( 1, count, CV_64FC2 ));
    _Mxy.reset(cvCreateMat( 1, count, CV_64FC2 ));

    // non-planar structure. Use DLT method
    CV_CheckGE(count, 6, "DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences.");
    double* L;
    double LL[12*12], LW[12], LV[12*12], sc;
    CvMat _LL = cvMat( 12, 12, CV_64F, LL );
    CvMat _LW = cvMat( 12, 1, CV_64F, LW );
    CvMat _LV = cvMat( 12, 12, CV_64F, LV );
    CvMat _RRt, _RR, _tt;
    CvPoint3D64f* M = (CvPoint3D64f*)matM->data.db;
    CvPoint2D64f* mn = (CvPoint2D64f*)_m->data.db;

    matL.reset(cvCreateMat( 2*count, 12, CV_64F ));
    L = matL->data.db;

    for( i = 0; i < count; i++, L += 24 )
    {
        double x = -mn[i].x, y = -mn[i].y;
        L[0] = L[16] = M[i].x;
        L[1] = L[17] = M[i].y;
        L[2] = L[18] = M[i].z;
        L[3] = L[19] = 1.;
        L[4] = L[5] = L[6] = L[7] = 0.;
        L[12] = L[13] = L[14] = L[15] = 0.;
        L[8] = x*M[i].x;
        L[9] = x*M[i].y;
        L[10] = x*M[i].z;
        L[11] = x;
        L[20] = y*M[i].x;
        L[21] = y*M[i].y;
        L[22] = y*M[i].z;
        L[23] = y;
    }

    cvMulTransposed( matL, &_LL, 1 );
    cvSVD( &_LL, &_LW, 0, &_LV, CV_SVD_MODIFY_A + CV_SVD_V_T );
    _RRt = cvMat( 3, 4, CV_64F, LV + 11*12 );
    cvGetCols( &_RRt, &_RR, 0, 3 );
    cvGetCol( &_RRt, &_tt, 3 );
    if( cvDet(&_RR) < 0 )
        cvScale( &_RRt, &_RRt, -1 );
    sc = cvNorm(&_RR);
    cvSVD( &_RR, &matW, &matU, &matV, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T );
    cvGEMM( &matU, &matV, 1, 0, 0, &matR, CV_GEMM_A_T );
    cvScale( &_tt, &_t, cvNorm(&matR)/sc );
    cvRodrigues2( &matR, &_r );

    cvReshape( matM, matM, 3, 1 );
    //cvReshape( _mn, _mn, 2, 1 );

    _r = cvMat( rvec->rows, rvec->cols,
                CV_MAKETYPE(CV_64F,CV_MAT_CN(rvec->type)), param );
    _t = cvMat( tvec->rows, tvec->cols,
                CV_MAKETYPE(CV_64F,CV_MAT_CN(tvec->type)), param + 3 );

    cvConvert( &_r, rvec );
    cvConvert( &_t, tvec );

    succ = true;

    return succ;
}


Ptr<PnPSolverDLT> PnPSolverDLT::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverDLT>(_withGeometricTests);
}

void PnPSolverDLT::solve(cv::InputArray _opoints, cv::InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    cv::Mat rVec(3,1,CV_64FC1);
    cv::Mat tVec(3,1,CV_64FC1);

    if (solveCImpl(_opoints, _ipoints,
                   rVec, tVec))
    {
        rVecs.push_back(rVec);
        tVecs.push_back(tVec);
    }
}

int PnPSolverDLT::minPointNumber() const
{
    return 6;
}
int PnPSolverDLT::maxPointNumber() const
{
    return -1;
}


bool PnPSolverDLT::requires3DObject() const
{
    return true;

}
bool PnPSolverDLT::requiresPlanarObject() const
{
    return false; //this is true only when the number of points is 4
}

bool PnPSolverDLT::requiresPlanarTagObject() const
{
    return false; //this is true only when the number of points is 4
}

PnPSolverEPnP3D::PnPSolverEPnP3D(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

Ptr<PnPSolverEPnP3D> PnPSolverEPnP3D::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverEPnP3D>(_withGeometricTests);
}

void PnPSolverEPnP3D::solve(cv::InputArray _opoints, cv::InputArray _ipoints,
                            CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    const auto K = makeIdentityIntrinsic();
    std::vector<std::tuple<cv::Mat, cv::Mat> >  ret;
    epnp solver(K, _opoints.getMat(), _ipoints.getMat());
    Mat rvec, tvec, R;
    solver.compute_pose(R, tvec);
    Rodrigues(R, rvec);
    rVecs.push_back(rvec);
    tVecs.push_back(tvec);
}

int PnPSolverEPnP3D::minPointNumber() const
{
    return 4; // unstable with 3 and 4 points
}

int PnPSolverEPnP3D::maxPointNumber() const
{
    return -1;
}

bool PnPSolverEPnP3D::requires3DObject() const
{
    return true;
}

bool PnPSolverEPnP3D::requiresPlanarObject() const
{
    return false;
}

bool PnPSolverEPnP3D::requiresPlanarTagObject() const
{
    return false; //this is true only when the number of points is 4
}

PnPSolverDLS::PnPSolverDLS(bool _withGeometricTests): PnPSolver(_withGeometricTests)
{

}

Ptr<PnPSolverDLS> PnPSolverDLS::create(bool _withGeometricTests)
{
    return makePtr<PnPSolverDLS>(_withGeometricTests);
}

void PnPSolverDLS::solve(cv::InputArray _opoints, cv::InputArray _ipoints,
                         CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    dls d(_opoints.getMat(),_ipoints.getMat());
    std::vector<cv::Mat> Rs;
    d.compute_poses(Rs, tVecs);
    for (size_t i = 0; i < Rs.size(); i++)
    {
        cv::Mat r;
        Rodrigues(Rs[i], r);
        rVecs.push_back(r);
    }
}

int PnPSolverDLS::minPointNumber() const
{
    return 4; // unstable with 3 and 4 points
}

int PnPSolverDLS::maxPointNumber() const
{
    return -1;
}

bool PnPSolverDLS::requires3DObject() const
{
    return false;
}

bool PnPSolverDLS::requiresPlanarObject() const
{
    return false;
}

bool PnPSolverDLS::requiresPlanarTagObject() const
{
    return false; //this is true only when the number of points is 4
}




PnPSolverAutoSelect1::PnPSolverAutoSelect1(bool withGeomTests): PnPSolver(withGeomTests)
{

}

Ptr<PnPSolverAutoSelect1> PnPSolverAutoSelect1::create(bool withGeomTests)
{
    return makePtr<PnPSolverAutoSelect1>(withGeomTests);
}

void PnPSolverAutoSelect1::solve(InputArray _opoints, InputArray _ipoints, CV_OUT std::vector<Mat> & rVecs, CV_OUT std::vector<Mat> & tVecs) const
{
    CV_INSTRUMENT_REGION();
    const double coplaneThresh = 1.0e-3;
    cv::Mat singValues;
    get3DPointsetShape(_opoints, singValues);
    bool isCoplanar = singValues.at<double>(2) < singValues.at<double>(1) * coplaneThresh;
    Ptr<PnPSolver> solver;
    if (isCoplanar)
    {
        solver = PnPSolverIPPE::create();
    }
    else {
        solver = PnPSolverDLT::create();
    }
    solver->solveProblem(_opoints,_ipoints,rVecs,tVecs);
}

int PnPSolverAutoSelect1::minPointNumber() const
{
    return 4;
}

int PnPSolverAutoSelect1::maxPointNumber() const
{
    return -1;
}

bool PnPSolverAutoSelect1::requires3DObject() const
{
    return false;
}

bool PnPSolverAutoSelect1::requiresPlanarObject() const
{
    return false;
}

bool PnPSolverAutoSelect1::requiresPlanarTagObject() const
{
    return false; //this is true only when the number of points is 4
}
