#include <opencv2/calib3d.hpp>
#include "linalg.hpp"
#include "cvCalibrationFork.hpp"

using namespace cv;

static void subMatrix(const cv::Mat& src, cv::Mat& dst, const std::vector<uchar>& cols,
                      const std::vector<uchar>& rows);
static const char* cvDistCoeffErr = "Distortion coefficients must be 1x4, 4x1, 1x5, 5x1, 1x8, 8x1, 1x12, 12x1, 1x14 or 14x1 floating-point vector";

static void cvEvaluateJtJ2(CvMat* _JtJ,
                            const CvMat* camera_matrix,
                            const CvMat* distortion_coeffs,
                            const CvMat* object_points,
                            const CvMat* param,
                            const CvMat* npoints,
                            int flags, int NINTRINSIC, double aspectRatio)
{
    int i, pos, ni, total = 0, npstep = 0, maxPoints = 0;

    npstep = npoints->rows == 1 ? 1 : npoints->step/CV_ELEM_SIZE(npoints->type);
    int nimages = npoints->rows*npoints->cols;
    for( i = 0; i < nimages; i++ )
    {
        ni = npoints->data.i[i*npstep];
        if( ni < 4 )
        {
            CV_Error_( CV_StsOutOfRange, ("The number of points in the view #%d is < 4", i));
        }
        maxPoints = MAX( maxPoints, ni );
        total += ni;
    }

    Mat _Ji( maxPoints*2, NINTRINSIC, CV_64FC1, Scalar(0));
    Mat _Je( maxPoints*2, 6, CV_64FC1 );
    Mat _err( maxPoints*2, 1, CV_64FC1 );
    Mat _m( 1, total, CV_64FC2 );
    const Mat matM = cvarrToMat(object_points);

    cvZero(_JtJ);
    for(i = 0, pos = 0; i < nimages; i++, pos += ni )
    {
        CvMat _ri, _ti;
        ni = npoints->data.i[i*npstep];

        cvGetRows( param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
        cvGetRows( param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );

        CvMat _Mi(matM.colRange(pos, pos + ni));
        CvMat _mi(_m.colRange(pos, pos + ni));

        _Je.resize(ni*2); _Ji.resize(ni*2); _err.resize(ni*2);
        CvMat _dpdr(_Je.colRange(0, 3));
        CvMat _dpdt(_Je.colRange(3, 6));
        CvMat _dpdf(_Ji.colRange(0, 2));
        CvMat _dpdc(_Ji.colRange(2, 4));
        CvMat _dpdk(_Ji.colRange(4, NINTRINSIC));
        CvMat _mp(_err.reshape(2, 1));

        cvProjectPoints2( &_Mi, &_ri, &_ti, camera_matrix, distortion_coeffs, &_mp, &_dpdr, &_dpdt,
                          (flags & CALIB_FIX_FOCAL_LENGTH) ? 0 : &_dpdf,
                          (flags & CALIB_FIX_PRINCIPAL_POINT) ? 0 : &_dpdc, &_dpdk,
                          (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0);
        cvSub( &_mp, &_mi, &_mp );
        Mat JtJ(cvarrToMat(_JtJ));
        // see HZ: (A6.14) for details on the structure of the Jacobian
        JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC)) += _Ji.t() * _Ji;
        JtJ(Rect(NINTRINSIC + i * 6, NINTRINSIC + i * 6, 6, 6)) = _Je.t() * _Je;
        JtJ(Rect(NINTRINSIC + i * 6, 0, 6, NINTRINSIC)) = _Ji.t() * _Je;
    }
}

double cvfork::cvCalibrateCamera2( const CvMat* objectPoints,
                    const CvMat* imagePoints, const CvMat* npoints,
                    CvSize imageSize, CvMat* cameraMatrix, CvMat* distCoeffs,
                    CvMat* rvecs, CvMat* tvecs, CvMat* stdDevs, CvMat* perViewErrors, int flags, CvTermCriteria termCrit )
{
    {
        const int NINTRINSIC = CV_CALIB_NINTRINSIC;
        double reprojErr = 0;

        Matx33d A;
        double k[14] = {0};
        CvMat matA = cvMat(3, 3, CV_64F, A.val), _k;
        int i, nimages, maxPoints = 0, ni = 0, pos, total = 0, nparams, npstep, cn;
        double aspectRatio = 0.;

        // 0. check the parameters & allocate buffers
        if( !CV_IS_MAT(objectPoints) || !CV_IS_MAT(imagePoints) ||
            !CV_IS_MAT(npoints) || !CV_IS_MAT(cameraMatrix) || !CV_IS_MAT(distCoeffs) )
            CV_Error( CV_StsBadArg, "One of required vector arguments is not a valid matrix" );

        if( imageSize.width <= 0 || imageSize.height <= 0 )
            CV_Error( CV_StsOutOfRange, "image width and height must be positive" );

        if( CV_MAT_TYPE(npoints->type) != CV_32SC1 ||
            (npoints->rows != 1 && npoints->cols != 1) )
            CV_Error( CV_StsUnsupportedFormat,
                "the array of point counters must be 1-dimensional integer vector" );
        if(flags & CV_CALIB_TILTED_MODEL)
        {
            //when the tilted sensor model is used the distortion coefficients matrix must have 14 parameters
            if (distCoeffs->cols*distCoeffs->rows != 14)
                CV_Error( CV_StsBadArg, "The tilted sensor model must have 14 parameters in the distortion matrix" );
        }
        else
        {
            //when the thin prism model is used the distortion coefficients matrix must have 12 parameters
            if(flags & CV_CALIB_THIN_PRISM_MODEL)
                if (distCoeffs->cols*distCoeffs->rows != 12)
                    CV_Error( CV_StsBadArg, "Thin prism model must have 12 parameters in the distortion matrix" );
        }

        nimages = npoints->rows*npoints->cols;
        npstep = npoints->rows == 1 ? 1 : npoints->step/CV_ELEM_SIZE(npoints->type);

        if( rvecs )
        {
            cn = CV_MAT_CN(rvecs->type);
            if( !CV_IS_MAT(rvecs) ||
                (CV_MAT_DEPTH(rvecs->type) != CV_32F && CV_MAT_DEPTH(rvecs->type) != CV_64F) ||
                ((rvecs->rows != nimages || (rvecs->cols*cn != 3 && rvecs->cols*cn != 9)) &&
                (rvecs->rows != 1 || rvecs->cols != nimages || cn != 3)) )
                CV_Error( CV_StsBadArg, "the output array of rotation vectors must be 3-channel "
                    "1xn or nx1 array or 1-channel nx3 or nx9 array, where n is the number of views" );
        }

        if( tvecs )
        {
            cn = CV_MAT_CN(tvecs->type);
            if( !CV_IS_MAT(tvecs) ||
                (CV_MAT_DEPTH(tvecs->type) != CV_32F && CV_MAT_DEPTH(tvecs->type) != CV_64F) ||
                ((tvecs->rows != nimages || tvecs->cols*cn != 3) &&
                (tvecs->rows != 1 || tvecs->cols != nimages || cn != 3)) )
                CV_Error( CV_StsBadArg, "the output array of translation vectors must be 3-channel "
                    "1xn or nx1 array or 1-channel nx3 array, where n is the number of views" );
        }

        if( stdDevs )
        {
            cn = CV_MAT_CN(stdDevs->type);
            if( !CV_IS_MAT(stdDevs) ||
                (CV_MAT_DEPTH(stdDevs->type) != CV_32F && CV_MAT_DEPTH(stdDevs->type) != CV_64F) ||
                ((stdDevs->rows != (nimages*6 + NINTRINSIC) || stdDevs->cols*cn != 1) &&
                (stdDevs->rows != 1 || stdDevs->cols != (nimages*6 + NINTRINSIC) || cn != 1)) )
                CV_Error( CV_StsBadArg, "the output array of standard deviations vectors must be 1-channel "
                    "1x(n*6 + NINTRINSIC) or (n*6 + NINTRINSIC)x1 array, where n is the number of views" );
        }

        if( (CV_MAT_TYPE(cameraMatrix->type) != CV_32FC1 &&
            CV_MAT_TYPE(cameraMatrix->type) != CV_64FC1) ||
            cameraMatrix->rows != 3 || cameraMatrix->cols != 3 )
            CV_Error( CV_StsBadArg,
                "Intrinsic parameters must be 3x3 floating-point matrix" );

        if( (CV_MAT_TYPE(distCoeffs->type) != CV_32FC1 &&
            CV_MAT_TYPE(distCoeffs->type) != CV_64FC1) ||
            (distCoeffs->cols != 1 && distCoeffs->rows != 1) ||
            (distCoeffs->cols*distCoeffs->rows != 4 &&
            distCoeffs->cols*distCoeffs->rows != 5 &&
            distCoeffs->cols*distCoeffs->rows != 8 &&
            distCoeffs->cols*distCoeffs->rows != 12 &&
            distCoeffs->cols*distCoeffs->rows != 14) )
            CV_Error( CV_StsBadArg, cvDistCoeffErr );

        for( i = 0; i < nimages; i++ )
        {
            ni = npoints->data.i[i*npstep];
            if( ni < 4 )
            {
                CV_Error_( CV_StsOutOfRange, ("The number of points in the view #%d is < 4", i));
            }
            maxPoints = MAX( maxPoints, ni );
            total += ni;
        }

        Mat matM( 1, total, CV_64FC3 );
        Mat _m( 1, total, CV_64FC2 );

        if(CV_MAT_CN(objectPoints->type) == 3) {
            cvarrToMat(objectPoints).convertTo(matM, CV_64F);
        } else {
            convertPointsHomogeneous(cvarrToMat(objectPoints), matM);
        }

        if(CV_MAT_CN(imagePoints->type) == 2) {
            cvarrToMat(imagePoints).convertTo(_m, CV_64F);
        } else {
            convertPointsHomogeneous(cvarrToMat(imagePoints), _m);
        }

        nparams = NINTRINSIC + nimages*6;
        Mat _Ji( maxPoints*2, NINTRINSIC, CV_64FC1, Scalar(0));
        Mat _Je( maxPoints*2, 6, CV_64FC1 );
        Mat _err( maxPoints*2, 1, CV_64FC1 );

        _k = cvMat( distCoeffs->rows, distCoeffs->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(distCoeffs->type)), k);
        if( distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 8 )
        {
            if( distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 5 )
                flags |= CALIB_FIX_K3;
            flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
        }
        const double minValidAspectRatio = 0.01;
        const double maxValidAspectRatio = 100.0;

        // 1. initialize intrinsic parameters & LM solver
        if( flags & CALIB_USE_INTRINSIC_GUESS )
        {
            cvConvert( cameraMatrix, &matA );
            if( A(0, 0) <= 0 || A(1, 1) <= 0 )
                CV_Error( CV_StsOutOfRange, "Focal length (fx and fy) must be positive" );
            if( A(0, 2) < 0 || A(0, 2) >= imageSize.width ||
                A(1, 2) < 0 || A(1, 2) >= imageSize.height )
                CV_Error( CV_StsOutOfRange, "Principal point must be within the image" );
            if( fabs(A(0, 1)) > 1e-5 )
                CV_Error( CV_StsOutOfRange, "Non-zero skew is not supported by the function" );
            if( fabs(A(1, 0)) > 1e-5 || fabs(A(2, 0)) > 1e-5 ||
                fabs(A(2, 1)) > 1e-5 || fabs(A(2,2)-1) > 1e-5 )
                CV_Error( CV_StsOutOfRange,
                    "The intrinsic matrix must have [fx 0 cx; 0 fy cy; 0 0 1] shape" );
            A(0, 1) = A(1, 0) = A(2, 0) = A(2, 1) = 0.;
            A(2, 2) = 1.;

            if( flags & CALIB_FIX_ASPECT_RATIO )
            {
                aspectRatio = A(0, 0)/A(1, 1);

                if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                    CV_Error( CV_StsOutOfRange,
                        "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
            }
            cvConvert( distCoeffs, &_k );
        }
        else
        {
            Scalar mean, sdv;
            meanStdDev(matM, mean, sdv);
            if( fabs(mean[2]) > 1e-5 || fabs(sdv[2]) > 1e-5 )
                CV_Error( CV_StsBadArg,
                "For non-planar calibration rigs the initial intrinsic matrix must be specified" );
            for( i = 0; i < total; i++ )
                matM.at<Point3d>(i).z = 0.;

            if( flags & CALIB_FIX_ASPECT_RATIO )
            {
                aspectRatio = cvmGet(cameraMatrix,0,0);
                aspectRatio /= cvmGet(cameraMatrix,1,1);
                if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                    CV_Error( CV_StsOutOfRange,
                        "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
            }
            CvMat _matM(matM), m(_m);
            cvInitIntrinsicParams2D( &_matM, &m, npoints, imageSize, &matA, aspectRatio );
        }

        //CvLevMarq solver( nparams, 0, termCrit );
        cvfork::CvLevMarqFork solver( nparams, 0, termCrit );
        Mat allErrors(1, total, CV_64FC2);

        if(flags & CALIB_USE_LU) {
            solver.solveMethod = DECOMP_LU;
        }
        else if(flags & CALIB_USE_QR)
            solver.solveMethod = DECOMP_QR;

        {
        double* param = solver.param->data.db;
        uchar* mask = solver.mask->data.ptr;

        param[0] = A(0, 0); param[1] = A(1, 1); param[2] = A(0, 2); param[3] = A(1, 2);
        std::copy(k, k + 14, param + 4);

        if( flags & CV_CALIB_FIX_FOCAL_LENGTH )
            mask[0] = mask[1] = 0;
        if( flags & CV_CALIB_FIX_PRINCIPAL_POINT )
            mask[2] = mask[3] = 0;
        if( flags & CV_CALIB_ZERO_TANGENT_DIST )
        {
            param[6] = param[7] = 0;
            mask[6] = mask[7] = 0;
        }
        if( !(flags & CALIB_RATIONAL_MODEL) )
            flags |= CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6;
        if( !(flags & CV_CALIB_THIN_PRISM_MODEL))
            flags |= CALIB_FIX_S1_S2_S3_S4;
        if( !(flags & CV_CALIB_TILTED_MODEL))
            flags |= CALIB_FIX_TAUX_TAUY;

        mask[ 4] = !(flags & CALIB_FIX_K1);
        mask[ 5] = !(flags & CALIB_FIX_K2);
        mask[ 8] = !(flags & CALIB_FIX_K3);
        mask[ 9] = !(flags & CALIB_FIX_K4);
        mask[10] = !(flags & CALIB_FIX_K5);
        mask[11] = !(flags & CALIB_FIX_K6);

        if(flags & CALIB_FIX_S1_S2_S3_S4)
        {
            mask[12] = 0;
            mask[13] = 0;
            mask[14] = 0;
            mask[15] = 0;
        }
        if(flags & CALIB_FIX_TAUX_TAUY)
        {
            mask[16] = 0;
            mask[17] = 0;
        }
        }

        // 2. initialize extrinsic parameters
        for( i = 0, pos = 0; i < nimages; i++, pos += ni )
        {
            CvMat _ri, _ti;
            ni = npoints->data.i[i*npstep];

            cvGetRows( solver.param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
            cvGetRows( solver.param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );

            CvMat _Mi(matM.colRange(pos, pos + ni));
            CvMat _mi(_m.colRange(pos, pos + ni));

            cvFindExtrinsicCameraParams2( &_Mi, &_mi, &matA, &_k, &_ri, &_ti );
        }

        // 3. run the optimization
        for(;;)
        {
            const CvMat* _param = 0;
            CvMat *_JtJ = 0, *_JtErr = 0;
            double* _errNorm = 0;
            bool proceed = solver.updateAlt( _param, _JtJ, _JtErr, _errNorm );
            double *param = solver.param->data.db, *pparam = solver.prevParam->data.db;

            if( flags & CALIB_FIX_ASPECT_RATIO )
            {
                param[0] = param[1]*aspectRatio;
                pparam[0] = pparam[1]*aspectRatio;
            }

            A(0, 0) = param[0]; A(1, 1) = param[1]; A(0, 2) = param[2]; A(1, 2) = param[3];
            std::copy(param + 4, param + 4 + 14, k);

            if( !proceed ) {
                //do errors estimation
                if(stdDevs) {
                    Ptr<CvMat> JtJ(cvCreateMat(nparams, nparams, CV_64F));
                    CvMat cvMatM(matM);
                    cvEvaluateJtJ2(JtJ, &matA, &_k, &cvMatM, solver.param, npoints, flags, NINTRINSIC, aspectRatio);

                    Mat mask = cvarrToMat(solver.mask);
                    int nparams_nz = countNonZero(mask);
                    Mat JtJinv, JtJN;
                    JtJN.create(nparams_nz, nparams_nz, CV_64F);
                    subMatrix(cvarrToMat(JtJ), JtJN, mask, mask);
                    completeSymm(JtJN, false);
    #ifndef USE_LAPACK
                    cv::invert(JtJN, JtJinv, DECOMP_SVD);
    #else
                    cvfork::invert(JtJN, JtJinv, DECOMP_SVD);
    #endif
                    double sigma2 = norm(allErrors, NORM_L2SQR) / (total - nparams_nz);
                    Mat stdDevsM = cvarrToMat(stdDevs);
                    int j = 0;
                    for (int s = 0; s < nparams; s++)
                        if(mask.data[s]) {
                            stdDevsM.at<double>(s) = std::sqrt(JtJinv.at<double>(j,j)*sigma2);
                            j++;
                        }
                        else
                            stdDevsM.at<double>(s) = 0;
                }
                break;
            }

            reprojErr = 0;

            for( i = 0, pos = 0; i < nimages; i++, pos += ni )
            {
                CvMat _ri, _ti;
                ni = npoints->data.i[i*npstep];

                cvGetRows( solver.param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
                cvGetRows( solver.param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );

                CvMat _Mi(matM.colRange(pos, pos + ni));
                CvMat _mi(_m.colRange(pos, pos + ni));
                CvMat _me(allErrors.colRange(pos, pos + ni));

                _Je.resize(ni*2); _Ji.resize(ni*2); _err.resize(ni*2);
                CvMat _dpdr(_Je.colRange(0, 3));
                CvMat _dpdt(_Je.colRange(3, 6));
                CvMat _dpdf(_Ji.colRange(0, 2));
                CvMat _dpdc(_Ji.colRange(2, 4));
                CvMat _dpdk(_Ji.colRange(4, NINTRINSIC));
                CvMat _mp(_err.reshape(2, 1));

                if( solver.state == CvLevMarq::CALC_J )
                {
                     cvProjectPoints2( &_Mi, &_ri, &_ti, &matA, &_k, &_mp, &_dpdr, &_dpdt,
                                      (flags & CALIB_FIX_FOCAL_LENGTH) ? 0 : &_dpdf,
                                      (flags & CALIB_FIX_PRINCIPAL_POINT) ? 0 : &_dpdc, &_dpdk,
                                      (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0);
                }
                else
                    cvProjectPoints2( &_Mi, &_ri, &_ti, &matA, &_k, &_mp );

                cvSub( &_mp, &_mi, &_mp );

                if( solver.state == CvLevMarq::CALC_J )
                {
                    Mat JtJ(cvarrToMat(_JtJ)), JtErr(cvarrToMat(_JtErr));

                    // see HZ: (A6.14) for details on the structure of the Jacobian
                    JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC)) += _Ji.t() * _Ji;
                    JtJ(Rect(NINTRINSIC + i * 6, NINTRINSIC + i * 6, 6, 6)) = _Je.t() * _Je;
                    JtJ(Rect(NINTRINSIC + i * 6, 0, 6, NINTRINSIC)) = _Ji.t() * _Je;

                    JtErr.rowRange(0, NINTRINSIC) += _Ji.t() * _err;
                    JtErr.rowRange(NINTRINSIC + i * 6, NINTRINSIC + (i + 1) * 6) = _Je.t() * _err;

                }
                if (stdDevs || perViewErrors)
                    cvCopy(&_mp, &_me);
                reprojErr += norm(_err, NORM_L2SQR);
            }

            if( _errNorm )
                *_errNorm = reprojErr;
        }

        // 4. store the results
        cvConvert( &matA, cameraMatrix );
        cvConvert( &_k, distCoeffs );

        for( i = 0, pos = 0; i < nimages; i++)
        {
            CvMat src, dst;
            if( perViewErrors )
            {
                ni = npoints->data.i[i*npstep];
                perViewErrors->data.db[i] = std::sqrt(cv::norm(allErrors.colRange(pos, pos + ni), NORM_L2SQR) / ni);
                pos+=ni;
            }

            if( rvecs )
            {
                src = cvMat( 3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6 );
                if( rvecs->rows == nimages && rvecs->cols*CV_MAT_CN(rvecs->type) == 9 )
                {
                    dst = cvMat( 3, 3, CV_MAT_DEPTH(rvecs->type),
                        rvecs->data.ptr + rvecs->step*i );
                    cvRodrigues2( &src, &matA );
                    cvConvert( &matA, &dst );
                }
                else
                {
                    dst = cvMat( 3, 1, CV_MAT_DEPTH(rvecs->type), rvecs->rows == 1 ?
                        rvecs->data.ptr + i*CV_ELEM_SIZE(rvecs->type) :
                        rvecs->data.ptr + rvecs->step*i );
                    cvConvert( &src, &dst );
                }
            }
            if( tvecs )
            {
                src = cvMat( 3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6 + 3 );
                dst = cvMat( 3, 1, CV_MAT_DEPTH(tvecs->type), tvecs->rows == 1 ?
                        tvecs->data.ptr + i*CV_ELEM_SIZE(tvecs->type) :
                        tvecs->data.ptr + tvecs->step*i );
                cvConvert( &src, &dst );
             }
        }

        return std::sqrt(reprojErr/total);
    }
}


static Mat prepareCameraMatrix(Mat& cameraMatrix0, int rtype)
{
    Mat cameraMatrix = Mat::eye(3, 3, rtype);
    if( cameraMatrix0.size() == cameraMatrix.size() )
        cameraMatrix0.convertTo(cameraMatrix, rtype);
    return cameraMatrix;
}

static Mat prepareDistCoeffs(Mat& distCoeffs0, int rtype)
{
    Mat distCoeffs = Mat::zeros(distCoeffs0.cols == 1 ? Size(1, 14) : Size(14, 1), rtype);
    if( distCoeffs0.size() == Size(1, 4) ||
       distCoeffs0.size() == Size(1, 5) ||
       distCoeffs0.size() == Size(1, 8) ||
       distCoeffs0.size() == Size(1, 12) ||
       distCoeffs0.size() == Size(1, 14) ||
       distCoeffs0.size() == Size(4, 1) ||
       distCoeffs0.size() == Size(5, 1) ||
       distCoeffs0.size() == Size(8, 1) ||
       distCoeffs0.size() == Size(12, 1) ||
       distCoeffs0.size() == Size(14, 1) )
    {
        Mat dstCoeffs(distCoeffs, Rect(0, 0, distCoeffs0.cols, distCoeffs0.rows));
        distCoeffs0.convertTo(dstCoeffs, rtype);
    }
    return distCoeffs;
}

static void collectCalibrationData( InputArrayOfArrays objectPoints,
                                    InputArrayOfArrays imagePoints1,
                                    InputArrayOfArrays imagePoints2,
                                    Mat& objPtMat, Mat& imgPtMat1, Mat* imgPtMat2,
                                    Mat& npoints )
{
    int nimages = (int)objectPoints.total();
    int i, j = 0, ni = 0, total = 0;
    CV_Assert(nimages > 0 && nimages == (int)imagePoints1.total() &&
        (!imgPtMat2 || nimages == (int)imagePoints2.total()));

    for( i = 0; i < nimages; i++ )
    {
        ni = objectPoints.getMat(i).checkVector(3, CV_32F);
        if( ni <= 0 )
            CV_Error(CV_StsUnsupportedFormat, "objectPoints should contain vector of vectors of points of type Point3f");
        int ni1 = imagePoints1.getMat(i).checkVector(2, CV_32F);
        if( ni1 <= 0 )
            CV_Error(CV_StsUnsupportedFormat, "imagePoints1 should contain vector of vectors of points of type Point2f");
        CV_Assert( ni == ni1 );

        total += ni;
    }

    npoints.create(1, (int)nimages, CV_32S);
    objPtMat.create(1, (int)total, CV_32FC3);
    imgPtMat1.create(1, (int)total, CV_32FC2);
    Point2f* imgPtData2 = 0;

    if( imgPtMat2 )
    {
        imgPtMat2->create(1, (int)total, CV_32FC2);
        imgPtData2 = imgPtMat2->ptr<Point2f>();
    }

    Point3f* objPtData = objPtMat.ptr<Point3f>();
    Point2f* imgPtData1 = imgPtMat1.ptr<Point2f>();

    for( i = 0; i < nimages; i++, j += ni )
    {
        Mat objpt = objectPoints.getMat(i);
        Mat imgpt1 = imagePoints1.getMat(i);
        ni = objpt.checkVector(3, CV_32F);
        npoints.at<int>(i) = ni;
        memcpy( objPtData + j, objpt.ptr(), ni*sizeof(objPtData[0]) );
        memcpy( imgPtData1 + j, imgpt1.ptr(), ni*sizeof(imgPtData1[0]) );

        if( imgPtData2 )
        {
            Mat imgpt2 = imagePoints2.getMat(i);
            int ni2 = imgpt2.checkVector(2, CV_32F);
            CV_Assert( ni == ni2 );
            memcpy( imgPtData2 + j, imgpt2.ptr(), ni*sizeof(imgPtData2[0]) );
        }
    }
}

double cvfork::calibrateCamera(InputArrayOfArrays _objectPoints,
                            InputArrayOfArrays _imagePoints,
                            Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                            OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, OutputArray _stdDeviations, OutputArray _perViewErrors, int flags, TermCriteria criteria )
{
    int rtype = CV_64F;
    Mat cameraMatrix = _cameraMatrix.getMat();
    cameraMatrix = prepareCameraMatrix(cameraMatrix, rtype);
    Mat distCoeffs = _distCoeffs.getMat();
    distCoeffs = prepareDistCoeffs(distCoeffs, rtype);
    if( !(flags & CALIB_RATIONAL_MODEL) &&
    (!(flags & CALIB_THIN_PRISM_MODEL)) &&
    (!(flags & CALIB_TILTED_MODEL)))
        distCoeffs = distCoeffs.rows == 1 ? distCoeffs.colRange(0, 5) : distCoeffs.rowRange(0, 5);

    int nimages = int(_objectPoints.total());
    CV_Assert( nimages > 0 );
    Mat objPt, imgPt, npoints, rvecM, tvecM, stdDeviationsM, errorsM;

    bool rvecs_needed = _rvecs.needed(), tvecs_needed = _tvecs.needed(),
            stddev_needed = _stdDeviations.needed(), errors_needed = _perViewErrors.needed();

    bool rvecs_mat_vec = _rvecs.isMatVector();
    bool tvecs_mat_vec = _tvecs.isMatVector();

    if( rvecs_needed ) {
        _rvecs.create(nimages, 1, CV_64FC3);

        if(rvecs_mat_vec)
            rvecM.create(nimages, 3, CV_64F);
        else
            rvecM = _rvecs.getMat();
    }

    if( tvecs_needed ) {
        _tvecs.create(nimages, 1, CV_64FC3);

        if(tvecs_mat_vec)
            tvecM.create(nimages, 3, CV_64F);
        else
            tvecM = _tvecs.getMat();
    }

    if( stddev_needed ) {
        _stdDeviations.create(nimages*6 + CV_CALIB_NINTRINSIC, 1, CV_64F);
        stdDeviationsM = _stdDeviations.getMat();
    }

    if( errors_needed) {
        _perViewErrors.create(nimages, 1, CV_64F);
        errorsM = _perViewErrors.getMat();
    }

    collectCalibrationData( _objectPoints, _imagePoints, noArray(),
                            objPt, imgPt, 0, npoints );
    CvMat c_objPt = objPt, c_imgPt = imgPt, c_npoints = npoints;
    CvMat c_cameraMatrix = cameraMatrix, c_distCoeffs = distCoeffs;
    CvMat c_rvecM = rvecM, c_tvecM = tvecM, c_stdDev = stdDeviationsM, c_errors = errorsM;

    double reprojErr = cvfork::cvCalibrateCamera2(&c_objPt, &c_imgPt, &c_npoints, imageSize,
                                          &c_cameraMatrix, &c_distCoeffs,
                                          rvecs_needed ? &c_rvecM : NULL,
                                          tvecs_needed ? &c_tvecM : NULL,
                                          stddev_needed ? &c_stdDev : NULL,
                                          errors_needed ? &c_errors : NULL, flags, criteria );

    // overly complicated and inefficient rvec/ tvec handling to support vector<Mat>
    for(int i = 0; i < nimages; i++ )
    {
        if( rvecs_needed && rvecs_mat_vec)
        {
            _rvecs.create(3, 1, CV_64F, i, true);
            Mat rv = _rvecs.getMat(i);
            memcpy(rv.ptr(), rvecM.ptr(i), 3*sizeof(double));
        }
        if( tvecs_needed && tvecs_mat_vec)
        {
            _tvecs.create(3, 1, CV_64F, i, true);
            Mat tv = _tvecs.getMat(i);
            memcpy(tv.ptr(), tvecM.ptr(i), 3*sizeof(double));
        }
    }

    cameraMatrix.copyTo(_cameraMatrix);
    distCoeffs.copyTo(_distCoeffs);

    return reprojErr;
}

double cvfork::calibrateCameraCharuco(InputArrayOfArrays _charucoCorners, InputArrayOfArrays _charucoIds,
                              Ptr<aruco::CharucoBoard> &_board, Size imageSize,
                              InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                              OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, OutputArray _stdDeviations, OutputArray _perViewErrors,
                              int flags, TermCriteria criteria) {

    CV_Assert(_charucoIds.total() > 0 && (_charucoIds.total() == _charucoCorners.total()));

    // Join object points of charuco corners in a single vector for calibrateCamera() function
    std::vector< std::vector< Point3f > > allObjPoints;
    allObjPoints.resize(_charucoIds.total());
    for(unsigned int i = 0; i < _charucoIds.total(); i++) {
        unsigned int nCorners = (unsigned int)_charucoIds.getMat(i).total();
        CV_Assert(nCorners > 0 && nCorners == _charucoCorners.getMat(i).total()); //actually nCorners must be > 3 for calibration
        allObjPoints[i].reserve(nCorners);

        for(unsigned int j = 0; j < nCorners; j++) {
            int pointId = _charucoIds.getMat(i).ptr< int >(0)[j];
            CV_Assert(pointId >= 0 && pointId < (int)_board->chessboardCorners.size());
            allObjPoints[i].push_back(_board->chessboardCorners[pointId]);
        }
    }

    return cvfork::calibrateCamera(allObjPoints, _charucoCorners, imageSize, _cameraMatrix, _distCoeffs,
                           _rvecs, _tvecs, _stdDeviations, _perViewErrors, flags, criteria);
}


static void subMatrix(const cv::Mat& src, cv::Mat& dst, const std::vector<uchar>& cols,
                      const std::vector<uchar>& rows) {
    int nonzeros_cols = cv::countNonZero(cols);
    cv::Mat tmp(src.rows, nonzeros_cols, CV_64FC1);

    for (int i = 0, j = 0; i < (int)cols.size(); i++)
    {
        if (cols[i])
        {
            src.col(i).copyTo(tmp.col(j++));
        }
    }

    int nonzeros_rows  = cv::countNonZero(rows);
    dst.create(nonzeros_rows, nonzeros_cols, CV_64FC1);
    for (int i = 0, j = 0; i < (int)rows.size(); i++)
    {
        if (rows[i])
        {
            tmp.row(i).copyTo(dst.row(j++));
        }
    }
}

void cvfork::CvLevMarqFork::step()
{
    using namespace cv;
    const double LOG10 = log(10.);
    double lambda = exp(lambdaLg10*LOG10);
    int nparams = param->rows;

    Mat _JtJ = cvarrToMat(JtJ);
    Mat _mask = cvarrToMat(mask);

    int nparams_nz = countNonZero(_mask);
    if(!JtJN || JtJN->rows != nparams_nz) {
        // prevent re-allocation in every step
        JtJN.reset(cvCreateMat( nparams_nz, nparams_nz, CV_64F ));
        JtJV.reset(cvCreateMat( nparams_nz, 1, CV_64F ));
        JtJW.reset(cvCreateMat( nparams_nz, 1, CV_64F ));
    }

    Mat _JtJN = cvarrToMat(JtJN);
    Mat _JtErr = cvarrToMat(JtJV);
    Mat_<double> nonzero_param = cvarrToMat(JtJW);

    subMatrix(cvarrToMat(JtErr), _JtErr, std::vector<uchar>(1, 1), _mask);
    subMatrix(_JtJ, _JtJN, _mask, _mask);

    if( !err )
        completeSymm( _JtJN, completeSymmFlag );
#if 1
    _JtJN.diag() *= 1. + lambda;
#else
    _JtJN.diag() += lambda;
#endif
#ifndef USE_LAPACK
    cv::solve(_JtJN, _JtErr, nonzero_param, solveMethod);
#else
    cvfork::solve(_JtJN, _JtErr, nonzero_param, solveMethod);
#endif

    int j = 0;
    for( int i = 0; i < nparams; i++ )
        param->data.db[i] = prevParam->data.db[i] - (mask->data.ptr[i] ? nonzero_param(j++) : 0);
}

cvfork::CvLevMarqFork::CvLevMarqFork(int nparams, int nerrs, CvTermCriteria criteria0, bool _completeSymmFlag)
{
    init(nparams, nerrs, criteria0, _completeSymmFlag);
}

cvfork::CvLevMarqFork::~CvLevMarqFork()
{
    clear();
}

bool cvfork::CvLevMarqFork::updateAlt( const CvMat*& _param, CvMat*& _JtJ, CvMat*& _JtErr, double*& _errNorm )
{
    CV_Assert( !err );
    if( state == DONE )
    {
        _param = param;
        return false;
    }

    if( state == STARTED )
    {
        _param = param;
        cvZero( JtJ );
        cvZero( JtErr );
        errNorm = 0;
        _JtJ = JtJ;
        _JtErr = JtErr;
        _errNorm = &errNorm;
        state = CALC_J;
        return true;
    }

    if( state == CALC_J )
    {
        cvCopy( param, prevParam );
        step();
        _param = param;
        prevErrNorm = errNorm;
        errNorm = 0;
        _errNorm = &errNorm;
        state = CHECK_ERR;
        return true;
    }

    assert( state == CHECK_ERR );
    if( errNorm > prevErrNorm )
    {
        if( ++lambdaLg10 <= 16 )
        {
            step();
            _param = param;
            errNorm = 0;
            _errNorm = &errNorm;
            state = CHECK_ERR;
            return true;
        }
    }

    lambdaLg10 = MAX(lambdaLg10-1, -16);
    if( ++iters >= criteria.max_iter ||
        cvNorm(param, prevParam, CV_RELATIVE_L2) < criteria.epsilon )
    {
        //printf("iters %i\n", iters);
        _param = param;
        state = DONE;
        return false;
    }

    prevErrNorm = errNorm;
    cvZero( JtJ );
    cvZero( JtErr );
    _param = param;
    _JtJ = JtJ;
    _JtErr = JtErr;
    state = CALC_J;
    return true;
}
