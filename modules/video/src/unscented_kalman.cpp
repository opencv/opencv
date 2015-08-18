#include "precomp.hpp"

namespace cv
{

void KalmanFilterUnscentedParams::
    init( int dp, int mp, int cp, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                StateFunction _f, MeasFunction _h, int type )
{
    CV_Assert( dp > 0 && mp > 0 );
    DP = dp;
    MP = mp;
    CP = std::max( cp, 0 );
    CV_Assert( type == CV_32F || type == CV_64F );
    dataType = type;

    this->f = _f;
    this->h = _h;

    stateInit = Mat::zeros( DP, 1, type );
    errorCovInit = Mat::eye( DP, DP, type );

    processNoiseCov = processNoiseCovDiag*Mat::eye( DP, DP, type );
    measurementNoiseCov = measurementNoiseCovDiag*Mat::eye( MP, MP, type );

    alpha = 1e-3;
    k = 0;
    beta = 2.0;
}

KalmanFilterUnscentedParams::
    KalmanFilterUnscentedParams( int dp, int mp, int cp, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                 StateFunction _f, MeasFunction _h, int type )
{
    init( dp, mp, cp, processNoiseCovDiag, measurementNoiseCovDiag, _f, _h, type );
}


class KalmanFilterUnscented: public KalmanFilterInterface
{

    int DP;                                     // dimensionality of the state vector    
    int MP;                                     // dimensionality of the measurement vector  
    int CP;                                     // dimensionality of the control vector  
    int dataType;                               // type of elements of vectors and matrices

    Mat state;                                  // estimate of the system state (x*), DP x 1
    Mat errorCov;                               // estimate of the state cross-covariance matrix (P), DP x DP

    Mat processNoiseCov;                        // process noise cross-covariance matrix (Q), DP x DP
    Mat measurementNoiseCov;                    // measurement noise cross-covariance matrix (R), MP x MP

    StateFunction f;                            // function for computing the next state from the previous state, f(x, u, q), 
                                                // x - previous state vector,
                                                // u - control vector,
                                                // q - process noise vector   
    
    MeasFunction h;                             // function for computing the measurement from the state, h(x, r)
                                                // x - state vector,
                                                // r - measurement noise vector  

// Parameters of algorithm
    double alpha;                               // parameter, default is 1e-3
    double k;                                   // parameter, default is 0
    double beta;                                // parameter, default is 2.0

    double lambda;                              // internal parameter, lambda = alpha*alpha*( DP + k ) - DP;
    double tmpLambda;                           // internal parameter, tmpLambda = alpha*alpha*( DP + k );

// Auxillary members
    Mat measurementEstimate;                    // estimate of current measurement (y*), MP x 1

    Mat sigmaPoints;                            // set of sigma points ( x_i, i = 1..2*DP+1 ), DP x 2*DP+1

    Mat transitionSPFuncVals;                   // set of state function values at sigma points ( f_i, i = 1..2*DP+1 ), DP x 2*DP+1
    Mat measurementSPFuncVals;                  // set of measurement function values at sigma points ( h_i, i = 1..2*DP+1 ), MP x 2*DP+1

    Mat transitionSPFuncValsCenter;             // set of state function values at sigma points minus estimate of state ( fc_i, i = 1..2*DP+1 ), DP x 2*DP+1
    Mat measurementSPFuncValsCenter;            // set of measurement function values at sigma points minus estimate of measurement ( hc_i, i = 1..2*DP+1 ), MP x 2*DP+1

    Mat Wm;                                     // vector of weights for estimate mean, 2*DP+1 x 1
    Mat Wc;                                     // matrix of weights for estimate covariance, 2*DP+1 x 2*DP+1

    Mat gain;                                   // Kalman gain matrix (K), DP x MP
    Mat xyCov;                                  // estimate of the covariance between x* and y* (Sxy), DP x MP
    Mat yyCov;                                  // estimate of the y* cross-covariance matrix (Syy), MP x MP

    Mat r;                                      // zero vector of process noise for getting transitionSPFuncVals, 
    Mat q;                                      // zero vector of measurement noise for getting measurementSPFuncVals


//get sigma points 
    Mat getSigmaPoints( const Mat& mean, const Mat& covMatrix, double coef );

public:

    KalmanFilterUnscented( const KalmanFilterUnscentedParams& params );
    ~KalmanFilterUnscented();

// perform prediction step
// control - the optional control vector, CP x 1
    Mat predict( const Mat& control = Mat() );

// perform correction step
// measurement - current measurement vector, MP x 1
    Mat correct( const Mat& measurement );

//  Get system parameters
    Mat getProcessNoiseCov() const;
    Mat getMeasurementNoiseCov() const;

//  Get the state estimate
    Mat getState() const;
};

KalmanFilterUnscented::KalmanFilterUnscented(const KalmanFilterUnscentedParams& params)
{
    alpha = params.alpha;
    beta = params.beta;
    k = params.k;

    DP = params.DP;
    MP = params.MP;
    CP = params.CP;
    dataType = params.dataType;

    f = params.f;
    h = params.h;

    state = params.stateInit.clone();
    errorCov = params.errorCovInit.clone();

    processNoiseCov = params.processNoiseCov.clone();
    measurementNoiseCov = params.measurementNoiseCov.clone();

    measurementEstimate = Mat::zeros( MP, 1, dataType);

    q = Mat::zeros( DP, 1, dataType);
    r = Mat::zeros( MP, 1, dataType);

    gain = Mat::zeros( DP, DP, dataType );

    transitionSPFuncVals = Mat::zeros( DP, 2*DP+1, dataType );
    measurementSPFuncVals = Mat::zeros( MP, 2*DP+1, dataType );

    transitionSPFuncValsCenter = Mat::zeros( DP, 2*DP+1, dataType );
    measurementSPFuncValsCenter = Mat::zeros( MP, 2*DP+1, dataType );

    lambda = alpha*alpha*( DP + k ) - DP;
    tmpLambda = lambda + DP;

    double tmp2Lambda = 0.5/tmpLambda;

    Wm = tmp2Lambda * Mat::ones( 2*DP+1, 1, dataType );
    Wm.at<CURRENT_TYPE>(0,0) = lambda/tmpLambda;

    Wc = tmp2Lambda * Mat::eye( 2*DP+1, 2*DP+1, dataType );
    Wc.at<CURRENT_TYPE>(0,0) =  lambda/tmpLambda + 1.0 - alpha*alpha + beta;
}

KalmanFilterUnscented::~KalmanFilterUnscented()
{
    state.release();
    errorCov.release();

    processNoiseCov.release();
    measurementNoiseCov.release();

    measurementEstimate.release();

    sigmaPoints.release();

    transitionSPFuncVals.release();
    measurementSPFuncVals.release();

    transitionSPFuncValsCenter.release();
    measurementSPFuncValsCenter.release();

    Wm.release();
    Wc.release();

    gain.release();
    xyCov.release();
    yyCov.release();

    r.release();
    q.release();
}

Mat KalmanFilterUnscented::getSigmaPoints(const Mat &mean, const Mat &covMatrix, double coef)
{
// x_0 = mean
// x_i = mean + coef * cholesky( covMatrix ), i = 1..n
// x_(i+n) = mean - coef * cholesky( covMatrix ), i = 1..n

    int n = mean.rows;
    Mat points = repeat(mean, 1, 2*n+1);

    Mat covMatrixL = covMatrix.clone();
    covMatrixL.setTo(0);

// covMatrixL = cholesky( covMatrix )
    choleskyDecomposition( covMatrix.ptr<CURRENT_TYPE>(), covMatrix.step, covMatrix.rows, covMatrixL.ptr<CURRENT_TYPE>() );
    covMatrixL = coef * covMatrixL;

    Mat p_plus = points( Rect( 1, 0, n, n ) );
    Mat p_minus = points( Rect( n+1, 0, n, n ) );

    add(p_plus, covMatrixL, p_plus);
    subtract(p_minus, covMatrixL, p_minus);

    return points;
}

Mat KalmanFilterUnscented::predict(const Mat& control)
{
// get sigma points from x* and P
    sigmaPoints = getSigmaPoints( state, errorCov, sqrt( tmpLambda ) );

// compute f-function values at sigma points
// f_i = f(x_i, control, 0), i = 0..2*DP
    Mat x, fx;
    for ( int i = 0; i<2*DP+1; i++)
    {
        x = sigmaPoints( Rect( i, 0, 1, DP) );
        fx = transitionSPFuncVals( Rect( i, 0, 1, DP) );
        f( x, control, q, fx );
    }
// compute the estimate of state as mean f-function value at sigma point
// x* = SUM_{i=0}^{2*DP}( Wm[i]*f_i )
    state = transitionSPFuncVals * Wm;

// compute f-function values at sigma points minus estimate of state
// fc_i = f_i - x*, i = 0..2*DP
    subtract(transitionSPFuncVals, repeat( state, 1, 2*DP+1 ), transitionSPFuncValsCenter);

// compute the estimate of the state cross-covariance matrix
// P = SUM_{i=0}^{2*DP}( Wc[i]*fc_i*fc_i.t ) + Q
    errorCov = transitionSPFuncValsCenter * Wc * transitionSPFuncValsCenter.t() + processNoiseCov;

    return state.clone();
}

Mat KalmanFilterUnscented::correct(const Mat& measurement)
{
// get sigma points from x* and P
    sigmaPoints = getSigmaPoints( state, errorCov, sqrt( tmpLambda ) );

// compute h-function values at sigma points
// h_i = h(x_i, 0), i = 0..2*DP
    Mat x, hx;
    for ( int i = 0; i<2*DP+1; i++)
    {
        x = sigmaPoints( Rect( i, 0, 1, DP) );
        hx = measurementSPFuncVals( Rect( i, 0, 1, MP) );
        h( x, r, hx );       
    }

// compute the estimate of measurement as mean h-function value at sigma point
// y* = SUM_{i=0}^{2*DP}( Wm[i]*h_i )
    measurementEstimate = measurementSPFuncVals * Wm;

// compute h-function values at sigma points minus estimate of state
// hc_i = h_i - y*, i = 0..2*DP
    subtract(measurementSPFuncVals, repeat( measurementEstimate, 1, 2*DP+1 ), measurementSPFuncValsCenter);

// compute the estimate of the y* cross-covariance matrix
// Syy = SUM_{i=0}^{2*DP}( Wc[i]*hc_i*hc_i.t ) + R
    yyCov = measurementSPFuncValsCenter * Wc * measurementSPFuncValsCenter.t() + measurementNoiseCov;

// compute the estimate of the covariance between x* and y*
// Sxy = SUM_{i=0}^{2*DP}( Wc[i]*fc_i*hc_i.t )
    xyCov = transitionSPFuncValsCenter * Wc * measurementSPFuncValsCenter.t();

// compute the Kalman gain matrix
// K = Sxy * Syy^(-1)
    gain = xyCov * yyCov.inv(DECOMP_SVD);

// compute the corrected estimate of state
// x* = x* + K*(y - y*), y - current measurement
    state = state + gain * ( measurement - measurementEstimate );

// compute the corrected estimate of the state cross-covariance matrix
// P = P - K*Sxy.t
    errorCov = errorCov - gain * xyCov.t();

    return state.clone();
}

Mat KalmanFilterUnscented::getProcessNoiseCov() const
{
    return processNoiseCov.clone();
}

Mat KalmanFilterUnscented::getMeasurementNoiseCov() const
{
    return measurementNoiseCov.clone();
}

Mat KalmanFilterUnscented::getState() const
{
    return state.clone();
}

cv::Ptr<KalmanFilterInterface> createUnscKalmanFilter(const KalmanFilterUnscentedParams &params)
{
    cv::Ptr<KalmanFilterInterface> kfu( new KalmanFilterUnscented(params) );
    return kfu;
}

} //cv
