#pragma once
#include <limits>

#define CURRENT_TYPE double

namespace cv
{

/** @brief Cholesky decomposition
* The function performs Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>. 
* @param A - the Hermitian, positive-definite matrix,
* @param astep - size of row in A,
* @param asize - number of cols and rows in A,
* @param L - the lower triangular matrix, A = L*Lt.
*/
template<typename _Tp> bool
choleskyDecomposition( const _Tp* A, size_t astep, const int asize, _Tp* L )
{
    int i, j, k;
    double s;
    astep /= sizeof(A[0]);
    for( i = 0; i < asize; i++ )

    {
        for( j = 0; j < i; j++ )
        {
            s = A[i*astep + j];
            for( k = 0; k < j; k++ )
                s -= L[i*astep + k]*L[j*astep + k];
            L[i*astep + j] = (_Tp)(s/L[j*astep + j]);
        }
        s = A[i*astep + i];
        for( k = 0; k < i; k++ )
        {
            double t = L[i*astep + k];
            s -= t*t;
        }
        if( s < std::numeric_limits<_Tp>::epsilon() )
            return false;
        L[i*astep + i] = (_Tp)(std::sqrt(s));
    }

   for( i = 0; i < asize; i++ )
       for( j = i+1; j < asize; j++ )
       {
           L[i*astep + j] = 0.0;
       }
       
    return true;
}

/** @brief Interface.
* The interface for Kalman filter, Unscented Kalman filter and Augmented Unscented Kalman filter.

*/
class CV_EXPORTS_W KalmanFilterInterface
{
public:

    CV_WRAP virtual ~KalmanFilterInterface(){}

    /** The function performs prediction step of the algorithm
    * @param control - the current control vector,    
    * @return - the predicted estimate of the state.
    */
    CV_WRAP virtual Mat predict( const Mat& control = Mat() ) = 0;

    /** The function performs correction step of the algorithm
    * @param measurement - the current measurement vector,    
    * @return - the corrected estimate of the state.
    */
    CV_WRAP virtual Mat correct( const Mat& measurement ) = 0;

    /** 
    * @return - the process noise cross-covariance matrix.
    */
    CV_WRAP virtual Mat getProcessNoiseCov() const = 0;

    /** 
    * @return - the measurement noise cross-covariance matrix.
    */
    CV_WRAP virtual Mat getMeasurementNoiseCov() const = 0;

    /** 
    * @return - the current estimate of the state.
    */
    CV_WRAP virtual Mat getState() const = 0;
};


/** @brief Kalman filter parameters.
* The class for initialization parameters of Kalman filter
*/
class CV_EXPORTS_W KalmanFilterParams
{
public:
    CV_PROP_RW Mat stateInit;                              //!< Initial state, DP x 1, default is zero.
    CV_PROP_RW Mat errorCovInit;                           //!< Initial state cross-covariance matrix, DP x DP, default is identity.

    CV_PROP_RW Mat transitionMatrix;                       //!< State transition matrix (A), DP x DP.
    CV_PROP_RW Mat controlMatrix;                          //!< Control matrix (B) (not used if there is no control), DP x CP.
    CV_PROP_RW Mat measurementMatrix;                      //!< Measurement matrix (H), DP x MP.

    CV_PROP_RW Mat processNoiseCov;                        //!< Process noise cross-covariance matrix, DP x DP.
    CV_PROP_RW Mat measurementNoiseCov;                    //!< Measurement noise cross-covariance matrix, MP x MP.

    CV_PROP_RW int DP;                                     //!< Dimensionality of the state vector.
    CV_PROP_RW int MP;                                     //!< Dimensionality of the measurement vector.
    CV_PROP_RW int CP;                                     //!< Dimensionality of the control vector.  
    CV_PROP_RW int dataType;                               //!< Type of elements of vectors and matrices. 


    /** The constructors.
     */
    CV_WRAP KalmanFilterParams(){}

    /** 
    * @param dp - dimensionality of the state vector,    
    * @param mp - dimensionality of the measurement vector,  
    * @param cp - dimensionality of the control vector,  
    * @param processNoiseCovDiag - value of elements on main diagonal process noise cross-covariance matrix,
    * @param measurementNoiseCovDiag - value of elements on main diagonal measurement noise cross-covariance matrix,
    * @param type - type of the created matrices that should be CV_32F or CV_64F.
    */
    CV_WRAP KalmanFilterParams( int dp, int mp, int cp, double processNoiseCovDiag, double measurementNoiseCovDiag, int type = CV_64F );
     
    /** The function for initialization of Kalman filter
    * @param dp - dimensionality of the state vector,    
    * @param mp - dimensionality of the measurement vector,  
    * @param cp - dimensionality of the control vector,  
    * @param processNoiseCovDiag - value of elements on main diagonal process noise cross-covariance matrix,
    * @param measurementNoiseCovDiag - value of elements on main diagonal measurement noise cross-covariance matrix,
    * @param type - type of the created matrices that should be CV_32F or CV_64F.
    */
    CV_WRAP void init( int dp, int mp, int cp, double processNoiseCovDiag, double measurementNoiseCovDiag, int type = CV_64F );
};

/** The function for computing the next state from the previous state
* @param x_k - previous state vector,
* @param u_k - control vector,
* @param v_k - noise vector,
* @param x_kplus1 - next state vector.
*/
typedef void( * StateFunction )( const Mat& x_k, const Mat& u_k, const Mat& v_k, Mat& x_kplus1 );

/** The function for computing the measurement from the state
* @param x_k - state vector,
* @param n_k - noise vector,
* @param z_k - measurement vector.
*/
typedef void( * MeasFunction )( const Mat& x_k, const Mat& n_k, Mat& z_k );


/** @brief Unscented Kalman filter parameters.
* The class for initialization parameters of Unscented Kalman filter
*/
class CV_EXPORTS_W KalmanFilterUnscentedParams
{
public:

    CV_PROP_RW int DP;                                     //!< Dimensionality of the state vector.    
    CV_PROP_RW int MP;                                     //!< Dimensionality of the measurement vector. 
    CV_PROP_RW int CP;                                     //!< Dimensionality of the control vector.
    CV_PROP_RW int dataType;                               //!< Type of elements of vectors and matrices, default is CV_64F.

    CV_PROP_RW Mat stateInit;                              //!< Initial state, DP x 1, default is zero. 
    CV_PROP_RW Mat errorCovInit;                           //!< State estimate cross-covariance matrix, DP x DP, default is identity.
     
    CV_PROP_RW Mat processNoiseCov;                        //!< Process noise cross-covariance matrix, DP x DP.
    CV_PROP_RW Mat measurementNoiseCov;                    //!< Measurement noise cross-covariance matrix, MP x MP.

    // Parameters of algorithm
    CV_PROP_RW double alpha;                               //!< Default is 1e-3.
    CV_PROP_RW double k;                                   //!< Default is 0.
    CV_PROP_RW double beta;                                //!< Default is 2.0.

    //Functions
    CV_PROP_RW StateFunction f;                            //!< Function for computing the next state from the previous state  
    CV_PROP_RW MeasFunction h;                             //!< Function for computing the measurement from the state

    /** The constructors.
    */
    CV_WRAP KalmanFilterUnscentedParams(){}

    /** 
    * @param dp - dimensionality of the state vector,    
    * @param mp - dimensionality of the measurement vector,  
    * @param cp - dimensionality of the control vector,  
    * @param processNoiseCovDiag - value of elements on main diagonal process noise cross-covariance matrix,
    * @param measurementNoiseCovDiag - value of elements on main diagonal measurement noise cross-covariance matrix,
    * @param type - type of the created matrices that should be CV_32F or CV_64F.
    */
    CV_WRAP KalmanFilterUnscentedParams( int DP, int MP, int CP, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                StateFunction _f, MeasFunction _h, int type = CV_64F );

    /** The function for initialization of Unscented Kalman filter
    * @param dp - dimensionality of the state vector,    
    * @param mp - dimensionality of the measurement vector,  
    * @param cp - dimensionality of the control vector,  
    * @param processNoiseCovDiag - value of elements on main diagonal process noise cross-covariance matrix,
    * @param measurementNoiseCovDiag - value of elements on main diagonal measurement noise cross-covariance matrix,
    * @param type - type of the created matrices that should be CV_32F or CV_64F.
    */
    CV_WRAP void init( int DP, int MP, int CP, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                StateFunction _f, MeasFunction _h, int type = CV_64F );
};

/** @brief Augmented Unscented Kalman filter parameters.
* The class for initialization parameters of Augmented Unscented Kalman filter
*/
class CV_EXPORTS_W KalmanFilterUnscentedAugmentedParams: public KalmanFilterUnscentedParams
{
public:

    CV_WRAP KalmanFilterUnscentedAugmentedParams(){}

    /** 
    * @param dp - dimensionality of the state vector,    
    * @param mp - dimensionality of the measurement vector,  
    * @param cp - dimensionality of the control vector,  
    * @param processNoiseCovDiag - value of elements on main diagonal process noise cross-covariance matrix,
    * @param measurementNoiseCovDiag - value of elements on main diagonal measurement noise cross-covariance matrix,
    * @param type - type of the created matrices that should be CV_32F or CV_64F.
    */
    CV_WRAP KalmanFilterUnscentedAugmentedParams( int DP, int MP, int CP, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                StateFunction _f, MeasFunction _h, int type = CV_64F );

    /** The function for initialization of Augmented Unscented Kalman filter
    * @param dp - dimensionality of the state vector,    
    * @param mp - dimensionality of the measurement vector,  
    * @param cp - dimensionality of the control vector,  
    * @param processNoiseCovDiag - value of elements on main diagonal process noise cross-covariance matrix,
    * @param measurementNoiseCovDiag - value of elements on main diagonal measurement noise cross-covariance matrix,
    * @param type - type of the created matrices that should be CV_32F or CV_64F.
    */
    CV_WRAP void init( int DP, int MP, int CP, double processNoiseCovDiag, double measurementNoiseCovDiag,
                                StateFunction _f, MeasFunction _h, int type = CV_64F );
};



/** Kalman Filter factory method
* The class implements a standard Kalman filter <http://en.wikipedia.org/wiki/Kalman_filter>. 
* However, you can modify transitionMatrix, controlMatrix, and measurementMatrix to get an extended Kalman filter functionality.
* @param params - an object of the KalmanFilterParams class containing KF parameters. 
* @return - pointer to the object of the KalmanFilter class implementing KalmanFilterInterface.
*/
CV_WRAP cv::Ptr<KalmanFilterInterface> createKalmanFilter( const KalmanFilterParams &params );
/** Unscented Kalman Filter factory method
* The class implements an Unscented Kalman filter <https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter>.
* @param params - an object of the KalmanFilterUnscentedParams class containing UKF parameters.
* @return - pointer to the object of the KalmanFilterUnscented class implementing KalmanFilterInterface.
*/
CV_WRAP cv::Ptr<KalmanFilterInterface> createUnscKalmanFilter( const KalmanFilterUnscentedParams &params );
/** Augmented Unscented Kalman Filter factory method
* The class implements an Augmented Unscented Kalman filter http://becs.aalto.fi/en/research/bayes/ekfukf/documentation.pdf, page 31-33.
* AUKF is more accurate than UKF but its computational complexity is larger.
* @param params - an object of the KalmanFilterUnscentedAugmentedParams class containing AUKF parameters. 
* @return - pointer to the object of the KalmanFilterUnscentedAugmented class implementing KalmanFilterInterface.
*/
CV_WRAP cv::Ptr<KalmanFilterInterface> createAugUnscKalmanFilter( const KalmanFilterUnscentedAugmentedParams &params );



} // cv


