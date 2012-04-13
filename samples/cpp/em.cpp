#include "opencv2/legacy/legacy.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

int main( int /*argc*/, char** /*argv*/ )
{
    const int N = 4;
    const int N1 = (int)sqrt((double)N);
    const Scalar colors[] =
    {
        Scalar(0,0,255), Scalar(0,255,0), 
        Scalar(0,255,255),Scalar(255,255,0)
    };

    int i, j;
    int nsamples = 100;
    Mat samples( nsamples, 2, CV_32FC1 );
    Mat labels;
    Mat img = Mat::zeros( Size( 500, 500 ), CV_8UC3 );
    Mat sample( 1, 2, CV_32FC1 );
    CvEM em_model;
    CvEMParams params;

    samples = samples.reshape(2, 0);
    for( i = 0; i < N; i++ )
    {
        // form the training samples
        Mat samples_part = samples.rowRange(i*nsamples/N, (i+1)*nsamples/N );
        
        Scalar mean(((i%N1)+1)*img.rows/(N1+1),
                    ((i/N1)+1)*img.rows/(N1+1));
        Scalar sigma(30,30);
        randn( samples_part, mean, sigma );
    }
    samples = samples.reshape(1, 0);

    // initialize model parameters
    params.covs      = NULL;
    params.means     = NULL;
    params.weights   = NULL;
    params.probs     = NULL;
    params.nclusters = N;
    params.cov_mat_type       = CvEM::COV_MAT_SPHERICAL;
    params.start_step         = CvEM::START_AUTO_STEP;
    params.term_crit.max_iter = 300;
    params.term_crit.epsilon  = 0.1;
    params.term_crit.type     = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;

    // cluster the data
    em_model.train( samples, Mat(), params, &labels );

#if 0
    // the piece of code shows how to repeatedly optimize the model
    // with less-constrained parameters
    //(COV_MAT_DIAGONAL instead of COV_MAT_SPHERICAL)
    // when the output of the first stage is used as input for the second one.
    CvEM em_model2;
    params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
    params.start_step = CvEM::START_E_STEP;
    params.means = em_model.get_means();
    params.covs = (const CvMat**)em_model.get_covs();
    params.weights = em_model.get_weights();

    em_model2.train( samples, Mat(), params, &labels );
    // to use em_model2, replace em_model.predict()
    // with em_model2.predict() below
#endif
    // classify every image pixel
    for( i = 0; i < img.rows; i++ )
    {
        for( j = 0; j < img.cols; j++ )
        {
            sample.at<float>(0) = (float)j;
            sample.at<float>(1) = (float)i;
            int response = cvRound(em_model.predict( sample ));
            Scalar c = colors[response];

            circle( img, Point(j, i), 1, c*0.75, CV_FILLED );
        }
    }

    //draw the clustered samples
    for( i = 0; i < nsamples; i++ )
    {
        Point pt(cvRound(samples.at<float>(i, 0)), cvRound(samples.at<float>(i, 1)));
        circle( img, pt, 1, colors[labels.at<int>(i)], CV_FILLED );
    }

    imshow( "EM-clustering result", img );
    waitKey(0);

    return 0;
}
