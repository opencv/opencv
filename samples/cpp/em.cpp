#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

using namespace cv;
using namespace cv::ml;

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

    // cluster the data
    Ptr<EM> em_model = EM::train( samples, noArray(), labels, noArray(),
            EM::Params(N, EM::COV_MAT_SPHERICAL,
                       TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 300, 0.1)));

    // classify every image pixel
    for( i = 0; i < img.rows; i++ )
    {
        for( j = 0; j < img.cols; j++ )
        {
            sample.at<float>(0) = (float)j;
            sample.at<float>(1) = (float)i;
            int response = cvRound(em_model->predict2( sample, noArray() )[1]);
            Scalar c = colors[response];

            circle( img, Point(j, i), 1, c*0.75, FILLED );
        }
    }

    //draw the clustered samples
    for( i = 0; i < nsamples; i++ )
    {
        Point pt(cvRound(samples.at<float>(i, 0)), cvRound(samples.at<float>(i, 1)));
        circle( img, pt, 1, colors[labels.at<int>(i)], FILLED );
    }

    imshow( "EM-clustering result", img );
    waitKey(0);

    return 0;
}
