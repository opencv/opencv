#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video.hpp"
#include <fstream>

using namespace std;
using namespace cv;

const String keys = "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1               }"
        "{@image2        |      | image2               }"
        "{@algorithm     |      | [farneback, simpleflow, tvl1 or deepflow] }"
        "{@groundtruth   |      | path to the .flo file  (optional) }"
        "{m measure      |endpoint| error measure - [endpoint or angular] }"
        "{r region       |all   | region to compute stats about [all, discontinuities, untextured] }"
        "{d display      |      | display additional info images (pauses program execution) }";

inline bool isFlowCorrect( const Point2f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9);
}
inline bool isFlowCorrect( const Point3f u )
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && !cvIsNaN(u.z) && (fabs(u.x) < 1e9) && (fabs(u.y) < 1e9)
            && (fabs(u.z) < 1e9);
}
///based on TV-L1 test, needs improvement
static Mat endpointError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);
    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1 = flow1(i, j);
            const Point2f u2 = flow2(i, j);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
            {
                const Point2f diff = u1 - u2;
                result.at<float>(i, j) = sqrt(diff.ddot(diff)); //distance
            } else
                result.at<float>(i, j) = NAN;
        }
    }
    return result;
}
static Mat angularError( const Mat_<Point2f>& flow1, const Mat_<Point2f>& flow2 )
{
    Mat result(flow1.size(), CV_32FC1);

    for ( int i = 0; i < flow1.rows; ++i )
    {
        for ( int j = 0; j < flow1.cols; ++j )
        {
            const Point2f u1_2d = flow1(i, j);
            const Point2f u2_2d = flow2(i, j);
            const Point3f u1(u1_2d.x, u1_2d.y, 1);
            const Point3f u2(u2_2d.x, u2_2d.y, 1);

            if ( isFlowCorrect(u1) && isFlowCorrect(u2) )
                result.at<float>(i, j) = acos((u1.ddot(u2)) / (norm(u1) * norm(u2)));
            else
                result.at<float>(i, j) = NAN;
        }
    }
    return result;
}
// what fraction of pixels have errors higher than given threshold?
static float stat_RX( Mat errors, float threshold, Mat mask )
{
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    int count = 0, all = 0;
    for ( int i = 0; i < errors.rows; ++i )
    {
        for ( int j = 0; j < errors.cols; ++j )
        {
            if ( mask.at<char>(i, j) != 0 )
            {
                ++all;
                if ( errors.at<float>(i, j) > threshold )
                    ++count;
            }
        }
    }
    return 1.0 * count / all;
}
static float stat_AX( Mat hist, double cutoff_count, double max_value )
{
    double counter = 0;
    int bin = 0;
    int bin_count = hist.rows;
    while ( bin < bin_count && counter < cutoff_count )
    {
        counter += hist.at<float>(bin, 0);
        ++bin;
    }
    return (1. * bin / bin_count) * max_value;
}
static void calculateStats( Mat errors, Mat mask = Mat(), bool display_images = false )
{
    float R_thresholds[] = { 0.5, 1, 2, 5, 10 };
    float A_thresholds[] = { 0.5, 0.75, 0.95 };
    if ( mask.empty() )
        mask = Mat::ones(errors.size(), CV_8U);
    CV_Assert(errors.size() == mask.size());
    CV_Assert(mask.depth() == CV_8U);

    //masking out NaNs - if not done before
    Mat nan_mask = Mat(errors != errors);
    bitwise_not(nan_mask, nan_mask);
    bitwise_and(nan_mask, mask, mask);

    //displaying the mask
    if(display_images)
    {
        namedWindow( "Region mask", WINDOW_AUTOSIZE );
        imshow( "Region mask", mask );
    }
//    long unsigned int masked_out_pixels = mask.total() - countNonZero(mask);
//    printf("Not analyzed pixels (%%): %lu (%.1f%%)\n", masked_out_pixels, 100.0 * masked_out_pixels / mask.total() );

    //mean and std computation
    Scalar s_mean, s_std;
    float mean, std;
    meanStdDev(errors, s_mean, s_std, mask);
    mean = s_mean[0];
    std = s_std[0];
    printf("Average: %.2f\nStandard deviation: %.2f\n", mean, std);

    //RX stats - displayed in percent
    float R;
    int R_thresholds_count = sizeof(R_thresholds) / sizeof(float);
    for ( int i = 0; i < R_thresholds_count; ++i )
    {
        R = stat_RX(errors, R_thresholds[i], mask);
        printf("R%.1f: %.2f%%\n", R_thresholds[i], R * 100);
    }

    //AX stats
    double max_value;
    minMaxLoc(errors, NULL, &max_value, NULL, NULL, mask);

    Mat hist;
    const int n_images = 1;
    const int channels[] = { 0 };
    const int n_dimensions = 1;
    const int hist_bins[] = { 1024 };
    const float iranges[] = { 0, (float) max_value };
    const float* ranges[] = { iranges };
    const bool uniform = true;
    const bool accumulate = false;
    calcHist(&errors, n_images, channels, mask, hist, n_dimensions, hist_bins, ranges, uniform,
            accumulate);
    int all_pixels = countNonZero(mask);
    int cutoff_count;
    float A;
    int A_thresholds_count = sizeof(A_thresholds) / sizeof(float);
    for ( int i = 0; i < A_thresholds_count; ++i )
    {
        cutoff_count = round(A_thresholds[i] * all_pixels);
        A = stat_AX(hist, cutoff_count, max_value);
        printf("A%.2f: %.2f\n", A_thresholds[i], A);
    }
    if(display_images) // wait for the user to see all the images
        waitKey(0);
}
int main( int argc, char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("OpenCV optical flow evaluation");
    if ( parser.has("help") )
    {
        parser.printMessage();
        return 0;
    }
    String i1_path = parser.get<String>(0);
    String i2_path = parser.get<String>(1);
    String method = parser.get<String>(2);
    String groundtruth_path = parser.get<String>(3);
    String error_measure = parser.get<String>("measure");
    String region = parser.get<String>("region");
    bool display_images = parser.has("display");

    if ( !parser.check() )
    {
        parser.printErrors();
        return 0;
    }

    Mat i1, i2;
    Mat_<Point2f> flow, ground_truth;
    Mat computed_errors;
    i1 = imread(i1_path, 1);
    i2 = imread(i2_path, 1);

    if ( !i1.data || !i2.data )
    {
        printf("No image data \n");
        return -1;
    }
    if ( i1.size() != i2.size() || i1.channels() != i2.channels() )
    {
        printf("Dimension mismatch between input images\n");
        return -1;
    }
    // 8-bit images expected by all algorithms
    if ( i1.depth() != CV_8U )
        i1.convertTo(i1, CV_8U);
    if ( i2.depth() != CV_8U )
        i2.convertTo(i2, CV_8U);

    if ( (method == "farneback" || method == "tvl1" || method == "deepflow") && i1.channels() == 3 )
    {   // 1-channel images are expected
        cvtColor(i1, i1, COLOR_BGR2GRAY);
        cvtColor(i2, i2, COLOR_BGR2GRAY);
    } else if ( method == "simpleflow" && i1.channels() == 1 )
    {   // 3-channel images expected
        cvtColor(i1, i1, COLOR_GRAY2BGR);
        cvtColor(i2, i2, COLOR_GRAY2BGR);
    }

    flow = Mat(i1.size[0], i1.size[1], CV_32FC2);
    Ptr<DenseOpticalFlow> algorithm;

    if ( method == "farneback" )
        algorithm = createOptFlow_Farnebacks();
    else if ( method == "simpleflow" )
        algorithm = createOptFlow_SimpleFlow();
    else if ( method == "tvl1" )
        algorithm = createOptFlow_DualTVL1();
    else if ( method == "deepflow" )
        algorithm = createOptFlow_DeepFlow();
    else
        printf("Wrong method!\n");

    double startTick, time;
    startTick = (double) getTickCount(); // measure time
    algorithm->calc(i1, i2, flow);
    time = ((double) getTickCount() - startTick) / getTickFrequency();
    printf("\nTime [s]: %.3f\n", time);

    if ( !groundtruth_path.empty() )
    { // compare to ground truth
        ground_truth = readOpticalFlow(groundtruth_path);
        if ( flow.size() != ground_truth.size() || flow.channels() != 2
                || ground_truth.channels() != 2 )
        {
            printf("Dimension mismatch between the computed flow and the provided ground truth\n");
            return -1;
        }

        if ( error_measure == "endpoint" )
            computed_errors = endpointError(flow, ground_truth);
        else if ( error_measure == "angular" )
            computed_errors = angularError(flow, ground_truth);
        else
        {
            printf("Invalid error measure! Available options: endpoint, angular\n");
            return -1;
        }

        Mat mask;
        if( region == "all" )
            mask = Mat::ones(ground_truth.size(), CV_8U) * 255;
        else if ( region == "discontinuities" )
        {
            Mat truth_merged, grad_x, grad_y, gradient;
            vector<Mat> truth_split;
            split(ground_truth, truth_split);
            truth_merged = truth_split[0] + truth_split[1];

            Sobel( truth_merged, grad_x, CV_16S, 1, 0, -1, 1, 0, BORDER_REPLICATE );
            grad_x = abs(grad_x);
            Sobel( truth_merged, grad_y, CV_16S, 0, 1, 1, 1, 0, BORDER_REPLICATE );
            grad_y = abs(grad_y);
            addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); //approximation!

            Scalar s_mean;
            s_mean = mean(gradient);
            double threshold = s_mean[0]; // threshold value arbitrary
            mask = gradient > threshold;
            dilate(mask, mask, Mat::ones(9, 9, CV_8U));
        }
        else if ( region == "untextured" )
        {
            Mat i1_grayscale, grad_x, grad_y, gradient;
            if( i1.channels() == 3 )
                cvtColor(i1, i1_grayscale, COLOR_BGR2GRAY);
            else
                i1_grayscale = i1;
            Sobel( i1_grayscale, grad_x, CV_16S, 1, 0, 7 );
            grad_x = abs(grad_x);
            Sobel( i1_grayscale, grad_y, CV_16S, 0, 1, 7 );
            grad_y = abs(grad_y);
            addWeighted(grad_x, 0.5, grad_y, 0.5, 0, gradient); //approximation!
            GaussianBlur(gradient, gradient, Size(5,5), 1, 1);

            Scalar s_mean;
            s_mean = mean(gradient);
            // arbitrary threshold value used - could be determined statistically from the image?
            double threshold = 1000;
            mask = gradient < threshold;
            dilate(mask, mask, Mat::ones(3, 3, CV_8U));
        }

        else
        {
            printf("Invalid region selected! Available options: all, discontinuities, untextured");
            return -1;
        }

        printf("Using %s error measure\n", error_measure.c_str());
        calculateStats(computed_errors, mask, display_images);

    }
    return 0;

}
