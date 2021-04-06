/**
 * @brief Sample code showing how to segment overlapping objects using Laplacian filtering, in addition to Watershed and Distance Transformation
 * @author OpenCV Team
 */

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
    //! [load_image]
    // Load the image
    CommandLineParser parser( argc, argv, "{@input | cards.png | input image}" );
    Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if( src.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }

    // Show the source image
    imshow("Source Image", src);
    //! [load_image]

    //! [black_bg]
    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    Mat mask;
    inRange(src, Scalar(255, 255, 255), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(0, 0, 0), mask);

    // Show output image
    imshow("Black Background Image", src);
    //! [black_bg]

    //! [sharp]
    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3,3) <<
                  1,  1, 1,
                  1, -8, 1,
                  1,  1, 1); // an approximation of second derivative, a quite strong kernel

    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;

    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow( "New Sharped Image", imgResult );
    //! [sharp]

    //! [bin]
    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
    imshow("Binary Image", bw);
    //! [bin]

    //! [dist]
    // Perform the distance transform algorithm
    Mat dist;
    distanceTransform(bw, dist, DIST_L2, 3);

    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1.0, NORM_MINMAX);
    imshow("Distance Transform Image", dist);
    //! [dist]

    //! [peaks]
    // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);

    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8U);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
    //! [peaks]

    //! [seeds]
    // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32S);

    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(markers, contours, static_cast<int>(i), Scalar(static_cast<int>(i)+1), -1);
    }

    // Draw the background marker
    circle(markers, Point(5,5), 3, Scalar(255), -1);
    Mat markers8u;
    markers.convertTo(markers8u, CV_8U, 10);
    imshow("Markers", markers8u);
    //! [seeds]

    //! [watershed]
    // Perform the watershed algorithm
    watershed(imgResult, markers);

    Mat mark;
    markers.convertTo(mark, CV_8U);
    bitwise_not(mark, mark);
    //    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
    // image looks like at that point

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);

        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
            {
                dst.at<Vec3b>(i,j) = colors[index-1];
            }
        }
    }

    // Visualize the final image
    imshow("Final Result", dst);
    //! [watershed]

    waitKey();
    return 0;
}
