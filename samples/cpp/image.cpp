#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/miniflann.hpp"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;
using namespace cv::flann;

static void help()
{
    cout <<
    "\nThis program shows how to use cv::Mat and IplImages converting back and forth.\n"
    "It shows reading of images, converting to planes and merging back, color conversion\n"
    "and also iterating through pixels.\n"
    "Call:\n"
    "./image [image-name Default: lena.jpg]\n" << endl;
}

// enable/disable use of mixed API in the code below.
#define DEMO_MIXED_API_USE 1

int main( int argc, char** argv )
{
    help();
    const char* imagename = argc > 1 ? argv[1] : "lena.jpg";
#if DEMO_MIXED_API_USE
    Ptr<IplImage> iplimg = cvLoadImage(imagename); // Ptr<T> is safe ref-conting pointer class
    if(iplimg.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename);
        return -1;
    }
    Mat img(iplimg); // cv::Mat replaces the CvMat and IplImage, but it's easy to convert
    // between the old and the new data structures (by default, only the header
    // is converted, while the data is shared)
#else
    Mat img = imread(imagename); // the newer cvLoadImage alternative, MATLAB-style function
    if(img.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename);
        return -1;
    }
#endif

    if( !img.data ) // check if the image has been loaded properly
        return -1;

    Mat img_yuv;
    cvtColor(img, img_yuv, CV_BGR2YCrCb); // convert image to YUV color space. The output image will be created automatically

    vector<Mat> planes; // Vector is template vector class, similar to STL's vector. It can store matrices too.
    split(img_yuv, planes); // split the image into separate color planes

#if 1
    // method 1. process Y plane using an iterator
    MatIterator_<uchar> it = planes[0].begin<uchar>(), it_end = planes[0].end<uchar>();
    for(; it != it_end; ++it)
    {
        double v = *it*1.7 + rand()%21-10;
        *it = saturate_cast<uchar>(v*v/255.);
    }

    // method 2. process the first chroma plane using pre-stored row pointer.
    // method 3. process the second chroma plane using individual element access
    for( int y = 0; y < img_yuv.rows; y++ )
    {
        uchar* Uptr = planes[1].ptr<uchar>(y);
        for( int x = 0; x < img_yuv.cols; x++ )
        {
            Uptr[x] = saturate_cast<uchar>((Uptr[x]-128)/2 + 128);
            uchar& Vxy = planes[2].at<uchar>(y, x);
            Vxy = saturate_cast<uchar>((Vxy-128)/2 + 128);
        }
    }

#else
    Mat noise(img.size(), CV_8U); // another Mat constructor; allocates a matrix of the specified size and type
    randn(noise, Scalar::all(128), Scalar::all(20)); // fills the matrix with normally distributed random values;
                                                     // there is also randu() for uniformly distributed random number generation
    GaussianBlur(noise, noise, Size(3, 3), 0.5, 0.5); // blur the noise a bit, kernel size is 3x3 and both sigma's are set to 0.5

    const double brightness_gain = 0;
    const double contrast_gain = 1.7;
#if DEMO_MIXED_API_USE
    // it's easy to pass the new matrices to the functions that only work with IplImage or CvMat:
    // step 1) - convert the headers, data will not be copied
    IplImage cv_planes_0 = planes[0], cv_noise = noise;
    // step 2) call the function; do not forget unary "&" to form pointers
    cvAddWeighted(&cv_planes_0, contrast_gain, &cv_noise, 1, -128 + brightness_gain, &cv_planes_0);
#else
    addWeighted(planes[0], contrast_gain, noise, 1, -128 + brightness_gain, planes[0]);
#endif
    const double color_scale = 0.5;
    // Mat::convertTo() replaces cvConvertScale. One must explicitly specify the output matrix type (we keep it intact - planes[1].type())
    planes[1].convertTo(planes[1], planes[1].type(), color_scale, 128*(1-color_scale));
    // alternative form of cv::convertScale if we know the datatype at compile time ("uchar" here).
    // This expression will not create any temporary arrays and should be almost as fast as the above variant
    planes[2] = Mat_<uchar>(planes[2]*color_scale + 128*(1-color_scale));

    // Mat::mul replaces cvMul(). Again, no temporary arrays are created in case of simple expressions.
    planes[0] = planes[0].mul(planes[0], 1./255);
#endif

    // now merge the results back
    merge(planes, img_yuv);
    // and produce the output RGB image
    cvtColor(img_yuv, img, CV_YCrCb2BGR);

    // this is counterpart for cvNamedWindow
    namedWindow("image with grain", CV_WINDOW_AUTOSIZE);
#if DEMO_MIXED_API_USE
    // this is to demonstrate that img and iplimg really share the data - the result of the above
    // processing is stored in img and thus in iplimg too.
    cvShowImage("image with grain", iplimg);
#else
    imshow("image with grain", img);
#endif
    waitKey();

    return 0;
    // all the memory will automatically be released by Vector<>, Mat and Ptr<> destructors.
}
