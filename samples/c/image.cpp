#include "cv.h" // include standard OpenCV headers, same as before
#include "highgui.h"
#include "ml.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/flann/flann.hpp>

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;
using namespace cv::flann;

#define RAD2DEG 57.295779513082321
void print32FMat(const CvMat&mat)
{
    float *data = mat.data.fl;
    
    for (int r = 0; r < mat.rows; ++r)
    {
        for (int c = 0; c < mat.cols; ++c)
        {
            printf("%+1.9f\t", data[r*mat.cols+c]);
        }
        printf("\n");
    }
}

#if 0

// enable/disable use of mixed API in the code below.
#define DEMO_MIXED_API_USE 1

int main( int argc, char** argv )
{
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

#else

int main(int argc, char *argv[])
{
    /*double a = 56004.409155979447;
    double b = -15158.994132169822;
    double c = 215540.83745481662;
    
    {
        double A[4];
        double InvA[4];
        CvMat matA, matInvA;
        
        A[0] = a;
        A[1] = A[2] = b;
        A[3] = c;
        
        cvInitMatHeader( &matA, 2, 2, CV_64F, A );
        cvInitMatHeader( &matInvA, 2, 2, CV_64FC1, InvA );
        
        cvInvert( &matA, &matInvA, CV_SVD );
        
        printf("%g\t%g\n%g\t%g\n", InvA[0], InvA[1], InvA[2], InvA[3]);
    }*/
    
    //Mat img = imread("/Users/vp/work/ocv/opencv/samples/c/left04.jpg", 1);
    //Vec<string, 4> v;
    
    /*Mat img = Mat::zeros(20, 20, CV_8U);
	img(Range(0,10),Range(0,10)) = Scalar(255);
    img.at<uchar>(10,10)=255;
	img(Range(11,20),Range(11,20)) = Scalar(255);
	vector<Point2f> corner(1, Point2f(9.5,9.5));
	cornerSubPix(img, corner, Size(5,5), Size(-1,-1), TermCriteria(3, 30, 0.001));
	printf("Corner at (%g, %g)", corner[0].x, corner[0].y);*/
    
    /*Mat large, large2, gray;
    resize(img, large, img.size()*3, 0, 0, CV_INTER_LANCZOS4);
    cvtColor(large, gray, CV_BGR2GRAY);
    vector<Point2f> corners;
    bool found = findChessboardCorners(gray, Size(9,6), corners);
    cornerSubPix(gray, corners, Size(11,11), Size(-1,-1),
                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01));
    
    drawChessboardCorners(large, Size(9,6), Mat(corners), false);
    //rectangle(img, Point(150,100), Point(250,200), Scalar(0,255,0), 1);
    resize(large(Rect(150*3,100*3,100*3,100*3)), large2, Size(), 4, 4, CV_INTER_CUBIC);
    imshow("test", large2);
    waitKey(0);*/
    
    /*int N=4;
    CvEM em_model;
    CvEMParams params;
    int nsamples=500;
    for (int D=2; D < 15; D++)
    {
        CvMat* samples = cvCreateMat( nsamples, D, CV_32FC1 );
        for (int s=0; s < nsamples;s++)
            for (int d=0; d <D;d++)
                cvmSet(samples, s, d, (double)s+d);
        // initialize model's parameters
        params.covs = NULL;
        params.means = NULL;
        params.weights = NULL;
        params.probs = NULL;
        params.nclusters = N;
        params.cov_mat_type = CvEM::COV_MAT_DIAGONAL;
        params.start_step = CvEM::START_AUTO_STEP;
        params.term_crit.max_iter = 100;
        params.term_crit.epsilon = 0.01;
        params.term_crit.type = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
        em_model.train( samples, 0, params);
        const CvMat* w = em_model.get_weights();
        double sum=0;
        for (int i=0; i < N; i++)
            sum+=cvmGet(w, 0, i);
        printf("D = %d, sum = %f\n", D, sum);
        cvReleaseMat( &samples );
    }*/
    /*Mat a(1, 0, CV_32F);
    cout << " empty=" << a.empty() << " data=" << (size_t)a.data << endl;*/
    
    //XOR input
    /*double in[]={ 0 ,0,
        1, 0,
        0, 1,
        1, 1};
    double out[]={ 0,
        1,
        1,
        0};
    
    const int IVS = 2; // Input Vector Size
    const int OVS = 1; // Output Vector Size
    const int HN = 10; // Number of Hidden nodes
    const int NV= 4;   //Number of Training Vector
    
    int layer[] = { IVS, HN, OVS};
    
    CvMat *input =cvCreateMat( NV, IVS, CV_64FC1);
    CvMat *output =cvCreateMat( NV, OVS, CV_64FC1);
    CvMat *p_output =cvCreateMat( NV, OVS, CV_64FC1);
    CvMat *layersize =cvCreateMat( 1 , 3 , CV_32SC1);
    
    cvInitMatHeader(input, NV, IVS, CV_64FC1, in);
    cvInitMatHeader(output, NV, OVS, CV_64FC1, out);
    cvInitMatHeader(layersize, 1, 3, CV_32SC1, layer);
    
    CvANN_MLP train_model(layersize, CvANN_MLP::SIGMOID_SYM,1,1);
    std::cout<< " =========== =========== =========== =========== ==========="<<std::endl;
    std::cout<< "  * First Iteration with initialzation of weights"<<std::endl;
    std::cout<< " =========== =========== =========== =========== ==========="<<std::endl;
    int iter = train_model.train(  input,
                                 output, 
                                 NULL, 
                                 0,
                                 CvANN_MLP_TrainParams( cvTermCriteria ( CV_TERMCRIT_ITER |
                                                                        CV_TERMCRIT_EPS,
                                                                        5000,0.000001),
                                                       CvANN_MLP_TrainParams::BACKPROP,
                                                       0.1,0.1),
                                 0
                                 //+ CvANN_MLP::NO_OUTPUT_SCALE
                                 );
    
    std::cout << " * iteration :"<<iter<<std::endl;
    train_model.predict( input, p_output );
    for(int i=0; i<NV;i++){
        std::cout<< CV_MAT_ELEM(*input,double,i,0) << " ," << CV_MAT_ELEM(*input,double,i,1)
        << " : " << CV_MAT_ELEM(*p_output,double,i,0) <<std::endl;
        
    }
    train_model.save( "firstModel.xml");
    std::cout<< " =========== =========== =========== =========== ==========="<<std::endl;
    std::cout<< "  * Second Iteration without initialzation of weights"<<std::endl;
    std::cout<< " =========== =========== =========== =========== ==========="<<std::endl;
    
    int iter2;
    for(int i=0;i<5;i++)
    {
        iter2 = train_model.train(  input,
                                  output, 
                                  NULL, 
                                  0,
                                  CvANN_MLP_TrainParams( cvTermCriteria ( CV_TERMCRIT_ITER |
                                                                         CV_TERMCRIT_EPS,
                                                                         5000,0.0000001),
                                                        CvANN_MLP_TrainParams::BACKPROP,
                                                        0.1,0.1),
                                  0
                                  +CvANN_MLP::UPDATE_WEIGHTS
                                  //+ CvANN_MLP::NO_OUTPUT_SCALE
                                  );
    }
    std::cout << " * iteration :"<<iter2<<std::endl;
    train_model.save( "secondModel.xml");
    
    train_model.predict( input, p_output );
    for(int i=0; i<NV;i++){
        std::cout<< CV_MAT_ELEM(*input,double,i,0) << " ," << CV_MAT_ELEM(*input,double,i,1)
        << " : " << CV_MAT_ELEM(*p_output,double,i,0) <<std::endl;
        
    }*/
    
    /*cv::Size imageSize;
    int Nimg, Npts;
    vector<vector<cv::Point3f> > objectPoints;
    vector<vector<cv::Point2f> >imagePoints;
    cv::FileStorage f("/Users/vp/Downloads/calib_debug.2.yml",cv::FileStorage::READ);
    cv::FileNodeIterator it = f["img_sz"].begin(); it >> imageSize.width >> imageSize.height;
    Nimg = (int) f ["NofImages"];
    Npts = (int) f["NofPoints"];
    for (int i=0; i<Nimg;i++) {
        std::stringstream imagename; imagename << "image" << i;
        cv::FileNode img = f[imagename.str()];
        vector <cv::Point3f> ov;
        vector <cv::Point2f> iv;
        for (int j=0; j<Npts; j++) {
            std::stringstream nodename; nodename << "node" << j;
            cv::FileNode pnt = img[nodename.str()];
            cv::Point3f op;
            cv::Point2f ip;
            cv::FileNodeIterator ot = pnt["objPnt"].begin(); ot >> op.x >> op.y >> op.z;
            cv::FileNodeIterator it = pnt["imgPnt"].begin(); it >> ip.x >> ip.y;
            iv.push_back(ip);
            ov.push_back(op);
        }
        imagePoints.push_back(iv);
        objectPoints.push_back(ov);
    }
    cv::Mat M,D;
    vector<cv::Mat> R,T;
    cv::calibrateCamera(objectPoints, imagePoints, imageSize, M, D,R,T,
                        CV_CALIB_FIX_ASPECT_RATIO + 1*CV_CALIB_FIX_K3 + 1*CV_CALIB_ZERO_TANGENT_DIST);
    cv::FileStorage fo("calib_output.yml",cv::FileStorage::WRITE);
    //fo << "M" << M;
    cout << "M: " << M;*/
    
    /*Mat img = imread("/Users/vp/Downloads/test5.tif", CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_COLOR), img16;
    cout << "img.depth()=" << img.depth() << endl;
    if(img.depth() == CV_16U)
        img16 = img;
    else
        img.convertTo(img16, CV_16U, 256, 0);
    imshow("test", img16);
    imwrite("/Users/vp/tmp/test16_2.png", img16);
    waitKey();*/
    
    /*Mat img(600,800,CV_8UC3);
    img = Scalar::all(0);
    IplImage _img = img;
    
    CvFont font = cvFont(2,1);
    cvPutText(&_img, "Test", cvPoint(100, 100), &font, cvScalar(0,0,255));
    
    imshow("test", img);
    waitKey();*/
    
    /*IplImage* img = cvCreateImage(cvSize(800,600), 8, 3);
    cvZero(img);
    CvFont font = cvFont(2,1);
    cvPutText(img, "Test", cvPoint(100, 100), &font, cvScalar(0,0,255)); 
    cvNamedWindow("test", 1);
    cvShowImage("test", img);
    cvWaitKey(0);*/
    /*int sz[] = {1, 5, 5};
    CvMatND* src = cvCreateMatND(3, sz, CV_64F);
    CvMatND* dst = cvCreateMatND(3, sz, CV_64F);
    CvRNG rng = cvRNG(-1);
    cvRandArr(&rng, src, CV_RAND_UNI, cvScalarAll(-100), cvScalarAll(100));
    cvAddS(src, cvScalar(100), dst, 0);
    cvSave("_input.xml", src);
    cvSave("_output.xml", dst);*/
    
    /*
    /// random data generation :
    Mat data(100,10,CV_32FC1);
    randn(data, 0.0, 1.0);
    /// Creating the ANN engine
    AutotunedIndexParams autoParams(0.9,0.5,0.2,1);
    Index index(data,autoParams);
    /// Creating a query
    SearchParams searchParams(5);
    vector<float> query, dist;
    vector<int> foundIndice;
    foundIndice.push_back(0);
    dist.push_back(0);
    for(int i = 0 ; i < 10 ; i++)
    {
        query.push_back(data.at<float>(2,i));
    }
    /// Do a reaserch : result must be equal to 2.
    index.knnSearch(query, foundIndice, dist, 1, searchParams);
    cout << "Found indice (must be 2) : " << foundIndice[0] << endl;
    /// save params
    index.save(string("test"));
    */
    
    /*namedWindow("orig", CV_WINDOW_AUTOSIZE);
	namedWindow("canny", CV_WINDOW_AUTOSIZE);
	namedWindow("hough", CV_WINDOW_AUTOSIZE);
    
	Mat orig = cv::imread("/Users/vp/Downloads/1.jpg", 0);
	//equalizeHist(orig, orig);
    Mat hough;
	cvtColor(orig, hough, CV_GRAY2BGR);
    
	Mat canny;
	Canny(orig, canny, 100, 50); // reproduce Canny-Filtering as in Hough-Circles
    
	int bestRad = 20;
	int minRad = bestRad / 1.3;
	int maxRad = bestRad * 1.3;
    
	vector<Vec3f> circles;  // detect circles
	HoughCircles(orig, circles, CV_HOUGH_GRADIENT,
                 1,   // accu-scaling
                 20,  // minDist
                 100, // CannyParam
                 10,  // minAccuCount
                 minRad,
                 maxRad);
	// Draw Circles into image in gray
	for( size_t i = 0; i < circles.size(); i++ )
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		circle( hough, center, 3, Scalar(0,0,255), -1, 8, 0 );
		// draw the circle outline
		circle( hough, center, radius, Scalar(0,255,255), 1, 8, 0 );
	}
    
	// Draw reference circles
	Point c(bestRad * 3, bestRad * 3);
	circle(hough, c, bestRad, 255);
	circle(hough, c, minRad, 255);
	circle(hough, c, maxRad, 255);
    
    
    
	cv::imshow("orig", orig);
	cv::imshow("canny", canny);
	cv::imshow("hough", hough);
    
	cv::waitKey();*/
    
    /*int npoints = 4;
    CvMat *OP = cvCreateMat(1, npoints, CV_32FC3);
    CvPoint3D32f *op = (CvPoint3D32f *)OP->data.fl;
    
    CvMat *OP2 = cvCreateMat(1, npoints, CV_32FC3);
    CvPoint3D32f *op2 = (CvPoint3D32f *)OP2->data.fl;
    
    CvMat *IP = cvCreateMat(1, npoints, CV_32FC2);
    CvPoint2D32f *ip = (CvPoint2D32f *)IP->data.fl;
    
    CvMat *IP2 = cvCreateMat(1, npoints, CV_32FC2);
    CvPoint2D32f *ip2 = (CvPoint2D32f *)IP2->data.fl;
    
    CvMat *IP0 = cvCreateMat(1, npoints, CV_32FC2);
    
    float rv[3], rv2[3];
    float rotMat[9];
    float t[3], t2[3];
    float tRotMat[16];
    
    double kMat[9];
    
    CvMat K = cvMat(3, 3, CV_64F, kMat);
    CvMat T = cvMat(3, 1, CV_32F, t);
    CvMat RV = cvMat(3, 1, CV_32F, rv);
    CvMat T2 = cvMat(3, 1, CV_32F, t2);
    CvMat RV2 = cvMat(3, 1, CV_32F, rv2);
    CvMat R = cvMat(3, 3, CV_32F, rotMat);
    
    float r0, r1, r2;
    
    kMat[0] = 659.88;
    kMat[1] = 0.00; 
    kMat[2] = 320.40; 
    kMat[3] = 0.00; 
    kMat[4] = 657.53; 
    kMat[5] = 240.98; 
    kMat[6] = 0.00; 
    kMat[7] = 0.00;
    kMat[8] = 1.00;
    
    ip[0].x = 277.56; ip[0].y = 184.03; 
    ip[1].x = 329.00; ip[1].y = 199.04; 
    ip[2].x = 405.96; ip[2].y = 205.96; 
    ip[3].x = 364.00; ip[3].y = 187.97;
    
    op[0].x = -189.00; op[0].y = 171.00; 
    op[1].x = -280.00; op[1].y = 265.00; 
    op[2].x = -436.00; op[2].y = 316.00; 
    op[3].x = -376.00; op[3].y = 209.00;
    
    ip2[0].x = 277.56; ip2[0].y = 184.03; 
    ip2[1].x = 328.00; ip2[1].y = 199.11; 
    ip2[2].x = 405.89; ip2[2].y = 206.89; 
    ip2[3].x = 366.00; ip2[3].y = 187.93;
    
    op2[0].x = -194.00; op2[0].y = 168.00; 
    op2[1].x = -281.00; op2[1].y = 267.00; 
    op2[2].x = -433.00; op2[2].y = 321.00; 
    op2[3].x = -372.00; op2[3].y = 208.00;
    
    //ip[4].x = 405.89; ip[4].y = 206.89; 
    //op[4].x = -433.00; op[4].y = 321.00; 
    //ip2[4].x = 364.00; ip2[4].y = 187.97;
    //op2[4].x = -376.00; op2[4].y = 209.00;
    
    cvFindExtrinsicCameraParams2(OP, IP, &K, 
                                 NULL, //&D, 
                                 &RV, &T, 0);
    
    cvRodrigues2(&RV, &R, 0);
    
    printf("--first--\n");
    print32FMat(R);
    
    cvFindExtrinsicCameraParams2(OP2, IP2, &K, 
                                 NULL, //&D, 
                                 &RV2, &T2, 0);
    
    cvRodrigues2(&RV2, &R, 0);
    printf("---second---\n");
    print32FMat(R);
    
    double err;
    cvProjectPoints2(OP, &RV, &T, &K, NULL, IP0);
    err = cvNorm(IP, IP0, CV_L2);
    printf("\n\nfirst avg reprojection error = %g\n", sqrt(err*err/npoints));
    
    cvProjectPoints2(OP2, &RV2, &T2, &K, NULL, IP0);
    err = cvNorm(IP2, IP0, CV_L2);
    printf("second avg reprojection error = %g\n", sqrt(err*err/npoints));
    
    cvProjectPoints2(OP, &RV2, &T2, &K, NULL, IP0);
    err = cvNorm(IP, IP0, CV_L2);
    printf("\n\nsecond->first cross reprojection error = %g\n", sqrt(err*err/npoints));
    
    cvProjectPoints2(OP2, &RV, &T, &K, NULL, IP0);
    err = cvNorm(IP2, IP0, CV_L2);
    printf("first->second cross reprojection error = %g\n", sqrt(err*err/npoints));
    */
    /*Mat img = imread("/Users/vp/work/ocv/opencv/samples/c/baboon.jpg", 1);
    vector<Point2f> corners;
    double t0 = 0, t;
    
    for( size_t i = 0; i < 50; i++ )
    {
        corners.clear();
        t = (double)getTickCount();
        goodFeaturesToTrack(img, corners, 1000, 0.01, 10);
        t = (double)getTickCount() - t;
        if( i == 0 || t0 > t )
            t0 = t;
    }
    printf("minimum running time = %gms\n", t0*1000./getTickFrequency());
    
    Mat imgc;
    cvtColor(img, imgc, CV_GRAY2BGR);
    
    for( size_t i = 0; i < corners.size(); i++ )
    {
        circle(imgc, corners[i], 3, Scalar(0,255,0), -1);
    }
    imshow("corners", imgc);*/
    /*Mat imgf, imgf2, img2;
    img.convertTo(imgf, CV_64F, 1./255);
    resize(imgf, imgf2, Size(), 0.7, 0.7, CV_INTER_LANCZOS4);
    imgf2.convertTo(img2, CV_8U, 255);
    imshow("test", img2);
    
    waitKey();*/
    
    /*Mat src = imread("/Users/vp/work/ocv/opencv/samples/c/fruits.jpg", 1);
    //if( argc != 2 || !(src=imread(argv[1], 1)).data )
    //    return -1;
    
    Mat hsv;
    cvtColor(src, hsv, CV_BGR2HSV);
    
    // let's quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};
    
    calcHist( &hsv, 1, channels, Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
    double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    
    int scale = 10;
    Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
    
    for( int h = 0; h < hbins; h++ )
        for( int s = 0; s < sbins; s++ )
        {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal*255/maxVal);
            rectangle( histImg, Point(h*scale, s*scale),
                        Point( (h+1)*scale - 1, (s+1)*scale - 1),
                        Scalar::all(intensity),
                        CV_FILLED );
        }
    
    namedWindow( "Source", 1 );
    imshow( "Source", src );
    
    namedWindow( "H-S Histogram", 1 );
    imshow( "H-S Histogram", histImg );
    waitKey();*/
    
    /*Mat_<double> a(3, 3);
    a << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    Mat_<double> b(3, 1);
    b << 1, 2, 3;
    Mat c;
    hconcat(a, b, c);
    cout << c;*/
    
    /*Mat img = imread("/Users/vp/work/ocv/opencv/samples/c/lena.jpg", 1), img2;
    cv::imshow( "Original Image D", img );
    
    if( img.channels()==3 )
    {
        Mat L,u,v;
        Mat luv;
        vector<Mat> splitted;
        Mat img_float0, img_float;
        
        img.convertTo( img_float0, CV_32F, 1./255, 0 );
        cvtColor( img_float0, luv, CV_BGR2Luv);
        
        cv::split( luv, splitted );
        
        L = (Mat)splitted[0];
        u = (Mat)splitted[1];
        v = (Mat)splitted[2];	
        
        vector<Mat> res;
        
        res.push_back( L );
        res.push_back( u );
        res.push_back( v );
        
        cv::merge( res, luv );
        
        cvtColor( luv, img_float, CV_Luv2BGR );
        
        printf("diff = %g\n", cv::norm(img_float0, img_float, CV_C));
        
        img_float.convertTo( img2, CV_8U, 255. );
    }
    
    cv::imshow( "After Darken", img2 );
    cv::absdiff(img, img2, img2);
    img2 *= 255;
    cv::imshow("Magnified difference", img2);
    
    waitKey();*/
    
    /*const char* imgFilename = "/Users/vp/Downloads/tsukuba.png";
    
    Mat bgr = imread( imgFilename );
    Mat gray = imread( imgFilename, 0 ), gray_;
    cvtColor( bgr, gray_, CV_BGR2GRAY );
    
    int N = countNonZero( gray != gray_ );
    printf( "Count non zero = %d / %d\n", N, gray.cols * gray.rows );
    
    Mat diff = abs( gray-gray_ );
    double maxVal = 0;
    minMaxLoc( diff, 0, &maxVal, 0, 0);
    printf( "Max abs diff = %f\n", maxVal);*/
    /*Mat img = imread("/Users/vp/Downloads/r_forearm_cam_rect_crop.png", 1);
    vector<Point2f> corners;
    Mat big;
    resize(img, big, Size(), 1, 1);
    bool found = findChessboardCorners(big, Size(5,4), corners);
    drawChessboardCorners(big, Size(5,4), Mat(corners), found);
    imshow("test", big);
    waitKey();*/
    
    /*float x[] = {0, 1};
    float y[] = {0, 1};
    CvMat mx = cvMat(2, 1, CV_32F, x);
    CvMat my = cvMat(2, 1, CV_32F, y);
    CvNormalBayesClassifier b;
    bool s = b.train(&mx, &my, 0, 0, false);*/
    
    /*float responseData[] = {1, 1, 1, 0, 0, 0};
    float intdata[] = { 1, 0, 0, 1,
        1, 0, 1, 0,
        
        1, 1, 0, 0,
        
        0, 0, 0, 1,
        
        0, 0, 1, 0,
        
        0, 1, 0, 0};
    
    CvMat data = cvMat(6, 4, CV_32FC1, intdata);
    
    CvMat responses = cvMat(6, 1, CV_32FC1, responseData);
    
    CvNormalBayesClassifier bc;
    
    bool succ = bc.train(&data, &responses, 0, 0, false);
    float testData[] = {1.0, 1, 0, 0};
    float dummy[] = {0};
    CvMat test = cvMat(1, 4, CV_32FC1, testData);
    
    CvMat testResults = cvMat(1, 6, CV_32FC1, 0);
    
    float whatsthis = bc.predict(&test, &testResults);*/
    
    int sz[] = {10, 20, 30};
    Mat m(3, sz, CV_32F);
    randu(m, Scalar::all(-10), Scalar::all(10));
    double maxVal0, maxVal = -FLT_MAX;
    minMaxIdx(m, 0, &maxVal0, 0, 0);
    
    MatConstIterator_<float> it = m.begin<float>(), it_end = m.end<float>();
    
    for( ; it != it_end; ++it )
    {
        if( maxVal < *it )
            maxVal = *it;
    }
    
    printf("maxval(minmaxloc) = %g, maxval(iterator) = %g\n", maxVal0, maxVal);
    return 0;
}

#endif
