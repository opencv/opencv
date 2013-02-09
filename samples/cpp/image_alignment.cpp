/*
 * This sample demonstrates the use of the function
 * findTransformECC that implements the image alignment ECC algorithm
 *
 *  The ground transformation for the image pair <cameramanTemplate.png, 
 *  cameramanImage.png> is homography. However, an affine transformation 
 *  sufficiently compensates for the deformation. Hence, both motion types 
 *  can be tested with the above image pair. 
 *
 * Authors: G. Evangelidis, M. Asbach
 */
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>


#include <stdio.h>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


#define HOMO_VECTOR(H, x, y)\
H.at<float>(0,0) = (float)(x);\
H.at<float>(1,0) = (float)(y);\
H.at<float>(2,0) = 1.;

#define GET_HOMO_VALUES(X, x, y)\
(x) = static_cast<float> (X.at<float>(0,0)/X.at<float>(2,0));\
(y) = static_cast<float> (X.at<float>(1,0)/X.at<float>(2,0));


static void help(void)
{
	  
	  cerr << "Usage: ecc -i image.pgm -t template.pgm -o warp.ecc [options]" << endl
         << endl
         << "  number of iteration:              -n N                                           (default:     50)" << endl
         << "  termination epsilon:              -eps X.XX                                      (default:  0.001)" << endl
         << "  verbose output flag:              -v [0|1]                                       (default:      0)" << endl
         << "  geometric transformation:         -m [translation|euclidean|affine|homograhpy]   (default: affine)" << endl
         << "  output (warped) image:            -oim <filename>" << endl
         << "  warp initialization file:         -init <filename>" << endl
         << endl
         << "  Example: ecc -i image.pgm -t template.pgm -o finalWarp.txt -m homography -n 75 -v 1 -init initWarp.txt -oim warped.pgm"  << endl << flush;
}

int readWarp(string iFilename, Mat& warp, int motionType){

	CV_Assert(warp.type()==CV_32FC1);
	float* matPtr = warp.ptr<float>(0);
	int numOfElements;
	if (motionType==MOTION_HOMOGRAPHY)
		numOfElements=9;
	else
		numOfElements=6;

	int i;
	int ret_value;

	ifstream myfile(iFilename.c_str());
	if (myfile.is_open()){
		for(i=0; i<numOfElements; i++){
			myfile >> matPtr[i];
		}
		ret_value = 1;
	}
	else {
		cout << "Unable to open file '" << iFilename.c_str() << "'!" << endl;
		ret_value = 0;
	}
	return ret_value;
}

int saveWarp(string fileName, const Mat& warp, int motionType)
{

	CV_Assert(warp.type()==CV_32FC1);

	const float* matPtr = warp.ptr<float>(0);
	int ret_value;

	ofstream outfile(fileName.c_str());
	if( !outfile ) {
		cerr << "error in saving "
			<< "Couldn't open file '" << fileName.c_str() << "'!" << endl;
		ret_value = 0;
	}
	else {//save the warp's elements
		outfile << matPtr[0] << " " << matPtr[1] << " " << matPtr[2] << endl;
		outfile << matPtr[3] << " " << matPtr[4] << " " << matPtr[5] << endl;
		if (motionType==MOTION_HOMOGRAPHY){
			outfile << matPtr[6] << " " << matPtr[7] << " " << matPtr[8] << endl;
		}
		ret_value = 1;
	}
	return ret_value;

}


static void draw_warped_roi(Mat& image, const int width, const int height, Mat& W)
{
	Point2f top_left, top_right, bottom_left, bottom_right;
  
	Mat  H = Mat (3, 1, CV_32F);
	Mat  U = Mat (3, 1, CV_32F);
  
	Mat warp_mat = Mat::eye (3, 3, CV_32F);
  
  for (int y = 0; y < W.rows; y++)
    for (int x = 0; x < W.cols; x++)
		warp_mat.at<float>(y,x) = W.at<float>(y,x);
  
	//warp the corners of rectangle
  
	// top-left
	HOMO_VECTOR(H, 1, 1);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_left.x, top_left.y);
  
	// top-right
	HOMO_VECTOR(H, width, 1);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, top_right.x, top_right.y);
  
	// bottom-left
	HOMO_VECTOR(H, 1, height);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_left.x, bottom_left.y);
  
	// bottom-right
	HOMO_VECTOR(H, width, height);
	gemm(warp_mat, H, 1, 0, 0, U);
	GET_HOMO_VALUES(U, bottom_right.x, bottom_right.y);
  
	// draw the warped perimeter
	line(image, top_left, top_right, Scalar(255,0,255));
	line(image, top_right, bottom_right, Scalar(255,0,255));
	line(image, bottom_right, bottom_left, Scalar(255,0,255));
	line(image, bottom_left, top_left, Scalar(255,0,255));
  

	
}



int main (const int argc, const char * argv[])
{
  // algorithm parameters and commandline arguments
  int          number_of_iterations = 50;
  double       termination_eps      = 0.001;
  const char * motion               = "affine";
  const char * warp_init            = NULL;
  const char * warp_to_file         = NULL;
  const char * image_to_file        = "warped.png";
  bool         verbose              = false;
  Mat          target_image         ;
  Mat          template_image       ;
  Mat          warp_matrix          ;
  
  // print help message
  if (argc < 7) 
  {

    help();
    return 1;
  }
  
  // handle command line arguments
  for (int arg = 1; arg < argc; arg++)
  {
    if (! strcmp(argv[arg], "-i"))
    {
      target_image = imread(argv[++arg], 0);//
      if (target_image.empty())
      {
        cerr << "Cannot load image." << endl << flush;
        return -1;
      }
    }
    if (! strcmp(argv[arg], "-t"))
    {
      template_image = imread(argv[++arg], 0);
      if (template_image.empty())
      {
        cerr << "Invalid image filenames (images cannot be loaded)..." << endl << flush;
        return -1;
      }
    }
    if (! strcmp(argv[arg],"-o"))
      warp_to_file=(argv[++arg]);
    if (! strcmp(argv[arg],"-n"))
      number_of_iterations=atoi(argv[++arg]);
    if (! strcmp(argv[arg],"-eps"))
      termination_eps=atof(argv[++arg]);
    if (! strcmp(argv[arg],"-m"))
    {
      motion=(argv[++arg]);
      if ((strcmp(motion,"affine")) && (strcmp(motion,"homography")) && (strcmp(motion,"translation")) && (strcmp(motion,"euclidean")))
      {
        cerr << "Invalid transformation name." << endl << flush;
        return -1;
      }
    }
    if (! strcmp(argv[arg],"-v"))
      verbose = (atoi(argv[++arg])!=0);
    if (! strcmp(argv[arg],"-init"))
      warp_init=(argv[++arg]);
    if (! strcmp(argv[arg],"-oim"))
      image_to_file=(argv[++arg]);
  }
  if (! warp_to_file)
  {
    cerr << "Output filename (-o [filename]) is missing." << endl << flush;
   
	return -1;
  }

    //enable the transformation mode
  int mode_temp;
  if (!strcmp(motion,"translation"))
  	  mode_temp = MOTION_TRANSLATION;
  else if (!strcmp(motion,"euclidean"))
  	  mode_temp = MOTION_EUCLIDEAN;
  else if (!strcmp(motion,"affine"))
	  mode_temp = MOTION_AFFINE;
  else 
	  mode_temp = MOTION_HOMOGRAPHY;

  const int warp_mode = mode_temp;

  // initialize the warp matrix or load it appropriately
  if (warp_mode == MOTION_HOMOGRAPHY)
	  warp_matrix = Mat::eye(3, 3, CV_32F); //identity warp
  else
	  warp_matrix = (Mat_<float>(2,3, CV_32F) << 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
  
  if (warp_init){
	  int readflag = readWarp(warp_init, warp_matrix, warp_mode);
	  if ((!readflag) || warp_matrix.empty()) 
		     {
	      cerr << "-> Check warp initialization file" << endl << flush;
		  return -1;
    }
  }
  
  if (! warp_init)
  {

	  printf("\n ->Perfomarnce Warning: If the deformation is strong, the identity warp may not "
		  "be a good initialization.\n\n");

	  if ((target_image.cols != template_image.cols) || 
		  (target_image.rows !=template_image.rows))
	  {
		  printf("\n ->Perfomarnce Warning: Identity warp ideally assumes images of similar "
		  "size.\n\n");
	  }
  }
  

  if (warp_mode != MOTION_HOMOGRAPHY)
      warp_matrix.rows = 2;
  
  // start timing
  const double tic_init = (double) getTickCount ();


  double cc = findTransformECC (template_image, target_image, warp_matrix, warp_mode, 
							   TermCriteria (TermCriteria::COUNT+TermCriteria::EPS, 
                                          number_of_iterations, termination_eps));

  if (cc == -1)
  {
    cerr << "The execution was interrupted. The correlation value is going to be minimized." << endl;
    cerr << "Check the warp initialization and/or the size of images." << endl << flush;
  }
  
  // end timing
  const double toc_final  = (double) getTickCount ();
  const double total_time = (toc_final-tic_init)/(getTickFrequency());
  if (verbose){
	  cout << "Alignment time (" << motion << " transformation): " 
		  << total_time << " sec" << endl << flush; 
	//  cout << "Final correlation: " << cc << endl << flush;

  }
    
  // save the final warp matrix
  saveWarp(warp_to_file, warp_matrix, warp_mode);


  if (verbose){
    cout << "The final warp has been saved in the file: " << warp_to_file << endl << flush;
  }
  
  // save the final warped image
  Mat warped_image = Mat(template_image.rows, template_image.cols, CV_32FC1);
  if (warp_mode != MOTION_HOMOGRAPHY)
    warpAffine      (target_image, warped_image, warp_matrix, warped_image.size(), 
					CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
  else
    warpPerspective (target_image, warped_image, warp_matrix, warped_image.size(), 
					CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS+CV_WARP_INVERSE_MAP);
  
  //save the warped image
  imwrite(image_to_file, warped_image);
  
  
  // display resulting images
  if (verbose)
  {

    cout << "The warped image has been saved in the file: " << image_to_file << endl << flush;
	
    namedWindow ("image",    CV_WINDOW_AUTOSIZE);
    namedWindow ("template", CV_WINDOW_AUTOSIZE);
    namedWindow ("warped image",   CV_WINDOW_AUTOSIZE);
	namedWindow ("error (black: no error)", CV_WINDOW_AUTOSIZE);
    
    moveWindow  ("template", 350, 350);
    moveWindow  ("warped image",   600, 300);
	moveWindow  ("error (black: no error)", 900, 300);
    
    // draw boundaries of corresponding regions
	Mat identity_matrix = Mat::eye(3,3,CV_32F);

    draw_warped_roi (target_image,   template_image.cols-2, template_image.rows-2, warp_matrix);
    draw_warped_roi (template_image, template_image.cols-2, template_image.rows-2, identity_matrix);
    
	
	Mat errorImage; 
	subtract(template_image, warped_image, errorImage);
	double max_of_error;
	minMaxLoc(errorImage, NULL, &max_of_error);

    // show images
    cout << "Press any key to exit the demo (you might need to click on the images before)." << endl << flush;
	
	imshow ("image",    target_image);
    waitKey (200);
    imshow ("template", template_image);
    waitKey (200);
    imshow ("warped image",   warped_image);
    waitKey(200);
	imshow ("error (black: no error)",  abs(errorImage)*255/max_of_error);
    waitKey(0);
    
  }
  
  // done
  return 0;
}

