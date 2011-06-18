//Yannick Verdie 2010

//--- Please read help() below: ---

#include <iostream>
#include <vector>

#include <opencv/highgui.h>
#include <GL/gl.h>
#include <opencv/cxcore.h>
#include <opencv/cv.h>
using namespace std;
using namespace cv;
void help()
{
	cout << "\nThis demo demonstrates the use of the Qt enhanced version of the highgui GUI interface\n"
			"  and dang if it doesn't throw in the use of of the POSIT 3D tracking algorithm too\n"
			"It works off of the video: cube4.avi\n"
			"Using OpenCV version %s\n" << CV_VERSION << "\n\n"
" 1). This demo is mainly based on work from Javier Barandiaran Martirena\n"
"     See this page http://opencv.willowgarage.com/wiki/Posit.\n"
" 2). This is a demo to illustrate how to use **OpenGL Callback**.\n"
" 3). You need Qt binding to compile this sample with OpenGL support enabled.\n"
" 4). The features' detection is very basic and could highly be improved \n"
"     (basic thresholding tuned for the specific video) but 2).\n"
" 5) THANKS TO Google Summer of Code 2010 for supporting this work!\n" << endl;
}

#define FOCAL_LENGTH 600
#define CUBE_SIZE 10

void renderCube(float size)
{
	glBegin(GL_QUADS);
	// Front Face
	glNormal3f( 0.0f, 0.0f, 1.0f);
	glVertex3f( 0.0f,  0.0f,  0.0f);
	glVertex3f( size,  0.0f,  0.0f);
	glVertex3f( size,  size,  0.0f);
	glVertex3f( 0.0f,  size,  0.0f);
	// Back Face
	glNormal3f( 0.0f, 0.0f,-1.0f);
	glVertex3f( 0.0f,  0.0f, size);
	glVertex3f( 0.0f,  size, size);
	glVertex3f( size,  size, size);
	glVertex3f( size,  0.0f, size);		
	// Top Face
	glNormal3f( 0.0f, 1.0f, 0.0f);
	glVertex3f( 0.0f,  size,  0.0f);
	glVertex3f( size,  size,  0.0f);
	glVertex3f( size,  size, size);
	glVertex3f( 0.0f,  size, size);
	// Bottom Face
	glNormal3f( 0.0f,-1.0f, 0.0f);
	glVertex3f( 0.0f,  0.0f,  0.0f);
	glVertex3f( 0.0f,  0.0f, size);
	glVertex3f( size,  0.0f, size);
	glVertex3f( size,  0.0f,  0.0f);
	// Right face
	glNormal3f( 1.0f, 0.0f, 0.0f);
	glVertex3f( size,  0.0f, 0.0f);
	glVertex3f( size,  0.0f, size);
	glVertex3f( size,  size, size);
	glVertex3f( size,  size, 0.0f);
	// Left Face
	glNormal3f(-1.0f, 0.0f, 0.0f);
	glVertex3f( 0.0f,  0.0f, 0.0f);
	glVertex3f( 0.0f,  size, 0.0f);
	glVertex3f( 0.0f,  size, size);
	glVertex3f( 0.0f,  0.0f, size);
	glEnd();
}


void on_opengl(void* param)
{
	//Draw the object with the estimated pose
	glLoadIdentity();
	glScalef( 1.0f, 1.0f, -1.0f);
	glMultMatrixf( (float*)param );
	glEnable( GL_LIGHTING );
	glEnable( GL_LIGHT0 );
	glEnable( GL_BLEND );
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	renderCube( CUBE_SIZE );
	glDisable(GL_BLEND);
	glDisable( GL_LIGHTING );
}

void initPOSIT(std::vector<CvPoint3D32f> *modelPoints)
{
	//Create the model pointss
	modelPoints->push_back(cvPoint3D32f(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0)
	modelPoints->push_back(cvPoint3D32f(0.0f, 0.0f, CUBE_SIZE));
	modelPoints->push_back(cvPoint3D32f(CUBE_SIZE, 0.0f, 0.0f));
	modelPoints->push_back(cvPoint3D32f(0.0f, CUBE_SIZE, 0.0f));
}

void foundCorners(vector<CvPoint2D32f> *srcImagePoints,IplImage* source, IplImage* grayImage)
{
	cvCvtColor(source,grayImage,CV_RGB2GRAY);
	cvSmooth( grayImage, grayImage,CV_GAUSSIAN,11);
	cvNormalize(grayImage, grayImage, 0, 255, CV_MINMAX);
	cvThreshold( grayImage, grayImage, 26, 255, CV_THRESH_BINARY_INV);//25

	Mat MgrayImage = grayImage;
	//For debug
	//MgrayImage = MgrayImage.clone();//deep copy
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(MgrayImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Point p;
	vector<CvPoint2D32f> srcImagePoints_temp(4,cvPoint2D32f(0,0));

	if (contours.size() == srcImagePoints_temp.size())
	{

		for(int i = 0 ; i<contours.size(); i++ )
		{

			p.x = p.y = 0;

			for(int j = 0 ; j<contours[i].size(); j++ )
				p+=contours[i][j];

			srcImagePoints_temp.at(i)=cvPoint2D32f(float(p.x)/contours[i].size(),float(p.y)/contours[i].size());
		}

		//Need to keep the same order
		//> y = 0
		//> x = 1
		//< x = 2
		//< y = 3

		//get point 0;
		int index = 0;
		for(int i = 1 ; i<srcImagePoints_temp.size(); i++ )
		{
			if (srcImagePoints_temp.at(i).y > srcImagePoints_temp.at(index).y)
				index = i;
		}
		srcImagePoints->at(0) = srcImagePoints_temp.at(index);

		//get point 1;
		index = 0;
		for(int i = 1 ; i<srcImagePoints_temp.size(); i++ )
		{
			if (srcImagePoints_temp.at(i).x > srcImagePoints_temp.at(index).x)
				index = i;
		}
		srcImagePoints->at(1) = srcImagePoints_temp.at(index);

		//get point 2;
		index = 0;
		for(int i = 1 ; i<srcImagePoints_temp.size(); i++ )
		{
			if (srcImagePoints_temp.at(i).x < srcImagePoints_temp.at(index).x)
				index = i;
		}
		srcImagePoints->at(2) = srcImagePoints_temp.at(index);

		//get point 3;
		index = 0;
		for(int i = 1 ; i<srcImagePoints_temp.size(); i++ )
		{
			if (srcImagePoints_temp.at(i).y < srcImagePoints_temp.at(index).y)
				index = i;
		}
		srcImagePoints->at(3) = srcImagePoints_temp.at(index);

		Mat Msource = source;
		stringstream ss;
		for(int i = 0 ; i<srcImagePoints_temp.size(); i++ )
		{
			ss<<i;
			circle(Msource,srcImagePoints->at(i),5,CV_RGB(255,0,0));
			putText( Msource, ss.str(), srcImagePoints->at(i),CV_FONT_HERSHEY_SIMPLEX,1,CV_RGB(255,0,0));
			ss.str("");

			//new coordinate system in the middle of the frame and reversed (camera coordinate system)
			srcImagePoints->at(i) = cvPoint2D32f(srcImagePoints_temp.at(i).x-source->width/2,source->height/2-srcImagePoints_temp.at(i).y);
		}
	}

}

void createOpenGLMatrixFrom(float *posePOSIT,const CvMatr32f &rotationMatrix, const CvVect32f &translationVector)
{


	//coordinate system returned is relative to the first 3D input point	
	for (int f=0; f<3; f++)
	{
		for (int c=0; c<3; c++)
		{
			posePOSIT[c*4+f] = rotationMatrix[f*3+c];	//transposed
		}
	}	
	posePOSIT[3] = 0.0;
	posePOSIT[7] = 0.0;	
	posePOSIT[11] = 0.0;
	posePOSIT[12] = translationVector[0];
	posePOSIT[13] = translationVector[1]; 
	posePOSIT[14] = translationVector[2];
	posePOSIT[15] = 1.0;	
}

int main(int argc, char *argv[])
{
	CvCapture* video = cvCaptureFromFile("cube4.avi");
	CV_Assert(video);

	IplImage* source = cvCreateImage(cvGetSize(cvQueryFrame(video)),8,3);
	IplImage* grayImage = cvCreateImage(cvGetSize(cvQueryFrame(video)),8,1);

	cvNamedWindow("original",CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cvNamedWindow("POSIT",CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	displayOverlay("POSIT", "We lost the 4 corners' detection quite often (the red circles disappear). This demo is only to illustrate how to use OpenGL callback.\n -- Press ESC to exit.", 10000);
	//For debug
	//cvNamedWindow("tempGray",CV_WINDOW_AUTOSIZE);
	float OpenGLMatrix[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	cvCreateOpenGLCallback("POSIT",on_opengl,OpenGLMatrix);

	vector<CvPoint3D32f> modelPoints;
	initPOSIT(&modelPoints);

	//Create the POSIT object with the model points
	CvPOSITObject* positObject = cvCreatePOSITObject( &modelPoints[0], (int)modelPoints.size() );

	CvMatr32f rotation_matrix = new float[9];
	CvVect32f translation_vector = new float[3];	
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 100, 1.0e-4f);

	vector<CvPoint2D32f> srcImagePoints(4,cvPoint2D32f(0,0));


	while(cvWaitKey(33) != 27)
	{
		source=cvQueryFrame(video);
		cvShowImage("original",source);

		foundCorners(&srcImagePoints,source,grayImage);
		cvPOSIT( positObject, &srcImagePoints[0], FOCAL_LENGTH, criteria, rotation_matrix, translation_vector );
		createOpenGLMatrixFrom(OpenGLMatrix,rotation_matrix,translation_vector);

		cvShowImage("POSIT",source);
		//For debug
		//cvShowImage("tempGray",grayImage);

		if (cvGetCaptureProperty(video,CV_CAP_PROP_POS_AVI_RATIO)>0.99)
			cvSetCaptureProperty(video,CV_CAP_PROP_POS_AVI_RATIO,0);
	}

	cvDestroyAllWindows();
	cvReleaseImage(&grayImage);
	cvReleaseCapture(&video);
	cvReleasePOSITObject(&positObject);

	return 0;
}
