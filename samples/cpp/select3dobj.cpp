#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

static void print_help()
{
    printf("Usage: select3dobj -w <board_width -h <board_height> [-s <square_size>]\n"
           "\t-i <intrinsics_filename> -o <output_prefix> [video_filename/cameraId]\n");
}

struct CameraData
{
	Size imageSize;
	Size boardSize;
	double squareSize;
	Mat distCoeffs;
	Mat cameraMatrix;
    vector<Point3f> objPoints;
};

Point mouseLoc;
int mouseEvent = -1;
int mouseButtonState = 0;

static void onMouse(int event, int x, int y, int flags, void*)
{
    mouseEvent = event;
    mouseLoc = Point(x,y);
    mouseButtonState = flags;
}

static bool readCameraMatrix(const string& filename, CameraData& calibrated)
{
    FileStorage fs(filename, FileStorage::READ);
    fs["image_width"] >> calibrated.imageSize.width;
    fs["image_height"] >> calibrated.imageSize.height;
    fs["board_width"] >> calibrated.boardSize.width;
    fs["board_height"] >> calibrated.boardSize.height;
    fs["square_size"] >> calibrated.squareSize;
    fs["distortion_coefficients"] >> calibrated.distCoeffs;
    if( calibrated.distCoeffs.type() != CV_64F )
        calibrated.distCoeffs = Mat_<double>(calibrated.distCoeffs);
    if( calibrated.cameraMatrix.type() != CV_64F )
        calibrated.cameraMatrix = Mat_<double>(calibrated.cameraMatrix);
    
    fs["camera_matrix"] >> calibrated.cameraMatrix;
    
    calibrated.objPoints.resize(0);
    
    for( int i = 0; i < calibrated.boardSize.height; i++ )
        for( int j = 0; j < calibrated.boardSize.width; j++ )
            calibrated.objPoints.push_back(
                Point3f(float(j*calibrated.squareSize),
                        float(i*calibrated.squareSize), 0));
    return true;
}

static Point3f image2plane(Point2f imgpt, const Mat& R, const Mat& tvec, const Mat& cameraMatrix, double Z)
{
    Mat R1 = R.clone();
    R1.col(2) = R1.col(2)*Z + tvec;
    Mat_<double> v = (cameraMatrix*R1).inv()*(Mat_<double>(3,1) << imgpt.x, imgpt.y, 1);
    double iw = fabs(v(2,0)) > DBL_EPSILON ? 1./v(2,0) : 0;
    return Point3f(v(0,0)*iw, v(1,0)*iw, Z);
}

int main(int argc, char** argv)
{
    const char* imgFilename = 0;//"frame.jpg";
    const float eps = 1e-3f;
    
    if(argc < 5)
    {
        print_help();
        return 0;
    }
    const char* intrinsicsFilename = 0;
    const char* outprefix = 0;
	const char* videoFilename = 0;
	int cameraId = 0;
	Size boardSize;
	double squareSize = 0;
	bool paused = false;
    vector<Point3f> objpts(4);
    vector<Point2f> imgpts(4);
    vector<Point2f> mousepts(4);
    int nobjpt = 0;
    
    for( int i = 1; i < argc; i++ )
    {
        if( strcmp(argv[i], "-i") == 0 )
			intrinsicsFilename = argv[++i];
		else if( strcmp(argv[i], "-o") == 0 )
			outprefix = argv[++i];
		else if( strcmp(argv[i], "-w") == 0 )
		{
			if(sscanf(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0)
			{
				printf("Incorrect -w parameter (must be a positive integer)\n");
				print_help();
				return 0;
			}
		}
		else if( strcmp(argv[i], "-h") == 0 )
		{
			if(sscanf(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0)
			{
				printf("Incorrect -h parameter (must be a positive integer)\n");
				print_help();
				return 0;
			}
		}
		else if( strcmp(argv[i], "-s") == 0 )
		{
			if(sscanf(argv[++i], "%lf", &squareSize) != 1 || squareSize <= 0)
			{
				printf("Incorrect -w parameter (must be a positive real number)\n");
				print_help();
				return 0;
			}
		}
		else if( argv[i][0] != '-' )
		{
			if( isdigit(argv[i][0]))
				sscanf(argv[i], "%d", &cameraId);
			else
				videoFilename = argv[i];
		}
		else
		{
			printf("Incorrect option\n");
			print_help();
			return 0;
		}
    }
    
	if( !intrinsicsFilename || !outprefix ||
		boardSize.width <= 0 || boardSize.height <= 0 ||
		squareSize <= 0 )
	{
		printf("One of required parameters are missing\n");
		print_help();
		return 0;
	}
	
	CameraData calibrated;
    readCameraMatrix(intrinsicsFilename, calibrated);
	calibrated.boardSize = boardSize;
	calibrated.squareSize = squareSize;
    
	VideoCapture cap;
    if( !imgFilename )
    {
        if( videoFilename )
            cap.open(videoFilename);
        else
            cap.open(0);
            
        if( !cap.isOpened() )
        {
            printf("Can not initialize video capture\n");
            print_help();
            return 0;
        }
    }
    
    const char* outbarename = 0;
    {
        outbarename = strrchr(outprefix, '/');
        const char* tmp = strrchr(outprefix, '\\');
        char cmd[1000];
        sprintf(cmd, "mkdir %s", outprefix);
        if( tmp && tmp > outbarename )
            outbarename = tmp;
        if( outbarename )
        {
            cmd[6 + outbarename - outprefix] = '\0';
            system(cmd);
            outbarename++;
        }
        else
            outbarename = outprefix;
    }
	
	Mat frame0, frame, shownFrame, selectedObjMask, selectedObjFrame, mapxy, R, rvec, tvec;
    vector<Point2f> boardCorners;
    vector<Point3f> tempobj(8);
    vector<Point2f> tempimg(8);
    vector<Point> temphull(8);
    
	namedWindow("Video", 1);
    namedWindow("Selected Object", 1);
    setMouseCallback("Video", onMouse, 0);
    bool boardFound = false;
    
    char path[1000];
    sprintf(path, "%s_index.txt", outprefix);
    FILE* fframes = fopen(path, "a+t");
    if(!fframes)
    {
        printf("Can not open path for writing. Permission denied?\n");
        return 0;
    }
    int frameIdx = 0;
	
	for(;;)
	{
        bool objselected = false;
        int nOutlinePt = 0;
        
		if( !paused )
		{
			if( imgFilename )
            {
                frame0 = imread(string(imgFilename), 1);
                paused = true;
            }
            else
                cap >> frame0;
			if( !frame0.data )
				break;
            if( !frame.data )
            {
                if( frame0.size() != calibrated.imageSize )
                {
                    // adjust the camera matrix for the new resolution
                    calibrated.cameraMatrix.at<double>(0,0) *= frame.cols/calibrated.imageSize.width;
                    calibrated.cameraMatrix.at<double>(0,2) *= frame.cols/calibrated.imageSize.width;
                    calibrated.cameraMatrix.at<double>(1,1) *= frame.rows/calibrated.imageSize.height;
                    calibrated.cameraMatrix.at<double>(1,2) *= frame.rows/calibrated.imageSize.height;
                    calibrated.imageSize = frame0.size();
                }
                Mat dummy;
                // initialize undistortion maps
                initUndistortRectifyMap(calibrated.cameraMatrix, calibrated.distCoeffs, Mat(),
                                        calibrated.cameraMatrix, calibrated.imageSize,
                                        CV_32FC2, mapxy, dummy );
                calibrated.distCoeffs = Mat::zeros(5, 1, CV_64F);
                selectedObjMask = Mat::zeros(frame0.size(), CV_8U);
                selectedObjFrame = frame0.clone();
            }
            remap(frame0, frame, mapxy, Mat(), INTER_LINEAR);
            boardFound = findChessboardCorners(frame, calibrated.boardSize, boardCorners);
            
            if( boardFound )
            {
                solvePnP(Mat(calibrated.objPoints), Mat(boardCorners), calibrated.cameraMatrix,
                         calibrated.distCoeffs, rvec, tvec, false);
                Rodrigues(rvec, R);
            }
		}
        frame.copyTo(shownFrame);
        selectedObjFrame = Scalar::all(0);
		
		if( boardFound )
        {
            float Z = 0.f;
            bool dragging = (mouseButtonState & CV_EVENT_FLAG_LBUTTON) != 0;
            int npt = nobjpt;
            
            drawChessboardCorners(shownFrame, calibrated.boardSize, Mat(boardCorners), true);
            
            if( (mouseEvent == CV_EVENT_LBUTTONDOWN ||
                mouseEvent == CV_EVENT_LBUTTONUP ||
                dragging) && nobjpt < 4 )
            {
                // update object box
                mousepts[npt] = mouseLoc;

                /*if(!paused)
                    imwrite("frame.jpg", frame0);*/
                paused = true;
                if( nobjpt < 2 )
                    imgpts[npt] = mousepts[npt];
                else
                {
                    tempobj.resize(1);
                    int nearestIdx = npt-1;
                    /*for( int i = 1; i < npt; i++ )
                        if( norm(mousepts[npt] - mousepts[i]) < norm(mousepts[npt] - imgpts[nearestIdx]) )
                            nearestIdx = i;*/
                    
                    if( npt == 2 )
                    {
                        float dx = objpts[1].x - objpts[0].x, dy = objpts[1].y - objpts[0].y;
                        float len = 1.f/std::sqrt(dx*dx+dy*dy);
                        tempobj[0] = Point3f(dy*len + objpts[nearestIdx].x, -dx*len + objpts[nearestIdx].y, 0.f);
                    }
                    else
                        tempobj[0] = Point3f(objpts[nearestIdx].x, objpts[nearestIdx].y, 1.f);

                    projectPoints(Mat(tempobj), rvec, tvec, calibrated.cameraMatrix,
                                  calibrated.distCoeffs, tempimg);
                    
                    Point2f a = mousepts[nearestIdx], b = tempimg[0],
                        m = mousepts[npt], d1 = b - a, d2 = m - a;
                    float n1 = norm(d1), n2 = norm(d2);
                    if( n1*n2 < eps )
                        imgpts[npt] = a;
                    else
                    {
                        Z = d1.dot(d2)/(n1*n1);
                        imgpts[npt] = d1*Z + a;
                    }
                }
                objpts[npt] = image2plane(imgpts[npt], R, tvec,
                                          calibrated.cameraMatrix, npt<3 ? 0 : Z);
                
                if( (npt == 0 && mouseEvent == CV_EVENT_LBUTTONDOWN) ||
                    (npt > 0 && norm(objpts[npt] - objpts[npt-1]) > eps &&
                    mouseEvent == CV_EVENT_LBUTTONUP) )
                {
                    nobjpt++;
                    if( nobjpt < 4 )
                    {
                        imgpts[nobjpt] = imgpts[nobjpt-1];
                        objpts[nobjpt] = objpts[nobjpt-1];
                        mousepts[nobjpt] = mousepts[nobjpt-1];
                    }
                }
                
                mouseEvent = -1; // reset the event
            }
            
            // draw object box (or a part of it)
            tempobj.resize(8);
            tempobj[0] = objpts[0];
            tempobj[1] = objpts[1];
            tempobj[2] = objpts[2];
            tempobj[3] = (objpts[2] - objpts[1]) + objpts[0];
            Z = objpts[3].z;
            tempobj[4] = tempobj[0] + Point3f(0,0,Z);
            tempobj[5] = tempobj[1] + Point3f(0,0,Z);
            tempobj[6] = tempobj[2] + Point3f(0,0,Z);
            tempobj[7] = tempobj[3] + Point3f(0,0,Z);
            
            projectPoints(Mat(tempobj), rvec, tvec, calibrated.cameraMatrix, Mat(), tempimg);

            if( npt == 0 && nobjpt == 0 )
                nOutlinePt = 0;
            else if( npt == 0 )
            {
                nOutlinePt = 1;
                circle(shownFrame, tempimg[0], 3, Scalar(0,255,0), -1, CV_AA);
            }
            else if( npt == 1 )
            {
                nOutlinePt = 2;
                line(shownFrame, tempimg[0], tempimg[1], Scalar(0,255,0), 3, CV_AA);
                circle(shownFrame, tempimg[0], 3, Scalar(0,255,0), -1, CV_AA);
                circle(shownFrame, tempimg[1], 3, Scalar(0,255,0), -1, CV_AA);
            }
            else
            {
                nOutlinePt = npt == 2 ? 4 : 8;
                for( int i = 0; i < nOutlinePt; i++ )
                {
                    circle(shownFrame, tempimg[i], 3, Scalar(0,255,0), -1, CV_AA);
                    line(shownFrame, tempimg[i], tempimg[(i+1)%4 + (i/4)*4], Scalar(0,255,0), 3, CV_AA);
                    line(shownFrame, tempimg[i], tempimg[i%4], Scalar(0,255,0), 3, CV_AA);
                }
            }
            
            if( nOutlinePt > 2 )
            {
                convexHull(Mat_<Point>(Mat(tempimg).rowRange(0,nOutlinePt)), temphull);
                selectedObjMask = Scalar::all(0);
                fillConvexPoly(selectedObjMask, &temphull[0], temphull.size(),
                               Scalar::all(255), 8, 0);
                frame.copyTo(selectedObjFrame, selectedObjMask);
                objselected = true;
            }
        }

        imshow("Video", shownFrame);
        imshow("Selected Object", selectedObjFrame);
		int c = waitKey(30);
		if( (c & 255) == 27 )
            nobjpt = 0;
        if( c == ' ' )
            paused = !paused;
        if( c == 'q' || c == 'Q' )
            break;
        if( (c == '\r' || c == '\n') && objselected && nOutlinePt == 8 )
        {
            Rect r = boundingRect(Mat(temphull));
            
            for(;;frameIdx++)
            {
                sprintf(path, "%s%04d.jpg", outprefix, frameIdx);
                FILE* f = fopen(path, "rb");
                if( !f )
                    break;
                fclose(f);
            }
            
            imwrite(path, selectedObjFrame(r));
            fprintf(fframes, "%s%04d.jpg", outbarename, frameIdx);
            for( int i = 0; i < 8; i++ )
                fprintf(fframes, " (%.2f %.2f %.2f)", objpts[i].x, objpts[i].y, objpts[i].z);
            fprintf(fframes, "\n");
            frameIdx++;
        }
	}

    fclose(fframes);
    return 0;
}
