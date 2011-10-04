#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#ifdef WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

#ifdef HAVE_CVCONFIG_H 
#include <cvconfig.h> 
#endif

#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

using namespace std;
using namespace cv;

void help()
{
    cout << "This program demonstrated the use of the latentSVM detector." << endl <<
            "It reads in a trained object models and then uses them to detect the object in an images" << endl <<
             endl <<
            "Call:" << endl <<
            "./latentsvm_multidetect <images_folder> <models_folder> [<threads_number>]" << endl <<
            "Example of models_folder is opencv_extra/testdata/cv/latentsvmdetector/models_VOC2007" << endl <<
             endl <<
            "Keys:" << endl <<
            "'n' - to go to the next image;" << endl <<
            "'esc' - to quit." << endl <<
            endl;
}

void detectAndDrawObjects( Mat& image, LatentSvmDetector& detector, const vector<Scalar>& colors, int numThreads )
{
    vector<LatentSvmDetector::ObjectDetection> detections;

    TickMeter tm;
    tm.start();
    detector.detect( image, detections, 0.5f, numThreads);
    tm.stop();

    cout << "Detection time = " << tm.getTimeSec() << " sec" << endl;

    const vector<string> classNames = detector.getClassNames();
    CV_Assert( colors.size() == classNames.size() );

    for( size_t i = 0; i < detections.size(); i++ )
    {
        const LatentSvmDetector::ObjectDetection& od = detections[i];
        rectangle( image, od.rect, colors[od.classID], 2 );
        putText( image, classNames[od.classID], Point(od.rect.x+2,od.rect.y+9), FONT_HERSHEY_SIMPLEX, 0.35, colors[od.classID], 1 );
    }
}

void readDirectory( const string& directoryName, vector<string>& filenames, bool addDirectoryName=true )
{
    filenames.clear();

#ifdef WIN32
    struct _finddata_t s_file;
    string str = directoryName + "\\*.*";

	intptr_t h_file = _findfirst( str.c_str(), &s_file );
	if( h_file != static_cast<intptr_t>(-1.0) )
    {
        do
        {
            if( addDirectoryName )
                filenames.push_back(directoryName + "\\" + s_file.name);
            else
                filenames.push_back((string)s_file.name);
        }
        while( _findnext( h_file, &s_file ) == 0 );
    }
    _findclose( h_file );
#else
    DIR* dir = opendir( directoryName.c_str() );
    if( dir != NULL )
    {
        struct dirent* dent;
        while( (dent = readdir(dir)) != NULL )
        {
            if( addDirectoryName )
                filenames.push_back( directoryName + "/" + string(dent->d_name) );
            else
                filenames.push_back( string(dent->d_name) );
        }
    }
#endif

    sort( filenames.begin(), filenames.end() );
}

void fillRngColors( vector<Scalar>& colors )
{
    Mat m = Mat(colors).reshape(1,1);
    randu( m, 0, 255 );
}

int main(int argc, char* argv[])
{
	help();

    string images_folder, models_folder;
    int numThreads = -1;
    if( argc > 2 )
	{
        images_folder = argv[1];
        models_folder = argv[2];
        if( argc > 3 ) numThreads = atoi(argv[3]);
	}

    vector<string> images_filenames, models_filenames;
    readDirectory( images_folder, images_filenames );
    readDirectory( models_folder, models_filenames );

    LatentSvmDetector detector( models_filenames );
    if( detector.empty() )
    {
        cout << "Models cann't be loaded" << endl;
        exit(-1);
    }

    vector<Scalar> colors( detector.getClassNames().size() );
    fillRngColors( colors );

    for( size_t i = 0; i < images_filenames.size(); i++ )
    {
        Mat image = imread( images_filenames[i] );
        if( image.empty() )  continue;

        cout << "Process image " << images_filenames[i] << endl;
        detectAndDrawObjects( image, detector, colors, numThreads );

        imshow( "result", image );

        while(1)
        {
            int c = waitKey();
            if( (char)c == 'n')
                break;
            else if( (char)c == '\x1b' )
                exit(0);
        }
    }
    
	return 0;
}
