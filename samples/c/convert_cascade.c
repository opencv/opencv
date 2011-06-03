#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <ctype.h>
#include <stdio.h>

void help()
{
    printf("\n This sample demonstrates cascade's convertation \n"
    "Usage:\n"
    "./convert_cascade --size=\"<width>x<height>\"<convertation size> \n"
    "                   input_cascade_path \n"
    "                   output_cascade_filename\n"
    "Example: \n"
    "./convert_cascade --size=640x480 ../../opencv/data/haarcascades/haarcascade_eye.xml ../../opencv/data/haarcascades/test_cascade.xml \n"
    );
}

int main( int argc, char** argv )
{
    const char* size_opt = "--size=";
    char comment[1024];
    CvHaarClassifierCascade* cascade = 0;
    CvSize size;

    help();

    if( argc != 4 || strncmp( argv[1], size_opt, strlen(size_opt) ) != 0 )
    {
    	help();
        return -1;
    }

    sscanf( argv[1], "--size=%ux%u", &size.width, &size.height );
    cascade = cvLoadHaarClassifierCascade( argv[2], size );

    if( !cascade )
    {
        fprintf( stderr, "Input cascade could not be found/opened\n" );
        return -1;
    }

    sprintf( comment, "Automatically converted from %s, window size = %dx%d", argv[2], size.width, size.height );
    cvSave( argv[3], cascade, 0, comment, cvAttrList(0,0) );
    return 0;
}

#ifdef _EiC
main(1,"facedetect.c");
#endif
