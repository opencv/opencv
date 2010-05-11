#ifdef _CH_
#pragma package <opencv>
#endif

#define CV_NO_BACKWARD_COMPATIBILITY

#ifndef _EiC
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#endif

int main( int argc, char** argv )
{
    const char* size_opt = "--size=";
    char comment[1024];
    CvHaarClassifierCascade* cascade = 0;
    CvSize size;

    if( argc != 4 || strncmp( argv[1], size_opt, strlen(size_opt) ) != 0 )
    {
        printf( "Usage: convert_cascade --size=\"<width>x<height>\" input_cascade_path output_cascade_filename\n" );
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
