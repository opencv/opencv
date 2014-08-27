#include "cvsamplesoutput.h"

#include <cstdio>

#include "_cvcommon.h"
#include "opencv2/opencv.hpp"

/* print statistic info */
#define CV_VERBOSE 1

IOutput::IOutput()
    : currentIdx(0)
{}

void IOutput::findFilePathPart(char **partOfPath, char *fullPath)
{
    *partOfPath = strrchr( fullPath, '\\' );
    if( *partOfPath == NULL )
    {
        *partOfPath = strrchr( fullPath, '/' );
    }
    if( *partOfPath == NULL )
    {
        *partOfPath = fullPath;
    }
    else
    {
        *partOfPath += 1;
    }
}

IOutput* IOutput::createOutput(const char *filename,
                               IOutput::OutputType type)
{
    IOutput* output = 0;
    switch (type) {
    case IOutput::PNG_DATASET:
        output = new PngDatasetOutput();
        break;
    case IOutput::JPG_DATASET:
        output = new JpgDatasetOutput();
        break;
    default:
#if CV_VERBOSE
        fprintf( stderr, "Invalid output type, valid types are: PNG_TRAINING_SET, JPG_TEST_SET");
#endif /* CV_VERBOSE */
        return 0;
    }

    if ( output->init( filename ) )
        return output;
    else
        return 0;
}

bool PngDatasetOutput::init( const char* annotationsListFileName )
{
    IOutput::init( annotationsListFileName );

    if(imgFileName == imgFullPath)
    {
        #if CV_VERBOSE
                fprintf( stderr, "Invalid path to annotations file: %s\n"
                                 "It should contain a parent directory name\n", imgFullPath );
        #endif /* CV_VERBOSE */
        return false;
    }


    const char* annotationsdirname = "/annotations/";
    const char* positivesdirname = "/pos/";

    imgFileName[-1] = '\0'; //erase slash at the end of the path
    imgFileName -= 1;

    //copy path to dataset top-level dir
    strcpy(annotationFullPath, imgFullPath);
    //find the name of annotation starting from the top-level dataset dir
    findFilePathPart(&annotationRelativePath, annotationFullPath);
    if( !strcmp( annotationRelativePath, ".." ) || !strcmp( annotationRelativePath, "." ) )
    {
        #if CV_VERBOSE
                fprintf( stderr, "Invalid path to annotations file: %s\n"
                                 "It should contain a parent directory name\n", annotationsListFileName );
        #endif /* CV_VERBOSE */
        return false;
    }
    //find the name of output image starting from the top-level dataset dir
    findFilePathPart(&imgRelativePath, imgFullPath);
    annotationFileName = annotationFullPath + strlen(annotationFullPath);

    sprintf(annotationFileName, "%s", annotationsdirname);
    annotationFileName += strlen(annotationFileName);
    sprintf(imgFileName, "%s", positivesdirname);
    imgFileName += strlen(imgFileName);

    if( !icvMkDir( annotationFullPath ) )
    {
        #if CV_VERBOSE
                fprintf( stderr, "Unable to create directory hierarchy: %s\n", annotationFullPath );
        #endif /* CV_VERBOSE */
        return false;
    }
    if( !icvMkDir( imgFullPath ) )
    {
        #if CV_VERBOSE
                fprintf( stderr, "Unable to create directory hierarchy: %s\n", imgFullPath );
        #endif /* CV_VERBOSE */
        return false;
    }

    return true;
}

bool PngDatasetOutput::write( const CvMat& img,
                                  const CvRect& boundingBox )
{
    CvRect bbox = scaleBoundingBox(cvGetSize(&img), boundingBox);

    sprintf( imgFileName,
             "%04d_%04d_%04d_%04d_%04d",
             ++currentIdx,
             bbox.x,
             bbox.y,
             bbox.width,
             bbox.height );

    sprintf( annotationFileName, "%s.txt", imgFileName );
    fprintf( annotationsList, "%s\n", annotationRelativePath );

    FILE* annotationFile = fopen( annotationFullPath, "w" );
    if(annotationFile == 0)
    {
        return false;
    }

    sprintf( imgFileName + strlen(imgFileName), ".%s", extension );



    fprintf( annotationFile,
             "Image filename : \"%s\"\n"
             "Bounding box for object 1 \"PASperson\" (Xmin, Ymin) - (Xmax, Ymax) : (%d, %d) - (%d, %d)",
             imgRelativePath,
             bbox.x,
             bbox.y,
             bbox.x + bbox.width,
             bbox.y + bbox.height );
    fclose( annotationFile );

    writeImage(img);

    return true;
}

void PngDatasetOutput::writeImage(const CvMat &img) const
{
    CvSize origsize = cvGetSize(&img);

    if( origsize.height > destImgHeight || origsize.width > destImgWidth )
    {
        CvMat result = cvMat( destImgHeight, destImgWidth, CV_8UC1,
                              cvAlloc( sizeof( uchar ) * destImgHeight * destImgWidth ) );
        cvResize(&img, &result);
        cvSaveImage( imgFullPath, &result );
        cvFree( &(result.data.ptr) );
    }
    else
    {
        cvSaveImage( imgFullPath, &img);
    }

    return;
}

CvRect PngDatasetOutput::scaleBoundingBox(const CvSize& imgSize, const CvRect& bbox)
{
    double scale = MAX( (float) destImgWidth / imgSize.width,
                        (float) destImgHeight / imgSize.height );
    CvRect boundingBox = bbox;
    int border = 5;
    if( scale < 1. )
    {
        boundingBox.x = bbox.x  * scale;
        boundingBox.y  = bbox.y  * scale;
        boundingBox.width = bbox.width * scale;
        boundingBox.height = bbox.height * scale;
    }
    boundingBox.x -= border;
    boundingBox.y -= border;
    boundingBox.width += 2*border;
    boundingBox.height += 2*border;

    return boundingBox;
}

IOutput::~IOutput()
{
    if(annotationsList)
    {
        fclose(annotationsList);
    }
}

bool IOutput::init(const char *filename)
{
    assert( filename != NULL );

    if( !icvMkDir( filename ) )
    {

#if CV_VERBOSE
        fprintf( stderr, "Unable to create directory hierarchy: %s\n", filename );
#endif /* CV_VERBOSE */

        return false;
    }

    annotationsList = fopen( filename, "w" );
    if( annotationsList == NULL )
    {
#if CV_VERBOSE
        fprintf( stderr, "Unable to create info file: %s\n", filename );
#endif /* CV_VERBOSE */
        return false;
    }
    strcpy( imgFullPath, filename );

    findFilePathPart( &imgFileName, imgFullPath );

    return true;
}

bool JpgDatasetOutput::write( const CvMat& img,
                               const CvRect& boundingBox )
{
    sprintf( imgFileName, "%04d_%04d_%04d_%04d_%04d.jpg",
             ++currentIdx,
             boundingBox.x,
             boundingBox.y,
             boundingBox.width,
             boundingBox.height );

   fprintf( annotationsList, "%s %d %d %d %d %d\n",
            imgFileName,
            1,
            boundingBox.x,
            boundingBox.y,
            boundingBox.width,
            boundingBox.height );

    cvSaveImage( imgFullPath, &img);

    return true;
}
