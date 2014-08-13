#ifndef CVSAMPLESOUTPUT_H
#define CVSAMPLESOUTPUT_H

#include <memory>
#include <cstdio>

#include "_cvcommon.h"

struct CvMat;
struct CvRect;

class IOutput
{
public:
    virtual bool init( const char* filename );
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox ) =0;

    virtual ~IOutput();

public:
    enum OutputType {PNG_TRAINING_SET, JPG_TEST_SET};
protected:
    /* finds the beginning of the last token in the path */
    void findFilePathPart( char** partofpath, char* fullpath );
protected:
    int  currentIdx;
    char imgFullPath[PATH_MAX];
    char* imgFileName;
    FILE* annotationsList;
};

IOutput *createOutput( const char* filename, IOutput::OutputType type );

class PngTrainingSetOutput: public IOutput
{
friend IOutput *createOutput( const char* filename, IOutput::OutputType type );
public:
    virtual bool init(const char* annotationsListFileName );
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox);

    virtual ~PngTrainingSetOutput(){}
private:
    PngTrainingSetOutput()
        : extension("png")
        , destImgWidth(640)
        , destImgHeight(480)
    {}

    void writeImage( const CvMat& img ) const;

    CvRect scaleBoundingBox( int destImgWidth,
                             int destImgHeight,
                             const CvRect& bbox);
private:

    char annotationFullPath[PATH_MAX];
    char* annotationFileName;
    char* annotationRelativePath;
    char* imgRelativePath;
    const char* extension;

    int destImgWidth;
    int destImgHeight ;
};

class TestSamplesOutput: public IOutput
{
friend IOutput *createOutput( const char* filename, IOutput::OutputType type );
public:
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox );
    virtual ~TestSamplesOutput(){}
private:
    TestSamplesOutput(){}
};
#endif // CVSAMPLESOUTPUT_H
