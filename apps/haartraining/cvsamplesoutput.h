#ifndef CVSAMPLESOUTPUT_H
#define CVSAMPLESOUTPUT_H

#include "ioutput.h"

class PngTrainingSetOutput: public IOutput
{
    friend IOutput* IOutput::createOutput(const char *filename, OutputType type);
public:
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox);

    virtual ~PngTrainingSetOutput(){}
private:
    PngTrainingSetOutput()
        : extension("png")
        , destImgWidth(640)
        , destImgHeight(480)
    {}

    virtual bool init(const char* annotationsListFileName );

    void writeImage( const CvMat& img ) const;

    CvRect scaleBoundingBox(const CvSize& imgSize,
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
    friend IOutput* IOutput::createOutput(const char *filename, OutputType type);
public:
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox );
    virtual ~TestSamplesOutput(){}
private:
    TestSamplesOutput(){}
};
#endif // CVSAMPLESOUTPUT_H
