#ifndef CVSAMPLESOUTPUT_H
#define CVSAMPLESOUTPUT_H

#include "ioutput.h"

class PngDatasetOutput: public IOutput
{
    friend IOutput* IOutput::createOutput(const char *filename, OutputType type);
public:
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox);

    virtual ~PngDatasetOutput(){}
private:
    PngDatasetOutput()
        : extension("png")
        , destImgWidth(640)
        , destImgHeight(480)
    {}

    virtual bool init(const char* annotationsListFileName );

    CvRect addBoundingboxBorder(const CvRect& bbox) const;
private:

    char annotationFullPath[PATH_MAX];
    char* annotationFileName;
    char* annotationRelativePath;
    char* imgRelativePath;
    const char* extension;

    int destImgWidth;
    int destImgHeight ;
};

class JpgDatasetOutput: public IOutput
{
    friend IOutput* IOutput::createOutput(const char *filename, OutputType type);
public:
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox );
    virtual ~JpgDatasetOutput(){}
private:
    JpgDatasetOutput(){}
};
#endif // CVSAMPLESOUTPUT_H
