#ifndef IOUTPUT_H
#define IOUTPUT_H

#include <cstdio>

#include "_cvcommon.h"

struct CvMat;
struct CvRect;

class IOutput
{
public:
    enum OutputType {PNG_DATASET, JPG_DATASET};
public:
    virtual bool write( const CvMat& img,
                        const CvRect& boundingBox ) =0;

    virtual ~IOutput();

    static IOutput* createOutput( const char *filename, OutputType type );
protected:
    IOutput();
    /* finds the beginning of the last token in the path */
    void findFilePathPart( char **partOfPath, char *fullPath );
    virtual bool init( const char* filename );
protected:
    int  currentIdx;
    char imgFullPath[PATH_MAX];
    char* imgFileName;
    FILE* annotationsList;
};

#endif // IOUTPUT_H
