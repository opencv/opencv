/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008, Nils Hasler, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// Author: Nils Hasler <hasler@mpi-inf.mpg.de>
//
//         Max-Planck-Institut Informatik

//
// capture video from a sequence of images
// the filename when opening can either be a printf pattern such as
// video%04d.png or the first frame of the sequence i.e. video0001.png
//

#include "precomp.hpp"
#include <sys/stat.h>

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

#ifndef _MAX_PATH
#define _MAX_PATH 1024
#endif

class CvCapture_Images : public CvCapture
{
public:
    CvCapture_Images()
    {
        filename = NULL;
        currentframe = firstframe = 0;
        length = 0;
        frame = NULL;
        grabbedInOpen = false;
    }

    virtual ~CvCapture_Images()
    {
        close();
    }

    virtual bool open(const char* _filename);
    virtual void close();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual IplImage* retrieveFrame(int);

protected:
    char*  filename; // actually a printf-pattern
    unsigned currentframe;
    unsigned firstframe; // number of first frame
    unsigned length; // length of sequence

    IplImage* frame;
    bool grabbedInOpen;
};


void CvCapture_Images::close()
{
    if( filename )
    {
        free(filename);
        filename = NULL;
    }
    currentframe = firstframe = 0;
    length = 0;
    cvReleaseImage( &frame );
}


bool CvCapture_Images::grabFrame()
{
    char str[_MAX_PATH];
    sprintf(str, filename, firstframe + currentframe);

    if (grabbedInOpen)
    {
        grabbedInOpen = false;
        ++currentframe;

        return frame != NULL;
    }

    cvReleaseImage(&frame);
    frame = cvLoadImage(str, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
    if( frame )
        currentframe++;

    return frame != NULL;
}

IplImage* CvCapture_Images::retrieveFrame(int)
{
    return grabbedInOpen ? NULL : frame;
}

double CvCapture_Images::getProperty(int id) const
{
    switch(id)
    {
    case CV_CAP_PROP_POS_MSEC:
        CV_WARN("collections of images don't have framerates\n");
        return 0;
    case CV_CAP_PROP_POS_FRAMES:
        return currentframe;
    case CV_CAP_PROP_FRAME_COUNT:
        return length;
    case CV_CAP_PROP_POS_AVI_RATIO:
        return (double)currentframe / (double)(length - 1);
    case CV_CAP_PROP_FRAME_WIDTH:
        return frame ? frame->width : 0;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return frame ? frame->height : 0;
    case CV_CAP_PROP_FPS:
        CV_WARN("collections of images don't have framerates\n");
        return 1;
    case CV_CAP_PROP_FOURCC:
        CV_WARN("collections of images don't have 4-character codes\n");
        return 0;
    }
    return 0;
}

bool CvCapture_Images::setProperty(int id, double value)
{
    switch(id)
    {
    case CV_CAP_PROP_POS_MSEC:
    case CV_CAP_PROP_POS_FRAMES:
        if(value < 0) {
            CV_WARN("seeking to negative positions does not work - clamping\n");
            value = 0;
        }
        if(value >= length) {
            CV_WARN("seeking beyond end of sequence - clamping\n");
            value = length - 1;
        }
        currentframe = cvRound(value);
        if (currentframe != 0)
            grabbedInOpen = false; // grabbed frame is not valid anymore
        return true;
    case CV_CAP_PROP_POS_AVI_RATIO:
        if(value > 1) {
            CV_WARN("seeking beyond end of sequence - clamping\n");
            value = 1;
        } else if(value < 0) {
            CV_WARN("seeking to negative positions does not work - clamping\n");
            value = 0;
        }
        currentframe = cvRound((length - 1) * value);
        if (currentframe != 0)
            grabbedInOpen = false; // grabbed frame is not valid anymore
        return true;
    }
    CV_WARN("unknown/unhandled property\n");
    return false;
}

static char* icvExtractPattern(const char *filename, unsigned *offset)
{
    char *name = (char *)filename;

    if( !filename )
        return 0;

    // check whether this is a valid image sequence filename
    char *at = strchr(name, '%');
    if(at)
    {
        int dummy;
        if(sscanf(at + 1, "%ud", &dummy) != 1)
            return 0;
        name = strdup(filename);
    }
    else // no pattern filename was given - extract the pattern
    {
        at = name;

        // ignore directory names
        char *slash = strrchr(at, '/');
        if (slash) at = slash + 1;

#ifdef _WIN32
        slash = strrchr(at, '\\');
        if (slash) at = slash + 1;
#endif

        while (*at && !isdigit(*at)) at++;

        if(!*at)
            return 0;

        sscanf(at, "%u", offset);

        int size = (int)strlen(filename) + 20;
        name = (char *)malloc(size);
        strncpy(name, filename, at - filename);
        name[at - filename] = 0;

        strcat(name, "%0");

        int i;
        char *extension;
        for(i = 0, extension = at; isdigit(at[i]); i++, extension++)
            ;
        char places[10];
        sprintf(places, "%dd", i);

        strcat(name, places);
        strcat(name, extension);
    }

    return name;
}


bool CvCapture_Images::open(const char * _filename)
{
    unsigned offset = 0;
    close();

    filename = icvExtractPattern(_filename, &offset);
    if(!filename)
        return false;

    // determine the length of the sequence
    length = 0;
    char str[_MAX_PATH];
    for(;;)
    {
        sprintf(str, filename, offset + length);
        struct stat s;
        if(stat(str, &s))
        {
            if(length == 0 && offset == 0) // allow starting with 0 or 1
            {
                offset++;
                continue;
            }
        }

        if(!cvHaveImageReader(str))
            break;

        length++;
    }

    if(length == 0)
    {
        close();
        return false;
    }

    firstframe = offset;

    // grab frame to enable properties retrieval
    bool grabRes = grabFrame();
    grabbedInOpen = true;
    currentframe = 0;

    return grabRes;
}


CvCapture* cvCreateFileCapture_Images(const char * filename)
{
    CvCapture_Images* capture = new CvCapture_Images;

    if( capture->open(filename) )
        return capture;

    delete capture;
    return NULL;
}

//
//
// image sequence writer
//
//
class CvVideoWriter_Images : public CvVideoWriter
{
public:
    CvVideoWriter_Images()
    {
        filename = 0;
        currentframe = 0;
    }
    virtual ~CvVideoWriter_Images() { close(); }

    virtual bool open( const char* _filename );
    virtual void close();
    virtual bool writeFrame( const IplImage* );

protected:
    char* filename;
    unsigned currentframe;
};

bool CvVideoWriter_Images::writeFrame( const IplImage* image )
{
    char str[_MAX_PATH];
    sprintf(str, filename, currentframe);
    int ret = cvSaveImage(str, image);

    currentframe++;

    return ret > 0;
}

void CvVideoWriter_Images::close()
{
    if( filename )
    {
        free( filename );
        filename = 0;
    }
    currentframe = 0;
}


bool CvVideoWriter_Images::open( const char* _filename )
{
    unsigned offset = 0;

    close();

    filename = icvExtractPattern(_filename, &offset);
    if(!filename)
        return false;

    char str[_MAX_PATH];
    sprintf(str, filename, 0);
    if(!cvHaveImageWriter(str))
    {
        close();
        return false;
    }

    currentframe = offset;
    return true;
}


CvVideoWriter* cvCreateVideoWriter_Images( const char* filename )
{
    CvVideoWriter_Images *writer = new CvVideoWriter_Images;

    if( writer->open( filename ))
        return writer;

    delete writer;
    return 0;
}
