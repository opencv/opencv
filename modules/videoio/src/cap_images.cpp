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
#include "opencv2/imgcodecs.hpp"
#include <sys/stat.h>

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

#ifndef _MAX_PATH
#define _MAX_PATH 1024
#endif

namespace cv
{

class CvCapture_Images: public IVideoCapture
{
public:
    void init()
    {
        filename.clear();
        frame.release();
        currentframe = firstframe = 0;
        length = 0;
        grabbedInOpen = false;
    }
    CvCapture_Images()
    {
        init();
    }
    CvCapture_Images(const String& _filename)
    {
        init();
        open(_filename);
    }

    virtual ~CvCapture_Images() CV_OVERRIDE
    {
        close();
    }
    virtual double getProperty(int) const CV_OVERRIDE;
    virtual bool setProperty(int, double) CV_OVERRIDE;
    virtual bool grabFrame() CV_OVERRIDE;
    virtual bool retrieveFrame(int, OutputArray) CV_OVERRIDE;
    virtual bool isOpened() const CV_OVERRIDE;
    virtual int getCaptureDomain() /*const*/ CV_OVERRIDE { return cv::CAP_IMAGES; }

    bool open(const String&);
    void close();
protected:

    std::string filename; // actually a printf-pattern
    unsigned currentframe;
    unsigned firstframe; // number of first frame
    unsigned length; // length of sequence

    Mat frame;
    bool grabbedInOpen;
};

void CvCapture_Images::close()
{
    init();
}

bool CvCapture_Images::grabFrame()
{
    char str[_MAX_PATH];

    if( filename.empty() )
        return false;

    sprintf(str, filename.c_str(), firstframe + currentframe);

    if (grabbedInOpen)
    {
        grabbedInOpen = false;
        ++currentframe;

        return !frame.empty();
    }

    frame = imread(str, IMREAD_UNCHANGED);
    if( !frame.empty() )
        currentframe++;

    return !frame.empty();
}

bool CvCapture_Images::retrieveFrame(int, OutputArray out)
{
    frame.copyTo(out);
    return grabbedInOpen ? false : !frame.empty();
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
        return frame.cols;
    case CV_CAP_PROP_FRAME_HEIGHT:
        return frame.rows;
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

static std::string extractPattern(const std::string& filename, unsigned& offset)
{
    std::string name;

    if( filename.empty() )
        return std::string();

    // check whether this is a valid image sequence filename
    char *at = strchr((char*)filename.c_str(), '%');
    if(at)
    {
        unsigned int dummy;
        if(sscanf(at + 1, "%ud", &dummy) != 1)
            return std::string();
        name = filename;
    }
    else // no pattern filename was given - extract the pattern
    {
        at = (char*)filename.c_str();

        // ignore directory names
        char *slash = strrchr(at, '/');
        if (slash) at = slash + 1;

#ifdef _WIN32
        slash = strrchr(at, '\\');
        if (slash) at = slash + 1;
#endif

        while (*at && !isdigit(*at)) at++;

        if(!*at)
            return std::string();

        sscanf(at, "%u", &offset);

        name = filename.substr(0, at - filename.c_str());
        name += "%0";

        int i;
        char *extension;
        for(i = 0, extension = at; isdigit(at[i]); i++, extension++)
            ;
        char places[13] = {0};
        sprintf(places, "%dd", i);

        name += places;
        name += extension;
    }

    return name;
}


bool CvCapture_Images::open(const std::string& _filename)
{
    unsigned offset = 0;
    close();

    filename = extractPattern(_filename, offset);
    if( filename.empty() )
        return false;

    // determine the length of the sequence
    length = 0;
    char str[_MAX_PATH];
    for(;;)
    {
        sprintf(str, filename.c_str(), offset + length);
        struct stat s;
        if(stat(str, &s))
        {
            if(length == 0 && offset == 0) // allow starting with 0 or 1
            {
                offset++;
                continue;
            }
        }

        if(!haveImageReader(str))
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

bool CvCapture_Images::isOpened() const
{
    return !filename.empty();
}

Ptr<IVideoCapture> create_Images_capture(const std::string &filename)
{
    return makePtr<CvCapture_Images>(filename);
}

//
//
// image sequence writer
//
//
class CvVideoWriter_Images CV_FINAL : public CvVideoWriter
{
public:
    CvVideoWriter_Images()
    {
        filename.clear();
        currentframe = 0;
    }
    virtual ~CvVideoWriter_Images() { close(); }

    virtual bool open( const char* _filename );
    virtual void close();
    virtual bool setProperty( int, double ); // FIXIT doesn't work: IVideoWriter interface only!
    virtual bool writeFrame( const IplImage* ) CV_OVERRIDE;

    int getCaptureDomain() const CV_OVERRIDE { return cv::CAP_IMAGES; }
protected:
    std::string filename;
    unsigned currentframe;
    std::vector<int> params;
};

bool CvVideoWriter_Images::writeFrame( const IplImage* image )
{
    char str[_MAX_PATH];
    sprintf(str, filename.c_str(), currentframe);
    std::vector<int> image_params = params;
    image_params.push_back(0); // append parameters 'stop' mark
    image_params.push_back(0);
    cv::Mat img = cv::cvarrToMat(image);
    bool ret = cv::imwrite(str, img, image_params);

    currentframe++;

    return ret;
}

void CvVideoWriter_Images::close()
{
    filename.clear();
    currentframe = 0;
    params.clear();
}


bool CvVideoWriter_Images::open( const char* _filename )
{
    unsigned offset = 0;

    close();

    filename = cv::extractPattern(_filename, offset);
    if(filename.empty())
        return false;

    char str[_MAX_PATH];
    sprintf(str, filename.c_str(), 0);
    if(!cv::haveImageWriter(str))
    {
        close();
        return false;
    }

    currentframe = offset;
    params.clear();
    return true;
}


bool CvVideoWriter_Images::setProperty( int id, double value )
{
    if (id >= cv::CAP_PROP_IMAGES_BASE && id < cv::CAP_PROP_IMAGES_LAST)
    {
        params.push_back( id - cv::CAP_PROP_IMAGES_BASE );
        params.push_back( static_cast<int>( value ) );
        return true;
    }
    return false; // not supported
}

Ptr<IVideoWriter> create_Images_writer(const std::string &filename, int, double, const Size &, bool)
{
    CvVideoWriter_Images *writer = new CvVideoWriter_Images;

    if( writer->open( filename.c_str() ))
        return makePtr<LegacyWriter>(writer);

    delete writer;
    return 0;
}

} // cv::
