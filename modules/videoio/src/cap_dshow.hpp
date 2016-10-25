/*M///////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//M*/

#ifndef _CAP_DSHOW_HPP_
#define _CAP_DSHOW_HPP_

#include "precomp.hpp"

#ifdef HAVE_DSHOW

class videoInput;
namespace cv
{

class VideoCapture_DShow : public IVideoCapture
{
public:
    VideoCapture_DShow(int index);
    virtual ~VideoCapture_DShow();

    virtual double getProperty(int propIdx) const;
    virtual bool setProperty(int propIdx, double propVal);

    virtual bool grabFrame();
    virtual bool retrieveFrame(int outputType, OutputArray frame);
    virtual int getCaptureDomain();
    virtual bool isOpened() const;
protected:
    void open(int index);
    void close();

    int m_index, m_width, m_height, m_fourcc;
    int m_widthSet, m_heightSet;
    static videoInput g_VI;
};

}

#endif //HAVE_DSHOW
#endif //_CAP_DSHOW_HPP_