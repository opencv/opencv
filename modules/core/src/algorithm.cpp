/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "precomp.hpp"

namespace cv
{

Algorithm::Algorithm()
{
    CV_TRACE_FUNCTION();
}

Algorithm::~Algorithm()
{
    CV_TRACE_FUNCTION();
}

void Algorithm::write(FileStorage& fs, const String& name) const
{
    CV_TRACE_FUNCTION();
    if(name.empty())
    {
        write(fs);
        return;
    }
    fs << name << "{";
    write(fs);
    fs << "}";
}

#if CV_VERSION_MAJOR < 5
void Algorithm::write(const Ptr<FileStorage>& fs, const String& name) const
{
    CV_Assert(fs);
    write(*fs, name);
}
#endif

void Algorithm::save(const String& filename) const
{
    CV_TRACE_FUNCTION();
    FileStorage fs(filename, FileStorage::WRITE);
    fs << getDefaultName() << "{";
    write(fs);
    fs << "}";
}

String Algorithm::getDefaultName() const
{
    CV_TRACE_FUNCTION();
    return String("my_object");
}

void Algorithm::writeFormat(FileStorage& fs) const
{
    CV_TRACE_FUNCTION();
    fs << "format" << (int)3;
}

}

/* End of file. */
