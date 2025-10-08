// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
C++ wrappers over OpenVX 1.x C API ("openvx-debug" module)
Details: TBD
*/

#pragma once
#ifndef IVX_LIB_DEBUG_HPP
#define IVX_LIB_DEBUG_HPP

#include "ivx.hpp"

namespace ivx
{
namespace debug
{
/*
* "openvx-debug" module
*/

//
void fReadImage(vx_context c, const std::string& path, vx_image img)
{
    IVX_CHECK_STATUS( vxuFReadImage(c, (vx_char*)path.c_str(), img) );
}

//
void fWriteImage(vx_context c, vx_image img, const std::string& path)
{
    IVX_CHECK_STATUS( vxuFWriteImage(c, img, (vx_char*)path.c_str()) );
}

} // namespace debug
} // namespace ivx

#endif //IVX_LIB_DEBUG_HPP
