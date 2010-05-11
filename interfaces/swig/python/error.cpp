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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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


// 2004-03-17, Gabriel Schreiber <schreiber@ient.rwth-aachen.de>
//             Mark Asbach       <asbach@ient.rwth-aachen.de>
//             Institute of Communications Engineering, RWTH Aachen University


// Python header.
// It may be required that this header is the first to be included
// (see Python documentation for details)
#include "Python.h"

#include "error.h"
#include <sstream>
#include <iostream>

// OpenCV headers
#include "cxcore.h"
#include "cxerror.h"


//=========================================================================
int SendErrorToPython
    (
    int status, 
    const char* func_name, 
    const char* err_msg,
    const char* file_name, 
    int line, 
    void* /*userdata*/
    )
    throw(int)
    {
    std::stringstream message;
    message   
        << " openCV Error:"
        << "\n        Status=" << cvErrorStr(status)
        << "\n        function name=" << (func_name ? func_name : "unknown")
        << "\n        error message=" << (err_msg ? err_msg : "unknown")
        << "\n        file_name=" << (file_name ? file_name : "unknown")
        << "\n        line=" << line
        << std::flush;

    // Clear OpenCV's error status for the next call!
    cvSetErrStatus( CV_StsOk );
    
    // Set Python Error.
    // ATTENTION: this only works if the function returning to
    // Python returns 0 instead of a PyObject (see also "custom_typemaps.i"        
    PyErr_SetString(PyExc_RuntimeError, message.str().c_str());
    throw 1;    
    return 0;
    }

    
//=========================================================================
void* void_ptr_generator()
    { 
    return 0;
    }

//=========================================================================
void** void_ptrptr_generator()
    { 
    return 0;
    }

//=========================================================================
CvErrorCallback function_ptr_generator()
    {
    return &SendErrorToPython;
    }

