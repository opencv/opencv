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


// 2004-03-16, Mark Asbach       <asbach@ient.rwth-aachen.de>
//             Institute of Communications Engineering, RWTH Aachen University

%module(package="opencv") ml

%{
  #include <ml.h>
	#include <cxtypes.h>
	#include <cv.h>
	#include <highgui.h>
	#include "pyhelpers.h"
	#include "pycvseq.hpp"
%}

// include python-specific files
%include "./nointpb.i"
%include "./pytypemaps.i"
%include "exception.i"

%import "../general/cv.i"

%include "../general/memory.i"
%include "../general/typemaps.i"

%newobject cvCreateCNNConvolutionLayer;
%newobject cvCreateCNNSubSamplingLayer;
%newobject cvCreateCNNFullConnectLayer;
%newobject cvCreateCNNetwork;
%newobject cvTrainCNNClassifier;

%newobject cvCreateCrossValidationEstimateModel;


%pythoncode 
%{

__doc__ = """Machine Learning

The Machine Learning library (ML) is a set of classes and functions for 
statistical classification, regression and clustering of data.

Most of the classification and regression algorithms are implemented as classes. 
As the algorithms have different sets of features (like ability to handle missing 
measurements, or categorical input variables etc.), there is only little common 
ground between the classes. This common ground is defined by the class CvStatModel 
that all the other ML classes are derived from.

This wrapper was semi-automatically created from the C/C++ headers and therefore
contains no Python documentation. Because all identifiers are identical to their
C/C++ counterparts, you can consult the standard manuals that come with OpenCV.
"""

%}

%extend CvEM
{
   PyObject * get_covs()
   {
       CvMat ** pointers = const_cast<CvMat **> (self->get_covs());
       int n = self->get_nclusters();

       PyObject * result = PyTuple_New(n);
       for (int i=0; i<n; ++i)
       {
           PyObject * obj = SWIG_NewPointerObj(pointers[i], $descriptor(CvMat *), 0);
           PyTuple_SetItem(result, i, obj);
           //Py_DECREF(obj);
       }
       
       return result;
   }
}

%ignore CvEM::get_covs;

%include "ml.h"
