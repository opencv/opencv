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


// 2004-03-16, Gabriel Schreiber <schreiber@ient.rwth-aachen.de>
//             Mark Asbach       <asbach@ient.rwth-aachen.de>
//             Institute of Communications Engineering, RWTH Aachen University
// 2008-04-09  Xavier Delacour <xavier.delacour@gmail.com>

// todo remove these..
#pragma SWIG nowarn=312,362,303,365,366,367,368,370,371,372,451,454,503

%{
#include <cxtypes.h>
#include <cv.h>
#include <highgui.h>
#include "octhelpers.h"
#include "octcvseq.hpp"
%}

// include octave-specific files
%include "./octtypemaps.i"
%include "exception.i"

// the wrapping code to enable the use of Octave-based mouse callbacks
%{
  // This encapsulates the octave callback and user_data for mouse callback
  struct OctCvMouseCBData {
    octave_value oct_func;
    octave_value user_data;
  };
  // This encapsulates the octave callback and user_data for mouse callback
  // C helper function which is responsible for calling
  // the Octave real trackbar callback function
  static void icvOctOnMouse (int event, int x, int y,
			     int flags, OctCvMouseCBData * param) {
    octave_value oct_func(param->oct_func);
    if (!oct_func.is_function() && !oct_func.is_function_handle())
      return;

    octave_value_list args;
    args.append(octave_value(event));
    args.append(octave_value(x));
    args.append(octave_value(y));
    args.append(octave_value(flags));
    args.append(param->user_data);
    oct_func.subsref ("(", std::list<octave_value_list>(1, args), 0);
  }
%}

// adapt cvSetMouseCallback to use octave callback
%rename (cvSetMouseCallbackOld) cvSetMouseCallback;
%rename (cvSetMouseCallback) cvSetMouseCallbackOct;
%inline {
  void cvSetMouseCallbackOct( const char* window_name, octave_value on_mouse, octave_value param = octave_value() ){
    OctCvMouseCBData * oct_callback = new OctCvMouseCBData;
    oct_callback->oct_func = on_mouse;
    oct_callback->user_data = param;
    cvSetMouseCallback( window_name, (CvMouseCallback) icvOctOnMouse, (void *) oct_callback );
  }
}

// The following code enables trackbar callbacks from octave.  Unfortunately, there is no 
// way to distinguish which trackbar the event originated from, so must hard code a 
// fixed number of unique c callback functions using the macros below
%{
  // C helper function which is responsible for calling
  // the Octave real trackbar callback function
  static void icvOctOnTrackbar( octave_value oct_cb_func, int pos) {
    if (!oct_cb_func.is_function() && !oct_cb_func.is_function_handle())
      return;

    octave_value_list args;
    args.append(octave_value(pos));
    oct_cb_func.subsref ("(", std::list<octave_value_list>(1, args), 0);
  }

#define ICV_OCT_MAX_CB 10

  struct OctCvTrackbar {
    CvTrackbarCallback cv_func;
    octave_value oct_func;
    octave_value oct_pos;
  };

  static int my_trackbar_cb_size=0;
  extern OctCvTrackbar my_trackbar_cb_funcs[ICV_OCT_MAX_CB];
  %}

// Callback table entry
%define %ICV_OCT_CB_TAB_ENTRY(idx)
{(CvTrackbarCallback) icvOctTrackbarCB##idx, octave_value(), octave_value() }
%enddef

// Table of callbacks
%define %ICV_OCT_CB_TAB
%{
  OctCvTrackbar my_trackbar_cb_funcs[ICV_OCT_MAX_CB] = {
    %ICV_OCT_CB_TAB_ENTRY(0),
    %ICV_OCT_CB_TAB_ENTRY(1),
    %ICV_OCT_CB_TAB_ENTRY(2),
    %ICV_OCT_CB_TAB_ENTRY(3),
    %ICV_OCT_CB_TAB_ENTRY(4),
    %ICV_OCT_CB_TAB_ENTRY(5),
    %ICV_OCT_CB_TAB_ENTRY(6),
    %ICV_OCT_CB_TAB_ENTRY(7),
    %ICV_OCT_CB_TAB_ENTRY(8),
    %ICV_OCT_CB_TAB_ENTRY(9)
  };
%}	 
%enddef

// Callback definition
%define %ICV_OCT_CB_IMPL(idx) 
%{
static void icvOctTrackbarCB##idx(int pos){                                      
  icvOctOnTrackbar(my_trackbar_cb_funcs[idx].oct_func, pos);
}                                                                               
%}
%enddef

%ICV_OCT_CB_IMPL(0);
%ICV_OCT_CB_IMPL(1);
%ICV_OCT_CB_IMPL(2);
%ICV_OCT_CB_IMPL(3);
%ICV_OCT_CB_IMPL(4);
%ICV_OCT_CB_IMPL(5);
%ICV_OCT_CB_IMPL(6);
%ICV_OCT_CB_IMPL(7);
%ICV_OCT_CB_IMPL(8);
%ICV_OCT_CB_IMPL(9);

%ICV_OCT_CB_TAB;

// typemap to memorize the Octave callback when doing cvCreateTrackbar ()
%typemap(in) CvTrackbarCallback {
  if(my_trackbar_cb_size == ICV_OCT_MAX_CB){
    SWIG_exception(SWIG_IndexError, "Exceeded maximum number of trackbars");
  }

  my_trackbar_cb_size++;

  // memorize the Octave address of the callback function
  my_trackbar_cb_funcs[my_trackbar_cb_size-1].oct_func = (octave_value) $input;

  // prepare to call the C function who will register the callback
  $1 = my_trackbar_cb_funcs[ my_trackbar_cb_size-1 ].cv_func;
}


%include "../general/highgui.i"
%include "adapters.i"

