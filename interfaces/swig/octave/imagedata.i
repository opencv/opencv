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

// 2006-08-29  Roman Stanchak -- converted to use CvMat rather than IplImage


%{

/// Accessor to convert a Octave string into the imageData.
void CvMat_imageData_set(CvMat * self, octave_value object)
{
  /*
  char* oct_string = OctString_AsString(object);
	int depth = CV_MAT_DEPTH(self->type);
	int cn = CV_MAT_CN(self->type);

	if (depth == CV_8U && cn==3){
		// RGB case
		// The data is reordered beause OpenCV uses BGR instead of RGB

		for (long line = 0; line < self->rows; ++line)
			for (long pixel = 0; pixel < self->cols; ++pixel)
			{
				// In OpenCV the beginning of the lines are aligned
				// to 4 Bytes. So use step instead of cols.
				long position = line*self->step + pixel*3;
				long sourcepos = line*self->cols*3 + pixel*3;
				self->data.ptr[position  ] = oct_string[sourcepos+2];
				self->data.ptr[position+1] = oct_string[sourcepos+1];
				self->data.ptr[position+2] = oct_string[sourcepos  ];
			}
	}
	else if (depth == CV_8U && cn==1)
	{
		// Grayscale 8bit case

		for (long line = 0; line < self->rows; ++line)
		{
			// In OpenCV the beginning of the lines are aligned
			// to 4 Bytes. So use step instead of cols.
			memcpy
				(
				 self->data.ptr + line*self->step,
				 oct_string + line*self->cols,
				 self->step
				);
		}
	}
	else if ( depth == CV_32F )
	{
		// float (32bit) case
		for (long line = 0; line < self->rows; ++line)
		{
			// here we don not have to care about alignment as the Floats are
			// as long as the alignment
			memcpy
				(
				 self->data.ptr + line*self->step,
				 oct_string + line*self->cols*sizeof(float),
				 self->step
				);
		}
	}
	else if ( depth == CV_64F )
	{
		// double (64bit) case
		for (long line = 0; line < self->rows; ++line)
		{
			// here we don not have to care about alignment as the Floats are
			// as long as the alignment
			memcpy
				(
				 self->data.ptr + line*self->step,
				 oct_string + line*self->cols*sizeof(double),
				 self->step
				);
		}
	}
	else
	{
	  // make some noise
	  SendErrorToOctave (SWIG_TypeError, 
                       "CvMat_imageData_set", 
                       "cannot convert string data to this image format",
                       __FILE__, __LINE__, NULL);
	}
  */
}

/// Accessor to convert the imageData into a Octave string.
octave_value CvMat_imageData_get(CvMat * self) 
{
  /*
	if (!self->data.ptr)
	{
		OctErr_SetString(OctExc_TypeError, "Data pointer of CvMat is NULL");
		return NULL;
	}		 
	return OctString_FromStringAndSize((const char *)self->data.ptr, self->rows*self->step);
  */
  return octave_value();
}

%}

// add virtual member variable to CvMat
%extend CvMat {
	octave_value imageData;
};
