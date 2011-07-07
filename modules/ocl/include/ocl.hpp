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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
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

#ifndef OCL_HPP
#define OCL_HPP

#include "precomp.hpp"


//Root level namespace
namespace cv{

	//This namespace will contain all the source code for the OpenCL module
	namespace ocl{

		extern cl_context ocl_context;
		extern cl_command_queue ocl_cmd_queue;
	
		class OCL_EXPORTS OclMat{
		
		public:
			//! default constructor
			OclMat();
			
			//! constructs OclMat of the specified size and type (supported types are CV_8UC1, CV_8UC3, CV_16UC1, CV_16UC3, CV_64FC3 etc)
            OclMat(int rows, int cols, int type);
			OclMat(Size size, int type);
            //! constucts OclMat and fills it with the specified value _s.
            OclMat(int rows, int cols, int type, const Scalar& s);
            OclMat(Size size, int type, const Scalar& s);
            //! copy constructor
            OclMat(const OclMat& m);

			//! builds OclMat from Mat. Perfom blocking upload to device.
            explicit OclMat (const Mat& m);

            //! destructor - calls release()
            ~OclMat();

			//Releases the OpenCL context, command queue and the data buffer
			void release();

            //! assignment operators
            OclMat& operator = (const OclMat& m);
            //! assignment operator. Perfom blocking upload to device.
            OclMat& operator = (const Mat& m);

			//! sets every OclMatelement to s
            OclMat& operator = (const Scalar& s);

			//! sets some of the OclMat elements to s, according to the mask
            OclMat& setTo(const Scalar& s);

			//! pefroms blocking upload data to GpuMat.
            void upload(const cv::Mat& m);

            //! downloads data from device to host memory. Blocking calls.
            operator Mat();
            void download(cv::Mat& m);

			//! returns the size of element in bytes.
			size_t elemSize() const;
            //! returns the size of element channel in bytes.
            size_t elemSize1() const;
            //! returns element type, similar to CV_MAT_TYPE(cvMat->type)
            int type() const;
            //! returns element type, similar to CV_MAT_DEPTH(cvMat->type)
            int depth() const;
            //! returns element type, similar to CV_MAT_CN(cvMat->type)
            int channels() const;
            //! returns step/elemSize1()
            size_t step1() const;
            //! returns GpuMatrix size:
            // width == number of columns, height == number of rows
            Size size() const;
            //! returns true if OclMat data is NULL
            bool empty() const;

			int flags;

			 //! the number of rows and columns
            int rows, cols;
            //! a distance between successive rows in bytes; includes the gap if any
            size_t step;
            //! pointer to the data of type cl_mem
            cl_mem data;

			int* refcount;

			//! allocates new OclMat data unless the OclMat already has specified size and type.
            // previous data is unreferenced if needed.
            void create(int rows, int cols, int type);
            void create(Size size, int type);

			void _upload(size_t size, void* src);
			void _download(size_t size, void* dst);

		};

		//Creates the OpenCL context and command queue
		OCL_EXPORTS void init();

	}
}

#endif