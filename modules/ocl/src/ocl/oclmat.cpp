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
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
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

#include "ocl.hpp"

namespace cv{

	namespace ocl{

		cl_context ocl_context;
		cl_command_queue ocl_cmd_queue;
		bool initialized = false;

		inline OclMat::OclMat() : rows(0), cols(0), step(0), data(0), refcount(0) {}

		inline OclMat::OclMat(int _rows, int _cols, int _type): flags(0), rows(0), cols(0), step(0), data(0), refcount(0){
		
			if(_rows > 0 && _cols > 0)
				create(_rows, _cols, _type);
		}

		inline OclMat::OclMat(Size size, int _type): flags(0), rows(0), cols(0), step(0), data(0), refcount(0){
			
			if(size.height > 0 && size.width > 0)
				create(size, _type);
		}

		inline OclMat::OclMat(const OclMat& m) 
		: flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount){
			
			if(refcount)
				CV_XADD(refcount, 1);
		}

		inline OclMat::OclMat(const Mat& m)
		: flags(0), rows(0), cols(0), step(0), data(0), refcount(0) { upload(m); }

		inline OclMat::~OclMat(){ release(); }

		void OclMat::_upload(size_t size, void* src){

			this->data = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, src, NULL);
		}

		void OclMat::_download(size_t size, void* dst){
		
			cl_int err = clEnqueueReadBuffer(ocl_cmd_queue, data, CL_TRUE, 0, size, dst, 0, NULL, NULL);
		}


		void OclMat::release(){
		
			if( refcount && CV_XADD(refcount, -1) == 1 )
			{
				free(refcount);
				clReleaseMemObject(data);
			}
			clReleaseMemObject(data);
			data = 0;
			step = rows = cols = 0;
			refcount = 0;
		}

		void OclMat::upload(const Mat& m){
		
			create(m.rows, m.cols, m.type());
			int ch = channels();
			int d = elemSize();
			size_t s = rows*cols*ch*d;
			if(initialized)
				this->_upload(s, m.data);
			else{
				init();
				this->_upload(s, m.data);
#ifdef _DEBUG
				printf("Context and Command queues not initialized. First call cv::ocl::init()");
#endif
			}
		}

		void OclMat::download(Mat& m){
		
			size_t s = rows*cols*channels()*elemSize();
			m.create(rows, cols, type());
			if(initialized){
					this->_download(s, m.data);
			}

			else{
				init();
				this->_download(s, m.data);
#ifdef _DEBUG
				printf("Context and Command queues not initialized. First call cv::ocl::init()");
#endif

				}

		}

		void init(){

			if(!initialized){
				cv::ocl::util::createContext(&ocl_context, &ocl_cmd_queue, false);
				initialized = true;
			}
		}

		void OclMat::create(int _rows, int _cols, int _type){
    
			if(!initialized)
				init();

			_type &= TYPE_MASK;
    
			if( rows == _rows && cols == _cols && type() == _type && data )
				return;
			if( data )
				release();
   
			if( _rows > 0 && _cols > 0 ){
				flags = Mat::MAGIC_VAL + _type;
				rows = _rows;
				cols = _cols;

       size_t esz = elemSize();
	   int ch = channels();

	   step = esz*cols*ch;

	   size_t size = esz*rows*cols*ch;
	   data = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, size, NULL, NULL);

	   if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        /*
        refcount = (int*)fastMalloc(sizeof(*refcount));
        *refcount = 1;
		*/
			}
		}

		void OclMat::create(Size size, int _type){
		
			return create(size.height, size.width, _type);
		}

		inline OclMat::operator Mat()
		{
			Mat m;
			download(m);
		    return m;
		}

		inline OclMat& OclMat::operator = (const OclMat& m)
		{
			if( this != &m )
			{
				if( m.refcount )
				CV_XADD(m.refcount, 1);
				release();
				flags = m.flags;
				rows = m.rows; cols = m.cols;
				step = m.step; data = m.data;
				data = m.data;
				refcount = m.refcount;
			}
			return *this;
		}

		inline OclMat::OclMat(int _rows, int _cols, int _type, const Scalar& _s)
		: flags(0), rows(0), cols(0), step(0), data(0), refcount(0)
		{
			if(_rows > 0 && _cols > 0)
			{
				create(_rows, _cols, _type);
				*this = _s;
			}
		}

		inline OclMat& OclMat::operator = (const Mat& m) { upload(m); return *this; }

		OclMat& OclMat::operator = (const Scalar& s)
		{
			setTo(s);
			return *this;
		}

		OclMat& OclMat::setTo(const Scalar& s){

			//if (s[0] == 0.0 && s[1] == 0.0 && s[2] == 0.0 && s[3] == 0.0)
			//{
			
				size_t sz = rows*cols*channels()*elemSize();
				//void* ptr = (void*)malloc(sz);
				//memset(ptr, s[0], sz);
				//clEnqueueWriteBuffer(ocl_cmd_queue, data, CL_TRUE, 0, sz, ptr, NULL, NULL, NULL);
				//free(ptr); ptr = 0;
				return *this;
			//}
		}

		inline size_t OclMat::elemSize() const{ return CV_ELEM_SIZE(flags); }
		inline size_t OclMat::elemSize1() const{ return CV_ELEM_SIZE1(flags); }
		inline int OclMat::type() const { return CV_MAT_TYPE(flags); }
		inline int OclMat::depth() const{ return CV_MAT_DEPTH(flags); }
		inline int OclMat::channels() const{ return CV_MAT_CN(flags); }
		inline size_t OclMat::step1() const{ return step/elemSize1(); }
		inline Size OclMat::size() const{ return Size(cols, rows); }
		inline bool OclMat::empty() const{ return data == 0; }
	}
}

	