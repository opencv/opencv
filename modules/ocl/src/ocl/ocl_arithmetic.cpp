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


#include "ocl.hpp"
#include <iostream>
#include <fstream>
using namespace std;

namespace cv{
	namespace ocl{

	void writeBinaries(cl_program cpProgram)
	{
        ofstream myfile("kernel.ptx");

        cl_uint program_num_devices = 1;
        clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &program_num_devices, NULL);

		if (program_num_devices == 0)
			return;

        size_t binaries_sizes[1];

        clGetProgramInfo(cpProgram,   CL_PROGRAM_BINARY_SIZES, program_num_devices*sizeof(size_t), binaries_sizes, NULL);

	   char **binaries = new char*[1];
		binaries[0] = 0;

        for (size_t i = 0; i < 1; i++)
                        binaries[i] = new char[binaries_sizes[i]+1];


        clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, program_num_devices*sizeof(size_t), binaries, NULL);
        
        if(myfile.is_open())
        {
                for (size_t i = 0; i < program_num_devices; i++)
                {
                                myfile << binaries[i];
                }
        }
        myfile.close();

        for (size_t i = 0; i < program_num_devices; i++)
                        delete [] binaries;

        delete [] binaries;
}

	OCL_EXPORTS void add(const OclMat& a, const OclMat& b, OclMat& sum){
		
			if(a.rows != b.rows || a.cols != b.cols)
				return;

			int size = a.rows*a.cols;

			cl_program program;
			cl_kernel kernel;
			size_t global_work_size[1];
			size_t local_work_size[1];
			cl_uint size_ret = 0;
			cl_int err;

			const char* add_kernel_source[] =   {\
				"__kernel void add_kernel (__global const uchar *a, __global const uchar* b, __global uchar* c)"\
				"{"\
				"int tid = get_global_id(0);"\
				"c[tid] = a[tid] + b[tid];"\
				"}"
    };

			program = clCreateProgramWithSource(ocl_context, 1, (const char**)&add_kernel_source, NULL, NULL);
///////////////////
			writeBinaries(program);
///////////////////


			//err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	/*		FILE* fp = fopen("kernel.ptx", "r");
			fseek (fp , 0 , SEEK_END);
			const size_t lSize = ftell(fp);
			rewind(fp);
			unsigned char* buffer;
			buffer = (unsigned char*) malloc (lSize);
			fread(buffer, 1, lSize, fp);
			fclose(fp);

			size_t cb;
			err = clGetContextInfo(ocl_context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
			cl_device_id *devices = (cl_device_id*)malloc(cb);
			clGetContextInfo(ocl_context, CL_CONTEXT_DEVICES, cb, devices, NULL);

			cl_int status;
			program = clCreateProgramWithBinary(ocl_context, 1, devices, 
                                &lSize, (const unsigned char**)&buffer, 
                                &status, &err);
        
			err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#ifdef _DEBUG
			if(err != CL_SUCCESS){
				printf("(Error code: %d)Build failed, check the program source...\n",err);
				return;
			}
#endif

			//writeBinaries(program);
			
			kernel = clCreateKernel(program, "add_kernel", NULL);

			err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &a.data);
			err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &a.data);
			err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &a.data);
#ifdef _DEBUG
			if(err != CL_SUCCESS){
				printf("(Error code: %d)Failed at setting kernel arguments...\n",err);
			return;
			}
#endif

			global_work_size[0] = size;
			local_work_size[0]= 1;

			err = clEnqueueNDRangeKernel(ocl_cmd_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
#ifdef _DEBUG
			if(err != CL_SUCCESS){
				printf("(Error code: %d)Kernel execution failed...\n",err);
			return;
			}
#endif

			clReleaseKernel(kernel);
			clReleaseProgram(program);*/

		}
	}
}