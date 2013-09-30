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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_OPENCL_CORE_HPP__
#define __OPENCV_OPENCL_CORE_HPP__

#include "opencv2/core.hpp"

namespace cv { namespace cl {

class CV_EXPORTS Platform
{
    //enum {};    
    Platform();
    ~Platform();
    Platform(const Platform& p);
    Platform& operator = (const Platform& p);
    
    void* ptr() const;
    static Platform& getDefault();
protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS Context
{
    enum
    {
        DEVICE_TYPE_DEFAULT     = (1 << 0),
        DEVICE_TYPE_CPU         = (1 << 1),
        DEVICE_TYPE_GPU         = (1 << 2),
        DEVICE_TYPE_ACCELERATOR = (1 << 3),
        DEVICE_TYPE_ALL         = 0xFFFFFFFF
    };
    
    Context(const Platform& p, int ctype);
    ~Context();
    Context(const Context& c);
    Context& operator = (const Context& c);
    
    static Context& getDefault();
    
    template<typename T> T deviceProp(int prop) const { ... }
    
    void* ptr() const;
protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS Queue
{
public:
    Queue(const Context& ctx);
    ~Queue();
    Queue(const Queue& q);
    Queue& operator = (const Queue& q);
    
    void push(const Kernel& k, Size globSize, Size localSize);
    
    static Queue& getDefault();
    
    void finish();
    void* ptr() const;
    
protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS Kernel
{
public:
    Kernel(Program& prog, const char* kname, const char* buildopts, const Context& ctx);
    ~Kernel();
    Kernel(const Kernel& k);
    Kernel& operator = (const Kernel& k);
    
    void set(int i, const void* value, size_t vsize);
    template<typename _Tp> void set(int i, const _Tp& value) { set(i, &value, sizeof(value)); }    
    
    void run(Queue& q, )

    void* ptr() const;
protected:
    struct Impl;
    Impl* p;
};

class CV_EXPORTS KernelArgSetter
{
public:    
    KernelArgSetter(Kernel* k, int i) : kernel(k), idx(i) {}
    template<typename _Tp> KernelArgSetter operator , (const _Tp& value)
    { kernel->set(idx, value); return KernelArgSetter(k, idx+1); }
    
protected:
    Kernel* kernel;
    int idx;
};

inline KernelArgSetter operator << (Kernel& k, const T& value)
{
    k.set(0, value);
    return KernelArgSetter(&k, 1);
}

class CV_EXPORTS Program
{
    Program(const char* prog);
    ~Program();
    Program(const Program& prog);
    Program& operator = (const Program& prog);
    
    bool build(const char* buildopts, const Context& ctx);
    
    String getErrMsg() const;
    
    // non-constant method
    Kernel get(const char* name, const char* buildopts, const Context& ctx);
    
protected:
    struct Impl;
    Impl* p;
};


}}

#endif


