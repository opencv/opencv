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

#ifndef __OPENCV_OPENCL_HPP__
#define __OPENCV_OPENCL_HPP__

#include "opencv2/core.hpp"

namespace cv { namespace ocl {

CV_EXPORTS bool haveOpenCL();
CV_EXPORTS bool useOpenCL();
CV_EXPORTS void setUseOpenCL(bool flag);
CV_EXPORTS void finish();

class CV_EXPORTS Platform
{
public:
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


class CV_EXPORTS Queue;

class CV_EXPORTS Context
{
public:
    enum
    {
        DEVICE_TYPE_DEFAULT     = (1 << 0),
        DEVICE_TYPE_CPU         = (1 << 1),
        DEVICE_TYPE_GPU         = (1 << 2),
        DEVICE_TYPE_ACCELERATOR = (1 << 3),
        DEVICE_TYPE_ALL         = 0xFFFFFFFF
    };

    Context();
    Context(const Platform& p, int dtype);
    ~Context();
    Context(const Context& c);
    Context& operator = (const Context& c);

    bool create(const Platform& p, int dtype);
    size_t ndevices() const;
    void* device(size_t idx) const;
    
    static Context& getDefault();
    static Queue& getDefaultQueue();
    
    template<typename _Tp> _Tp deviceProp(void* d, int prop) const
    { _Tp val; deviceProp(d, prop, &val, sizeof(val)); return val; }
    
    void* ptr() const;
protected:
    void deviceProp(void* d, int prop, void* val, size_t sz);

    struct Impl;
    Impl* p;
};


class CV_EXPORTS Queue
{
public:
    Queue();
    Queue(const Context& ctx);
    ~Queue();
    Queue(const Queue& q);
    Queue& operator = (const Queue& q);
    
    bool create(const Context& ctx);
    void finish();
    void* ptr() const;
    
protected:
    struct Impl;
    Impl* p;
};


class CV_EXPORTS Buffer
{
public:
    enum
    {
        MEM_READ_WRITE=(1 << 0),
        MEM_WRITE_ONLY=(1 << 1),
        MEM_READ_ONLY=(1 << 2),
        MEM_USE_HOST_PTR=(1 << 3),
        MEM_ALLOC_HOST_PTR=(1 << 4),
        MEM_COPY_HOST_PTR=(1 << 5),

        MAP_READ=(1 << 0),
        MAP_WRITE=(1 << 1),
        MAP_WRITE_INVALIDATE_REGION=(1 << 2)
    };

    static void* create(Context& ctx, int flags, size_t size, void* hostptr);
    static void release(void* handle);
    static void retain(void* handle);

    static void read(Queue& q, void* handle, size_t offset, size_t size, void* dst, bool async);
    static void read(Queue& q, void* handle, size_t offset[3], size_t size[3], size_t step[2],
                     void* dst, size_t dststep[2], bool async);

    static void write(Queue& q, void* handle, size_t offset, size_t size, const void* src, bool async);
    static void write(Queue& q, void* handle, size_t offset[3], size_t size[3], size_t step[2],
                     const void* src, size_t srcstep[2], bool async);

    static void fill(Queue& q, void* handle, const void* pattern,
                     size_t pattern_size, size_t offset, size_t size, bool async);

    static void copy(Queue& q, void* srchandle, size_t srcoffset, void* dsthandle, size_t dstoffset, size_t size, bool async);
    static void copy(Queue& q, void* srchandle, size_t srcoffset[3], size_t srcstep[2],
                     void* dsthandle, size_t dstoffset[3], size_t dststep[2],
                     size_t size[3], bool async);

    static void* map(Queue& q, void* handle, int mapflags, size_t offset, size_t size, bool async);
    static void unmap(Queue& q, void* ptr, bool async);
};


class CV_EXPORTS KernelArg
{
public:
    enum { LOCAL=1, READ_ONLY=2, WRITE_ONLY=4, READ_WRITE=6, CONSTANT=8 };
    KernelArg(int _flags, UMat* _m, void* _obj=0, size_t _sz=0);

    static KernelArg Local() { return KernelArg(LOCAL, 0); }
    static KernelArg ReadOnly(const UMat& m) { return KernelArg(READ_ONLY, (UMat*)&m); }
    static KernelArg WriteOnly(const UMat& m) { return KernelArg(WRITE_ONLY, (UMat*)&m); }
    static KernelArg Constant(const Mat& m);
    template<typename _Tp> static KernelArg Constant(const _Tp* arr, size_t n)
    { return KernelArg(CONSTANT, 0, (void*)arr, n); }

    int flags;
    UMat* m;
    void* obj;
    size_t sz;
};

class CV_EXPORTS Program;

class CV_EXPORTS Kernel
{
public:
    class CV_EXPORTS Callback
    {
    public:
        virtual ~Callback() {}
        virtual void operator()() = 0;
    };

    Kernel();
    Kernel(const Context& ctx, Program& prog, const char* kname, const char* buildopts);
    ~Kernel();
    Kernel(const Kernel& k);
    Kernel& operator = (const Kernel& k);

    void create(const Context& ctx, Program& prog, const char* kname, const char* buildopts);

    int set(int i, const void* value, size_t sz);
    int set(int i, const UMat& m);
    int set(int i, const KernelArg& arg);
    template<typename _Tp> int set(int i, const _Tp& value) { return set(i, &value, sizeof(value)); }

    void run(Queue& q, int dims, size_t offset[], size_t globalsize[], size_t localsize[],
             bool async, const Ptr<Callback>& cleanupCallback=Ptr<Callback>());
    void runTask(Queue& q, bool async, const Ptr<Callback>& cleanupCallback=Ptr<Callback>());

    void* ptr() const;

    struct Impl;

protected:
    Impl* p;
};


class CV_EXPORTS KernelArgSetter
{
public:    
    KernelArgSetter(Kernel* k, int i) : kernel(k), idx(i) {}
    template<typename _Tp> KernelArgSetter operator , (const _Tp& value)
    { return KernelArgSetter(kernel, kernel->set(idx, value)); }
    
protected:
    Kernel* kernel;
    int idx;
};

template<typename _Tp> inline KernelArgSetter operator << (Kernel& k, const _Tp& value)
{
    return KernelArgSetter(&k, k.set(0, value));
}

class CV_EXPORTS Program
{
public:
    Program(const char* prog);
    ~Program();
    Program(const Program& prog);
    Program& operator = (const Program& prog);
    
    void* build(const Context& ctx, const char* buildopts);
    
    String getErrMsg() const;
    
protected:
    struct Impl;
    Impl* p;
};

}}

#endif
