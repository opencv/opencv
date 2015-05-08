/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CORE_UTILITY_H__
#define __OPENCV_CORE_UTILITY_H__

#ifndef __cplusplus
#  error utility.hpp header must be compiled as C++
#endif

#include "opencv2/core.hpp"

namespace cv
{

#ifdef CV_COLLECT_IMPL_DATA
CV_EXPORTS void setImpl(int flags); // set implementation flags and reset storage arrays
CV_EXPORTS void addImpl(int flag, const char* func = 0); // add implementation and function name to storage arrays
// Get stored implementation flags and fucntions names arrays
// Each implementation entry correspond to function name entry, so you can find which implementation was executed in which fucntion
CV_EXPORTS int getImpl(std::vector<int> &impl, std::vector<String> &funName);

CV_EXPORTS bool useCollection(); // return implementation collection state
CV_EXPORTS void setUseCollection(bool flag); // set implementation collection state

#define CV_IMPL_PLAIN  0x01 // native CPU OpenCV implementation
#define CV_IMPL_OCL    0x02 // OpenCL implementation
#define CV_IMPL_IPP    0x04 // IPP implementation
#define CV_IMPL_MT     0x10 // multithreaded implementation

#define CV_IMPL_ADD(impl)                                                   \
    if(cv::useCollection())                                                 \
    {                                                                       \
        cv::addImpl(impl, CV_Func);                                         \
    }
#else
#define CV_IMPL_ADD(impl)
#endif

//! @addtogroup core_utils
//! @{

/** @brief  Automatically Allocated Buffer Class

 The class is used for temporary buffers in functions and methods.
 If a temporary buffer is usually small (a few K's of memory),
 but its size depends on the parameters, it makes sense to create a small
 fixed-size array on stack and use it if it's large enough. If the required buffer size
 is larger than the fixed size, another buffer of sufficient size is allocated dynamically
 and released after the processing. Therefore, in typical cases, when the buffer size is small,
 there is no overhead associated with malloc()/free().
 At the same time, there is no limit on the size of processed data.

 This is what AutoBuffer does. The template takes 2 parameters - type of the buffer elements and
 the number of stack-allocated elements. Here is how the class is used:

 \code
 void my_func(const cv::Mat& m)
 {
    cv::AutoBuffer<float> buf; // create automatic buffer containing 1000 floats

    buf.allocate(m.rows); // if m.rows <= 1000, the pre-allocated buffer is used,
                          // otherwise the buffer of "m.rows" floats will be allocated
                          // dynamically and deallocated in cv::AutoBuffer destructor
    ...
 }
 \endcode
*/
template<typename _Tp, size_t fixed_size = 1024/sizeof(_Tp)+8> class AutoBuffer
{
public:
    typedef _Tp value_type;

    //! the default constructor
    AutoBuffer();
    //! constructor taking the real buffer size
    AutoBuffer(size_t _size);

    //! the copy constructor
    AutoBuffer(const AutoBuffer<_Tp, fixed_size>& buf);
    //! the assignment operator
    AutoBuffer<_Tp, fixed_size>& operator = (const AutoBuffer<_Tp, fixed_size>& buf);

    //! destructor. calls deallocate()
    ~AutoBuffer();

    //! allocates the new buffer of size _size. if the _size is small enough, stack-allocated buffer is used
    void allocate(size_t _size);
    //! deallocates the buffer if it was dynamically allocated
    void deallocate();
    //! resizes the buffer and preserves the content
    void resize(size_t _size);
    //! returns the current buffer size
    size_t size() const;
    //! returns pointer to the real buffer, stack-allocated or head-allocated
    operator _Tp* ();
    //! returns read-only pointer to the real buffer, stack-allocated or head-allocated
    operator const _Tp* () const;

protected:
    //! pointer to the real buffer, can point to buf if the buffer is small enough
    _Tp* ptr;
    //! size of the real buffer
    size_t sz;
    //! pre-allocated buffer. At least 1 element to confirm C++ standard reqirements
    _Tp buf[(fixed_size > 0) ? fixed_size : 1];
};

/**  @brief Sets/resets the break-on-error mode.

When the break-on-error mode is set, the default error handler issues a hardware exception, which
can make debugging more convenient.

\return the previous state
 */
CV_EXPORTS bool setBreakOnError(bool flag);

extern "C" typedef int (*ErrorCallback)( int status, const char* func_name,
                                       const char* err_msg, const char* file_name,
                                       int line, void* userdata );


/** @brief Sets the new error handler and the optional user data.

  The function sets the new error handler, called from cv::error().

  \param errCallback the new error handler. If NULL, the default error handler is used.
  \param userdata the optional user data pointer, passed to the callback.
  \param prevUserdata the optional output parameter where the previous user data pointer is stored

  \return the previous error handler
*/
CV_EXPORTS ErrorCallback redirectError( ErrorCallback errCallback, void* userdata=0, void** prevUserdata=0);

/** @brief Returns a text string formatted using the printf-like expression.

The function acts like sprintf but forms and returns an STL string. It can be used to form an error
message in the Exception constructor.
@param fmt printf-compatible formatting specifiers.
 */
CV_EXPORTS String format( const char* fmt, ... );
CV_EXPORTS String tempfile( const char* suffix = 0);
CV_EXPORTS void glob(String pattern, std::vector<String>& result, bool recursive = false);

/** @brief OpenCV will try to set the number of threads for the next parallel region.

If threads == 0, OpenCV will disable threading optimizations and run all it's functions
sequentially. Passing threads \< 0 will reset threads number to system default. This function must
be called outside of parallel region.

OpenCV will try to run it's functions with specified threads number, but some behaviour differs from
framework:
-   `TBB` – User-defined parallel constructions will run with the same threads number, if
    another does not specified. If late on user creates own scheduler, OpenCV will be use it.
-   `OpenMP` – No special defined behaviour.
-   `Concurrency` – If threads == 1, OpenCV will disable threading optimizations and run it's
    functions sequentially.
-   `GCD` – Supports only values \<= 0.
-   `C=` – No special defined behaviour.
@param nthreads Number of threads used by OpenCV.
@sa getNumThreads, getThreadNum
 */
CV_EXPORTS void setNumThreads(int nthreads);

/** @brief Returns the number of threads used by OpenCV for parallel regions.

Always returns 1 if OpenCV is built without threading support.

The exact meaning of return value depends on the threading framework used by OpenCV library:
- `TBB` – The number of threads, that OpenCV will try to use for parallel regions. If there is
  any tbb::thread_scheduler_init in user code conflicting with OpenCV, then function returns
  default number of threads used by TBB library.
- `OpenMP` – An upper bound on the number of threads that could be used to form a new team.
- `Concurrency` – The number of threads, that OpenCV will try to use for parallel regions.
- `GCD` – Unsupported; returns the GCD thread pool limit (512) for compatibility.
- `C=` – The number of threads, that OpenCV will try to use for parallel regions, if before
  called setNumThreads with threads \> 0, otherwise returns the number of logical CPUs,
  available for the process.
@sa setNumThreads, getThreadNum
 */
CV_EXPORTS int getNumThreads();

/** @brief Returns the index of the currently executed thread within the current parallel region. Always
returns 0 if called outside of parallel region.

The exact meaning of return value depends on the threading framework used by OpenCV library:
- `TBB` – Unsupported with current 4.1 TBB release. May be will be supported in future.
- `OpenMP` – The thread number, within the current team, of the calling thread.
- `Concurrency` – An ID for the virtual processor that the current context is executing on (0
  for master thread and unique number for others, but not necessary 1,2,3,...).
- `GCD` – System calling thread's ID. Never returns 0 inside parallel region.
- `C=` – The index of the current parallel task.
@sa setNumThreads, getNumThreads
 */
CV_EXPORTS int getThreadNum();

/** @brief Returns full configuration time cmake output.

Returned value is raw cmake output including version control system revision, compiler version,
compiler flags, enabled modules and third party libraries, etc. Output format depends on target
architecture.
 */
CV_EXPORTS_W const String& getBuildInformation();

/** @brief Returns the number of ticks.

The function returns the number of ticks after the certain event (for example, when the machine was
turned on). It can be used to initialize RNG or to measure a function execution time by reading the
tick count before and after the function call. See also the tick frequency.
 */
CV_EXPORTS_W int64 getTickCount();

/** @brief Returns the number of ticks per second.

The function returns the number of ticks per second. That is, the following code computes the
execution time in seconds:
@code
    double t = (double)getTickCount();
    // do something ...
    t = ((double)getTickCount() - t)/getTickFrequency();
@endcode
 */
CV_EXPORTS_W double getTickFrequency();

/** @brief Returns the number of CPU ticks.

The function returns the current number of CPU ticks on some architectures (such as x86, x64,
PowerPC). On other platforms the function is equivalent to getTickCount. It can also be used for
very accurate time measurements, as well as for RNG initialization. Note that in case of multi-CPU
systems a thread, from which getCPUTickCount is called, can be suspended and resumed at another CPU
with its own counter. So, theoretically (and practically) the subsequent calls to the function do
not necessary return the monotonously increasing values. Also, since a modern CPU varies the CPU
frequency depending on the load, the number of CPU clocks spent in some code cannot be directly
converted to time units. Therefore, getTickCount is generally a preferable solution for measuring
execution time.
 */
CV_EXPORTS_W int64 getCPUTickCount();

/** @brief Available CPU features.

remember to keep this list identical to the one in cvdef.h
*/
enum CpuFeatures {
    CPU_MMX             = 1,
    CPU_SSE             = 2,
    CPU_SSE2            = 3,
    CPU_SSE3            = 4,
    CPU_SSSE3           = 5,
    CPU_SSE4_1          = 6,
    CPU_SSE4_2          = 7,
    CPU_POPCNT          = 8,

    CPU_AVX             = 10,
    CPU_AVX2            = 11,
    CPU_FMA3            = 12,

    CPU_AVX_512F        = 13,
    CPU_AVX_512BW       = 14,
    CPU_AVX_512CD       = 15,
    CPU_AVX_512DQ       = 16,
    CPU_AVX_512ER       = 17,
    CPU_AVX_512IFMA512  = 18,
    CPU_AVX_512PF       = 19,
    CPU_AVX_512VBMI     = 20,
    CPU_AVX_512VL       = 21,

    CPU_NEON            = 100
};

/** @brief Returns true if the specified feature is supported by the host hardware.

The function returns true if the host hardware supports the specified feature. When user calls
setUseOptimized(false), the subsequent calls to checkHardwareSupport() will return false until
setUseOptimized(true) is called. This way user can dynamically switch on and off the optimized code
in OpenCV.
@param feature The feature of interest, one of cv::CpuFeatures
 */
CV_EXPORTS_W bool checkHardwareSupport(int feature);

/** @brief Returns the number of logical CPUs available for the process.
 */
CV_EXPORTS_W int getNumberOfCPUs();


/** @brief Aligns a pointer to the specified number of bytes.

The function returns the aligned pointer of the same type as the input pointer:
\f[\texttt{(\_Tp*)(((size\_t)ptr + n-1) \& -n)}\f]
@param ptr Aligned pointer.
@param n Alignment size that must be a power of two.
 */
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

/** @brief Aligns a buffer size to the specified number of bytes.

The function returns the minimum number that is greater or equal to sz and is divisible by n :
\f[\texttt{(sz + n-1) \& -n}\f]
@param sz Buffer size to align.
@param n Alignment size that must be a power of two.
 */
static inline size_t alignSize(size_t sz, int n)
{
    CV_DbgAssert((n & (n - 1)) == 0); // n is a power of 2
    return (sz + n-1) & -n;
}

/** @brief Enables or disables the optimized code.

The function can be used to dynamically turn on and off optimized code (code that uses SSE2, AVX,
and other instructions on the platforms that support it). It sets a global flag that is further
checked by OpenCV functions. Since the flag is not checked in the inner OpenCV loops, it is only
safe to call the function on the very top level in your application where you can be sure that no
other OpenCV function is currently executed.

By default, the optimized code is enabled unless you disable it in CMake. The current status can be
retrieved using useOptimized.
@param onoff The boolean flag specifying whether the optimized code should be used (onoff=true)
or not (onoff=false).
 */
CV_EXPORTS_W void setUseOptimized(bool onoff);

/** @brief Returns the status of optimized code usage.

The function returns true if the optimized code is enabled. Otherwise, it returns false.
 */
CV_EXPORTS_W bool useOptimized();

static inline size_t getElemSize(int type) { return CV_ELEM_SIZE(type); }

/////////////////////////////// Parallel Primitives //////////////////////////////////

/** @brief Base class for parallel data processors
*/
class CV_EXPORTS ParallelLoopBody
{
public:
    virtual ~ParallelLoopBody();
    virtual void operator() (const Range& range) const = 0;
};

/** @brief Parallel data processor
*/
CV_EXPORTS void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes=-1.);

/////////////////////////////// forEach method of cv::Mat ////////////////////////////
template<typename _Tp, typename Functor> inline
void Mat::forEach_impl(const Functor& operation) {
    if (false) {
        operation(*reinterpret_cast<_Tp*>(0), reinterpret_cast<int*>(NULL));
        // If your compiler fail in this line.
        // Please check that your functor signature is
        //     (_Tp&, const int*)   <- multidimential
        //  or (_Tp&, void*)        <- in case of you don't need current idx.
    }

    CV_Assert(this->total() / this->size[this->dims - 1] <= INT_MAX);
    const int LINES = static_cast<int>(this->total() / this->size[this->dims - 1]);

    class PixelOperationWrapper :public ParallelLoopBody
    {
    public:
        PixelOperationWrapper(Mat_<_Tp>* const frame, const Functor& _operation)
            : mat(frame), op(_operation) {};
        virtual ~PixelOperationWrapper(){};
        // ! Overloaded virtual operator
        // convert range call to row call.
        virtual void operator()(const Range &range) const {
            const int DIMS = mat->dims;
            const int COLS = mat->size[DIMS - 1];
            if (DIMS <= 2) {
                for (int row = range.start; row < range.end; ++row) {
                    this->rowCall2(row, COLS);
                }
            } else {
                std::vector<int> idx(COLS); /// idx is modified in this->rowCall
                idx[DIMS - 2] = range.start - 1;

                for (int line_num = range.start; line_num < range.end; ++line_num) {
                    idx[DIMS - 2]++;
                    for (int i = DIMS - 2; i >= 0; --i) {
                        if (idx[i] >= mat->size[i]) {
                            idx[i - 1] += idx[i] / mat->size[i];
                            idx[i] %= mat->size[i];
                            continue; // carry-over;
                        }
                        else {
                            break;
                        }
                    }
                    this->rowCall(&idx[0], COLS, DIMS);
                }
            }
        };
    private:
        Mat_<_Tp>* const mat;
        const Functor op;
        // ! Call operator for each elements in this row.
        inline void rowCall(int* const idx, const int COLS, const int DIMS) const {
            int &col = idx[DIMS - 1];
            col = 0;
            _Tp* pixel = &(mat->template at<_Tp>(idx));

            while (col < COLS) {
                op(*pixel, const_cast<const int*>(idx));
                pixel++; col++;
            }
            col = 0;
        }
        // ! Call operator for each elements in this row. 2d mat special version.
        inline void rowCall2(const int row, const int COLS) const {
            union Index{
                int body[2];
                operator const int*() const {
                    return reinterpret_cast<const int*>(this);
                }
                int& operator[](const int i) {
                    return body[i];
                }
            } idx = {{row, 0}};
            // Special union is needed to avoid
            // "error: array subscript is above array bounds [-Werror=array-bounds]"
            // when call the functor `op` such that access idx[3].

            _Tp* pixel = &(mat->template at<_Tp>(idx));
            const _Tp* const pixel_end = pixel + COLS;
            while(pixel < pixel_end) {
                op(*pixel++, static_cast<const int*>(idx));
                idx[1]++;
            }
        };
        PixelOperationWrapper& operator=(const PixelOperationWrapper &) {
            CV_Assert(false);
            // We can not remove this implementation because Visual Studio warning C4822.
            return *this;
        };
    };

    parallel_for_(cv::Range(0, LINES), PixelOperationWrapper(reinterpret_cast<Mat_<_Tp>*>(this), operation));
}

/////////////////////////// Synchronization Primitives ///////////////////////////////

class CV_EXPORTS Mutex
{
public:
    Mutex();
    ~Mutex();
    Mutex(const Mutex& m);
    Mutex& operator = (const Mutex& m);

    void lock();
    bool trylock();
    void unlock();

    struct Impl;
protected:
    Impl* impl;
};

class CV_EXPORTS AutoLock
{
public:
    AutoLock(Mutex& m) : mutex(&m) { mutex->lock(); }
    ~AutoLock() { mutex->unlock(); }
protected:
    Mutex* mutex;
private:
    AutoLock(const AutoLock&);
    AutoLock& operator = (const AutoLock&);
};

class CV_EXPORTS TLSDataContainer
{
private:
    int key_;
protected:
    TLSDataContainer();
    virtual ~TLSDataContainer();
public:
    virtual void* createDataInstance() const = 0;
    virtual void deleteDataInstance(void* data) const = 0;

    void* getData() const;
};

template <typename T>
class TLSData : protected TLSDataContainer
{
public:
    inline TLSData() {}
    inline ~TLSData() {}
    inline T* get() const { return (T*)getData(); }
private:
    virtual void* createDataInstance() const { return new T; }
    virtual void deleteDataInstance(void* data) const { delete (T*)data; }
};

/** @brief Designed for command line parsing

The sample below demonstrates how to use CommandLineParser:
@code
    CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int N = parser.get<int>("N");
    double fps = parser.get<double>("fps");
    String path = parser.get<String>("path");

    use_time_stamp = parser.has("timestamp");

    String img1 = parser.get<String>(0);
    String img2 = parser.get<String>(1);

    int repeat = parser.get<int>(2);

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
@endcode

### Keys syntax

The keys parameter is a string containing several blocks, each one is enclosed in curley braces and
describes one argument. Each argument contains three parts separated by the `|` symbol:

-# argument names is a space-separated list of option synonyms (to mark argument as positional, prefix it with the `@` symbol)
-# default value will be used if the argument was not provided (can be empty)
-# help message (can be empty)

For example:

@code{.cpp}
    const String keys =
        "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1 for compare   }"
        "{@image2        |      | image2 for compare   }"
        "{@repeat        |1     | number               }"
        "{path           |.     | path to file         }"
        "{fps            | -1.0 | fps for output video }"
        "{N count        |100   | count of objects     }"
        "{ts timestamp   |      | use time stamp       }"
        ;
}
@endcode

### Usage

For the described keys:

@code{.sh}
    # Good call (3 positional parameters: image1, image2 and repeat; N is 200, ts is true)
    $ ./app -N=200 1.png 2.jpg 19 -ts

    # Bad call
    $ ./app -fps=aaa
    ERRORS:
    Exception: can not convert: [aaa] to [double]
@endcode
 */
class CV_EXPORTS CommandLineParser
{
public:

    /** @brief Constructor

    Initializes command line parser object

    @param argc number of command line arguments (from main())
    @param argv array of command line arguments (from main())
    @param keys string describing acceptable command line parameters (see class description for syntax)
    */
    CommandLineParser(int argc, const char* const argv[], const String& keys);

    /** @brief Copy constructor */
    CommandLineParser(const CommandLineParser& parser);

    /** @brief Assignment operator */
    CommandLineParser& operator = (const CommandLineParser& parser);

    /** @brief Destructor */
    ~CommandLineParser();

    /** @brief Returns application path

    This method returns the path to the executable from the command line (`argv[0]`).

    For example, if the application has been started with such command:
    @code{.sh}
    $ ./bin/my-executable
    @endcode
    this method will return `./bin`.
    */
    String getPathToApplication() const;

    /** @brief Access arguments by name

    Returns argument converted to selected type. If the argument is not known or can not be
    converted to selected type, the error flag is set (can be checked with @ref check).

    For example, define:
    @code{.cpp}
    String keys = "{N count||}";
    @endcode

    Call:
    @code{.sh}
    $ ./my-app -N=20
    # or
    $ ./my-app --count=20
    @endcode

    Access:
    @code{.cpp}
    int N = parser.get<int>("N");
    @endcode

    @param name name of the argument
    @param space_delete remove spaces from the left and right of the string
    @tparam T the argument will be converted to this type if possible

    @note You can access positional arguments by their `@`-prefixed name:
    @code{.cpp}
    parser.get<String>("@image");
    @endcode
     */
    template <typename T>
    T get(const String& name, bool space_delete = true) const
    {
        T val = T();
        getByName(name, space_delete, ParamType<T>::type, (void*)&val);
        return val;
    }

    /** @brief Access positional arguments by index

    Returns argument converted to selected type. Indexes are counted from zero.

    For example, define:
    @code{.cpp}
    String keys = "{@arg1||}{@arg2||}"
    @endcode

    Call:
    @code{.sh}
    ./my-app abc qwe
    @endcode

    Access arguments:
    @code{.cpp}
    String val_1 = parser.get<String>(0); // returns "abc", arg1
    String val_2 = parser.get<String>(1); // returns "qwe", arg2
    @endcode

    @param index index of the argument
    @param space_delete remove spaces from the left and right of the string
    @tparam T the argument will be converted to this type if possible
     */
    template <typename T>
    T get(int index, bool space_delete = true) const
    {
        T val = T();
        getByIndex(index, space_delete, ParamType<T>::type, (void*)&val);
        return val;
    }

    /** @brief Check if field was provided in the command line

    @param name argument name to check
    */
    bool has(const String& name) const;

    /** @brief Check for parsing errors

    Returns true if error occured while accessing the parameters (bad conversion, missing arguments,
    etc.). Call @ref printErrors to print error messages list.
     */
    bool check() const;

    /** @brief Set the about message

    The about message will be shown when @ref printMessage is called, right before arguments table.
     */
    void about(const String& message);

    /** @brief Print help message

    This method will print standard help message containing the about message and arguments description.

    @sa about
    */
    void printMessage() const;

    /** @brief Print list of errors occured

    @sa check
    */
    void printErrors() const;

protected:
    void getByName(const String& name, bool space_delete, int type, void* dst) const;
    void getByIndex(int index, bool space_delete, int type, void* dst) const;

    struct Impl;
    Impl* impl;
};

//! @} core_utils

//! @cond IGNORED

/////////////////////////////// AutoBuffer implementation ////////////////////////////////////////

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer()
{
    ptr = buf;
    sz = fixed_size;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer(size_t _size)
{
    ptr = buf;
    sz = fixed_size;
    allocate(_size);
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::AutoBuffer(const AutoBuffer<_Tp, fixed_size>& abuf )
{
    ptr = buf;
    sz = fixed_size;
    allocate(abuf.size());
    for( size_t i = 0; i < sz; i++ )
        ptr[i] = abuf.ptr[i];
}

template<typename _Tp, size_t fixed_size> inline AutoBuffer<_Tp, fixed_size>&
AutoBuffer<_Tp, fixed_size>::operator = (const AutoBuffer<_Tp, fixed_size>& abuf)
{
    if( this != &abuf )
    {
        deallocate();
        allocate(abuf.size());
        for( size_t i = 0; i < sz; i++ )
            ptr[i] = abuf.ptr[i];
    }
    return *this;
}

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::~AutoBuffer()
{ deallocate(); }

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::allocate(size_t _size)
{
    if(_size <= sz)
    {
        sz = _size;
        return;
    }
    deallocate();
    if(_size > fixed_size)
    {
        ptr = new _Tp[_size];
        sz = _size;
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::deallocate()
{
    if( ptr != buf )
    {
        delete[] ptr;
        ptr = buf;
        sz = fixed_size;
    }
}

template<typename _Tp, size_t fixed_size> inline void
AutoBuffer<_Tp, fixed_size>::resize(size_t _size)
{
    if(_size <= sz)
    {
        sz = _size;
        return;
    }
    size_t i, prevsize = sz, minsize = MIN(prevsize, _size);
    _Tp* prevptr = ptr;

    ptr = _size > fixed_size ? new _Tp[_size] : buf;
    sz = _size;

    if( ptr != prevptr )
        for( i = 0; i < minsize; i++ )
            ptr[i] = prevptr[i];
    for( i = prevsize; i < _size; i++ )
        ptr[i] = _Tp();

    if( prevptr != buf )
        delete[] prevptr;
}

template<typename _Tp, size_t fixed_size> inline size_t
AutoBuffer<_Tp, fixed_size>::size() const
{ return sz; }

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::operator _Tp* ()
{ return ptr; }

template<typename _Tp, size_t fixed_size> inline
AutoBuffer<_Tp, fixed_size>::operator const _Tp* () const
{ return ptr; }

#ifndef OPENCV_NOSTL
template<> inline std::string CommandLineParser::get<std::string>(int index, bool space_delete) const
{
    return get<String>(index, space_delete);
}
template<> inline std::string CommandLineParser::get<std::string>(const String& name, bool space_delete) const
{
    return get<String>(name, space_delete);
}
#endif // OPENCV_NOSTL

//! @endcond

} //namespace cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/core/core_c.h"
#endif

#endif //__OPENCV_CORE_UTILITY_H__
