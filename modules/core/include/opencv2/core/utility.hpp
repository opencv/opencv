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

#ifndef OPENCV_CORE_UTILITY_H
#define OPENCV_CORE_UTILITY_H

#ifndef __cplusplus
#  error utility.hpp header must be compiled as C++
#endif

#if defined(check)
#  warning Detected Apple 'check' macro definition, it can cause build conflicts. Please, include this header before any Apple headers.
#endif

#include "opencv2/core.hpp"
#include <ostream>

#include <functional>

#if !defined(_M_CEE)
#include <mutex>  // std::mutex, std::lock_guard
#endif

namespace cv
{

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
    cv::AutoBuffer<float> buf(1000); // create automatic buffer containing 1000 floats

    buf.allocate(m.rows); // if m.rows <= 1000, the pre-allocated buffer is used,
                          // otherwise the buffer of "m.rows" floats will be allocated
                          // dynamically and deallocated in cv::AutoBuffer destructor
    ...
 }
 \endcode
*/
#ifdef OPENCV_ENABLE_MEMORY_SANITIZER
template<typename _Tp, size_t fixed_size = 0> class AutoBuffer
#else
template<typename _Tp, size_t fixed_size = 1024/sizeof(_Tp)+8> class AutoBuffer
#endif
{
public:
    typedef _Tp value_type;

    //! the default constructor
    AutoBuffer();
    //! constructor taking the real buffer size
    explicit AutoBuffer(size_t _size);

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
    //! returns pointer to the real buffer, stack-allocated or heap-allocated
    inline _Tp* data() { return ptr; }
    //! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    inline const _Tp* data() const { return ptr; }

#if !defined(OPENCV_DISABLE_DEPRECATED_COMPATIBILITY) // use to .data() calls instead
    //! returns pointer to the real buffer, stack-allocated or heap-allocated
    operator _Tp* () { return ptr; }
    //! returns read-only pointer to the real buffer, stack-allocated or heap-allocated
    operator const _Tp* () const { return ptr; }
#else
    //! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
    inline _Tp& operator[] (size_t i) { CV_DbgCheckLT(i, sz, "out of range"); return ptr[i]; }
    //! returns a reference to the element at specified location. No bounds checking is performed in Release builds.
    inline const _Tp& operator[] (size_t i) const { CV_DbgCheckLT(i, sz, "out of range"); return ptr[i]; }
#endif

protected:
    //! pointer to the real buffer, can point to buf if the buffer is small enough
    _Tp* ptr;
    //! size of the real buffer
    size_t sz;
    //! pre-allocated buffer. At least 1 element to confirm C++ standard requirements
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

CV_EXPORTS String tempfile( const char* suffix = 0);
CV_EXPORTS void glob(String pattern, std::vector<String>& result, bool recursive = false);

/** @brief OpenCV will try to set the number of threads for subsequent parallel regions.

If threads == 1, OpenCV will disable threading optimizations and run all it's functions
sequentially. Passing threads \< 0 will reset threads number to system default.
The function is not thread-safe. It must not be called in parallel region or concurrent threads.

OpenCV will try to run its functions with specified threads number, but some behaviour differs from
framework:
-   `TBB` - User-defined parallel constructions will run with the same threads number, if
    another is not specified. If later on user creates his own scheduler, OpenCV will use it.
-   `OpenMP` - No special defined behaviour.
-   `Concurrency` - If threads == 1, OpenCV will disable threading optimizations and run its
    functions sequentially.
-   `GCD` - Supports only values \<= 0.
-   `C=` - No special defined behaviour.
@param nthreads Number of threads used by OpenCV.
@sa getNumThreads, getThreadNum
 */
CV_EXPORTS_W void setNumThreads(int nthreads);

/** @brief Returns the number of threads used by OpenCV for parallel regions.

Always returns 1 if OpenCV is built without threading support.

The exact meaning of return value depends on the threading framework used by OpenCV library:
- `TBB` - The number of threads, that OpenCV will try to use for parallel regions. If there is
  any tbb::thread_scheduler_init in user code conflicting with OpenCV, then function returns
  default number of threads used by TBB library.
- `OpenMP` - An upper bound on the number of threads that could be used to form a new team.
- `Concurrency` - The number of threads, that OpenCV will try to use for parallel regions.
- `GCD` - Unsupported; returns the GCD thread pool limit (512) for compatibility.
- `C=` - The number of threads, that OpenCV will try to use for parallel regions, if before
  called setNumThreads with threads \> 0, otherwise returns the number of logical CPUs,
  available for the process.
@sa setNumThreads, getThreadNum
 */
CV_EXPORTS_W int getNumThreads();

/** @brief Returns the index of the currently executed thread within the current parallel region. Always
returns 0 if called outside of parallel region.

@deprecated Current implementation doesn't corresponding to this documentation.

The exact meaning of the return value depends on the threading framework used by OpenCV library:
- `TBB` - Unsupported with current 4.1 TBB release. Maybe will be supported in future.
- `OpenMP` - The thread number, within the current team, of the calling thread.
- `Concurrency` - An ID for the virtual processor that the current context is executing on (0
  for master thread and unique number for others, but not necessary 1,2,3,...).
- `GCD` - System calling thread's ID. Never returns 0 inside parallel region.
- `C=` - The index of the current parallel task.
@sa setNumThreads, getNumThreads
 */
CV_EXPORTS_W int getThreadNum();

/** @brief Returns full configuration time cmake output.

Returned value is raw cmake output including version control system revision, compiler version,
compiler flags, enabled modules and third party libraries, etc. Output format depends on target
architecture.
 */
CV_EXPORTS_W const String& getBuildInformation();

/** @brief Returns library version string

For example "3.4.1-dev".

@sa getMajorVersion, getMinorVersion, getRevisionVersion
*/
CV_EXPORTS_W String getVersionString();

/** @brief Returns major library version */
CV_EXPORTS_W int getVersionMajor();

/** @brief Returns minor library version */
CV_EXPORTS_W int getVersionMinor();

/** @brief Returns revision field of the library version */
CV_EXPORTS_W int getVersionRevision();

/** @brief Returns the number of ticks.

The function returns the number of ticks after the certain event (for example, when the machine was
turned on). It can be used to initialize RNG or to measure a function execution time by reading the
tick count before and after the function call.
@sa getTickFrequency, TickMeter
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
@sa getTickCount, TickMeter
 */
CV_EXPORTS_W double getTickFrequency();

/** @brief a Class to measure passing time.

The class computes passing time by counting the number of ticks per second. That is, the following code computes the
execution time in seconds:
@snippet snippets/core_various.cpp TickMeter_total

It is also possible to compute the average time over multiple runs:
@snippet snippets/core_various.cpp TickMeter_average

@sa getTickCount, getTickFrequency
*/
class CV_EXPORTS_W TickMeter
{
public:
    //! the default constructor
    CV_WRAP TickMeter()
    {
        reset();
    }

    //! starts counting ticks.
    CV_WRAP void start()
    {
        startTime = cv::getTickCount();
    }

    //! stops counting ticks.
    CV_WRAP void stop()
    {
        int64 time = cv::getTickCount();
        if (startTime == 0)
            return;
        ++counter;
        sumTime += (time - startTime);
        startTime = 0;
    }

    //! returns counted ticks.
    CV_WRAP int64 getTimeTicks() const
    {
        return sumTime;
    }

    //! returns passed time in microseconds.
    CV_WRAP double getTimeMicro() const
    {
        return getTimeMilli()*1e3;
    }

    //! returns passed time in milliseconds.
    CV_WRAP double getTimeMilli() const
    {
        return getTimeSec()*1e3;
    }

    //! returns passed time in seconds.
    CV_WRAP double getTimeSec()   const
    {
        return (double)getTimeTicks() / getTickFrequency();
    }

    //! returns internal counter value.
    CV_WRAP int64 getCounter() const
    {
        return counter;
    }

    //! returns average FPS (frames per second) value.
    CV_WRAP double getFPS() const
    {
        const double sec = getTimeSec();
        if (sec < DBL_EPSILON)
            return 0.;
        return counter / sec;
    }

    //! returns average time in seconds
    CV_WRAP double getAvgTimeSec() const
    {
        if (counter <= 0)
            return 0.;
        return getTimeSec() / counter;
    }

    //! returns average time in milliseconds
    CV_WRAP double getAvgTimeMilli() const
    {
        return getAvgTimeSec() * 1e3;
    }

    //! resets internal values.
    CV_WRAP void reset()
    {
        startTime = 0;
        sumTime = 0;
        counter = 0;
    }

private:
    int64 counter;
    int64 sumTime;
    int64 startTime;
};

/** @brief output operator
@code
TickMeter tm;
tm.start();
// do something ...
tm.stop();
std::cout << tm;
@endcode
*/

static inline
std::ostream& operator << (std::ostream& out, const TickMeter& tm)
{
    return out << tm.getTimeSec() << "sec";
}

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

/** @brief Returns true if the specified feature is supported by the host hardware.

The function returns true if the host hardware supports the specified feature. When user calls
setUseOptimized(false), the subsequent calls to checkHardwareSupport() will return false until
setUseOptimized(true) is called. This way user can dynamically switch on and off the optimized code
in OpenCV.
@param feature The feature of interest, one of cv::CpuFeatures
 */
CV_EXPORTS_W bool checkHardwareSupport(int feature);

/** @brief Returns feature name by ID

Returns empty string if feature is not defined
*/
CV_EXPORTS_W String getHardwareFeatureName(int feature);

/** @brief Returns list of CPU features enabled during compilation.

Returned value is a string containing space separated list of CPU features with following markers:

- no markers - baseline features
- prefix `*` - features enabled in dispatcher
- suffix `?` - features enabled but not available in HW

Example: `SSE SSE2 SSE3 *SSE4.1 *SSE4.2 *FP16 *AVX *AVX2 *AVX512-SKX?`
*/
CV_EXPORTS_W std::string getCPUFeaturesLine();

/** @brief Returns the number of logical CPUs available for the process.
 */
CV_EXPORTS_W int getNumberOfCPUs();


/** @brief Aligns a pointer to the specified number of bytes.

The function returns the aligned pointer of the same type as the input pointer:
\f[\texttt{(_Tp*)(((size_t)ptr + n-1) & -n)}\f]
@param ptr Aligned pointer.
@param n Alignment size that must be a power of two.
 */
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    CV_DbgAssert((n & (n - 1)) == 0); // n is a power of 2
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

/** @brief Aligns a buffer size to the specified number of bytes.

The function returns the minimum number that is greater than or equal to sz and is divisible by n :
\f[\texttt{(sz + n-1) & -n}\f]
@param sz Buffer size to align.
@param n Alignment size that must be a power of two.
 */
static inline size_t alignSize(size_t sz, int n)
{
    CV_DbgAssert((n & (n - 1)) == 0); // n is a power of 2
    return (sz + n-1) & -n;
}

/** @brief Integer division with result round up.

Use this function instead of `ceil((float)a / b)` expressions.

@sa alignSize
*/
static inline int divUp(int a, unsigned int b)
{
    CV_DbgAssert(a >= 0);
    return (a + b - 1) / b;
}
/** @overload */
static inline size_t divUp(size_t a, unsigned int b)
{
    return (a + b - 1) / b;
}

/** @brief Round first value up to the nearest multiple of second value.

Use this function instead of `ceil((float)a / b) * b` expressions.

@sa divUp
*/
static inline int roundUp(int a, unsigned int b)
{
    CV_DbgAssert(a >= 0);
    return a + b - 1 - (a + b -1) % b;
}
/** @overload */
static inline size_t roundUp(size_t a, unsigned int b)
{
    return a + b - 1 - (a + b - 1) % b;
}

/** @brief Alignment check of passed values

Usage: `isAligned<sizeof(int)>(...)`

@note Alignment(N) must be a power of 2 (2**k, 2^k)
*/
template<int N, typename T> static inline
bool isAligned(const T& data)
{
    CV_StaticAssert((N & (N - 1)) == 0, "");  // power of 2
    return (((size_t)data) & (N - 1)) == 0;
}
/** @overload */
template<int N> static inline
bool isAligned(const void* p1)
{
    return isAligned<N>((size_t)p1);
}
/** @overload */
template<int N> static inline
bool isAligned(const void* p1, const void* p2)
{
    return isAligned<N>(((size_t)p1)|((size_t)p2));
}
/** @overload */
template<int N> static inline
bool isAligned(const void* p1, const void* p2, const void* p3)
{
    return isAligned<N>(((size_t)p1)|((size_t)p2)|((size_t)p3));
}
/** @overload */
template<int N> static inline
bool isAligned(const void* p1, const void* p2, const void* p3, const void* p4)
{
    return isAligned<N>(((size_t)p1)|((size_t)p2)|((size_t)p3)|((size_t)p4));
}

/** @brief Enables or disables the optimized code.

The function can be used to dynamically turn on and off optimized dispatched code (code that uses SSE4.2, AVX/AVX2,
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

static inline size_t getElemSize(int type) { return (size_t)CV_ELEM_SIZE(type); }

/////////////////////////////// Parallel Primitives //////////////////////////////////

/** @brief Base class for parallel data processors

@ingroup core_parallel
*/
class CV_EXPORTS ParallelLoopBody
{
public:
    virtual ~ParallelLoopBody();
    virtual void operator() (const Range& range) const = 0;
};

/** @brief Parallel data processor

@ingroup core_parallel
*/
CV_EXPORTS void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes=-1.);

//! @ingroup core_parallel
class ParallelLoopBodyLambdaWrapper : public ParallelLoopBody
{
private:
    std::function<void(const Range&)> m_functor;
public:
    inline
    ParallelLoopBodyLambdaWrapper(std::function<void(const Range&)> functor)
        : m_functor(functor)
    {
        // nothing
    }

    virtual void operator() (const cv::Range& range) const CV_OVERRIDE
    {
        m_functor(range);
    }
};

//! @ingroup core_parallel
static inline
void parallel_for_(const Range& range, std::function<void(const Range&)> functor, double nstripes=-1.)
{
    parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
}


/////////////////////////////// forEach method of cv::Mat ////////////////////////////
template<typename _Tp, typename Functor> inline
void Mat::forEach_impl(const Functor& operation) {
    if (false) {
        operation(*reinterpret_cast<_Tp*>(0), reinterpret_cast<int*>(0));
        // If your compiler fails in this line.
        // Please check that your functor signature is
        //     (_Tp&, const int*)   <- multi-dimensional
        //  or (_Tp&, void*)        <- in case you don't need current idx.
    }

    CV_Assert(!empty());
    CV_Assert(this->total() / this->size[this->dims - 1] <= INT_MAX);
    const int LINES = static_cast<int>(this->total() / this->size[this->dims - 1]);

    class PixelOperationWrapper :public ParallelLoopBody
    {
    public:
        PixelOperationWrapper(Mat_<_Tp>* const frame, const Functor& _operation)
            : mat(frame), op(_operation) {}
        virtual ~PixelOperationWrapper(){}
        // ! Overloaded virtual operator
        // convert range call to row call.
        virtual void operator()(const Range &range) const CV_OVERRIDE
        {
            const int DIMS = mat->dims;
            const int COLS = mat->size[DIMS - 1];
            if (DIMS <= 2) {
                for (int row = range.start; row < range.end; ++row) {
                    this->rowCall2(row, COLS);
                }
            } else {
                std::vector<int> idx(DIMS); /// idx is modified in this->rowCall
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
        }
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
        }
        PixelOperationWrapper& operator=(const PixelOperationWrapper &) {
            CV_Assert(false);
            // We can not remove this implementation because Visual Studio warning C4822.
            return *this;
        }
    };

    parallel_for_(cv::Range(0, LINES), PixelOperationWrapper(reinterpret_cast<Mat_<_Tp>*>(this), operation));
}

/////////////////////////// Synchronization Primitives ///////////////////////////////

#if !defined(_M_CEE)
#ifndef OPENCV_DISABLE_THREAD_SUPPORT
typedef std::recursive_mutex Mutex;
typedef std::lock_guard<cv::Mutex> AutoLock;
#else // OPENCV_DISABLE_THREAD_SUPPORT
// Custom (failing) implementation of `std::recursive_mutex`.
struct Mutex {
    void lock(){
        CV_Error(cv::Error::StsNotImplemented,
                 "cv::Mutex is disabled by OPENCV_DISABLE_THREAD_SUPPORT=ON");
    }
    void unlock(){
        CV_Error(cv::Error::StsNotImplemented,
                 "cv::Mutex is disabled by OPENCV_DISABLE_THREAD_SUPPORT=ON");
    }
};
// Stub for cv::AutoLock when threads are disabled.
struct AutoLock {
    AutoLock(Mutex &) { }
};
#endif // OPENCV_DISABLE_THREAD_SUPPORT
#endif // !defined(_M_CEE)


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

The keys parameter is a string containing several blocks, each one is enclosed in curly braces and
describes one argument. Each argument contains three parts separated by the `|` symbol:

-# argument names is a list of option synonyms separated by standard space characters ' ' (to mark argument as positional, prefix it with the `@` symbol)
-# default value will be used if the argument was not provided (can be empty)
-# help message (can be empty)

For example:

@code{.cpp}
    const String keys =
        "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1 for compare   }"
        "{@image2        |<none>| image2 for compare   }"
        "{@repeat        |1     | number               }"
        "{path           |.     | path to file         }"
        "{fps            | -1.0 | fps for output video }"
        "{N count        |100   | count of objects     }"
        "{ts timestamp   |      | use time stamp       }"
        ;
}
@endcode

Note that there are no default values for `help` and `timestamp` so we can check their presence using the `has()` method.
Arguments with default values are considered to be always present. Use the `get()` method in these cases to check their
actual value instead.
Note that whitespace characters other than standard spaces are considered part of the string.
Additionally, leading and trailing standard spaces around the help messages are ignored.

String keys like `get<String>("@image1")` return the empty string `""` by default - even with an empty default value.
Use the special `<none>` default value to enforce that the returned string must not be empty. (like in `get<String>("@image2")`)

### Usage

For the described keys:

@code{.sh}
    # Good call (3 positional parameters: image1, image2 and repeat; N is 200, ts is true)
    $ ./app -N=200 1.png 2.jpg 19 -ts

    # Bad call
    $ ./app -fps=aaa
    ERRORS:
    Parameter 'fps': can not convert: [aaa] to [double]
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

    For example, if the application has been started with such a command:
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

    Returns false if error occurred while accessing the parameters (bad conversion, missing arguments,
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

    /** @brief Print list of errors occurred

    @sa check
    */
    void printErrors() const;

protected:
    void getByName(const String& name, bool space_delete, Param type, void* dst) const;
    void getByIndex(int index, bool space_delete, Param type, void* dst) const;

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
    sz = _size;
    if(_size > fixed_size)
    {
        ptr = new _Tp[_size];
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

//! @endcond


// Basic Node class for tree building
template<class OBJECT>
class CV_EXPORTS Node
{
public:
    Node()
    {
        m_pParent  = 0;
    }
    Node(OBJECT& payload) : m_payload(payload)
    {
        m_pParent  = 0;
    }
    ~Node()
    {
        removeChilds();
        if (m_pParent)
        {
            int idx = m_pParent->findChild(this);
            if (idx >= 0)
                m_pParent->m_childs.erase(m_pParent->m_childs.begin() + idx);
        }
    }

    Node<OBJECT>* findChild(OBJECT& payload) const
    {
        for(size_t i = 0; i < this->m_childs.size(); i++)
        {
            if(this->m_childs[i]->m_payload == payload)
                return this->m_childs[i];
        }
        return NULL;
    }

    int findChild(Node<OBJECT> *pNode) const
    {
        for (size_t i = 0; i < this->m_childs.size(); i++)
        {
            if(this->m_childs[i] == pNode)
                return (int)i;
        }
        return -1;
    }

    void addChild(Node<OBJECT> *pNode)
    {
        if(!pNode)
            return;

        CV_Assert(pNode->m_pParent == 0);
        pNode->m_pParent = this;
        this->m_childs.push_back(pNode);
    }

    void removeChilds()
    {
        for(size_t i = 0; i < m_childs.size(); i++)
        {
            m_childs[i]->m_pParent = 0; // avoid excessive parent vector trimming
            delete m_childs[i];
        }
        m_childs.clear();
    }

    int getDepth()
    {
        int   count   = 0;
        Node *pParent = m_pParent;
        while(pParent) count++, pParent = pParent->m_pParent;
        return count;
    }

public:
    OBJECT                     m_payload;
    Node<OBJECT>*              m_pParent;
    std::vector<Node<OBJECT>*> m_childs;
};


namespace samples {

//! @addtogroup core_utils_samples
// This section describes utility functions for OpenCV samples.
//
// @note Implementation of these utilities is not thread-safe.
//
//! @{

/** @brief Try to find requested data file

Search directories:

1. Directories passed via `addSamplesDataSearchPath()`
2. OPENCV_SAMPLES_DATA_PATH_HINT environment variable
3. OPENCV_SAMPLES_DATA_PATH environment variable
   If parameter value is not empty and nothing is found then stop searching.
4. Detects build/install path based on:
   a. current working directory (CWD)
   b. and/or binary module location (opencv_core/opencv_world, doesn't work with static linkage)
5. Scan `<source>/{,data,samples/data}` directories if build directory is detected or the current directory is in source tree.
6. Scan `<install>/share/OpenCV` directory if install directory is detected.

@see cv::utils::findDataFile

@param relative_path Relative path to data file
@param required Specify "file not found" handling.
       If true, function prints information message and raises cv::Exception.
       If false, function returns empty result
@param silentMode Disables messages
@return Returns path (absolute or relative to the current directory) or empty string if file is not found
*/
CV_EXPORTS_W cv::String findFile(const cv::String& relative_path, bool required = true, bool silentMode = false);

CV_EXPORTS_W cv::String findFileOrKeep(const cv::String& relative_path, bool silentMode = false);

inline cv::String findFileOrKeep(const cv::String& relative_path, bool silentMode)
{
    cv::String res = findFile(relative_path, false, silentMode);
    if (res.empty())
        return relative_path;
    return res;
}

/** @brief Override search data path by adding new search location

Use this only to override default behavior
Passed paths are used in LIFO order.

@param path Path to used samples data
*/
CV_EXPORTS_W void addSamplesDataSearchPath(const cv::String& path);

/** @brief Append samples search data sub directory

General usage is to add OpenCV modules name (`<opencv_contrib>/modules/<name>/samples/data` -> `<name>/samples/data` + `modules/<name>/samples/data`).
Passed subdirectories are used in LIFO order.

@param subdir samples data sub directory
*/
CV_EXPORTS_W void addSamplesDataSearchSubDirectory(const cv::String& subdir);

//! @}
} // namespace samples

namespace utils {

CV_EXPORTS int getThreadID();

} // namespace

} //namespace cv

#ifdef CV_COLLECT_IMPL_DATA
#include "opencv2/core/utils/instrumentation.hpp"
#else
/// Collect implementation data on OpenCV function call. Requires ENABLE_IMPL_COLLECTION build option.
#define CV_IMPL_ADD(impl)
#endif

#endif //OPENCV_CORE_UTILITY_H
