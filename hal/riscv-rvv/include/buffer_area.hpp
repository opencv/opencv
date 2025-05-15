// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_RVV_UTILS_BUFFER_AREA_HPP
#define OPENCV_RVV_UTILS_BUFFER_AREA_HPP

#include <vector>

namespace cv { namespace utils {

//! @addtogroup core_utils
//! @{

/** @brief Manages memory block shared by muliple buffers.

This class allows to allocate one large memory block and split it into several smaller
non-overlapping buffers. In safe mode each buffer allocation will be performed independently,
this mode allows dynamic memory access instrumentation using valgrind or memory sanitizer.

Safe mode can be explicitly switched ON in constructor. It will also be enabled when compiling with
memory sanitizer support or in runtime with the environment variable `OPENCV_BUFFER_AREA_ALWAYS_SAFE`.

Example of usage:
@code
int * buf1 = 0;
double * buf2 = 0;
cv::util::BufferArea area;
area.allocate(buf1, 200); // buf1 = new int[200];
area.allocate(buf2, 1000, 64); // buf2 = new double[1000]; - aligned by 64
area.commit();
@endcode

@note This class is considered private and should be used only in OpenCV itself. API can be changed.
*/
class CV_EXPORTS BufferArea
{
public:
    /** @brief Class constructor.

    @param safe Enable _safe_ operation mode, each allocation will be performed independently.
    */
    BufferArea(bool safe = false);

    /** @brief Class destructor

    All allocated memory well be freed. Each bound pointer will be reset to NULL.
    */
    ~BufferArea();

    /** @brief Bind a pointer to local area.

    BufferArea will store reference to the pointer and allocation parameters effectively owning the
    pointer and allocated memory. This operation has the same parameters and does the same job
    as the operator `new`, except allocation can be performed later during the BufferArea::commit call.

    @param ptr Reference to a pointer of type T. Must be NULL
    @param count Count of objects to be allocated, it has the same meaning as in the operator `new`.
    @param alignment Alignment of allocated memory. same meaning as in the operator `new` (C++17).
                     Must be divisible by sizeof(T). Must be power of two.

    @note In safe mode allocation will be performed immediatly.
    */
    template <typename T>
    void allocate(T*&ptr, size_t count, ushort alignment = sizeof(T))
    {
        CV_Assert(ptr == NULL);
        CV_Assert(count > 0);
        CV_Assert(alignment > 0);
        CV_Assert(alignment % sizeof(T) == 0);
        CV_Assert((alignment & (alignment - 1)) == 0);
        allocate_((void**)(&ptr), static_cast<ushort>(sizeof(T)), count, alignment);
#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
        if (safe)
#endif
            CV_Assert(ptr != NULL);
    }

    /** @brief Fill one of buffers with zeroes

    @param ptr pointer to memory block previously added using BufferArea::allocate

    BufferArea::commit must be called before using this method
    */
    template <typename T>
    void zeroFill(T*&ptr)
    {
        CV_Assert(ptr);
        zeroFill_((void**)&ptr);
    }

    /** @brief Fill all buffers with zeroes

    BufferArea::commit must be called before using this method
    */
    void zeroFill();

    /** @brief Allocate memory and initialize all bound pointers

    Each pointer bound to the area with the BufferArea::allocate will be initialized and will be set
    to point to a memory block with requested size and alignment.

    @note Does nothing in safe mode as all allocations will be performed by BufferArea::allocate
    */
    void commit();

    /** @brief Release all memory and unbind all pointers

    All memory will be freed and all pointers will be reset to NULL and untied from the area allowing
    to call `allocate` and `commit` again.
    */
    void release();

private:
    BufferArea(const BufferArea &); // = delete
    BufferArea &operator=(const BufferArea &); // = delete
    void allocate_(void **ptr, ushort type_size, size_t count, ushort alignment);
    void zeroFill_(void **ptr);

private:
    class Block;
    std::vector<Block> blocks;
#ifndef OPENCV_ENABLE_MEMORY_SANITIZER
    void * oneBuf;
    size_t totalSize;
    const bool safe;
#endif
};

//! @}

}} // cv::utils::

#endif
