// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_KERNEL_DISPATCHER_HPP
#define OPENCV_DNN_SRC_CUDA_KERNEL_DISPATCHER_HPP

#include <cstddef>
#include <type_traits>

/* The performance of many kernels are highly dependent on the tensor rank. Instead of having
 * one kernel which can work with the maximally ranked tensors, we make one kernel for each supported
 * tensor rank. This is to ensure that the requirements of the maximally ranked tensors do not take a
 * toll on the performance of the operation for low ranked tensors. Hence, many kernels take the tensor
 * rank as a template parameter.
 *
 * The kernel is a template and we have different instantiations for each rank. This causes the following pattern
 * to arise frequently:
 *
 * if(rank == 3)
 *     kernel<T, 3>();
 * else if(rank == 2)
 *     kernel<T, 2>();
 * else
 *     kernel<T, 1>();
 *
 * The rank is a runtime variable. To facilitate creation of such structures, we use GENERATE_KERNEL_DISPATCHER.
 * This macro creates a function which selects the correct kernel instantiation at runtime.
 *
 * Example:
 *
 * // function which setups the kernel and launches it
 * template <class T, std::size_t Rank>
 * void launch_some_kernel(...);
 *
 * // creates the dispatcher named "some_dispatcher" which invokves the correct instantiation of "launch_some_kernel"
 * GENERATE_KERNEL_DISPATCHER(some_dispatcher, launch_some_kernel);
 *
 * // internal API function
 * template <class T>
 * void some(...) {
 *    // ...
 *    auto rank = input.rank();
 *    some_dispatcher<T, MIN_RANK, MAX_RANK>(rank, ...);
 * }
 */

/*
 * name     name of the dispatcher function that is generated
 * func     template function that requires runtime selection
 *
 * T        first template parameter to `func`
 * start    starting rank
 * end      ending rank (inclusive)
 *
 * Executes func<T, selector> based on runtime `selector` argument given `selector` lies
 * within the range [start, end]. If outside the range, no instantiation of `func` is executed.
 */
#define GENERATE_KERNEL_DISPATCHER(name,func);                                          \
    template <class T, std::size_t start, std::size_t end, class... Args> static        \
    typename std::enable_if<start == end, void>                                         \
    ::type name(int selector, Args&& ...args) {                                         \
        if(selector == start)                                                           \
            func<T, start>(std::forward<Args>(args)...);                                \
    }                                                                                   \
                                                                                        \
    template <class T, std::size_t start, std::size_t end, class... Args> static        \
    typename std::enable_if<start != end, void>                                         \
    ::type name(int selector, Args&& ...args) {                                         \
        if(selector == start)                                                           \
            func<T, start>(std::forward<Args>(args)...);                                \
        else                                                                            \
            name<T, start + 1, end, Args...>(selector, std::forward<Args>(args)...);    \
    }

// Same as GENERATE_KERNEL_DISPATCHER but takes two class template parameters T and TP1 instead of just T
#define GENERATE_KERNEL_DISPATCHER_2TP(name,func);                                              \
    template <class TP1, class TP2, std::size_t start, std::size_t end, class... Args> static   \
    typename std::enable_if<start == end, void>                                                 \
    ::type name(int selector, Args&& ...args) {                                                 \
        if(selector == start)                                                                   \
            func<TP1, TP2, start>(std::forward<Args>(args)...);                                 \
    }                                                                                           \
                                                                                                \
    template <class TP1, class TP2, std::size_t start, std::size_t end, class... Args> static   \
    typename std::enable_if<start != end, void>                                                 \
    ::type name(int selector, Args&& ...args) {                                                 \
        if(selector == start)                                                                   \
            func<TP1, TP2, start>(std::forward<Args>(args)...);                                 \
        else                                                                                    \
            name<TP1, TP2, start + 1, end, Args...>(selector, std::forward<Args>(args)...);     \
    }

#endif /* OPENCV_DNN_SRC_CUDA_KERNEL_DISPATCHER_HPP */
