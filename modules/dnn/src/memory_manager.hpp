// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

// There are several memory management models in this file. They are actual for
// pipelines where we completely know about intervals of time and memory sizes
// what will be required for every stage.

#ifndef __OPENCV_DNN_MEMORY_MANAGER_HPP__
#define __OPENCV_DNN_MEMORY_MANAGER_HPP__

#include <opencv2/dnn.hpp>

namespace cv { namespace dnn {

// Memory user entity. It has request for memory size and it wants
// to use it from the start iteration until the end iteration (inclusive both).
// User expects continuous memory block that won't be damaged by other users
// during mentioned period of time.
struct MemoryUser
{
    MemoryUser(uint64_t startIter = 0, uint64_t endIter = 0, uint64_t memSize = 0);

    uint64_t id;  // Optional field (is used internally).
    uint64_t startIter;
    uint64_t endIter;
    uint64_t memSize;

    bool isActual(uint64_t iter);
};

// Class that provides different memory management models.
// Expected that users has similar dimensionalities of memSize, in bytes.
class MemoryManager
{
public:
    // Solve memory mamagement task in the most optimal way. It distributes
    // pointers to memory blocks inside the single memory buffer.
    // Returns offsets in the same order as users and size of buffer that
    // must be allocated for pipeline (total memory usage).
    static void solveOpt(std::vector<MemoryUser> users,
                         std::vector<uint64_t>& memPoses, uint64_t* memUsage);

    // Except this model is not optimal, it is one and only way to reuse
    // memory on devices that have no pointers arithmetic. In example, we can't
    // offset OpenCL's cl_mem memory object in C++ runtime, but we can offset
    // destination pointer inside the kernel (pass cl_mem and offset pair).
    // To simplify memory reusing, this model solves memory management task in
    // the way when new users reuse memory block that was used before but
    // is free now. Otherwise we just allocate the new one memory block.
    // Returns offsets and indices of users that must be allocated on device
    // (the rest of users has offsets considering to allocated memory). Also
    // returns total memory usage.
    static void solveReuseOrCreate(std::vector<MemoryUser> users,
                                   std::vector<uint64_t>& memPoses,
                                   std::vector<int>& hostIds,
                                   uint64_t* memUsage);
};

}  // namespace dnn
}  // namespace cv

#endif  // __OPENCV_DNN_MEMORY_MANAGER_HPP__
