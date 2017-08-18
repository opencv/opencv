// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "memory_manager.hpp"

namespace cv { namespace dnn {

MemoryUser::MemoryUser(uint64_t startIter, uint64_t endIter, uint64_t memSize)
    : startIter(startIter), endIter(endIter), memSize(memSize) {}

bool MemoryUser::isActual(uint64_t iter)
{
    return startIter <= iter && iter <= endIter;
}

// Check if two intervals are intersected. Both bounds are inclusive.
// l_from      l_to
//   [----------]
//         [---------]
//        r_from    r_to
// It isn't necessary to keep 'left' and 'right', it's just for naming.
static bool isIntersection(uint64_t l_from, uint64_t l_to, uint64_t r_from,
                           uint64_t r_to)
{
    return l_to >= r_from && l_from <= r_to;
}

static bool compareByMemSize(const MemoryUser& l, const MemoryUser& r)
{
    return l.memSize > r.memSize;
}

static bool compareByMemPos(const std::pair<MemoryUser, uint64_t>& l,
                            const std::pair<MemoryUser, uint64_t>& r)
{
    return l.second < r.second;
}

// Check correctness.
static void internalCheck(const std::vector<MemoryUser>& users,
                          const std::vector<uint64_t>& memPoses)
{
    for (int i = 0; i < users.size(); ++i)
    {
        for (int j = i + 1; j < users.size(); ++j)
        {
            if (isIntersection(users[i].startIter, users[i].endIter,
                               users[j].startIter, users[j].endIter)) {
                uint64_t iMemPos = memPoses[users[i].id];
                uint64_t jMemPos = memPoses[users[j].id];
                CV_Assert(!isIntersection(iMemPos, iMemPos + users[i].memSize - 1,
                                          jMemPos, jMemPos + users[j].memSize - 1));
            }
        }
    }
}

void MemoryManager::solveOpt(std::vector<MemoryUser> users,
                             std::vector<uint64_t>& memPoses, uint64_t* memUsage)
{
    const int numUsers = users.size();
    memPoses.resize(numUsers, 0);
    for (int i = 0; i < numUsers; ++i)
        users[i].id = i;

    // Sort by memory size in descending order.
    std::sort(users.begin(), users.end(), compareByMemSize);

    *memUsage = 0;
    for (int i = 0; i < numUsers; ++i)
    {
        // Collect processed users that are actual at the same time.
        std::vector<std::pair<MemoryUser, uint64_t> > processedUsers;
        for (int j = 0; j < i; ++j)
        {
            if (isIntersection(users[i].startIter, users[i].endIter,
                               users[j].startIter, users[j].endIter))
            {
                std::pair<MemoryUser, uint64_t> term(users[j], memPoses[users[j].id]);
                processedUsers.push_back(term);
            }
        }
        std::sort(processedUsers.begin(), processedUsers.end(), compareByMemPos);

        uint64_t memPos = 0;
        for (int j = 0; j < processedUsers.size(); ++j)
        {
            if (isIntersection(memPos, memPos + users[i].memSize - 1,
                               processedUsers[j].second,
                               processedUsers[j].second + processedUsers[j].first.memSize - 1))
            {
                memPos = processedUsers[j].second + processedUsers[j].first.memSize;
            }
        }
        memPoses[users[i].id] = memPos;
        *memUsage = std::max(*memUsage, memPos + users[i].memSize);
    }
    internalCheck(users, memPoses);
}

void MemoryManager::solveReuseOrCreate(std::vector<MemoryUser> users,
                                       std::vector<uint64_t>& memPoses,
                                       std::vector<int>& hostIds,
                                       uint64_t* memUsage)
{
    hostIds.clear();
    memPoses.resize(users.size(), 0);
    for (int i = 0; i < users.size(); ++i)
        users[i].id = i;

    std::sort(users.begin(), users.end(), compareByMemSize);

    *memUsage = 0;
    std::vector<MemoryUser> layer;
    layer.reserve(users.size());
    do
    {
        for (int i = 0; i < users.size(); ++i)
        {
            bool addToLayer = true;
            for (int j = 0; addToLayer && j < layer.size(); ++j)
            {
                addToLayer = !isIntersection(users[i].startIter, users[i].endIter,
                                             layer[j].startIter, layer[j].endIter);
            }
            if (addToLayer)
            {
                memPoses[users[i].id] = *memUsage;
                layer.push_back(users[i]);
                users.erase(users.begin() + i);
                --i;
            }
        }
        hostIds.push_back(layer[0].id);
        *memUsage += layer[0].memSize;
        layer.clear();
    }
    while (!users.empty());
    internalCheck(users, memPoses);
}

}  // namespace dnn
}  // namespace cv
