// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#include "opencv2/core/utility.hpp"
#include "opencv2/flann/heap.h"

#include <vector>

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)
#include <thread>
#endif

namespace opencv_test { namespace {

using cvflann::Heap;

// Basic single-threaded behaviour: popMin returns the stored elements in
// ascending order and reports emptiness correctly.
TEST(Flann_Heap, popMin_returns_sorted)
{
    const int capacity = 8;
    Heap<int> heap(capacity);

    const int values[] = {5, 1, 4, 2, 8, 3, 7, 6};
    for (int v : values)
        heap.insert(v);

    EXPECT_EQ(heap.size(), capacity);

    int prev = -1, out = 0;
    int popped = 0;
    while (heap.popMin(out))
    {
        EXPECT_GT(out, prev);  // strictly increasing
        prev = out;
        ++popped;
    }
    EXPECT_EQ(popped, capacity);
    EXPECT_TRUE(heap.empty());
}

// Capacity is a hard limit: extra inserts are dropped, not stored.
TEST(Flann_Heap, insert_respects_capacity)
{
    const int capacity = 3;
    Heap<int> heap(capacity);
    for (int i = 0; i < 10; ++i)
        heap.insert(i);
    EXPECT_EQ(heap.size(), capacity);
}

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)

// getPooledInstance() is no longer used on the FLANN search hot path, but it
// remains public API, so keep it covered: many threads each key the shared
// pool with their own thread id and must get usable, correctly ordered heaps
// without tripping the internal use_count guard.
TEST(Flann_Heap, getPooledInstance_concurrent_usage)
{
    const int numThreads = 8;
    const int capacity = 32;
    std::vector<std::thread> threads;
    std::vector<char> ok(numThreads, 0);

    for (int t = 0; t < numThreads; ++t)
    {
        threads.emplace_back([&, t]()
        {
            bool good = true;
            for (int iter = 0; iter < 200 && good; ++iter)
            {
                cv::Ptr<Heap<int>> heap =
                    Heap<int>::getPooledInstance(cv::utils::getThreadID(), capacity);
                for (int v = capacity - 1; v >= 0; --v)
                    heap->insert(v);

                int prev = -1, out = 0, count = 0;
                while (heap->popMin(out))
                {
                    if (out <= prev) { good = false; break; }
                    prev = out;
                    ++count;
                }
                if (count != capacity) good = false;
            }
            ok[t] = good ? 1 : 0;
        });
    }
    for (auto& th : threads)
        th.join();

    for (int t = 0; t < numThreads; ++t)
        EXPECT_EQ((int)ok[t], 1) << "thread " << t << " produced incorrect heap results";
}

#endif // !OPENCV_DISABLE_THREAD_SUPPORT

}} // namespace
