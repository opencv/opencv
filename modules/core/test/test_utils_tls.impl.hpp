// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This is .hpp file included from test_utils.cpp

#ifdef CV_CXX11
#include <thread>  // std::thread
#endif

#include "opencv2/core/utils/tls.hpp"

namespace opencv_test { namespace {

class TLSReporter
{
public:
    static int g_last_id;
    static int g_allocated;

    int id;

    TLSReporter()
    {
        id = CV_XADD(&g_last_id, 1);
        CV_XADD(&g_allocated, 1);
    }
    ~TLSReporter()
    {
        CV_XADD(&g_allocated, -1);
    }
};

int TLSReporter::g_last_id = 0;
int TLSReporter::g_allocated = 0;

#ifdef CV_CXX11

template<typename T>
static void callNThreadsWithTLS(int N, TLSData<T>& tls)
{
    std::vector<std::thread> threads(N);
    for (int i = 0; i < N; i++)
    {
        threads[i] = std::thread([&]() {
            TLSReporter* pData = tls.get();
            (void)pData;
        });
    }
    for (int i = 0; i < N; i++)
    {
        threads[i].join();
    }
    threads.clear();
}

TEST(Core_TLS, HandleThreadTermination)
{
    const int init_id = TLSReporter::g_last_id;
    const int init_allocated = TLSReporter::g_allocated;

    const int N = 4;
    TLSData<TLSReporter> tls;

    // use TLS
    ASSERT_NO_THROW(callNThreadsWithTLS(N, tls));

    EXPECT_EQ(init_id + N, TLSReporter::g_last_id);
    EXPECT_EQ(init_allocated + 0, TLSReporter::g_allocated);
}


static void testTLSAccumulator(bool detachFirst)
{
    const int init_id = TLSReporter::g_last_id;
    const int init_allocated = TLSReporter::g_allocated;

    const int N = 4;
    TLSDataAccumulator<TLSReporter> tls;

    {  // empty TLS checks
        std::vector<TLSReporter*>& data0 = tls.detachData();
        EXPECT_EQ((size_t)0, data0.size());
        tls.cleanupDetachedData();
    }

    // use TLS
    ASSERT_NO_THROW(callNThreadsWithTLS(N, tls));

    EXPECT_EQ(init_id + N, TLSReporter::g_last_id);
    EXPECT_EQ(init_allocated + N, TLSReporter::g_allocated);

    if (detachFirst)
    {
        std::vector<TLSReporter*>& data1 = tls.detachData();
        EXPECT_EQ((size_t)N, data1.size());

        // no data through gather after detachData()
        std::vector<TLSReporter*> data2;
        tls.gather(data2);
        EXPECT_EQ((size_t)0, data2.size());

        tls.cleanupDetachedData();

        EXPECT_EQ(init_id + N, TLSReporter::g_last_id);
        EXPECT_EQ(init_allocated + 0, TLSReporter::g_allocated);
        EXPECT_EQ((size_t)0, data1.size());
    }
    else
    {
        std::vector<TLSReporter*> data2;
        tls.gather(data2);
        EXPECT_EQ((size_t)N, data2.size());

        std::vector<TLSReporter*>& data1 = tls.detachData();
        EXPECT_EQ((size_t)N, data1.size());

        tls.cleanupDetachedData();

        EXPECT_EQ((size_t)0, data1.size());
        // data2 is not empty, but it has invalid contents
        EXPECT_EQ((size_t)N, data2.size());
    }

    EXPECT_EQ(init_id + N, TLSReporter::g_last_id);
    EXPECT_EQ(init_allocated + 0, TLSReporter::g_allocated);
}

TEST(Core_TLS, AccumulatorHoldData_detachData) { testTLSAccumulator(true); }
TEST(Core_TLS, AccumulatorHoldData_gather) { testTLSAccumulator(false); }

#endif

}}  // namespace
