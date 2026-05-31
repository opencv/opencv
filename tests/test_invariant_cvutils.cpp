#include <gtest/gtest.h>
#include <cstdlib>
#include <climits>
#include <cstring>

// Pull in the actual production function
namespace cv { namespace hal { namespace ndsrvp {
    void* fastMalloc(size_t size);
    void  fastFree(void* ptr);
}}}

class FastMallocSecurityTest : public ::testing::TestWithParam<size_t> {};

TEST_P(FastMallocSecurityTest, RejectsOrSafelyHandlesOversizedInput) {
    // Invariant: fastMalloc must either return nullptr / throw on overflow,
    // or return a valid pointer to a buffer of at least the requested size.
    // It must NEVER silently allocate a tiny buffer for a near-SIZE_MAX request.
    size_t size = GetParam();

    // Sizes that would overflow size_t arithmetic should not produce a tiny buffer.
    // We detect the overflow condition: if size + sizeof(void*) + 64 wraps, skip write test.
    constexpr size_t align = 64; // CV_MALLOC_ALIGN is typically 64
    bool would_overflow = (size > SIZE_MAX - sizeof(void*) - align);

    void* ptr = nullptr;
    bool threw = false;
    try {
        ptr = cv::hal::ndsrvp::fastMalloc(size);
    } catch (...) {
        threw = true;
    }

    if (would_overflow) {
        // Must either throw or return null — never a silently undersized buffer
        EXPECT_TRUE(threw || ptr == nullptr)
            << "fastMalloc must not silently succeed on overflow-inducing size";
    } else {
        // Valid allocation: pointer must be non-null and writable up to 'size'
        ASSERT_NE(ptr, nullptr);
        // Touch first and last byte to confirm buffer is at least 'size' bytes
        if (size > 0) {
            static_cast<unsigned char*>(ptr)[0]      = 0xAB;
            static_cast<unsigned char*>(ptr)[size-1] = 0xCD;
        }
        cv::hal::ndsrvp::fastFree(ptr);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialInputs,
    FastMallocSecurityTest,
    ::testing::Values(
        SIZE_MAX,                          // exact exploit: wraps to tiny buffer
        SIZE_MAX - sizeof(void*) - 63,     // boundary: just at overflow edge
        SIZE_MAX / 2,                      // large but may still overflow after addition
        1024                               // valid input: normal allocation
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}