#include <gtest/gtest.h>
#include <cstdlib>
#include <cstdint>
#include <climits>

// Pull in the actual production function
namespace cv { namespace ndsrvp {
    void* fastMalloc(size_t size);
    void  fastFree(void* ptr);
}}

class FastMallocSecurityTest : public ::testing::TestWithParam<size_t> {};

TEST_P(FastMallocSecurityTest, NoOverflowOrOversizedAlloc) {
    // Invariant: fastMalloc must either return NULL / throw on overflow-inducing
    // sizes, or allocate a buffer large enough to hold 'size' bytes safely.
    // It must NEVER silently return a tiny buffer for a huge requested size.
    size_t requested = GetParam();

    // Sizes near SIZE_MAX will overflow the internal arithmetic; the function
    // must not return a non-null pointer in that case (it would be a tiny buf).
    constexpr size_t OVERFLOW_THRESHOLD = SIZE_MAX - 256;

    void* ptr = nullptr;
    bool threw = false;
    try {
        ptr = cv::ndsrvp::fastMalloc(requested);
    } catch (...) {
        threw = true;
    }

    if (requested >= OVERFLOW_THRESHOLD) {
        // Must not silently succeed with a wrapped-around allocation
        EXPECT_TRUE(ptr == nullptr || threw)
            << "fastMalloc returned non-null for overflow-inducing size "
            << requested << " — heap overflow risk";
    } else {
        // For legitimate sizes the allocation should succeed and be usable
        if (ptr != nullptr) {
            // Write one byte at start and end to confirm buffer is real
            static_cast<unsigned char*>(ptr)[0] = 0xAB;
            static_cast<unsigned char*>(ptr)[requested - 1] = 0xCD;
            cv::ndsrvp::fastFree(ptr);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialInputs,
    FastMallocSecurityTest,
    ::testing::Values(
        SIZE_MAX - 15,   // exact exploit: wraps to ~0 after +sizeof(void*)+align
        SIZE_MAX / 2,    // boundary: large but may still overflow on 32-bit
        64u              // valid: normal small allocation
    )
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}