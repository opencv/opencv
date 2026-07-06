#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Core_FP8, conversions)
{
    // Test conversion from float to fp8_t and back
    float values[] = {
        0.0f, -0.0f, 1.0f, -1.0f,
        2.0f, 0.5f, 0.125f, -0.125f,
        448.0f, -448.0f // Max normal values for E4M3
    };

    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); i++)
    {
        float orig = values[i];
        cv::fp8_t fp8_val(orig);
        float restored = (float)fp8_val;
        
        // The restored value should be exactly equal to the original for these simple powers of 2
        EXPECT_EQ(orig, restored) << "Failed at value: " << orig;
    }
}

TEST(Core_FP8, limits)
{
    // NaN
    float my_nan = std::numeric_limits<float>::quiet_NaN();
    cv::fp8_t fp8_nan(my_nan);
    float restored_nan = (float)fp8_nan;
    EXPECT_TRUE(std::isnan(restored_nan));

    // Infinity (E4M3 doesn't have standard infinity, it maps to NaN in the basic spec,
    // but some specs map it to max value. Let's see what our logic did:
    // our logic sets it to 0x7F which maps to NaN in E4M3).
    float my_inf = std::numeric_limits<float>::infinity();
    cv::fp8_t fp8_inf(my_inf);
    float restored_inf = (float)fp8_inf;
    EXPECT_TRUE(std::isnan(restored_inf));
}

}} // namespace
