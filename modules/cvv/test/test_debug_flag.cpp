#include "test_precomp.hpp"

/**
 * Tests whether cvv::debugMode() and cvv::setDebugFlag(bool)`  
 * (from /include/opencv2/debug_mode.hpp) behave correctly.
 */

TEST(DebugFlagTest, SetAndUnsetDebugMode)
{
	EXPECT_EQ(cvv::debugMode(), true);
	cvv::setDebugFlag(false);
	EXPECT_EQ(cvv::debugMode(), false);
	cvv::setDebugFlag(true);
	EXPECT_EQ(cvv::debugMode(), true);
}
