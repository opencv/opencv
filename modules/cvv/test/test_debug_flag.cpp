//#include <opencv2/ts/ts.hpp>
#include <gtest/gtest.h>

#include <thread>

#include "debug_mode.hpp"

/**
 * Tests whether cvv::debugMode() and cvv::setDebugFlag(bool)`  
 * (from /include/opencv2/debug_mode.hpp) behave correctly.
 */
class DebugFlagTest : public testing::Test
{
};

TEST_F(DebugFlagTest, SetAndUnsetDebugMode)
{
	EXPECT_EQ(cvv::debugMode(), true);
	cvv::setDebugFlag(false);
	EXPECT_EQ(cvv::debugMode(), false);
	cvv::setDebugFlag(true);
	EXPECT_EQ(cvv::debugMode(), true);
}
