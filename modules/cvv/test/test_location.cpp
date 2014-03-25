#include <gtest/gtest.h>

#include "call_meta_data.hpp"

/**
 * Tests whether the `CVVISUAL_LOCATION` macro (from /include/opencv2/call_meta_data.hpp)
 * works as expected, i.e. the instance of `cvv::impl::CallMetaData` as which it gets defined has the correct data.
 * The second test in this file checks wether a `cvv::impl::CallMataData` created by hand and with an empty
 * initializer list has no known location, as it is supposed to be.
 */
class LocationTest : public testing::Test
{
};

TEST_F(LocationTest, FileLineFunction)
{
	auto locationMacroResult = CVVISUAL_LOCATION;
	auto line = __LINE__ - 1;
	auto file = __FILE__;
	auto fun = CVVISUAL_FUNCTION_NAME_MACRO;
	EXPECT_EQ(locationMacroResult.isKnown, true);
	EXPECT_EQ(locationMacroResult.file, file);
	EXPECT_EQ(locationMacroResult.line, line);
	EXPECT_EQ(locationMacroResult.function, fun);
}

TEST_F(LocationTest, EmptyLocation)
{
	cvv::impl::CallMetaData loc{};
	EXPECT_EQ(loc.isKnown, false);
}
