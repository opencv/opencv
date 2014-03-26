#include "test_precomp.hpp"

#include "../src/util/util.hpp"

/**
 * Tests whether the `cvv::util::isAnyOf()` function (from /src/util/util.hpp) correctly recognises 
 * the first parameter as element or not element of the data structure in the second parameter 
 * for the following structures:
 * - Initializer lists with `int int`
 * - Initializer lists with `long int`
 * - Vectors of `int int`
 * - Vectors of `long int`
 */

using cvv::util::isAnyOf;

TEST(IsAnyOfTest, InitializerListIntInt)
{
	EXPECT_EQ(isAnyOf(3, { 1, 2, 3, 4 }), true);
	EXPECT_EQ(isAnyOf(3, { 1, 2, 4 }), false);
}

TEST(IsAnyOfTest, InitializerListLongInt)
{
	EXPECT_EQ(isAnyOf(3, { 1L, 2L, 3L, 4L }), true);
	EXPECT_EQ(isAnyOf(3, { 1L, 2L, 4L }), false);
}

TEST(IsAnyOfTest, VectorIntInt)
{
	EXPECT_EQ(isAnyOf(3, std::vector<int>{ 1, 2, 3, 4 }), true);
	EXPECT_EQ(isAnyOf(3, std::vector<int>{ 1, 2, 4 }), false);
}

TEST(IsAnyOfTest, VectorLongInt)
{
	EXPECT_EQ(isAnyOf(3, std::vector<long>{ 1, 2, 3, 4 }), true);
	EXPECT_EQ(isAnyOf(3, std::vector<long>{ 1, 2, 4 }), false);
}
