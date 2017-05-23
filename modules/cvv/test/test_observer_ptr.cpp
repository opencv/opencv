#include "test_precomp.hpp"

#include "../src/util/observer_ptr.hpp"

/**
 * Verifies that assigning `nullptr` and a nonzero value to a `cvv::util::ObserverPtr<Int>` 
 * (from /src/util/observer_ptr.hpp) work and that `isNull()` and `getPtr()` return the correct result.
 */

using cvv::util::ObserverPtr;

TEST(ObserverPtrTest, ConstructionAssignment)
{
	ObserverPtr<int> ptr = nullptr;
	EXPECT_TRUE(ptr.isNull());
	int x = 3;
	ptr = x;
	EXPECT_FALSE(ptr.isNull());
	EXPECT_EQ(&x, ptr.getPtr());
}
