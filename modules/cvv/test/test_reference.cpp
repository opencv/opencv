#include "test_precomp.hpp"

#include "../src/util/util.hpp"

/**
 * Makes sure that
 * - creating, reassigning and comparing `cvv::util::Reference<int>`s (from /src/util/util.hpp) works, 
 * 	as well as the
 * - `makeRef()` and `getPtr()` functions; that
 * - `Reference<const int>`s can be created with `makeRef(const int)`, `{const int}` initializer list
 * 	and via `Reference<const int>{int}` and, finally, that 
 * - `cvv::util::Reference`s of super classes can be constructed with derived classes and 
 * - `Reference`s to base classes can be `castTo()` `References` to the derived classes, but not the other way.
 */

using cvv::util::Reference;
using cvv::util::makeRef;

TEST(ReferenceTest, Construction)
{
	int i = 3;
	Reference<int> ref1{ i };
	EXPECT_EQ(*ref1, 3);
}

TEST(ReferenceTest, Reassignment)
{
	int i1 = 3;
	int i2 = 4;
	Reference<int> ref{ i1 };
	EXPECT_EQ(ref.getPtr(), &i1);
	EXPECT_EQ(*ref, 3);
	ref = Reference<int>{ i2 };
	EXPECT_EQ(*ref, 4);
	EXPECT_EQ(ref.getPtr(), &i2);
}

TEST(ReferenceTest, Comparing)
{
	int i1 = 1, i2 = 1;
	auto ref1 = makeRef(i1);
	auto ref2 = makeRef(i1);
	auto ref3 = makeRef(i2);

	EXPECT_EQ(ref1 == ref2, true);
	EXPECT_EQ(ref1 != ref2, false);
	EXPECT_EQ(ref1 == ref3, false);
	EXPECT_EQ(ref1 != ref3, true);
}

TEST(ReferenceTest, MakeRef)
{
	int i1 = 3, i2 = 4;
	auto ref = makeRef(i1);
	EXPECT_EQ(*ref, 3);
	EXPECT_EQ(ref.getPtr(), &i1);
	ref = makeRef(i2);
	EXPECT_EQ(*ref, 4);
	EXPECT_EQ(ref.getPtr(), &i2);
}

TEST(ReferenceTest, ConstRefs)
{
	const int i = 3;
	auto ref1 = makeRef(i);
	Reference<const int> ref2{ i };
	EXPECT_EQ(ref1, ref2);
}

TEST(ReferenceTest, ConstRefsFromMutable)
{
	int i;
	Reference<const int> ref{ i };
	EXPECT_EQ(ref.getPtr(), &i);
}

struct Base
{
	virtual ~Base() = default;
};
struct Derived : Base
{
};
struct Derived2 : Base
{
};

TEST(ReferenceTest, LiberalConstruction)
{
	Derived var;
	auto derivedRef = makeRef(var);
	Reference<Base> baseRef{ derivedRef };
	EXPECT_EQ(&var, baseRef.getPtr());
}

TEST(ReferenceTest, castTo)
{
	Derived var;
	Reference<Base> baseRef{ var };
	auto derivedRef = baseRef.castTo<Derived>();
	EXPECT_EQ(&var, derivedRef.getPtr());
	EXPECT_THROW(baseRef.castTo<Derived2>(), std::bad_cast);
	// should result in a compiler-error:
	// EXPECT_THROW(baseRef.castTo<std::vector<int>>(), std::bad_cast);
}
