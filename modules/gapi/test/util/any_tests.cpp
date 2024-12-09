// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include <opencv2/gapi/util/any.hpp>

namespace opencv_test
{

TEST(Any, basic)
{
   using namespace util;
   any a(8);
   auto casted_pointer =  any_cast<int>(&a);
   ASSERT_NE(nullptr, casted_pointer);
   ASSERT_EQ(8, *casted_pointer);

   *casted_pointer = 7;
   ASSERT_EQ(7, any_cast<int>(a));
}

TEST(Any, any_cast_ref_throws_on_empty)
{
   using namespace util;
   any a;

   ASSERT_THROW(util::any_cast<int>(a), bad_any_cast);
}

TEST(Any, copy)
{
   using namespace util;
   any a(8);

   ASSERT_EQ(8, any_cast<int>(a));

   any b (a);

   ASSERT_NE(nullptr, any_cast<int>(&b));
   ASSERT_EQ(8, any_cast<int>(b));
   ASSERT_EQ(8, any_cast<int>(a));
}

TEST(Any, copy_empty)
{
   using namespace util;
   any a;

   ASSERT_EQ(nullptr, any_cast<int>(&a));

   any b (a);

   ASSERT_EQ(nullptr, any_cast<int>(&a));
   ASSERT_EQ(nullptr, any_cast<int>(&b));
}

TEST(Any, move)
{
   using namespace util;
   any a(8);

   ASSERT_EQ(8, any_cast<int>(a));

   any b (std::move(a));

   ASSERT_NE(nullptr,  any_cast<int>(&b));
   ASSERT_EQ(8,  any_cast<int>(b));
   ASSERT_EQ(nullptr,  any_cast<int>(&a));
}

TEST(Any, swap)
{
   using namespace util;
   any a(8);
   any b(7);

   ASSERT_EQ(7, any_cast<int>(b));
   ASSERT_EQ(8, any_cast<int>(a));

   swap(a,b);

   ASSERT_EQ(8, any_cast<int>(b));
   ASSERT_EQ(7, any_cast<int>(a));
}

TEST(Any, move_assign)
{
   using namespace util;
   any a(8);
   any b;

   ASSERT_EQ(8, any_cast<int>(a));

   b = (std::move(a));

   ASSERT_NE(nullptr,  any_cast<int>(&b));
   ASSERT_EQ(8,  any_cast<int>(b));
   ASSERT_EQ(nullptr,  any_cast<int>(&a));
}

TEST(Any, copy_assign)
{
   using namespace util;
   any a(8);
   any b;

   ASSERT_EQ(8, any_cast<int>(a));
   ASSERT_EQ(nullptr,  any_cast<int>(&b));

   b = a;

   ASSERT_NE(nullptr, any_cast<int>(&b));
   ASSERT_EQ(8, any_cast<int>(b));
   ASSERT_EQ(8, any_cast<int>(a));
}

TEST(Any, get_ref_to_val_from_any)
{
   using namespace util;
   int x = 8;
   any a(x);

   int& casted_ref =  any_cast<int>(a);
   ASSERT_EQ(8, casted_ref);
}

TEST(Any, update_val_via_ref)
{
   using namespace util;
   int x = 8;
   any a(x);
   int& casted_ref = any_cast<int>(a);
   ASSERT_EQ(8, casted_ref);

   casted_ref = 7;
   ASSERT_EQ(7, any_cast<int>(a));
}
} // namespace opencv_test
