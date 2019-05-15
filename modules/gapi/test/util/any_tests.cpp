// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "opencv2/gapi/util/any.hpp"

namespace opencv_test
{

TEST(Any, basic)
{
   using namespace util;
   any a(8);
   auto casted_pointer =  any_cast<int>(&a);
   ASSERT_NE(nullptr, casted_pointer);
   ASSERT_EQ(*casted_pointer, 8);

   *casted_pointer = 7;
   ASSERT_EQ(any_cast<int>(a), 7);
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

   ASSERT_EQ(any_cast<int>(a), 8);

   any b (a);

   ASSERT_NE(nullptr, any_cast<int>(&b));
   ASSERT_EQ(8      , any_cast<int>(b));
   ASSERT_EQ(8      , any_cast<int>(a));
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

   ASSERT_EQ(any_cast<int>(a), 8);

   any b (std::move(a));

   ASSERT_NE(nullptr,  any_cast<int>(&b));
   ASSERT_EQ(8      ,  any_cast<int>(b));
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

   ASSERT_EQ(any_cast<int>(a), 8);

   b = (std::move(a));

   ASSERT_NE(nullptr,  any_cast<int>(&b));
   ASSERT_EQ(8      ,  any_cast<int>(b));
   ASSERT_EQ(nullptr,  any_cast<int>(&a));
}

TEST(Any, copy_assign)
{
   using namespace util;
   any a(8);
   any b;

   ASSERT_EQ(any_cast<int>(a), 8);
   ASSERT_EQ(nullptr,  any_cast<int>(&b));

   b = a;

   ASSERT_NE(nullptr, any_cast<int>(&b));
   ASSERT_EQ(8      , any_cast<int>(b));
   ASSERT_EQ(8      , any_cast<int>(a));
}

} // namespace opencv_test
