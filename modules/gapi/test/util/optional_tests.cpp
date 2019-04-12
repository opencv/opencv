// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "opencv2/gapi/util/optional.hpp"
#include <opencv2/gapi/util/compiler_hints.hpp> //suppress_unused_warning

namespace opencv_test
{

TEST(Optional, EmptyCtor)
{
    util::optional<int> o;
    EXPECT_FALSE(o.has_value());
    EXPECT_FALSE(static_cast<bool>(o));
}

TEST(Optional, ValueCTor)
{
    util::optional<int> o(42);
    EXPECT_TRUE(o.has_value());
    EXPECT_TRUE(static_cast<bool>(o));
}

TEST(Optional, MoveCtr)
{
    util::optional<std::string> os1(std::string("text"));
    EXPECT_TRUE(os1.has_value());

    util::optional<std::string> os2(std::move(os1));
    EXPECT_FALSE(os1.has_value());
    EXPECT_TRUE(os2.has_value());
    EXPECT_EQ("text", os2.value());
}

TEST(Optional, EmptyThrows)
{
    struct foo { int bar; };
    util::optional<foo> om;
    const util::optional<foo> oc;

    int dummy;

    EXPECT_THROW(dummy = om->bar,    util::bad_optional_access);
    EXPECT_THROW(dummy = oc->bar,    util::bad_optional_access);
    cv::util::suppress_unused_warning(dummy);
    EXPECT_THROW(*om,        util::bad_optional_access);
    EXPECT_THROW(*oc,        util::bad_optional_access);
    EXPECT_THROW(om.value(), util::bad_optional_access);
    EXPECT_THROW(oc.value(), util::bad_optional_access);
}

TEST(Optional, ValueNoThrow)
{
    struct foo { int bar; };
    util::optional<foo> om(foo{42});
    const util::optional<foo> oc(foo{42});

    int dummy;
    EXPECT_NO_THROW(dummy = om->bar);
    EXPECT_NO_THROW(dummy = oc->bar);
    cv::util::suppress_unused_warning(dummy);
    EXPECT_NO_THROW(*om);
    EXPECT_NO_THROW(*oc);
    EXPECT_NO_THROW(om.value());
    EXPECT_NO_THROW(oc.value());
}

TEST(Optional, Value)
{
    util::optional<int> oi(42);

    struct foo { int bar; };
    util::optional<foo> of(foo{42});

    EXPECT_EQ(42, oi.value());
    EXPECT_EQ(42, *oi);

    EXPECT_EQ(42, of.value().bar);
    EXPECT_EQ(42, of->bar);
}

TEST(Optional, Mutable)
{
    util::optional<int> oi(42);
    *oi = 43;
    EXPECT_EQ(43, *oi);

    struct foo { int bar; int baz; };
    util::optional<foo> of(foo{11,22});

    (*of).bar = 42;
    EXPECT_EQ(42, of->bar);
    EXPECT_EQ(22, of->baz);

    of->baz = 33;
    EXPECT_EQ(42, of->bar);
    EXPECT_EQ(33, of->baz);
}

TEST(Optional, MoveAssign)
{
    util::optional<int> e, i(42);

    EXPECT_FALSE(e.has_value());
    EXPECT_TRUE(i.has_value());
    EXPECT_EQ(42, *i);

    e = std::move(i);
    EXPECT_TRUE(e.has_value());
    EXPECT_FALSE(i.has_value());
    EXPECT_EQ(42, *e);
}

TEST(Optional, CopyAssign)
{
    util::optional<int> e;
    const util::optional<int> i(42);

    EXPECT_FALSE(e.has_value());
    EXPECT_TRUE(i.has_value());
    EXPECT_EQ(42, *i);

    e = i;
    EXPECT_TRUE(e.has_value());
    EXPECT_TRUE(i.has_value());
    EXPECT_EQ(42, *e);
    EXPECT_EQ(42, *i);
}

TEST(Optional, ValueOr)
{
    util::optional<int> e;
    EXPECT_FALSE(e.has_value());
    EXPECT_EQ(42, e.value_or(42));
    EXPECT_EQ(42, e.value_or(42.1));
}

TEST(Optional, Swap)
{
    util::optional<int> e, i(42);

    EXPECT_FALSE(e.has_value());
    EXPECT_TRUE(i.has_value());
    EXPECT_EQ(42, *i);

    e.swap(i);

    EXPECT_TRUE(e.has_value());
    EXPECT_FALSE(i.has_value());
    EXPECT_EQ(42, *e);
}

TEST(Optional, Reset)
{
    util::optional<int> i(42);
    EXPECT_TRUE(i.has_value());

    i.reset();
    EXPECT_FALSE(i.has_value());
}

TEST(Optional, MakeOptional)
{
    std::string s("text");
    auto os = util::make_optional(s);
    EXPECT_TRUE(os.has_value());
    EXPECT_EQ(s, os.value());
}

} // namespace opencv_test
