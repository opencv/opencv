/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef GTEST_CAN_COMPARE_NULL
#  define EXPECT_NULL(ptr) EXPECT_EQ(NULL, ptr)
#else
#  define EXPECT_NULL(ptr) EXPECT_TRUE(ptr == NULL)
#endif

using namespace cv;

namespace {

struct Reporter {
    Reporter(bool* deleted) : deleted_(deleted)
    { *deleted_ = false; }

    // the destructor is virtual, so that we can test dynamic_cast later
    virtual ~Reporter()
    { *deleted_ = true; }

private:
    bool* deleted_;

    Reporter(const Reporter&);
    Reporter& operator = (const Reporter&);
};

struct ReportingDeleter {
    ReportingDeleter(bool* deleted) : deleted_(deleted)
    { *deleted_ = false; }

    void operator()(void*)
    { *deleted_ = true; }

private:
    bool* deleted_;
};

int dummyObject;

}

TEST(Core_Ptr, default_ctor)
{
    Ptr<int> p;
    EXPECT_NULL(p.get());
}

TEST(Core_Ptr, owning_ctor)
{
    bool deleted = false;

    {
        Reporter* r = new Reporter(&deleted);
        Ptr<void> p(r);
        EXPECT_EQ(r, p.get());
    }
    EXPECT_TRUE(deleted);

    {
        Ptr<int> p(&dummyObject, ReportingDeleter(&deleted));
        EXPECT_EQ(&dummyObject, p.get());
    }
    EXPECT_TRUE(deleted);

    {
        Ptr<void> p((void*)0, ReportingDeleter(&deleted));
        EXPECT_NULL(p.get());
    }
    EXPECT_TRUE(deleted);  // Differ from OpenCV 3.4 (but conformant to std::shared_ptr, see below)

    {
        std::shared_ptr<void> p((void*)0, ReportingDeleter(&deleted));
        EXPECT_NULL(p.get());
    }
    EXPECT_TRUE(deleted);
}

TEST(Core_Ptr, sharing_ctor)
{
    bool deleted = false;

    {
        Ptr<Reporter> p1(new Reporter(&deleted));
        Ptr<Reporter> p2(p1);
        EXPECT_EQ(p1.get(), p2.get());
        p1.release();
        EXPECT_FALSE(deleted);
    }

    EXPECT_TRUE(deleted);

    {
        Ptr<Reporter> p1(new Reporter(&deleted));
        Ptr<void> p2(p1);
        EXPECT_EQ(p1.get(), p2.get());
        p1.release();
        EXPECT_FALSE(deleted);
    }

    EXPECT_TRUE(deleted);

    {
        Ptr<Reporter> p1(new Reporter(&deleted));
        Ptr<int> p2(p1, &dummyObject);
        EXPECT_EQ(&dummyObject, p2.get());
        p1.release();
        EXPECT_FALSE(deleted);
    }

    EXPECT_TRUE(deleted);
}

TEST(Core_Ptr, assignment)
{
    bool deleted1 = false, deleted2 = false;

    {
        Ptr<Reporter> p1(new Reporter(&deleted1));
        p1 = *&p1;
        EXPECT_FALSE(deleted1);
    }

    EXPECT_TRUE(deleted1);

    {
        Ptr<Reporter> p1(new Reporter(&deleted1));
        Ptr<Reporter> p2(new Reporter(&deleted2));
        p2 = p1;
        EXPECT_TRUE(deleted2);
        EXPECT_EQ(p1.get(), p2.get());
        p1.release();
        EXPECT_FALSE(deleted1);
    }

    EXPECT_TRUE(deleted1);

    {
        Ptr<Reporter> p1(new Reporter(&deleted1));
        Ptr<void> p2(new Reporter(&deleted2));
        p2 = p1;
        EXPECT_TRUE(deleted2);
        EXPECT_EQ(p1.get(), p2.get());
        p1.release();
        EXPECT_FALSE(deleted1);
    }

    EXPECT_TRUE(deleted1);
}

TEST(Core_Ptr, release)
{
    bool deleted = false;

    Ptr<Reporter> p1(new Reporter(&deleted));
    p1.release();
    EXPECT_TRUE(deleted);
    EXPECT_NULL(p1.get());
}

TEST(Core_Ptr, reset)
{
    bool deleted_old = false, deleted_new = false;

    {
        Ptr<void> p(new Reporter(&deleted_old));
        Reporter* r = new Reporter(&deleted_new);
        p.reset(r);
        EXPECT_TRUE(deleted_old);
        EXPECT_EQ(r, p.get());
    }

    EXPECT_TRUE(deleted_new);

    {
        Ptr<void> p(new Reporter(&deleted_old));
        p.reset(&dummyObject, ReportingDeleter(&deleted_new));
        EXPECT_TRUE(deleted_old);
        EXPECT_EQ(&dummyObject, p.get());
    }

    EXPECT_TRUE(deleted_new);
}

TEST(Core_Ptr, swap)
{
    bool deleted1 = false, deleted2 = false;

    {
        Reporter* r1 = new Reporter(&deleted1);
        Reporter* r2 = new Reporter(&deleted2);
        Ptr<Reporter> p1(r1), p2(r2);
        p1.swap(p2);
        EXPECT_EQ(r1, p2.get());
        EXPECT_EQ(r2, p1.get());
        EXPECT_FALSE(deleted1);
        EXPECT_FALSE(deleted2);
        p1.release();
        EXPECT_TRUE(deleted2);
    }

    EXPECT_TRUE(deleted1);

    {
        Reporter* r1 = new Reporter(&deleted1);
        Reporter* r2 = new Reporter(&deleted2);
        Ptr<Reporter> p1(r1), p2(r2);
        swap(p1, p2);
        EXPECT_EQ(r1, p2.get());
        EXPECT_EQ(r2, p1.get());
        EXPECT_FALSE(deleted1);
        EXPECT_FALSE(deleted2);
        p1.release();
        EXPECT_TRUE(deleted2);
    }

    EXPECT_TRUE(deleted1);
}

TEST(Core_Ptr, accessors)
{
    {
        Ptr<int> p;
        EXPECT_NULL(static_cast<int*>(p));
        EXPECT_TRUE(p.empty());
    }

    {
        Size* s = new Size();
        Ptr<Size> p(s);
        EXPECT_EQ(s, static_cast<Size*>(p));
        EXPECT_EQ(s, &*p);
        EXPECT_EQ(&s->width, &p->width);
        EXPECT_FALSE(p.empty());
    }
}

namespace {

struct SubReporterBase {
    virtual ~SubReporterBase() {}
    int padding;
};

/* multiple inheritance, so that casts do something interesting */
struct SubReporter : SubReporterBase, Reporter
{
    SubReporter(bool* deleted) : Reporter(deleted)
    {}
};

}

TEST(Core_Ptr, casts)
{
    bool deleted = false;

    {
        Ptr<const Reporter> p1(new Reporter(&deleted));
        Ptr<Reporter> p2 = p1.constCast<Reporter>();
        EXPECT_EQ(p1.get(), p2.get());
        p1.release();
        EXPECT_FALSE(deleted);
    }

    EXPECT_TRUE(deleted);

    {
        SubReporter* sr = new SubReporter(&deleted);
        Ptr<Reporter> p1(sr);
        // This next check isn't really for Ptr itself; it checks that Reporter
        // is at a non-zero offset within SubReporter, so that the next
        // check will give us more confidence that the cast actually did something.
        EXPECT_NE(static_cast<void*>(sr), static_cast<void*>(p1.get()));
        Ptr<SubReporter> p2 = p1.staticCast<SubReporter>();
        EXPECT_EQ(sr, p2.get());
        p1.release();
        EXPECT_FALSE(deleted);
    }

    EXPECT_TRUE(deleted);

    {
        SubReporter* sr = new SubReporter(&deleted);
        Ptr<Reporter> p1(sr);
        EXPECT_NE(static_cast<void*>(sr), static_cast<void*>(p1.get()));
        Ptr<void> p2 = p1.dynamicCast<void>();
        EXPECT_EQ(sr, p2.get());
        p1.release();
        EXPECT_FALSE(deleted);
    }

    EXPECT_TRUE(deleted);

    {
        Ptr<Reporter> p1(new Reporter(&deleted));
        Ptr<SubReporter> p2 = p1.dynamicCast<SubReporter>();
        EXPECT_NULL(p2.get());
        p1.release();
        EXPECT_TRUE(deleted);
    }

    EXPECT_TRUE(deleted);
}

TEST(Core_Ptr, comparisons)
{
    Ptr<int> p1, p2(new int), p3(new int);
    Ptr<int> p4(p2, p3.get());

    // Not using EXPECT_EQ here, since none of them are really "expected" or "actual".
    EXPECT_TRUE(p1 == p1);
    EXPECT_TRUE(p2 == p2);
    EXPECT_TRUE(p2 != p3);
    EXPECT_TRUE(p2 != p4);
    EXPECT_TRUE(p3 == p4);
}

TEST(Core_Ptr, make)
{
    bool deleted = true;

    {
        Ptr<void> p = makePtr<Reporter>(&deleted);
        EXPECT_FALSE(deleted);
    }

    EXPECT_TRUE(deleted);
}

}} // namespace

namespace {

struct SpeciallyDeletable
{
    SpeciallyDeletable() : deleted(false)
    {}
    bool deleted;
};

} // namespace

namespace cv {
template<> struct DefaultDeleter<SpeciallyDeletable>
{
    void operator()(SpeciallyDeletable * obj) const { obj->deleted = true; }
};
} // namespace

namespace opencv_test { namespace {

TEST(Core_Ptr, specialized_deleter)
{
    SpeciallyDeletable sd;

    { Ptr<void> p(&sd); }

    ASSERT_TRUE(sd.deleted);
}

TEST(Core_Ptr, specialized_deleter_via_reset)
{
    SpeciallyDeletable sd;

    {
        Ptr<SpeciallyDeletable> p;
        p.reset(&sd);
    }

    ASSERT_TRUE(sd.deleted);
}

}} // namespace
