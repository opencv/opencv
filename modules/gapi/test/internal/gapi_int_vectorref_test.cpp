// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

namespace opencv_test
{

typedef ::testing::Types<int, cv::Point, cv::Rect> VectorRef_Test_Types;

template<typename T> struct VectorRefT: public ::testing::Test { using Type = T; };

TYPED_TEST_CASE(VectorRefT, VectorRef_Test_Types);

TYPED_TEST(VectorRefT, Reset_Valid)
{
    using T = typename TestFixture::Type;
    cv::detail::VectorRefT<T> ref;       // vector ref created empty
    EXPECT_NO_THROW(ref.reset());        // 1st reset is OK (initializes)
    EXPECT_NO_THROW(ref.reset());        // 2nd reset is also OK (resets)
}

TYPED_TEST(VectorRefT, Reset_Invalid)
{
    using T = typename TestFixture::Type;
    std::vector<T> vec(42);              // create a std::vector of 42 elements
    cv::detail::VectorRefT<T> ref(vec);  // RO_EXT (since reference is const)
    EXPECT_ANY_THROW(ref.reset());       // data-bound vector ref can't be reset
}

TYPED_TEST(VectorRefT, ReadRef_External)
{
    using T = typename TestFixture::Type;
    const std::vector<T> vec(42);        // create a std::vector of 42 elements
    cv::detail::VectorRefT<T> ref(vec);  // RO_EXT (since reference is const)
    auto &vref = ref.rref();
    EXPECT_EQ(vec.data(), vref.data());
    EXPECT_EQ(vec.size(), vref.size());
}

TYPED_TEST(VectorRefT, ReadRef_Internal)
{
    using T = typename TestFixture::Type;
    cv::detail::VectorRefT<T> ref;
    ref.reset();                         // RW_OWN (reset on empty ref)
    auto &vref = ref.rref();             // read access is valid for RW_OWN
    EXPECT_EQ(0u, vref.size());          // by default vector is empty
}

TYPED_TEST(VectorRefT, WriteRef_External)
{
    using T = typename TestFixture::Type;
    std::vector<T> vec(42);               // create a std::vector of 42 elements
    cv::detail::VectorRefT<T> ref(vec);   // RW_EXT (since reference is not const)
    auto &vref = ref.wref();              // write access is valid with RW_EXT
    EXPECT_EQ(vec.data(), vref.data());
    EXPECT_EQ(vec.size(), vref.size());
}

TYPED_TEST(VectorRefT, WriteRef_Internal)
{
    using T = typename TestFixture::Type;
    cv::detail::VectorRefT<T> ref;
    ref.reset();                          // RW_OWN (reset on empty ref)
    auto &vref = ref.wref();              // write access is valid for RW_OWN
    EXPECT_EQ(0u, vref.size());           // empty vector by default
}

TYPED_TEST(VectorRefT, WriteToRO)
{
    using T = typename TestFixture::Type;
    const std::vector<T> vec(42);        // create a std::vector of 42 elements
    cv::detail::VectorRefT<T> ref(vec);  // RO_EXT (since reference is const)
    EXPECT_ANY_THROW(ref.wref());
}

TYPED_TEST(VectorRefT, ReadAfterWrite)
{
    using T = typename TestFixture::Type;
    std::vector<T> vec;                        // Initial data holder (empty vector)
    cv::detail::VectorRefT<T> writer(vec);     // RW_EXT

    const auto& ro_ref = vec;
    cv::detail::VectorRefT<T> reader(ro_ref);  // RO_EXT

    EXPECT_EQ(0u, writer.wref().size()); // Check the initial state
    EXPECT_EQ(0u, reader.rref().size());

    writer.wref().emplace_back();        // Check that write is successful
    EXPECT_EQ(1u, writer.wref().size());

    EXPECT_EQ(1u, vec.size());           // Check that changes are reflected to the original container
    EXPECT_EQ(1u, reader.rref().size()); // Check that changes are reflected to reader's view

    EXPECT_EQ(T(), vec.at(0));           // Check the value (must be default-initialized)
    EXPECT_EQ(T(), reader.rref().at(0));
    EXPECT_EQ(T(), writer.wref().at(0));
}

template<typename T> struct VectorRefU: public ::testing::Test { using Type = T; };

TYPED_TEST_CASE(VectorRefU, VectorRef_Test_Types);

template<class T> struct custom_struct { T a; T b; };

TYPED_TEST(VectorRefU, Reset_Valid)
{
    using T = typename TestFixture::Type;
    cv::detail::VectorRef ref;           // vector ref created empty
    EXPECT_NO_THROW(ref.reset<T>());     // 1st reset is OK (initializes)
    EXPECT_NO_THROW(ref.reset<T>());     // 2nd reset is also OK (resets)

    EXPECT_ANY_THROW(ref.reset<custom_struct<T> >()); // type change is not allowed
}

TYPED_TEST(VectorRefU, Reset_Invalid)
{
    using T = typename TestFixture::Type;
    std::vector<T> vec(42);              // create a std::vector of 42 elements
    cv::detail::VectorRef ref(vec);      // RO_EXT (since reference is const)
    EXPECT_ANY_THROW(ref.reset<T>());    // data-bound vector ref can't be reset
}

TYPED_TEST(VectorRefU, ReadRef_External)
{
    using T = typename TestFixture::Type;
    const std::vector<T> vec(42);        // create a std::vector of 42 elements
    cv::detail::VectorRef ref(vec);      // RO_EXT (since reference is const)
    auto &vref = ref.rref<T>();
    EXPECT_EQ(vec.data(), vref.data());
    EXPECT_EQ(vec.size(), vref.size());
}

TYPED_TEST(VectorRefU, ReadRef_Internal)
{
    using T = typename TestFixture::Type;
    cv::detail::VectorRef ref;
    ref.reset<T>();                      // RW_OWN (reset on empty ref)
    auto &vref = ref.rref<T>();          // read access is valid for RW_OWN
    EXPECT_EQ(0u, vref.size());          // by default vector is empty
}

TYPED_TEST(VectorRefU, WriteRef_External)
{
    using T = typename TestFixture::Type;
    std::vector<T> vec(42);             // create a std::vector of 42 elements
    cv::detail::VectorRef ref(vec);     // RW_EXT (since reference is not const)
    auto &vref = ref.wref<T>();         // write access is valid with RW_EXT
    EXPECT_EQ(vec.data(), vref.data());
    EXPECT_EQ(vec.size(), vref.size());
}

TYPED_TEST(VectorRefU, WriteRef_Internal)
{
    using T = typename TestFixture::Type;
    cv::detail::VectorRef ref;
    ref.reset<T>();                     // RW_OWN (reset on empty ref)
    auto &vref = ref.wref<T>();         // write access is valid for RW_OWN
    EXPECT_EQ(0u, vref.size());         // empty vector by default
}

TYPED_TEST(VectorRefU, WriteToRO)
{
    using T = typename TestFixture::Type;
    const std::vector<T> vec(42);       // create a std::vector of 42 elements
    cv::detail::VectorRef ref(vec);     // RO_EXT (since reference is const)
    EXPECT_ANY_THROW(ref.wref<T>());
}

TYPED_TEST(VectorRefU, ReadAfterWrite)
{
    using T = typename TestFixture::Type;
    std::vector<T> vec;                     // Initial data holder (empty vector)
    cv::detail::VectorRef writer(vec);      // RW_EXT

    const auto& ro_ref = vec;
    cv::detail::VectorRef reader(ro_ref);   // RO_EXT

    EXPECT_EQ(0u, writer.wref<T>().size()); // Check the initial state
    EXPECT_EQ(0u, reader.rref<T>().size());

    writer.wref<T>().emplace_back();        // Check that write is successful
    EXPECT_EQ(1u, writer.wref<T>().size());

    EXPECT_EQ(1u, vec.size());              // Check that changes are reflected to the original container
    EXPECT_EQ(1u, reader.rref<T>().size()); // Check that changes are reflected to reader's view

    EXPECT_EQ(T(), vec.at(0));              // Check the value (must be default-initialized)
    EXPECT_EQ(T(), reader.rref<T>().at(0));
    EXPECT_EQ(T(), writer.wref<T>().at(0));
}

TEST(VectorRefU, TypeCheck)
{
    cv::detail::VectorRef ref;
    ref.reset<int>(); // RW_OWN

    EXPECT_ANY_THROW(ref.reset<char>());
    EXPECT_ANY_THROW(ref.rref<char>());
    EXPECT_ANY_THROW(ref.wref<char>());
}

} // namespace opencv_test
