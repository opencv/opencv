// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "opencv2/core/stl/algorithm.hh"
#include "opencv2/core.hpp"

namespace opencv_test { namespace {

// The fixture for testing class Foo.
class CORE_stl_forward : public ::testing::Test {
 protected:

    void SetUp() override
    {
        mat = cv::Mat(13,13,CV_32S);
        mat_f = cv::Mat_<float>(13,13);


        intVec = std::vector<int>(13*13);
        intList = std::list<int>(13*13);

        subMat = mat(cv::Rect(2,2,7,7));
        matSub_f = mat_f(cv::Rect(2,2,7,7));

        std::iota(mat.begin<int>(), mat.end<int>(), -1);
        std::iota(mat_f.begin(), mat_f.end(), -1.0f);

        std::iota(intVec.begin(), intVec.end(), -1);
        std::iota(intList.begin(), intList.end(), -1);

        std::iota(subMat.begin(), subMat.end(), -1);
        std::iota(matSub_f.begin(), matSub_f.end(), -1.0f);
    }

    void TearDown() override
    {

    }
    cv::Mat mat;
    cv::Mat_<float> mat_f;

    std::vector<int> intVec;
    std::list<int> intList;

    cv::Mat_<int> subMat;
    cv::Mat_<float> matSub_f;
};

TEST_F(CORE_stl_forward, iterators_replacable)
{
        ///////////////////////////////////Contiguous Tests/////////////////////////////////
    //Test only one "span" where we can replace iterators. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__iterators__replaceable(mat.begin<int>(), mat.end<int>(),[](int val){return val*2;}));

    //Should be false because there is no cv::Mat iterator. Lambda doesn't have an effect
    EXPECT_FALSE(cv::detail::__iterators__replaceable(intVec.begin(), intVec.end(),[](int val){return val*2;}));

    //Should be true because there is only one cv::Mat iterator of a contiguous iterator. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__iterators__replaceable(intVec.begin(), intVec.end(), mat.begin<int>(),[](int val){return val*2;}));

    ///////////////////////////////////Submatrix Tests/////////////////////////////////
    //Test only one "span" where we can't replace iterators (not contiguous)
    EXPECT_FALSE(cv::detail::__iterators__replaceable(subMat.begin(), subMat.end()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__iterators__replaceable(intVec.begin(), intVec.end(), subMat.begin()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__iterators__replaceable(intVec.begin(), intVec.end(), subMat.begin()));
}

TEST_F(CORE_stl_forward, DISABLED_iterators_replacable_reverse_iterator)
{
    ///////////////////////////////////Contiguous Tests/////////////////////////////////
    //Test only one "span" where we can replace iterators. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__iterators__replaceable(mat.rbegin<uchar>(), mat.rend<uchar>(),[](uchar val){return val*2;}));

    //Should be false because there is no cv::Mat iterator. Lambda doesn't have an effect
    EXPECT_FALSE(cv::detail::__iterators__replaceable(intVec.rbegin(), intVec.rend(),[](uchar val){return val*2;}));

    //Should be true because there is no cv::Mat iterator of a non contiguous iterator. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__iterators__replaceable(intVec.rbegin(), intVec.rend(), mat.rbegin<uchar>(),[](uchar val){return val*2;}));

    ///////////////////////////////////Submatrix Tests/////////////////////////////////
    //Test only one "span" where we can't replace iterators (not contiguous)
    EXPECT_FALSE(cv::detail::__iterators__replaceable(subMat.rbegin(), subMat.rend()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__iterators__replaceable(intVec.begin(), intVec.end(), subMat.rbegin()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__iterators__replaceable(intVec.rbegin(), intVec.end(), subMat.rbegin()));
}

TEST_F(CORE_stl_forward, tuple_replacer)
{
    //Simple example of only opencv iterators being replaced by their pointers
    auto itReplace= cv::detail::make_tpl_replaced(mat_f.begin(), mat_f.end());
    static_assert(std::is_same<decltype(itReplace), std::tuple<float*,float*>>::value,"CV iterators not replaced with their pointers.");

    //Check the pointer values
    EXPECT_EQ((void*)std::get<0>(itReplace), (void*)mat_f.begin().ptr) << "In replaced tuple: Pointers not pointing to expected location";
    EXPECT_EQ((void*)std::get<1>(itReplace), (void*)mat_f.end().ptr) << "In replaced tuple: Pointers not pointing to expected location";


    //This seems like it shouldn't compile because it is not a valid operation to do this on submatrices.
    //This however is the purpose of the run-time function __iterators__replaceable! We can't do this at compile time
    auto itReplaceSub= cv::detail::make_tpl_replaced(subMat.begin(), subMat.end());
    static_assert(std::is_same<decltype(itReplaceSub), std::tuple<int*,int*>>::value,"CV iterators not replaced with their pointers.");

    //Mixed opencv and other iterator (list)
    auto itReplaceMixed= cv::detail::make_tpl_replaced(mat.begin<int>(), mat.end<int>(),intList.begin());
    static_assert(std::is_same<decltype(itReplaceMixed), std::tuple<int*,int*,decltype (intList.begin())>>::value,"CV iterators not replaced with their pointers.");

    //Turn order around: list first
    auto itReplaceMixed_order_reversed= cv::detail::make_tpl_replaced(intList.begin(),mat.begin<int>(), mat.end<int>());
    static_assert(std::is_same<decltype(itReplaceMixed_order_reversed), std::tuple<decltype (intList.begin()),int*,int*>>::value,"CV iterators not replaced with their pointers.");


    //Test with a lambda. decltype of a lambda isn't really specified. So we compare only the first elements
    auto itReplaced_lambda= cv::detail::make_tpl_replaced(intList.begin(),mat.begin<int>(), mat.end<int>(),[](int val){return 2*val;});
    static_assert(std::is_same<std::tuple_element<0,decltype(itReplaced_lambda)>::type,decltype(intList.begin())>::value,"CV iterators not replaced with their pointers.");
    static_assert(std::is_same<std::tuple_element<1,decltype(itReplaced_lambda)>::type,int*>::value,"CV iterators not replaced with their pointers.");
    static_assert(std::is_same<std::tuple_element<2,decltype(itReplaced_lambda)>::type,int*>::value,"CV iterators not replaced with their pointers.");
}


TEST_F(CORE_stl_forward, DISABLED_tuple_replacer_reverse_iterator)
{
    //Simple example of only opencv iterators being replaced by their pointers
    auto itReplace= cv::detail::make_tpl_replaced(mat_f.begin(), mat_f.end());
    static_assert(std::is_same<decltype(itReplace), std::tuple<float*,float*>>::value,"CV iterators not replaced with their pointers.");

    //Check the pointer values
    EXPECT_EQ((void*)std::get<0>(itReplace), (void*)mat_f.begin().ptr) << "In replaced tuple: Pointers not pointing to expected location";
    EXPECT_EQ((void*)std::get<1>(itReplace), (void*)mat_f.end().ptr) << "In replaced tuple: Pointers not pointing to expected location";


    //This seems like it shouldn't compile because it is not a valid operation to do this on submatrices.
    //This however is the purpose of the run-time function __iterators__replaceable! We can't do this at compile time
    auto itReplaceSub= cv::detail::make_tpl_replaced(subMat.begin(), subMat.end());
    static_assert(std::is_same<decltype(itReplaceSub), std::tuple<int*,int*>>::value,"CV iterators not replaced with their pointers.");

    //Mixed opencv and other iterator (list)
    auto itReplaceMixed= cv::detail::make_tpl_replaced(mat.begin<int>(), mat.end<int>(),intList.begin());
    static_assert(std::is_same<decltype(itReplaceMixed), std::tuple<int*,int*,decltype (intList.begin())>>::value,"CV iterators not replaced with their pointers.");

    //Turn order around: list first
    auto itReplaceMixed_order_reversed= cv::detail::make_tpl_replaced(intList.begin(),mat.begin<int>(), mat.end<int>());
    static_assert(std::is_same<decltype(itReplaceMixed_order_reversed), std::tuple<decltype (intList.begin()),int*,int*>>::value,"CV iterators not replaced with their pointers.");


    //Test with a lambda. decltype of a lambda isn't really specified. So we compare only the first elements
    auto itReplaced_lambda= cv::detail::make_tpl_replaced(intList.begin(),mat.begin<int>(), mat.end<int>(),[](int val){return 2*val;});
    static_assert(std::is_same<std::tuple_element<0,decltype(itReplaced_lambda)>::type,decltype(intList.begin())>::value,"CV iterators not replaced with their pointers.");
    static_assert(std::is_same<std::tuple_element<1,decltype(itReplaced_lambda)>::type,int*>::value,"CV iterators not replaced with their pointers.");
    static_assert(std::is_same<std::tuple_element<2,decltype(itReplaced_lambda)>::type,int*>::value,"CV iterators not replaced with their pointers.");
}


TEST_F(CORE_stl_forward, count_if_test)
{
    auto lambda = [](int val){return val >13 && val < 100;};

    //This test is with replacable iterators.
    EXPECT_TRUE(cv::detail::__iterators__replaceable(mat.begin<int>(), mat.end<int>(),lambda));

    //Test replaced iterators vs. normal stl algo
    EXPECT_EQ(experimental::count_if(mat.begin<int>(), mat.end<int>(),lambda), std::count_if(mat.begin<int>(), mat.end<int>(),lambda));
    EXPECT_EQ(experimental::count_if(mat.begin<int>(), mat.end<int>(),lambda), std::count_if((int*)mat.begin<int>().ptr, (int*)mat.end<int>().ptr,lambda));
}


}} // namespace
