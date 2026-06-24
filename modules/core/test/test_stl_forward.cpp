// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "opencv2/core/stl/algorithm.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/detail/cpp_features.hpp"
#include <numeric>

namespace opencv_test { namespace {

// The fixture for testing class Foo.
class CORE_stl_forward : public ::testing::Test {
 protected:

    void SetUp() override
    {
        size_t size = 13;
        mat = cv::Mat(static_cast<int>(size),static_cast<int>(size),CV_32S);
        mat_f = cv::Mat_<float>(static_cast<int>(size),static_cast<int>(size));


        intVec = std::vector<int>(size*size);
        intList = std::list<int>(size*size);

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

//This opencv feature requires return type auto a C++14 feature
#ifdef _stl_forward_cpp_features_present

TEST_F(CORE_stl_forward, iterators_replacable)
{
        ///////////////////////////////////Contiguous Tests/////////////////////////////////
    //Test only one "span" where we can replace iterators. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__it_replacable(mat.begin<int>(), mat.end<int>(),[](int val){return val*2;}));

    //Should be false because there is no cv::Mat iterator. Lambda doesn't have an effect
    EXPECT_FALSE(cv::detail::__it_replacable(intVec.begin(), intVec.end(),[](int val){return val*2;}));

    //Should be true because there is only one cv::Mat iterator of a contiguous iterator. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__it_replacable(intVec.begin(), intVec.end(), mat.begin<int>(),[](int val){return val*2;}));

    ///////////////////////////////////Submatrix Tests/////////////////////////////////
    //Test only one "span" where we can't replace iterators (not contiguous)
    EXPECT_FALSE(cv::detail::__it_replacable(subMat.begin(), subMat.end()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__it_replacable(intVec.begin(), intVec.end(), subMat.begin()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__it_replacable(intVec.begin(), intVec.end(), subMat.begin()));
}

TEST_F(CORE_stl_forward, iterators_replacable_reverse_iterator)
{
    ///////////////////////////////////Contiguous Tests/////////////////////////////////
    //Test only one "span" where we can replace iterators. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__it_replacable(mat.rbegin<int>(), mat.rend<int>(),[](uchar val){return val*2;}));

    //Should be false because there is no cv::Mat iterator. Lambda doesn't have an effect
    EXPECT_FALSE(cv::detail::__it_replacable(intVec.rbegin(), intVec.rend(),[](uchar val){return val*2;}));

    //Should be true because there is no cv::Mat iterator of a non contiguous iterator. Lambda doesn't have an effect
    EXPECT_TRUE(cv::detail::__it_replacable(intVec.begin(), intVec.end(), mat.begin<int>(),[](uchar val){return val*2;}));

    ///////////////////////////////////Submatrix Tests/////////////////////////////////
    //Test only one "span" where we can't replace iterators (not contiguous)
    EXPECT_FALSE(cv::detail::__it_replacable(subMat.rbegin(), subMat.rend()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__it_replacable(intVec.begin(), intVec.end(), subMat.rbegin()));

    //Should be false because there is a cv::Mat iterator of a non-contiguous matrix
    EXPECT_FALSE(cv::detail::__it_replacable(intVec.rbegin(), intVec.end(), subMat.rbegin()));
}

TEST_F(CORE_stl_forward, tuple_replacer_single)
{
    static_assert(std::is_same<decltype(cv::detail::get_replaced_val(mat_f.begin())), float*>::value, "Couldn't replace cv iterator");
    static_assert(std::is_same<decltype(cv::detail::get_replaced_val(mat.end<int>())), int*>::value, "Couldn't replace cv iterator");
    static_assert(!std::is_same<decltype(cv::detail::get_replaced_val(intVec.begin())), int*>::value, "Replaced non cv it while it shouldn't");
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

TEST_F(CORE_stl_forward, reverse_tuple_replacer_single)
{
    static_assert(std::is_same<decltype(cv::detail::get_replaced_val(mat_f.rbegin())), std::reverse_iterator<float*>>::value, "Couldn't replace cv iterator");
    static_assert(std::is_same<decltype(cv::detail::get_replaced_val(mat.rend<int>())), std::reverse_iterator<int*>>::value, "Couldn't replace cv iterator");
    static_assert(!std::is_same<decltype(cv::detail::get_replaced_val(intVec.rbegin())), int*>::value, "Replaced non cv it while it shouldn't");
}

TEST_F(CORE_stl_forward, reverse_tuple_replacer)
{
    //Simple example of only opencv iterators being replaced by their pointers
    auto itReplace= cv::detail::make_tpl_replaced(mat_f.rbegin(), mat_f.rend());
    static_assert(std::is_same<decltype(itReplace), std::tuple<std::reverse_iterator<float*>,std::reverse_iterator<float*>>>::value,"CV iterators not replaced with their pointers.");

    //Check the pointer values
    EXPECT_EQ((void*)std::get<0>(itReplace).base(), (void*)mat_f.end().ptr) << "In replaced tuple: Pointers not pointing to expected location";
    EXPECT_EQ((void*)std::get<1>(itReplace).base(), (void*)mat_f.begin().ptr) << "In replaced tuple: Pointers not pointing to expected location";


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

TEST_F(CORE_stl_forward, __get_first_cv_it_index)
{
    static_assert(cv::detail::__get_first_cv_it_index<decltype(mat.begin<int>())>() == 0, "Wrong index chosen");
    static_assert(cv::detail::__get_first_cv_it_index<decltype(mat.begin<int>()),decltype(mat.end<int>())>() == 0, "Wrong index chosen");

    static_assert(cv::detail::__get_first_cv_it_index<decltype(intVec.begin()), decltype(intVec.begin())>() == 0, "Wrong index chosen");
    static_assert(cv::detail::__get_first_cv_it_index<decltype(intVec.begin()), decltype(intVec.begin()), decltype(mat.begin<int>())>() == 2, "Wrong index chosen");
    static_assert(cv::detail::__get_first_cv_it_index<decltype(intVec.begin()), decltype(mat.begin<int>()), decltype(intVec.begin())>() == 1, "Wrong index chosen");

}

TEST_F(CORE_stl_forward, is_base_of_reverse)
{
    static_assert(cv::detail::is_base_of_reverse<cv::MatConstIterator, decltype(mat.rbegin<int>())>::value,"");
    static_assert(cv::detail::is_base_of_reverse<cv::MatConstIterator, decltype(mat.begin<int>())>::value,"");
    static_assert(cv::detail::is_base_of_reverse<cv::MatConstIterator, decltype(mat.rend<double>())>::value,"");

    static_assert(!cv::detail::is_base_of_reverse<cv::MatConstIterator, decltype(intVec.begin())>::value,"");
    static_assert(!cv::detail::is_base_of_reverse<cv::MatConstIterator, decltype(intVec.rbegin())>::value,"");
    static_assert(!cv::detail::is_base_of_reverse<cv::MatConstIterator, decltype(intVec.end())>::value,"");

    static_assert(!cv::detail::is_base_of_reverse<cv::MatConstIterator, int>::value,"");
}


TEST_F(CORE_stl_forward, __get_first_cv_it_index_reverse)
{
    static_assert(cv::detail::__get_first_cv_it_index<decltype(mat.rbegin<int>())>() == 0, "Wrong index chosen");
    static_assert(cv::detail::__get_first_cv_it_index<decltype(mat.rbegin<int>()),decltype(mat.rend<int>())>() == 0, "Wrong index chosen");

    static_assert(cv::detail::__get_first_cv_it_index<decltype(intVec.rbegin()), decltype(intVec.rbegin())>() == 0, "Wrong index chosen");

    static_assert(cv::detail::__get_first_cv_it_index<decltype(intVec.rbegin()), decltype(intVec.rbegin()), decltype(mat.rbegin<int>())>() == 2, "Wrong index chosen");
    static_assert(cv::detail::__get_first_cv_it_index<decltype(intVec.rbegin()), decltype(mat.rbegin<int>()), decltype(intVec.rbegin())>() == 1, "Wrong index chosen");
}



TEST_F(CORE_stl_forward, tuple_replacer_reverse_iterator)
{
    //Simple example of only opencv iterators being replaced by their pointers
    auto itReplace= cv::detail::make_tpl_replaced(mat_f.rbegin(), mat_f.rend());
    static_assert(std::is_same<decltype(itReplace), std::tuple<std::reverse_iterator<float*>,std::reverse_iterator<float*>>>::value,"CV iterators not replaced with their pointers.");

    //Check the pointer values
    EXPECT_EQ((void*)std::get<0>(itReplace).base(), (void*)mat_f.rbegin().base().ptr) << "In replaced tuple: Pointers not pointing to expected location";
    EXPECT_EQ((void*)std::get<1>(itReplace).base(), (void*)mat_f.rend().base().ptr) << "In replaced tuple: Pointers not pointing to expected location";


    //This seems like it shouldn't compile because it is not a valid operation to do this on submatrices.
    //This however is the purpose of the run-time function __iterators__replaceable! We can't do this at compile time
    auto itReplaceSub= cv::detail::make_tpl_replaced(subMat.rbegin(), subMat.rend());
    static_assert(std::is_same<decltype(itReplaceSub), std::tuple<std::reverse_iterator<int*>,std::reverse_iterator<int*>>>::value,"CV iterators not replaced with their pointers.");

    //Mixed opencv and other iterator (list)
    auto itReplaceMixed= cv::detail::make_tpl_replaced(mat.rbegin<int>(), mat.rend<int>(),intList.rbegin());
    static_assert(std::is_same<decltype(itReplaceMixed), std::tuple<std::reverse_iterator<int*>,std::reverse_iterator<int*>,decltype (intList.rbegin())>>::value,"CV iterators not replaced with their pointers.");

    //Turn order around: list first
    auto itReplaceMixed_order_reversed= cv::detail::make_tpl_replaced(intList.begin(),mat.rbegin<int>(), mat.rend<int>());
    static_assert(std::is_same<decltype(itReplaceMixed_order_reversed), std::tuple<decltype (intList.begin()),std::reverse_iterator<int*>,std::reverse_iterator<int*>>>::value,"CV iterators not replaced with their pointers.");


    //Test with a lambda. decltype of a lambda isn't really specified. So we compare only the first elements
    auto itReplaced_lambda= cv::detail::make_tpl_replaced(intList.begin(),mat.rbegin<int>(), mat.end<int>(),[](int val){return 2*val;});
    static_assert(std::is_same<std::tuple_element<0,decltype(itReplaced_lambda)>::type,decltype(intList.begin())>::value,"CV iterators not replaced with their pointers.");
    static_assert(std::is_same<std::tuple_element<1,decltype(itReplaced_lambda)>::type,std::reverse_iterator<int*>>::value,"CV iterators not replaced with their pointers.");
    static_assert(std::is_same<std::tuple_element<2,decltype(itReplaced_lambda)>::type,int*>::value,"CV iterators not replaced with their pointers.");
}


TEST_F(CORE_stl_forward, count_if_test)
{
    auto lambda = [](int val){return val >13 && val < 100;};

    //This test is with replacable iterators.
    EXPECT_TRUE(cv::detail::__it_replacable(mat.begin<int>(), mat.end<int>(),lambda));

    //Test replaced iterators vs. normal stl algo
    EXPECT_EQ(experimental::count_if(mat.begin<int>(), mat.end<int>(),lambda), std::count_if(mat.begin<int>(), mat.end<int>(),lambda));
    EXPECT_EQ(experimental::count_if(mat.rbegin<int>(), mat.rend<int>(),lambda), std::count_if(mat.rbegin<int>(), mat.rend<int>(),lambda));
    EXPECT_EQ(experimental::count_if(mat.begin<int>(), mat.end<int>(),lambda), std::count_if((int*)mat.begin<int>().ptr, (int*)mat.end<int>().ptr,lambda));
}

TEST_F(CORE_stl_forward, find_test)
{
    //Test replaced iterators vs. normal stl algo
    EXPECT_EQ(*experimental::find(mat_f.begin(), mat_f.end(),5),*std::find(mat_f.begin(), mat_f.end(),5));
    EXPECT_EQ(experimental::find(mat_f.begin(), mat_f.end(),5),std::find(mat_f.begin(), mat_f.end(),5));

    //Test reverse iterator vs stl algo
    EXPECT_EQ(*experimental::find(mat_f.rbegin(), mat_f.rend(),5), *std::find(mat_f.rbegin(), mat_f.rend(),5));
    EXPECT_EQ(experimental::find(mat_f.rbegin(), mat_f.rend(),5), std::find(mat_f.rbegin(), mat_f.rend(),5));

    EXPECT_EQ(*experimental::find(intVec.begin(), intVec.end(),5),*std::find(intVec.begin(), intVec.end(),5));
    EXPECT_EQ(*experimental::find(intVec.rbegin(), intVec.rend(),5),*std::find(intVec.rbegin(), intVec.rend(),5));

    std::ptrdiff_t replaced_dist = experimental::find(mat.begin<int>(), mat.end<int>(),10) - mat.begin<int>();
    std::ptrdiff_t orig_dist = std::find(mat.begin<int>(), mat.end<int>(),10) - mat.begin<int>();

    EXPECT_EQ(replaced_dist, orig_dist);
}

#endif //_stl_forward_cpp_features_present


}} // namespace
