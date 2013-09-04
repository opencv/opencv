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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@outlook.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <map>
#include <functional>
#include "test_precomp.hpp"

using namespace std;
using namespace cvtest;
using namespace testing;
using namespace cv;


namespace
{
IMPLEMENT_PARAM_CLASS(IsGreaterThan, bool)
IMPLEMENT_PARAM_CLASS(InputSize, int)
IMPLEMENT_PARAM_CLASS(SortMethod, int)


template<class T>
struct KV_CVTYPE{ static int toType() {return 0;} };

template<> struct KV_CVTYPE<int>  { static int toType() {return CV_32SC1;} };
template<> struct KV_CVTYPE<float>{ static int toType() {return CV_32FC1;} };
template<> struct KV_CVTYPE<Vec2i>{ static int toType() {return CV_32SC2;} };
template<> struct KV_CVTYPE<Vec2f>{ static int toType() {return CV_32FC2;} };

template<class key_type, class val_type>
bool kvgreater(pair<key_type, val_type> p1, pair<key_type, val_type> p2)
{
    return p1.first > p2.first;
}

template<class key_type, class val_type>
bool kvless(pair<key_type, val_type> p1, pair<key_type, val_type> p2)
{
    return p1.first < p2.first;
}

template<class key_type, class val_type>
void toKVPair(
    MatConstIterator_<key_type> kit,
    MatConstIterator_<val_type> vit,
    int vecSize,
    vector<pair<key_type, val_type> >& kvres
    )
{
    kvres.clear();
    for(int i = 0; i < vecSize; i ++)
    {
        kvres.push_back(make_pair(*kit, *vit));
        ++kit;
        ++vit;
    }
}

template<class key_type, class val_type>
void kvquicksort(Mat& keys, Mat& vals, bool isGreater = false)
{
    vector<pair<key_type, val_type> > kvres;
    toKVPair(keys.begin<key_type>(), vals.begin<val_type>(), keys.cols, kvres);

    if(isGreater)
    {
        std::sort(kvres.begin(), kvres.end(), kvgreater<key_type, val_type>);
    }
    else
    {
        std::sort(kvres.begin(), kvres.end(), kvless<key_type, val_type>);
    }
    key_type * kptr = keys.ptr<key_type>();
    val_type * vptr = vals.ptr<val_type>();
    for(int i = 0; i < keys.cols; i ++)
    {
        kptr[i] = kvres[i].first;
        vptr[i] = kvres[i].second;
    }
}

class SortByKey_STL
{
public:
    static void sort(cv::Mat&, cv::Mat&, bool is_gt);
private:
    typedef void (*quick_sorter)(cv::Mat&, cv::Mat&, bool);
    SortByKey_STL();
    quick_sorter quick_sorters[CV_64FC4][CV_64FC4];
    static SortByKey_STL instance;
};

SortByKey_STL SortByKey_STL::instance = SortByKey_STL();

SortByKey_STL::SortByKey_STL()
{
    memset(instance.quick_sorters, 0, sizeof(quick_sorters));
#define NEW_SORTER(KT, VT) \
    instance.quick_sorters[KV_CVTYPE<KT>::toType()][KV_CVTYPE<VT>::toType()] = kvquicksort<KT, VT>;

    NEW_SORTER(int, int);
    NEW_SORTER(int, Vec2i);
    NEW_SORTER(int, float);
    NEW_SORTER(int, Vec2f);

    NEW_SORTER(float, int);
    NEW_SORTER(float, Vec2i);
    NEW_SORTER(float, float);
    NEW_SORTER(float, Vec2f);
#undef NEW_SORTER
}

void SortByKey_STL::sort(cv::Mat& keys, cv::Mat& vals, bool is_gt)
{
    instance.quick_sorters[keys.type()][vals.type()](keys, vals, is_gt);
}

bool checkUnstableSorterResult(const Mat& gkeys_, const Mat& gvals_,
                               const Mat& /*dkeys_*/, const Mat& dvals_)
{
    int cn_val = gvals_.channels();
    int count  = gkeys_.cols;

    //for convenience we convert depth to float and channels to 1
    Mat gkeys, gvals, dkeys, dvals;
    gkeys_.reshape(1).convertTo(gkeys, CV_32F);
    gvals_.reshape(1).convertTo(gvals, CV_32F);
    //dkeys_.reshape(1).convertTo(dkeys, CV_32F);
    dvals_.reshape(1).convertTo(dvals, CV_32F);
    float * gkptr = gkeys.ptr<float>();
    float * gvptr = gvals.ptr<float>();
    //float * dkptr = dkeys.ptr<float>();
    float * dvptr = dvals.ptr<float>();

    for(int i = 0; i < count - 1; ++i)
    {
        int iden_count = 0;
        // firstly calculate the number of identical keys
        while(gkptr[i + iden_count] == gkptr[i + 1 + iden_count])
        {
            ++ iden_count;
        }

        // sort dv and gv
        int num_of_val = (iden_count + 1) * cn_val;
        std::sort(gvptr + i * cn_val, gvptr + i * cn_val + num_of_val);
        std::sort(dvptr + i * cn_val, dvptr + i * cn_val + num_of_val);

        // then check if [i, i + iden_count) is the same
        for(int j = 0; j < num_of_val; ++j)
        {
            if(gvptr[i + j] != dvptr[i + j])
            {
                return false;
            }
        }
        i += iden_count;
    }
    return true;
}
}

#define INPUT_SIZES  Values(InputSize(0x10), InputSize(0x100), InputSize(0x10000)) //2^4, 2^8, 2^16
#define KEY_TYPES    Values(MatType(CV_32SC1), MatType(CV_32FC1))
#define VAL_TYPES    Values(MatType(CV_32SC1), MatType(CV_32SC2), MatType(CV_32FC1), MatType(CV_32FC2))
#define SORT_METHODS Values(SortMethod(cv::ocl::SORT_BITONIC),SortMethod(cv::ocl::SORT_MERGE),SortMethod(cv::ocl::SORT_RADIX)/*,SortMethod(cv::ocl::SORT_SELECTION)*/)
#define F_OR_T       Values(IsGreaterThan(false), IsGreaterThan(true))

PARAM_TEST_CASE(SortByKey, InputSize, MatType, MatType, SortMethod, IsGreaterThan)
{
    InputSize input_size;
    MatType key_type, val_type;
    SortMethod method;
    IsGreaterThan is_gt;

    Mat mat_key, mat_val;
    virtual void SetUp()
    {
        input_size = GET_PARAM(0);
        key_type   = GET_PARAM(1);
        val_type   = GET_PARAM(2);
        method     = GET_PARAM(3);
        is_gt      = GET_PARAM(4);

        using namespace cv;
        // fill key and val
        mat_key = randomMat(Size(input_size, 1), key_type, INT_MIN, INT_MAX);
        mat_val = randomMat(Size(input_size, 1), val_type, INT_MIN, INT_MAX);
    }
};

TEST_P(SortByKey, Accuracy)
{
    using namespace cv;
    ocl::oclMat oclmat_key(mat_key);
    ocl::oclMat oclmat_val(mat_val);

    ocl::sortByKey(oclmat_key, oclmat_val, method, is_gt);
    SortByKey_STL::sort(mat_key, mat_val, is_gt);

    EXPECT_MAT_NEAR(mat_key, oclmat_key, 0.0);
    EXPECT_TRUE(checkUnstableSorterResult(mat_key, mat_val, oclmat_key, oclmat_val));
}
INSTANTIATE_TEST_CASE_P(OCL_SORT, SortByKey, Combine(INPUT_SIZES, KEY_TYPES, VAL_TYPES, SORT_METHODS, F_OR_T));
