// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

struct FastGemm1t_int8_t{
    std::vector<int> vec_shape;
    std::vector<int> wt_shape;
    std::vector<int> bias_shape;
    std::vector<int> multiplier_shape;

    int zeropoint;
    FastGemm1t_int8_t(std::vector<int> vec_shape_, std::vector<int>wt_shape_, std::vector<int>bias_shape_ = {}, std::vector<int>multiplier_shape_ = {}, int zero_point_ = 0)
        : vec_shape(vec_shape_), wt_shape(wt_shape_), bias_shape(bias_shape_), multiplier_shape(multiplier_shape_), zeropoint(zero_point_){} 

};

static inline void PrintTo(const FastGemm1t_int8_t & params, std::ostream* os){
    auto print_shape = [os] (const std::vector<int>& shape, const std::string tag){
        if(shape.empty()){
            return;
        }

        *os << tag << "=[";
        for (size_t i = 0; i < shape.size(); ++i){
            if(i == shape.size() - 1){
                *os << shape[i]<<"]";
                break;
            }
            *os << shape[i]<<",";
        }

    };
    print_shape(params.vec_shape, "vec");
    print_shape(params.wt_shape, ", wt");
}

static const FastGemm1t_int8_t test_fast_gemm1_configs[]={
    {{1, 192}, {1, 192},{1}},
    {{1, 1024},{192, 1024}, {192}},
    {{1, 64}, {192, 64}, {192}},
    {{1, 192}, {512, 192}, {512}},
    {{64, 192}, {192, 192}, {192}},
    {{64, 1024}, {192, 1024}, {192}},

    { {  768,  768 }, {  768,  768 }, {  768 }},
    { { 1024, 1024 }, { 1024, 1024 }, { 1024 }},
    { {   50,  768 }, {  2304, 768 }, {2304}},
    { {  197,  768 }, {  2304, 768}, {2304}},
    { {   50, 1024 }, { 3072, 1024 }, {3072}},
    { {  197, 1024 }, { 3072, 1024 }, {3072}},

    
};

class Layer_FastGemm1t : public  TestBaseWithParam<FastGemm1t_int8_t>{};

PERF_TEST_P_(Layer_FastGemm1t, fastgemm1t){

    const FastGemm1t_int8_t& p = GetParam();
    const int N = p.wt_shape[0], K = p.wt_shape[1];

    LayerParams lp;

    lp.type = "InnerProductInt8"; 
    lp.name = "TestInnerProductInt8";

    Mat wt(N, K, CV_8S);      
    randu(wt,   -127,    127);
    
    Mat bias(1, N, CV_32S);   
    randu(bias, -128,    128);
    
    Mat mult(1, N, CV_32F);   
    randu(mult, 0.001f,  0.01f);

    lp.blobs.push_back(wt);   
    lp.blobs.push_back(bias);
    lp.blobs.push_back(mult);


    lp.set("num_output",      N);
    lp.set("input_scale",     0.02f);
    lp.set("input_zeropoint", p.zeropoint);
    lp.set("scales",          0.03f);
    lp.set("zeropoints",      0);

    Net net;
    int id = net.addLayerToPrev(lp.name, lp.type, lp);
    net.connect(0, 0, id, 0);
    net.setInputsNames({"A"});


    Mat vec(p.vec_shape, CV_8S);   
    randu(vec, -127, 127);
    net.setInput(vec, "A");

    Mat out = net.forward();
    {
        out = net.forward(); // warmup
    }
    
    TEST_CYCLE() { 
        out = net.forward(); 
    }
    
    SANITY_CHECK_NOTHING();

}

INSTANTIATE_TEST_CASE_P(/**/, Layer_FastGemm1t, testing::ValuesIn(test_fast_gemm1_configs));

}
