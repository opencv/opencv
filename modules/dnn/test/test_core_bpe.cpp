#include "test_precomp.hpp"
#include "../src/tokenizertokens/core_bpe.hpp"

namespace opencv_test { namespace  {

using namespace cv::dnn::tokenizer;

class Test_CoreBPE : public ::testing::Test {
public:
    static ByteVecRankMap makeRanks() {
        ByteVecRankMap ranks;
        ranks.emplace(ByteVec{'a', 'b'}, 0);
        ranks.emplace(ByteVec{'c', 'd'}, 1);
        return ranks;
    }
};

TEST_F(Test_CoreBPE, bytePairSplit_Simple) {
    auto ranks = makeRanks();
    ByteVec piece = {'a', 'b', 'c', 'd'};
    auto parts = bytePairSplit(piece, ranks);
    std::vector<ByteVec> expected = { 
        ByteVec{'a', 'b'},
        ByteVec{'c', 'd'}
    };
    EXPECT_EQ(parts, expected) << "bytePairSplit should split \"abcd\" into [\"ab\",\"cd\"]";
}   

/*
** First time running test and getting a hang of testing my tokenizer correctness.
** For my knowledge and to verify correctnes. [Keep for now]

CTEST_FULL_OUTPUT
OpenCV version: 5.0.0-pre
OpenCV VCS version: 5.0.0-alpha-426-gc9a258504e-dirty
Build type: Debug
Compiler: /usr/bin/c++  (ver 15.0.0.15000309)
Algorithm hint: ALGO_HINT_ACCURATE
HAL: YES (carotene (ver 0.0.1))
[ INFO:0@0.108] global registry_parallel.impl.hpp:96 ParallelBackendRegistry core(parallel): Enabled backends(3, sorted by priority): ONETBB(1000); TBB(990); OPENMP(980)
Parallel framework: gcd (nthreads=8)
CPU features: NEON FP16 NEON_DOTPROD NEON_FP16 *NEON_BF16?
[ INFO:0@0.109] global ocl.cpp:1185 haveOpenCL Initialize OpenCL runtime...
[ INFO:0@0.110] global ocl.cpp:1191 haveOpenCL OpenCL: found 1 platforms
[ INFO:0@0.110] global ocl.cpp:983 getInitializedExecutionContext OpenCL: initializing thread execution context
[ INFO:0@0.110] global ocl.cpp:993 getInitializedExecutionContext OpenCL: creating new execution context...
[ INFO:0@0.200] global ocl.cpp:1011 getInitializedExecutionContext OpenCL: device=Apple M1
OpenCL Platforms: 
    Apple
        iGPU: Apple M1 (OpenCL 1.2 )
Current OpenCL device: 
    Type = iGPU
    Name = Apple M1
    Version = OpenCL 1.2 
    Driver version = 1.2 1.0
    Address bits = 64
    Compute units = 8
    Max work group size = 256
    Local memory size = 32 KB
    Max memory allocation size = 1 GB
    Double support = No
    Half support = No
    Host unified memory = Yes
    Device extensions:
        cl_APPLE_SetMemObjectDestructor
        cl_APPLE_ContextLoggingFunctions
        cl_APPLE_clut
        cl_APPLE_query_kernel_names
        cl_APPLE_gl_sharing
        cl_khr_gl_event
        cl_khr_byte_addressable_store
        cl_khr_global_int32_base_atomics
        cl_khr_global_int32_extended_atomics
        cl_khr_local_int32_base_atomics
        cl_khr_local_int32_extended_atomics
        cl_khr_3d_image_writes
        cl_khr_image2d_from_buffer
        cl_khr_depth_images
    Has AMD Blas = No
    Has AMD Fft = No
    Preferred vector width char = 1
    Preferred vector width short = 1
    Preferred vector width int = 1
    Preferred vector width long = 1
    Preferred vector width float = 1
    Preferred vector width double = 1
    Preferred vector width half = 0
TEST: Skip tests with tags: 'mem_6gb', 'verylong', 'debug_verylong', 'dnn_skip_opencv_backend', 'dnn_skip_cpu', 'dnn_skip_cpu_fp16', 'dnn_skip_ocl', 'dnn_skip_ocl_fp16', 'dnn_skip_onnx_conformance', 'dnn_skip_parser'
Note: Google Test filter = Test_CoreBPE.*
[==========] Running 1 test from 1 test case.
[----------] Global test environment set-up.
[----------] 1 test from Test_CoreBPE
[ RUN      ] Test_CoreBPE.bytePairSplit_Simple
[       OK ] Test_CoreBPE.bytePairSplit_Simple (0 ms)
[----------] 1 test from Test_CoreBPE (0 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test case ran. (1 ms total)
[  PASSED  ] 1 test.
*/

}}