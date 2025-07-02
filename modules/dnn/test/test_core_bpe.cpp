#include "test_precomp.hpp"
#include "../src/tokenizertokens/core_bpe.hpp"
// #include "../src/tokenizertokens/encoding_registry.hpp"
// #include "../src/tokenizertokens/encoding.cpp"

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

// Both following test cases bytePairSplit_Simple and BytePairSplit_Repeated are taken from the lib.rs file in tiktoken 
TEST_F(Test_CoreBPE, bytePairSplit_Simple) {
    auto ranks = makeRanks();
    ByteVec piece = {'a', 'b', 'c', 'd'};
    auto parts = bytePairSplit(piece, ranks);
    std::vector<ByteVec> expected = { 
        ByteVec{'a', 'b'},
        ByteVec{'c', 'd'}
    };
    EXPECT_EQ(parts, expected) << "bytePairSplit should split \"abcd\" into [\"ab\",\"cd\"]";
    // [PASSED]
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

TEST_F(Test_CoreBPE, BytePairSplit_Repeated) {
    auto ranks = makeRanks();
    ByteVec piece = {'a', 'b', 'a', 'b'};
    auto parts = bytePairSplit(piece, ranks);
    std::vector<ByteVec> expected = {
        ByteVec{'a', 'b'},
        ByteVec{'a', 'b'}
    };
    EXPECT_EQ(parts, expected) << "bytePairEncode(\"abcd\") should yield [0,1]";
    // [PASSED]
}

/*
[Outputs the same above information as the above test case but omitted here]

Note: Google Test filter = Test_CoreBPE.*
[==========] Running 2 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 2 tests from Test_CoreBPE
[ RUN      ] Test_CoreBPE.bytePairSplit_Simple
[       OK ] Test_CoreBPE.bytePairSplit_Simple (0 ms)
[ RUN      ] Test_CoreBPE.BytePairSplit_Repeated
[       OK ] Test_CoreBPE.BytePairSplit_Repeated (0 ms)
[----------] 2 tests from Test_CoreBPE (0 ms total)

[----------] Global test environment tear-down
[==========] 2 tests from 1 test case ran. (0 ms total)
[  PASSED  ] 2 tests.
*/


TEST_F(Test_CoreBPE, EncodeOrdinary_Simple) {
    auto ranks = makeRanks();
    std::unordered_map<std::string, Rank> special;

    // We choose a tiny regex that first matches "ab" or "cd" as whole,
    // falling back to matching any single char (.)
    static const std::string PAT = R"((?:ab|cd)|.)";
    auto bpe = CoreBPE::create(
        ranks.begin(), ranks.end(),
        special.begin(), special.end(),
        PAT
    );
    std::vector<Rank> out = bpe.encodeOrdinary("abcd");
    // 3) Verify: we should get exactly two tokens [0,1]
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0], 0u);  // "ab" → token 0
    EXPECT_EQ(out[1], 1u);  // "cd" → token 1
}

// TEST_F(Test_CoreBPE, EncodeOrdinary_RepeatedPairs) {
//     auto ranks = makeRanks();  
//     std::unordered_map<std::string, Rank> special;

//     auto bpe = CoreBPE::create(
//         ranks.begin(), ranks.end(),
//         special.begin(), special.end(),
//         CoreBPE::patternString() 
//     );

//     auto tokens = bpe.encodeOrdinary("abab");
//     std::vector<Rank> expected{0, 0};
//     EXPECT_EQ(tokens, expected)
//         << "encodeOrdinary(\"abab\") should yield [0,0] because it byte-pairs into [\"ab\",\"ab\"]";

// }

// TEST_F(Test_CoreBPE, EncodeOrdinary_MixedOverlap) {
//     auto ranks = makeRanks();  // { "ab":0, "cd":1 } 
//     std::unordered_map<std::string, Rank> special;

//     auto bpe = CoreBPE::create(
//         ranks.begin(), ranks.end(),
//         special.begin(), special.end(),
//         CoreBPE::patternString()  
//     );

//     auto tokens = bpe.encodeOrdinary("abcdab");
//     std::vector<Rank> expected{0, 1, 0};
//     EXPECT_EQ(tokens, expected)
//         << "encodeOrdinary(\"abcdab\") should yield [0,1,0] because it byte-pairs into [\"ab\",\"cd\",\"ab\"]";
// }

// TEST_F(Test_CoreBPE, EncodeUnstableNative_Compile) {
//     auto ranks = makeRanks();
//     std::unordered_map<std::string, Rank> special;
//     auto bpe = CoreBPE::create(
//         ranks.begin(), ranks.end(),
//         special.begin(), special.end(),
//         CoreBPE::patternString()
//     );

//     std::unordered_set<std::string> allowedSpecial;
//     auto result = bpe.encodeUnstableNative("abcd", allowedSpecial);
//     SUCCEED() << "Just checking that encodeUnstableNative(...) compiles and links";
//     // [PASSED]
// }


// TEST_F(Test_CoreBPE, EncodeUnstableNative_Fallback) {
//     auto ranks = makeRanks(); // { "ab":0, "cd":1 }
//     std::unordered_map<std::string, Rank> special;   // no special tokens
//     auto bpe = CoreBPE::create(
//         ranks.begin(), ranks.end(),
//         special.begin(), special.end(),
//         CoreBPE::patternString()
//     );
//     std::unordered_set<std::string> allowedSpecial; 

//     auto [ordinary, completions] =
//         bpe.encodeUnstableNative("abcd", allowedSpecial);

//     // 3) Rust behavior the last fallback piece (“cd”) is treated as unstable,
//     // so ordinary tokens get truncated away, and the only completion is [0,1].
//     EXPECT_TRUE(ordinary.empty())
//         << "Rust logic returns no ‘ordinary’ tokens when the final BPE fallback is treated as unstable";

//     ASSERT_EQ(completions.size(), 1u)
//         << "Expect exactly one completion sequence";

//     EXPECT_EQ(*completions.begin(), (std::vector<Rank>{0,1}))
//         << "That completion must be the full BPE fallback for \"abcd\": [0,1]";
// }

// TEST(Tokenizer, SimpleEncodeDecode) {
//     auto& enc = getEncoding("gpt2");
//     auto toks = enc.encode("hello world");
//     EXPECT_EQ(toks, (std::vector<Rank>{31373, 995}));
// }

}}