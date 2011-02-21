#include "test_precomp.hpp"

int main(int argc, char **argv)
{
    cvtest::TS::ptr()->init("gpu");
    ::testing::InitGoogleTest(&argc, argv);
#ifdef HAVE_CUDA
    return RUN_ALL_TESTS();
#else
    std::cerr << "opencv_test_gpu: OpenCV was compiled without GPU support\n";
    return -1;
#endif
}