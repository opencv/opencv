#include "precomp.hpp"

using namespace cv;

TEST(ArithmTest, add)
{
    typedef uchar _Tp;
    
    Mat A(30,30,DataType<_Tp>::type), B(A.size(), A.type()), C0, C;
    RNG rng(-1);
    rng.fill(A, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
    rng.fill(B, RNG::UNIFORM, Scalar::all(0), Scalar::all(256));
    C0.create(A.size(), A.type());
    int i, j, cols = A.cols*A.channels();
    for(i = 0; i < A.rows; i++)
    {
        const _Tp* aptr = A.ptr<_Tp>(i);
        const _Tp* bptr = B.ptr<_Tp>(i);
        _Tp* cptr = C0.ptr<_Tp>(i);
        for(j = 0; j < cols; j++)
            cptr[j] = saturate_cast<_Tp>(aptr[j] + bptr[j]);
    }
    add(A, B, C);
    EXPECT_EQ(norm(C, C0, NORM_INF), 0);
}
