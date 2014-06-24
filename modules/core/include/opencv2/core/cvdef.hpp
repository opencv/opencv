//
//  cvdef.hpp
//  OpenCV
//
//  Created by Jasper Shemilt on 05/09/2013.
//
//

#ifndef OpenCV_cvdef_hpp
#define OpenCV_cvdef_hpp


#ifndef __cplusplus
#  error core.hpp header must be compiled as C++
#endif



///////////////////////////// Bitwise and discrete math operations ///////////////////////////

template<typename _Tp> _Tp gComDivisor(_Tp u, _Tp v) {
    if (v)
        return gComDivisor<_Tp>(v, u % v);
    else
        return u < 0 ? -u : u; /* abs(u) */
};

template<typename _Tp> _Tp gComDivisor(_Tp a, _Tp b, _Tp c){
    return gComDivisor<_Tp>(gComDivisor<_Tp>(a, b), c);
};


template<typename _Tp> _Tp gComDivisor(_Tp a, _Tp* b, unsigned int size_b){
    if (size_b >= 2){
        gComDivisor<_Tp>(a, b[0]);
        return gComDivisor<_Tp>(gComDivisor<_Tp>(a, b[0]), b++, size_b-1);
    }
    else if(size_b == 1) {
        return gComDivisor<_Tp>(a, b[0]);
    }
    else {
        return a;
    }
};

template<typename _Tp> _Tp gComDivisor(_Tp* b, unsigned int size_b){
    //  std::cout << "b[0] = " << b[0] << " b[size_b-1] = " << b[size_b-1]<< " size_b = " << size_b << "\n";
    switch (size_b) {
        case 0:
            return _Tp();
            break;
        case 1:
            return b[0];
            break;
        case 2:
            return gComDivisor<_Tp>(b[0],b[1]);
            break;
        case 3:
            return gComDivisor<_Tp>(gComDivisor<_Tp>(b[0],b[1]),b[2]);
            break;
        case 4:
            return gComDivisor<_Tp>(gComDivisor<_Tp>(b[0],b[1]), gComDivisor<_Tp>(b[2],b[3]));
            break;
        default:
            return gComDivisor<_Tp>(gComDivisor<_Tp>(b,size_b/2), gComDivisor<_Tp>(b+(size_b)/2,(size_b+1)/2));
            break;
    }
};

unsigned int CV_INLINE mostSignificantBit(uint64_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xFFFFFFFF00000000) { r += 32/1; x >>= 32/1; }
    if (x & 0x00000000FFFF0000) { r += 32/2; x >>= 32/2; }
    if (x & 0x000000000000FF00) { r += 32/4; x >>= 32/4; }
    if (x & 0x00000000000000F0) { r += 32/8; x >>= 32/8; }
    return r + bval[x];
}
unsigned int CV_INLINE  mostSignificantBit(uint32_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xFFFF0000) { r += 16/1; x >>= 16/1; }
    if (x & 0x0000FF00) { r += 16/2; x >>= 16/2; }
    if (x & 0x000000F0) { r += 16/4; x >>= 16/4; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(uint16_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xFF00) { r += 8/1; x >>= 8/1; }
    if (x & 0x00F0) { r += 8/2; x >>= 8/2; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(uint8_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xF0) { r += 4/1; x >>= 4/1; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(int64_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x7FFFFFFF00000000) { r += 32/1; x >>= 32/1; }
    if (x & 0x00000000FFFF0000) { r += 32/2; x >>= 32/2; }
    if (x & 0x000000000000FF00) { r += 32/4; x >>= 32/4; }
    if (x & 0x00000000000000F0) { r += 32/8; x >>= 32/8; }
    return r + bval[x];
}
unsigned int CV_INLINE  mostSignificantBit(int32_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x7FFF0000) { r += 16/1; x >>= 16/1; }
    if (x & 0x0000FF00) { r += 16/2; x >>= 16/2; }
    if (x & 0x000000F0) { r += 16/4; x >>= 16/4; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(int16_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x7F00) { r += 8/1; x >>= 8/1; }
    if (x & 0x00F0) { r += 8/2; x >>= 8/2; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(int8_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x70) { r += 4/1; x >>= 4/1; }
    return r + bval[x];
}


/* f : number to convert.
 * num, denom: returned parts of the rational.
 * max_denom: max denominator value.  Note that machine floating point number
 *     has a finite resolution (10e-16 ish for 64 bit double), so specifying
 *     a "best match with minimal error" is often wrong, because one can
 *     always just retrieve the significand and return that divided by
 *     2**52, which is in a sense accurate, but generally not very useful:
 *     1.0/7.0 would be "2573485501354569/18014398509481984", for example.
 */
void CV_INLINE rat_approx(double f, int64_t max_denom, int64_t *num, int64_t *denom)
{
    /*  a: continued fraction coefficients. */
    int64_t a, h[3] = { 0, 1, 0 }, k[3] = { 1, 0, 0 };
    int64_t x, d, n = 1;
    int i, neg = 0;

    if (max_denom <= 1) { *denom = 1; *num = (int64_t) f; return; }

    if (f < 0) { neg = 1; f = -f; }

    while (f != floor(f)) { n <<= 1; f *= 2; }
    d = f;

    /* continued fraction and check denominator each step */
    for (i = 0; i < 64; i++) {
        a = n ? d / n : 0;
        if (i && !a) break;

        x = d; d = n; n = x % n;

        x = a;
        if (k[1] * a + k[0] >= max_denom) {
            x = (max_denom - k[0]) / k[1];
            if (x * 2 >= a || k[1] >= max_denom)
                i = 65;
            else
                break;
        }

        h[2] = x * h[1] + h[0]; h[0] = h[1]; h[1] = h[2];
        k[2] = x * k[1] + k[0]; k[0] = k[1]; k[1] = k[2];
    }
    *denom = k[1];
    *num = neg ? -h[1] : h[1];
}


#  endif
