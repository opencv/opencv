// check sanity of vsx aligned ld/st
// https://github.com/opencv/opencv/issues/13211

#include <altivec.h>

#define vsx_ld vec_vsx_ld
#define vsx_st vec_vsx_st

template<typename T>
static void fill(T& d, int from = 0, int to = 16)
{
   for (int i = from; i < to; i++)
        d[i] = i;
}

template<typename T, typename Tvec>
static bool check_data(T& d, Tvec& v, int from = 0, int to = 16)
{
    for (int i = from; i < to; i++)
    {
        if (d[i] != vec_extract(v, i))
            return false;
    }
    return true;
}

int main()
{
    unsigned char __attribute__ ((aligned (16))) rbuf[16];
    unsigned char __attribute__ ((aligned (16))) wbuf[16];
    __vector unsigned char a;

    // 1- check aligned load and store
    fill(rbuf);
    a = vec_ld(0, rbuf);
    if (!check_data(rbuf, a))
        return 1;
    vec_st(a, 0, wbuf);
    if (!check_data(wbuf, a))
        return 11;

    // 2- check mixing aligned load and unaligned store
    a = vec_ld(0, rbuf);
    vsx_st(a, 0, wbuf);
    if (!check_data(wbuf, a))
        return 2;

    // 3- check mixing unaligned load and aligned store
    a = vsx_ld(0, rbuf);
    vec_st(a, 0, wbuf);
    if (!check_data(wbuf, a))
        return 3;

    return 0;
}