#if defined(__VSX__)
    #if defined(__PPC64__) && defined(__LITTLE_ENDIAN__)
        #include <altivec.h>
    #else
        #error "OpenCV only supports little-endian mode"
    #endif
#else
    #error "VSX3 is not supported"
#endif

int main()
{
    __vector unsigned char a = vec_splats((unsigned char)1);
    __vector unsigned char b = vec_splats((unsigned char)2);
    __vector unsigned char r = vec_absd(a, b);
    return 0;
}
