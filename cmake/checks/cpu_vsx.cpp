#if defined(__VSX__)
    #if defined(__PPC64__) && defined(__LITTLE_ENDIAN__)
        #include <altivec.h>
    #else
        #error "OpenCV only supports little-endian mode"
    #endif
#else
    #error "VSX is not supported"
#endif

int main()
{
    __vector float testF = vec_splats(0.f);
    testF = vec_madd(testF, testF, testF);
    return 0;
}
