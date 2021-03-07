#if defined(__VSX__)
    #if defined(__PPC64__) && defined(__LITTLE_ENDIAN__)
        #include <altivec.h>
    #else
        #error "OpenCV only supports little-endian mode"
    #endif
#else
    #error "VSX is not supported"
#endif

/*
 * xlc and wide versions of clang don't support %x<n> in the inline asm template which fixes register number
 * when using any of the register constraints wa, wd, wf
*/
int main()
{
    __vector float vf;
    __vector signed int vi;
    __asm__ __volatile__ ("xvcvsxwsp %x0,%x1" : "=wa" (vf) : "wa" (vi));
    return 0;
}