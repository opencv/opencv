# if defined(__VSX__)
#   include <altivec.h>
# else
#   error "VSX is not supported"
# endif

int main()
{
    __vector float testF = vec_splats(0.f);
    testF = vec_madd(testF, testF, testF);
    return 0;
}
