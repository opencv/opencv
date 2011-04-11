#include "precomp.hpp"

#if ANDROID
int wcscasecmp(const wchar_t* lhs, const wchar_t* rhs)
{
  wint_t left, right;
  do {
    left = towlower(*lhs++);
    right = towlower(*rhs++);
  } while (left && left == right);
  return left == right;
}
#endif
