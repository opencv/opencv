#include <stdio.h>

int badFunction() {
    char str[20];
    gets(str); // Noncompliant; `str` buffer size is not checked and it is vulnerable to overflows
    return 0;
}
