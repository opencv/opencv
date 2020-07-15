#include <windows.h>

int main(int argc, char** argv)
{
    CreateWindow(NULL /*lpClassName*/, NULL /*lpWindowName*/, 0 /*dwStyle*/, 0 /*x*/,
                 0 /*y*/, 0 /*nWidth*/, 0 /*nHeight*/, NULL /*hWndParent*/, NULL /*hMenu*/,
                NULL /*hInstance*/,  NULL /*lpParam*/);
    DeleteDC(NULL);

    return 0;
}
