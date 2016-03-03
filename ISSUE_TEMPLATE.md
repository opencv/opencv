This is a template helping you to create an issue which can be processes as quickly as possible. Feel free to add additional information or remove not relevant points if you do not need them.

If you have a question rather than reporting a bug please go to http://answers.opencv.org where you get much faster responses.

### Please state the information for your system
- OpenCV version: 2.4 / 3.x
- Host OS: Linux (Ubuntu 14.04)  / Mac OS X 10.11.3 / Windows 10
- *(if needed, only cross-platform builds)* Target OS: host / Android 6.0 / ARM board / Raspberry Pi 2
- *(if needed)* Compiler & CMake: GCC 5.3 & CMake 3.5

### In which part of the OpenCV library you got the issue?
Examples:
- objdetect, highgui, imgproc, cuda, tests
- face recognition, resizing an image, reading an jpg image

### Expected behaviour

### Actual behaviour

### Additional description

### Code example to reproduce the issue / Steps to reproduce the issue
Please try to give a full example which will compile as is.
```
#include "opencv2/core.hpp"
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    double d[] = { 546,2435,7,4534,23423,3 };
    cout << "d = 0x" << reinterpret_cast<void*>(d) << endl;

    return 0;
}
```
