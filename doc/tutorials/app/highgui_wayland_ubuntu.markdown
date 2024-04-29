Using Wayland highgui-backend in Ubuntu {#tutorial_wayland_ubuntu}
=======================================

@tableofcontents

@prev_tutorial{tutorial_intelperc}

|    |    |
| -: | :- |
| Original author | Kumataro |
| Compatibility | OpenCV >= 4.10 |
| ^ | Ubuntu 24.04 |

Goal
-----
This tutorial is to use Wayland highgui-backend in Ubuntu 24.04.

Wayland highgui-backend is experimental implementation.

Setup
-----
- Setup Ubuntu 24.04.
- `sudo apt install build-essential git cmake` to build OpenCV.
- `sudo apt install libwayland-dev wayland-protocols libxkbcommon-dev` to enable Wayland highgui-backend.
- (Option) `sudo apt install ninja-build` (or remove `-GNinja` option for cmake command).
- (Option) `sudo apt install libwayland-egl1` to enable Wayland EGL library.

Get OpenCV from GitHub
----------------------

```bash
mkdir work
cd work
git clone --depth=1 https://github.com/opencv/opencv.git
```

@note
`--depth=1` option is to limit downloading commits. If you want to see more commit history, please remove this option.

Build/Install OpenCV with Wayland highgui-backend
-------------------------------------------------

Run `cmake` with `-DWITH_WAYLAND=ON` option to configure OpenCV.

```bash
cmake -S opencv -B build4-main -DWITH_WAYLAND=ON -GNinja
```

If succeeded, Wayland Client/Cursor/Protocols and Xkbcommon versions are shown. Wayland EGL is option.

```plaintext
--
--   GUI:                           Wayland
--     Wayland:                     (Experimental) YES
--       Wayland Client:            YES (ver 1.22.0)
--       Wayland Cursor:            YES (ver 1.22.0)
--       Wayland Protocols:         YES (ver 1.34)
--       Xkbcommon:                 YES (ver 1.6.0)
--       Wayland EGL(Option):       YES (ver 18.1.0)
--     GTK+:                        NO
--     VTK support:                 NO
```

Run `cmake --build` to build, and `sudo cmake --install` to install into your system.

```bash
cmake --build build4-main
sudo cmake --install build4-main
sudo ldconfig
```

Simple Application to try Wayland highgui-backend
-------------------------------------------------
Try this code, so you can see name of currentUIFrramework() and OpenCV logo window with Wayland highgui-backend.


```bash
// g++ main.cpp -o a.out -I /usr/local/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgcodecs
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

int main(void)
{
  std::cout << "cv::currentUIFramework() returns " << cv::currentUIFramework() << std::endl;

  cv::Mat src;
  src = cv::imread("opencv-logo.png");

  cv::namedWindow("src");

  int key = 0;
  do
  {
      cv::imshow("src", src );
      key = cv::waitKey(50);
  } while( key != 'q' );
  return 0;
}
```

Limitation/Known problem
------------------------
- cv::moveWindow() is not implementated. ( See. https://github.com/opencv/opencv/issues/25478 )
