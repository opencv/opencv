On Linux, install the development packages for FreeType,
Cairo, and GLib. For example, on Ubuntu / Debian, you would do:

    sudo apt-get install meson pkg-config ragel gtk-doc-tools gcc g++ libfreetype6-dev libglib2.0-dev libcairo2-dev

whereas on Fedora, RHEL, CentOS, and other Red Hat based systems you would do:

    sudo dnf install meson pkgconfig gtk-doc gcc gcc-c++ freetype-devel glib2-devel cairo-dev

and on ArchLinux and Manjaro:

    sudo pacman -Suy meson pkg-config ragel gcc freetype2 glib2 cairo

then use meson to build the project like `meson build && meson test -Cbuild`.

On macOS, `brew install pkg-config ragel gtk-doc freetype glib cairo meson` then use
meson like above.

On Windows, meson can build the project like above if a working MSVC's cl.exe (`vcvarsall.bat`)
or gcc/clang is already on your path, and if you use something like `meson build --wrap-mode=default`
it fetches and compiles most of the dependencies also.

Our CI configurations is also a good source of learning how to build HarfBuzz.

There is also amalgam source provided with HarfBuzz which reduces whole process of building
HarfBuzz like `g++ src/harfbuzz.cc -fno-exceptions` but there is not guarantee provided
with buildability and reliability of features you get.
