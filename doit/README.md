# python DoIt scripts for building opencv targets

Python DoIt is a tool to provide automatic task dependency management and execution.

For more information about pydoit see: http://pydoit.org

## Description of the opencv doit toolchain

In this directory is contained various python scripts to make it easier to build opencv for multiple platforms.

`doit` provides a pythonic way of defining various tasks and their dependencies. The tasks are all defined (or imported)
in `dodo.py`.

`doit` adds an extra layer of simplification to the opencv build system so that developers don't need to worry about the
various command line parameters required to make the opencv cmake build work.

If `doit list` is run in the doit directory within opencv, the following is output:

```
build_all           Build opencv for all platforms. This may take a while
build_for_android   Build all android architecture libraries
build_for_desktop   Build the desktop libraries
build_for_ios       Build all ios architecture libraries
desktop_cmake       Run cmake for the desktop
setup               Setup output directories
```

# build_for_desktop

Running `doit build_for_desktop` will build the opencv libraries for the current desktop platform into 'builds/desktop'

# build_for_android

Running `doit build_for_android` will build the opencv libraries for multiple android abis into 'builds/mobile/android'


