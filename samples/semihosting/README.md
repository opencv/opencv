# Arm semihosting

This folder contain a toolchain file and a couple of examples for
building OpenCV based applications that can run in an [Arm
semihosting](https://developer.arm.com/documentation/100863/latest)
setup.

OpenCV can be compiled to target a semihosting platform as follows:

```
cmake ../opencv/ \
    -DCMAKE_TOOLCHAIN_FILE=../opencv/platforms/semihosting/aarch64-semihosting.toolchain.cmake \
    -DSEMIHOSTING_TOOLCHAIN_PATH=/path/to/baremetal-toolchain/bin/ \
    -DBUILD_EXAMPLES=ON -GNinja
```

A barematel toolchain for targeting aarch64 semihosting can be found
[here](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-a/downloads),
under `aarch64-none-elf`.

The code of the examples in the `norm` and `histogram` folders can be
executed with qemu in Linux userspace:

```
    qemu-aarch64 ./bin/example_semihosting_histogram
    qemu-aarch64 ./bin/example_semihosting_norm
```
