# Overview

libdeflate is a library for fast, whole-buffer DEFLATE-based compression and
decompression.

The supported formats are:

- DEFLATE (raw)
- zlib (a.k.a. DEFLATE with a zlib wrapper)
- gzip (a.k.a. DEFLATE with a gzip wrapper)

libdeflate is heavily optimized.  It is significantly faster than the zlib
library, both for compression and decompression, and especially on x86
processors.  In addition, libdeflate provides optional high compression modes
that provide a better compression ratio than the zlib's "level 9".

libdeflate itself is a library.  The following command-line programs which use
this library are also included:

* `libdeflate-gzip`, a program which can be a drop-in replacement for standard
  `gzip` under some circumstances.  Note that `libdeflate-gzip` has some
  limitations; it is provided for convenience and is **not** meant to be the
  main use case of libdeflate.  It needs a lot of memory to process large files,
  and it omits support for some infrequently-used options of GNU gzip.

* `benchmark`, a test program that does round-trip compression and decompression
  of the provided data, and measures the compression and decompression speed.
  It can use libdeflate, zlib, or a combination of the two.

* `checksum`, a test program that checksums the provided data with Adler-32 or
  CRC-32, and optionally measures the speed.  It can use libdeflate or zlib.

For the release notes, see the [NEWS file](NEWS.md).

## Table of Contents

- [Building](#building)
  - [Using CMake](#using-cmake)
  - [Directly integrating the library sources](#directly-integrating-the-library-sources)
- [API](#api)
- [Bindings for other programming languages](#bindings-for-other-programming-languages)
- [DEFLATE vs. zlib vs. gzip](#deflate-vs-zlib-vs-gzip)
- [Compression levels](#compression-levels)
- [Motivation](#motivation)
- [License](#license)

# Building

## Using CMake

libdeflate uses [CMake](https://cmake.org/).  It can be built just like any
other CMake project, e.g. with:

    cmake -B build && cmake --build build

By default the following targets are built:

- The static library (normally called `libdeflate.a`)
- The shared library (normally called `libdeflate.so`)
- The `libdeflate-gzip` program, including its alias `libdeflate-gunzip`

Besides the standard CMake build and installation options, there are some
libdeflate-specific build options.  See `CMakeLists.txt` for the list of these
options.  To set an option, add `-DOPTION=VALUE` to the `cmake` command.

Prebuilt Windows binaries can be downloaded from
https://github.com/ebiggers/libdeflate/releases.

## Directly integrating the library sources

Although the official build system is CMake, care has been taken to keep the
library source files compilable directly, without a prerequisite configuration
step.  Therefore, it is also fine to just add the library source files directly
to your application, without using CMake.

You should compile both `lib/*.c` and `lib/*/*.c`.  You don't need to worry
about excluding irrelevant architecture-specific code, as this is already
handled in the source files themselves using `#ifdef`s.

It is strongly recommended to use either gcc or clang, and to use `-O2`.

If you are doing a freestanding build with `-ffreestanding`, you must add
`-DFREESTANDING` as well (matching what the `CMakeLists.txt` does).

# API

libdeflate has a simple API that is not zlib-compatible.  You can create
compressors and decompressors and use them to compress or decompress buffers.
See libdeflate.h for details.

There is currently no support for streaming.  This has been considered, but it
always significantly increases complexity and slows down fast paths.
Unfortunately, at this point it remains a future TODO.  So: if your application
compresses data in "chunks", say, less than 1 MB in size, then libdeflate is a
great choice for you; that's what it's designed to do.  This is perfect for
certain use cases such as transparent filesystem compression.  But if your
application compresses large files as a single compressed stream, similarly to
the `gzip` program, then libdeflate isn't for you.

Note that with chunk-based compression, you generally should have the
uncompressed size of each chunk stored outside of the compressed data itself.
This enables you to allocate an output buffer of the correct size without
guessing.  However, libdeflate's decompression routines do optionally provide
the actual number of output bytes in case you need it.

Windows developers: note that the calling convention of libdeflate.dll is
"cdecl".  (libdeflate v1.4 through v1.12 used "stdcall" instead.)

# Bindings for other programming languages

The libdeflate project itself only provides a C library.  If you need to use
libdeflate from a programming language other than C or C++, consider using the
following bindings:

* C#: [LibDeflate.NET](https://github.com/jzebedee/LibDeflate.NET)
* Go: [go-libdeflate](https://github.com/4kills/go-libdeflate)
* Java: [libdeflate-java](https://github.com/astei/libdeflate-java)
* Julia: [LibDeflate.jl](https://github.com/jakobnissen/LibDeflate.jl)
* Nim: [libdeflate-nim](https://github.com/gemesa/libdeflate-nim)
* Perl: [Gzip::Libdeflate](https://github.com/benkasminbullock/gzip-libdeflate)
* PHP: [ext-libdeflate](https://github.com/pmmp/ext-libdeflate)
* Python: [deflate](https://github.com/dcwatson/deflate)
* Ruby: [libdeflate-ruby](https://github.com/kaorimatz/libdeflate-ruby)
* Rust: [libdeflater](https://github.com/adamkewley/libdeflater)

Note: these are third-party projects which haven't necessarily been vetted by
the authors of libdeflate.  Please direct all questions, bugs, and improvements
for these bindings to their authors.

Also, unfortunately many of these bindings bundle or pin an old version of
libdeflate.  To avoid known issues in old versions and to improve performance,
before using any of these bindings please ensure that the bundled or pinned
version of libdeflate has been upgraded to the latest release.

# DEFLATE vs. zlib vs. gzip

The DEFLATE format ([rfc1951](https://www.ietf.org/rfc/rfc1951.txt)), the zlib
format ([rfc1950](https://www.ietf.org/rfc/rfc1950.txt)), and the gzip format
([rfc1952](https://www.ietf.org/rfc/rfc1952.txt)) are commonly confused with
each other as well as with the [zlib software library](http://zlib.net), which
actually supports all three formats.  libdeflate (this library) also supports
all three formats.

Briefly, DEFLATE is a raw compressed stream, whereas zlib and gzip are different
wrappers for this stream.  Both zlib and gzip include checksums, but gzip can
include extra information such as the original filename.  Generally, you should
choose a format as follows:

- If you are compressing whole files with no subdivisions, similar to the `gzip`
  program, you probably should use the gzip format.
- Otherwise, if you don't need the features of the gzip header and footer but do
  still want a checksum for corruption detection, you probably should use the
  zlib format.
- Otherwise, you probably should use raw DEFLATE.  This is ideal if you don't
  need checksums, e.g. because they're simply not needed for your use case or
  because you already compute your own checksums that are stored separately from
  the compressed stream.

Note that gzip and zlib streams can be distinguished from each other based on
their starting bytes, but this is not necessarily true of raw DEFLATE streams.

# Compression levels

An often-underappreciated fact of compression formats such as DEFLATE is that
there are an enormous number of different ways that a given input could be
compressed.  Different algorithms and different amounts of computation time will
result in different compression ratios, while remaining equally compatible with
the decompressor.

For this reason, the commonly used zlib library provides nine compression
levels.  Level 1 is the fastest but provides the worst compression; level 9
provides the best compression but is the slowest.  It defaults to level 6.
libdeflate uses this same design but is designed to improve on both zlib's
performance *and* compression ratio at every compression level.  In addition,
libdeflate's levels go [up to 12](https://xkcd.com/670/) to make room for a
minimum-cost-path based algorithm (sometimes called "optimal parsing") that can
significantly improve on zlib's compression ratio.

If you are using DEFLATE (or zlib, or gzip) in your application, you should test
different levels to see which works best for your application.

# Motivation

Despite DEFLATE's widespread use mainly through the zlib library, in the
compression community this format from the early 1990s is often considered
obsolete.  And in a few significant ways, it is.

So why implement DEFLATE at all, instead of focusing entirely on
bzip2/LZMA/xz/LZ4/LZX/ZSTD/Brotli/LZHAM/LZFSE/[insert cool new format here]?

To do something better, you need to understand what came before.  And it turns
out that most ideas from DEFLATE are still relevant.  Many of the newer formats
share a similar structure as DEFLATE, with different tweaks.  The effects of
trivial but very useful tweaks, such as increasing the sliding window size, are
often confused with the effects of nontrivial but less useful tweaks.  And
actually, many of these formats are similar enough that common algorithms and
optimizations (e.g. those dealing with LZ77 matchfinding) can be reused.

In addition, comparing compressors fairly is difficult because the performance
of a compressor depends heavily on optimizations which are not intrinsic to the
compression format itself.  In this respect, the zlib library sometimes compares
poorly to certain newer code because zlib is not well optimized for modern
processors.  libdeflate addresses this by providing an optimized DEFLATE
implementation which can be used for benchmarking purposes.  And, of course,
real applications can use it as well.

# License

libdeflate is [MIT-licensed](COPYING).

I am not aware of any patents or patent applications relevant to libdeflate.
