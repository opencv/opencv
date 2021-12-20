# OpenEXR Release Notes

* [Version 3.1.3](#version-313-october-27-2021) October 27, 2021
* [Version 3.1.2](#version-312-october-4-2021) October 4, 2021
* [Version 3.1.1](#version-311-august-2-2021) August 2, 2021
* [Version 3.1.0](#version-310-july-22-2021) July 22, 2021
* [Version 3.0.5](#version-305-july-1-2021) July 1, 2021
* [Version 3.0.4](#version-304-june-3-2021) June 3, 2021
* [Version 3.0.3](#version-303-may-18-2021) May 18, 2021
* [Version 3.0.2](#version-302-may-17-2021) May 17, 2021
* [Version 3.0.1](#version-301-april-1-2021) April 1, 2021
* [Version 3.0.1-beta](#version-301-beta-march-28-2021) March 28, 2021
* [Version 3.0.0-beta](#version-300-beta-march-16-2021) March 16, 2021
* [Version 2.5.7](#version-257-june-16-2021) June 16, 2021
* [Version 2.5.6](#version-256-may-17-2021) May 17, 2021
* [Version 2.5.5](#version-255-february-12-2021) February 12, 2021
* [Version 2.5.4](#version-254-december-31-2020) December 31, 2020
* [Version 2.5.3](#version-253-august-12-2020) August 12, 2020
* [Version 2.5.2](#version-252-june-15-2020) June 15, 2020
* [Version 2.5.1](#version-251-may-11-2020) May 11, 2020
* [Version 2.5.0](#version-250-may-6-2020) May 6, 2020
* [Version 2.4.3](#version-243-may-17-2021) May 17, 2021
* [Version 2.4.2](#version-242-june-15-2020) June 15, 2020
* [Version 2.4.1](#version-241-february-11-2020) February 11, 2020
* [Version 2.4.0](#version-240-september-19-2019) September 19, 2019
* [Version 2.3.0](#version-230-august-13-2018) August 13, 2018
* [Version 2.2.2](#version-222-april-30-2020) April 30, 2020
* [Version 2.2.1](#version-221-november-30-2017) November 30, 2017
* [Version 2.2.0](#version-220-august-10-2014) August 10, 2014
* [Version 2.1.0](#version-210-november-25-2013) November 25, 2013
* [Version 2.0.1](#version-201-july-11-2013) July 11, 2013
* [Version 2.0.0](#version-200-april-9-2013) April 9, 2013
* [Version 1.7.1](#version-171-july-31-2012) July 31, 2012
* [Version 1.7.0](#version-170-july-23-2010) July 23, 2010
* [Version 1.6.1](#version-161-october-22-2007) October 22, 2007
* [Version 1.6.0](#version-160-august-3,2007) August 3, 2007
* [Version 1.5.0](#version-150-december-15-2006) December 15, 2006
* [Version 1.4.0a](#version-140a-august-9-2006) August 9, 2006
* [Version 1.4.0](#version-140-august-2,2006) August 2, 2006
* [Version 1.3.1](#version-131-june-14-2006) June 14, 2006
* [Version 1.3.0](#version-130-june-8,2006) June 8, 2006
* [Version 1.2.2](#version-122-march-15-2005) March 15, 2005
* [Version 1.2.1](#version-121-june-6,2004) June 6, 2004
* [Version 1.2.0](#version-120-may-11-2004) May 11, 2004
* [Version 1.1.1](#version-111-march-27-2004) March 27, 2004
* [Version 1.1.0](#version-110-february-6-2004) February 6, 2004
* [Version 1.0.7](#version-107-january-7-2004) January 7, 2004
* [Version 1.0.6](#version-106)
* [Version 1.0.5](#version-105-april-3-2003) April 3, 2003
* [Version 1.0.4](#version-104)
* [Version 1.0.3](#version-103)
* [Version 1.0.2](#version-102)
* [Version 1.0.1](#version-101)
* [Version 1.0](#version-10)

## Version 3.1.3 (October 27, 2021)

Patch release with a change to default zip compression level:

* Default zip compression level is now 4 (instead of 6), which in our
  tests improves compression times by 2x with only a tiny drop in
  compression ratio.
* ``setDefaultZipCompression()`` and ``setDefaultDwaCompression()``
  now set default compression levels for writing.
* The Header how has ``zipCompressionLevel()`` and
  ``dwaCompressionLevel()`` to return the levels used for writing.

Also, various bug fixes, build improvements, and documentation
updates. In particular:

* Fixes a build failure with Imath prior to v3.1
* Fixes a bug in detecting invalid chromaticity values

Specific OSS-fuzz issues:

* OSS-fuzz [40091](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=40091)
  Heap-buffer-overflow in hufDecode
* OSS-fuzz [39997](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39997)
  Null-dereference in Imf_3_1::readCoreScanlinePart
* OSS-fuzz [39996](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39996)
  Heap-buffer-overflow in generic_unpack
* OSS-fuzz [39936](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39936)
  Heap-buffer-overflow in Imf_3_1::memstream_read
* OSS-fuzz [39836](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39836)
  Heap-buffer-overflow in internal_huf_decompress
* OSS-fuzz [39799](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39799)
  Heap-buffer-overflow in unpack_32bit
* OSS-fuzz [39754](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39754)
  Abrt in internal_decode_alloc_buffer
* OSS-fuzz [39737](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39737)
  Heap-buffer-overflow in unpack_16bit
* OSS-fuzz [39683](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39683)
  Null-dereference in Imf_3_1::readCoreScanlinePart
* OSS-fuzz [39630](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39630)
  Direct-leak in internal_decode_alloc_buffer
* OSS-fuzz [39623](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39623)
  Heap-buffer-overflow in unpack_16bit
* OSS-fuzz [39616](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39616)
  Heap-buffer-overflow in Imf_3_1::memstream_read
* OSS-fuzz [39604](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39604)
  Abrt in internal_decode_free_buffer
* OSS-fuzz [39601](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39601)
  Heap-buffer-overflow in internal_huf_decompress
* OSS-fuzz [39591](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39591)
  Integer-overflow in Imf_3_1::readCoreTiledPart
* OSS-fuzz [39579](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39579)
  Undefined-shift in internal_huf_decompress
* OSS-fuzz [39571](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39571)
  Heap-buffer-overflow in generic_unpack
* OSS-fuzz [39568](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39568)
  Null-dereference in Imf_3_1::readCoreScanlinePart
* OSS-fuzz [39542](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39542)
  Heap-buffer-overflow in Imf_3_1::memstream_read
* OSS-fuzz [39538](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39538)
  Heap-buffer-overflow in unpack_16bit_4chan_planar
* OSS-fuzz [39532](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39532)
  Heap-buffer-overflow in unpack_16bit_4chan_planar
* OSS-fuzz [39529](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39529)
  Null-dereference in Imf_3_1::readCoreTiledPart
* OSS-fuzz [39526](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39526)
  Integer-overflow in exr_read_tile_chunk_info
* OSS-fuzz [39522](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39522)
  Direct-leak in internal_decode_alloc_buffer
* OSS-fuzz [39472](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39472)
  Heap-buffer-overflow in unpack_16bit
* OSS-fuzz [39421](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39421)
  Stack-overflow in Imf_3_1::memstream_read
* OSS-fuzz [39399](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39399)
  Direct-leak in exr_attr_preview_init
* OSS-fuzz [39397](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39397)
  Timeout in openexr_exrcheck_fuzzer
* OSS-fuzz [39343](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39343)
  Null-dereference READ in ubsan_GetStackTrace
* OSS-fuzz [39342](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39342)
  Direct-leak in Imf_3_1::OpaqueAttribute::copy
* OSS-fuzz [39340](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39340)
  Stack-overflow in Imf_3_1::memstream_read
* OSS-fuzz [39332](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39332)
  Out-of-memory in openexr_exrcheck_fuzzer
* OSS-fuzz [39329](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39329)
  Negative-size-param in Imf_3_1::memstream_read
* OSS-fuzz [39328](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39328)
  Undefined-shift in internal_exr_compute_tile_information
* OSS-fuzz [39323](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39323)
  Integer-overflow in Imf_3_1::readCoreTiledPart

Merged Pull Requests:
* [1187](https://github.com/AcademySoftwareFoundation/openexr/pull/1187)
  Add size check to memory stream check program
* [1186](https://github.com/AcademySoftwareFoundation/openexr/pull/1186)
  Add extra tile validation
* [1185](https://github.com/AcademySoftwareFoundation/openexr/pull/1185)
  Fix test for bad chunk data to allow for 0-sample deep chunks
* [1184](https://github.com/AcademySoftwareFoundation/openexr/pull/1184)
  Fixes an issue computing the unpacked size of a chunk
* [1183](https://github.com/AcademySoftwareFoundation/openexr/pull/1183)
  Fix decoding of piz when y sampling is not the same for all channels
* [1182](https://github.com/AcademySoftwareFoundation/openexr/pull/1182)
  Require at least one channel
* [1180](https://github.com/AcademySoftwareFoundation/openexr/pull/1180)
  reduce iterations in testIDManifest to speed up
* [1178](https://github.com/AcademySoftwareFoundation/openexr/pull/1178)
  use std::abs in chromaticity sanity tests (fixes #1177)
* [1176](https://github.com/AcademySoftwareFoundation/openexr/pull/1176)
  Update CI builds
* [1174](https://github.com/AcademySoftwareFoundation/openexr/pull/1174)
  Update docs with link to EasyCLA
* [1173](https://github.com/AcademySoftwareFoundation/openexr/pull/1173)
  Fix misc issues due to OSS-fuzz
* [1172](https://github.com/AcademySoftwareFoundation/openexr/pull/1172)
  fix casts in readUInt shifts
* [1169](https://github.com/AcademySoftwareFoundation/openexr/pull/1169)
  Clean up error messages, check against packed size of 0, integer overflow
* [1168](https://github.com/AcademySoftwareFoundation/openexr/pull/1168)
  Refactor attribute size checks
* [1167](https://github.com/AcademySoftwareFoundation/openexr/pull/1167)
  Fix loop iterators in ImfCheckFile.cpp
* [1166](https://github.com/AcademySoftwareFoundation/openexr/pull/1166)
  fix int overflow in calc_level_size
* [1165](https://github.com/AcademySoftwareFoundation/openexr/pull/1165)
  Prevent read when offset past the end of the memstream
* [1164](https://github.com/AcademySoftwareFoundation/openexr/pull/1164)
  Also fail when the user provides a preview image that has a zero size coordinate
* [1163](https://github.com/AcademySoftwareFoundation/openexr/pull/1163)
  don't validate chunk size when file_size unknown
* [1161](https://github.com/AcademySoftwareFoundation/openexr/pull/1161)
  validate filesize before allocating chunk memory
* [1160](https://github.com/AcademySoftwareFoundation/openexr/pull/1160)
  validate dwaCompressionLevel attribute type
* [1150](https://github.com/AcademySoftwareFoundation/openexr/pull/1150)
  Enable Google OSS Fuzz to also test Core library
* [1149](https://github.com/AcademySoftwareFoundation/openexr/pull/1145)
  Enable ephemeral compression levels
* [1145](https://github.com/AcademySoftwareFoundation/openexr/pull/1145)
  Fix when compiling against pre-3.1 version of Imath
* [1125](https://github.com/AcademySoftwareFoundation/openexr/pull/1125)
  Zip: switch to compression level 4 instead of default 6

## Version 3.1.2 (October 4, 2021)

Patch release with various bug fixes, build improvements, and
documentation updates. In particular:

* Fix a test failure on arm7
* Proper handling of pthread with glibc 2.34+
* Miscellaneous fixes for handling of invalid input by the new
  OpenEXRCore library

With this version, the OpenEXR technical documentation formerly
distributed exclusivly as pdf's is now published online at
https://openexr.readthedocs.io, with the document source now
maintained as .rst files in the repo's docs subfolder.

Specific OSS-fuzz issues:

* OSS-fuzz [39196](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39196)
  Stack-buffer-overflow in dispatch_print_error
* OSS-fuzz [39198](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39198)
  Direct-leak in exr_attr_chlist_add_with_length
* OSS-fuzz [39206](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39206)
  Direct-leak in extract_attr_string_vector
* OSS-fuzz [39212](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39212)
  Heap-use-after-free in dispatch_print_error
* OSS-fuzz [39205](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39205)
  Timeout in openexr_exrcheck_fuzzer
* OSS-fuzz [38912](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=38912)
  Integer-overflow in Imf_3_1::bytesPerDeepLineTable
* OSS-fuzz [39084](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=39084)
  Divide-by-zero in Imf_3_1::RGBtoXYZ

Merged Pull Requests:

* [1159](https://github.com/AcademySoftwareFoundation/openexr/pull/1159)
  Fix unterminated string causing issue with print
* [1158](https://github.com/AcademySoftwareFoundation/openexr/pull/1158)
  Fix memory leak when unable to parse the channel list 
* [1157](https://github.com/AcademySoftwareFoundation/openexr/pull/1157)
  Fix leak when parsing header with duplicate attribute names 
* [1156](https://github.com/AcademySoftwareFoundation/openexr/pull/1156)
  Fixes a use-after-free when an invalid type string is provided
* [1155](https://github.com/AcademySoftwareFoundation/openexr/pull/1155)
  Fix hang when there is EOF while extracting string from attr type/name
* [1153](https://github.com/AcademySoftwareFoundation/openexr/pull/1153)
  Avoid div by zero with test for bad chromaticities in RGBtoXYZ
* [1152](https://github.com/AcademySoftwareFoundation/openexr/pull/1152)
  prevent overflow in bytesPerDeepLineTable 
* [1151](https://github.com/AcademySoftwareFoundation/openexr/pull/1151)
  Add additional text to ensure correct detection for threads 
* [1147](https://github.com/AcademySoftwareFoundation/openexr/pull/1147)
  Simplify the definition for bswap_32 for NetBSD 
* [1146](https://github.com/AcademySoftwareFoundation/openexr/pull/1146)
  Fix typo in comment in ImfChromaticities.h
* [1142](https://github.com/AcademySoftwareFoundation/openexr/pull/1142)
  Cleanup cmake thread detection 
* [1141](https://github.com/AcademySoftwareFoundation/openexr/pull/1141)
  Fix issue with unpacked size computation 
* [1138](https://github.com/AcademySoftwareFoundation/openexr/pull/1138)
  the HufDec struct used during decompression also contains a pointer 
* [1136](https://github.com/AcademySoftwareFoundation/openexr/pull/1136)
  Fixes #1135, test which assumed 64-bit pointer size 
* [1134](https://github.com/AcademySoftwareFoundation/openexr/pull/1134)
  Clean up enum declarations in OpenEXRCore 
* [1133](https://github.com/AcademySoftwareFoundation/openexr/pull/1133)
  Fix copy/paste error in special case unpack routine 
* [1132](https://github.com/AcademySoftwareFoundation/openexr/pull/1132)
  Build sphinx/doxygen docs with CMake 
* [1131](https://github.com/AcademySoftwareFoundation/openexr/pull/1131)
  Retire old docs 
* [1130](https://github.com/AcademySoftwareFoundation/openexr/pull/1130)
  Clean up OpenEXRCore doxygen comments 
* [1129](https://github.com/AcademySoftwareFoundation/openexr/pull/1129)
  Guard `__has_attribute` for compilers that don't support it 
* [1124](https://github.com/AcademySoftwareFoundation/openexr/pull/1124)
  Remove throw from ~IlmThread 
* [1123](https://github.com/AcademySoftwareFoundation/openexr/pull/1123)
  Require Imath 3.1 
* [1122](https://github.com/AcademySoftwareFoundation/openexr/pull/1122)
  Remove stray and unnecessary Imf:: namespace prefixes 
* [1120](https://github.com/AcademySoftwareFoundation/openexr/pull/1120)
  Docs: fixed wrong code examples in "how to read a file" section 
* [1111](https://github.com/AcademySoftwareFoundation/openexr/pull/1111)
  Fix document cross-references in .rst files and other gotchas 
* [1108](https://github.com/AcademySoftwareFoundation/openexr/pull/1108)
  Fix formatting in InterpretingDeepPixels.rst: 
* [1104](https://github.com/AcademySoftwareFoundation/openexr/pull/1104)
  'TheoryDeepPixels.rst' first pass converson from latex 
* [1042](https://github.com/AcademySoftwareFoundation/openexr/pull/1042)
  Fix broken link for releases 

## Version 3.1.1 (August 2, 2021)

Patch release that fixes build failures on various systems, introduces
CMake ``CMAKE_CROSSCOMPILING_EMULATOR`` support, and fixes a few other
minor issues.

Merged Pull Requests:

* [1117](https://github.com/AcademySoftwareFoundation/openexr/pull/1117)
  Improve handling of ``#include <*intrin.h>``
* [1116](https://github.com/AcademySoftwareFoundation/openexr/pull/1116)
  Fix up some printf warnings by using appropriate macros
* [1115](https://github.com/AcademySoftwareFoundation/openexr/pull/1115)
  Remove an old check for bsd behaviour, all the bsd-ish oses have the
  leXXtoh functions
* [1112](https://github.com/AcademySoftwareFoundation/openexr/pull/1112)
  Include ``<x86intrin.h>`` only if ``defined(__x86_64__)``
* [1109](https://github.com/AcademySoftwareFoundation/openexr/pull/1109)
  Remove commented-out code in internal_huf.c
* [1106](https://github.com/AcademySoftwareFoundation/openexr/pull/1106)
  ``CMAKE_CROSSCOMPILING_EMULATOR`` and misc. fixes

## Version 3.1.0 (July 22, 2021)

The 3.1 release of OpenEXR introduces a new library, OpenEXRCore,
which is the result of a significant re-thinking of how OpenEXR
manages file I/O and provides access to image data. It begins to
address long-standing scalability issues with multithreaded image
reading and writing.

The OpenEXRCore library provides thread-safe, non-blocking access to
files, which was not possible with the current API, where the
framebuffer management is separate from read requests. It is written
entirely in C and provides a new C-language API alongside the existing
C++ API. This new low-level API allows applications to do custom
unpacking of EXR data, such as on the GPU, while still benefiting from
efficient I/O, file validation, and other semantics. It provides
efficient direct access to EXR files in texturing applications. This C
library also introduces an easier path to implementing OpenEXR
bindings in other languages, such as Rust.

The 3.1 release represents a technology preview for upcoming
releases. The initial release is incremental; the existing API and
underlying behavior has not changed. The new API is available now for
performance validation testing, and then in future OpenEXR releases,
the C++ API will migrate to use the new core in stages.  It is not the
intention to entirely deprecate the C++ API, nor must all applications
re-implement EXR I/O in terms of the C library. The C API does not,
and will not, provide the rich set of utility classes that exist in
the C++ layer. The 3.1 release of the OpenEXRCore library simply
offers new functionality for specialty applications seeking the
highest possible performance. In the future, the ABI will evolve, but
the API will remain consistent, or only have additions.

Technical Design

The OpenEXRCore API introduces a ``context`` object to manage file
I/O. The context provides customization for I/O, memory allocation,
and error handling.  This makes it possible to use a decode and/or
encode pipeline to customize how the chunks are written and read, and
how they are packed or unpacked.

The OpenEXRCore library is built around the concept of “chunks”, or
atomic blocks of data in a file, the smallest unit of data to be read
or written.  The contents of a chunk vary from file to file based on
compression (i.e. zip and zips) and layout (scanline
vs. tiled). Currently this is either 1, 16, or 32 scanlines, or 1 tile
(or subset of a tile on edge boundaries / small mip level).

The OpenEXRCore library is specifically designed for multipart EXR
files. It will continue to produce legacy-compatible single part files
as needed, but the API assumes you are always dealing with a
multi-part file. It also fully supports attributes, although being C,
it lacks some of the C++ layer’s abstraction.

Limitations:

* No support yet for DWAA and DWAB compression during decode and
  encode pipelines. The low-level chunk I/O still works with DWAA and
  DWAB compressed files, but the encoder and decoder are not yet
  included in this release.

* For deep files, reading of deep data is functional, but the path for
  encoding deep data into chunk-level data (i.e. packing and
  compressing) is not yet complete.

* For both of these deficiencies, it is easy to define a custom
  routine to implement this, should it be needed prior to the library
  providing full support.

* No attempt to search through the file and find missing chunks is
  made when a corrupt chunk table is encountered. However, if a
  particular chunk is corrupt, this is handled such that the other
  chunks may be read without rendering the context unusable

Merged Pull Requests:

* [1097](https://github.com/AcademySoftwareFoundation/openexr/pull/1097) Include exported OpenEXR headers with "" instead of <>
* [1092](https://github.com/AcademySoftwareFoundation/openexr/pull/1092) Document current standard optional attributes
* [1088](https://github.com/AcademySoftwareFoundation/openexr/pull/1088) First draft of rst/readthedocs for C API/OpenEXRCore
* [1087](https://github.com/AcademySoftwareFoundation/openexr/pull/1087) Edit doxygen comments for consistency and style
* [1086](https://github.com/AcademySoftwareFoundation/openexr/pull/1086) Simplify names, improve error messages, fix imath usage in Core
* [1077](https://github.com/AcademySoftwareFoundation/openexr/pull/1077) Initial doxygen/sphinx/breathe/readthedocs docs
* [1076](https://github.com/AcademySoftwareFoundation/openexr/pull/1076) Refactor deep tests to separate file, fix deep chunk reads, ripmap reading
* [1074](https://github.com/AcademySoftwareFoundation/openexr/pull/1074) Add utility function for ease of use in other libraries
* [1073](https://github.com/AcademySoftwareFoundation/openexr/pull/1073) Use same struct scheme as box from imath for consistency
* [1069](https://github.com/AcademySoftwareFoundation/openexr/pull/1069) Clean up library VERSION and SOVERSION
* [1062](https://github.com/AcademySoftwareFoundation/openexr/pull/1062) Add missing "throw" before InputExc in IDManifest::init()
* [1045](https://github.com/AcademySoftwareFoundation/openexr/pull/1045) Fix #1039 The vtable for TiledRgbaInputFile was not properly tagged
* [1038](https://github.com/AcademySoftwareFoundation/openexr/pull/1038) fix/extend part number validation in MultiPart methods
* [1024](https://github.com/AcademySoftwareFoundation/openexr/pull/1024) Remove dead code in ImfB44Compressor.cpp
* [1020](https://github.com/AcademySoftwareFoundation/openexr/pull/1020) Fix comparison of integer expressions of different signedness warning
* [870](https://github.com/AcademySoftwareFoundation/openexr/pull/870) WIP: New C core

## Version 3.0.5 (July 1, 2021)

Patch release that fixes problems with library symlinks and
pkg-config, as well as miscellaneous bugs/security issues.

* [1064](https://github.com/AcademySoftwareFoundation/openexr/pull/1064) Use CMAKE_INSTALL_FULL_LIBDIR/INCLUDEDIR in pkgconfig for 3.*
* [1051](https://github.com/AcademySoftwareFoundation/openexr/pull/1051) Fix non-versioned library symlinks in debug build.
* [1050](https://github.com/AcademySoftwareFoundation/openexr/pull/1050) Use CMAKE_<CONFIG>_POSTFIX for .pc file lib suffix.
* [1045](https://github.com/AcademySoftwareFoundation/openexr/pull/1045) Fixes #1039: The vtable for TiledRgbaInputFile was not properly tagged as export 
* [1038](https://github.com/AcademySoftwareFoundation/openexr/pull/1038) fix/extend part number validation in MultiPart methods
* [1037](https://github.com/AcademySoftwareFoundation/openexr/pull/1037) verify data size in deepscanlines with NO_COMPRESSION
* [1036](https://github.com/AcademySoftwareFoundation/openexr/pull/1036) detect buffer overflows in RleUncompress
* The Imath auto-build version defaults to Imath v3.0.5.

## Version 3.0.4 (June 3, 2021)

Patch release that corrects a problem with the release version number
of v3.0.2/v3.0.3:

* [1025](https://github.com/AcademySoftwareFoundation/openexr/pull/1025) Set OPENEXR_VERSION from OpenEXR_VERSION variables
* [1028](https://github.com/AcademySoftwareFoundation/openexr/pull/1028) Fix break of OpenEXRConfig.h generation after PR 1025

## Version 3.0.3 (May 18, 2021)

Patch release that fixes a regression in v3.0.2 the prevented headers
from being installed properly.

## Version 3.0.2 (May 17, 2021)

Patch release with miscellaneous bug/build fixes, including:

* Fix TimeCode.frame max value
* Don't impose C++14 on downstream projects
* Restore fix to macOS universal 2 build lost from #854
* Imath auto-build version defaults to v3.0.2

Specific OSS-fuzz issues:

* OSS-fuzz [33741](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=33741) Integer-overflow in Imf_3_0::getScanlineChunkOffsetTableSize
* OSS-fuzz [32620](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=32620) Out-of-memory in openexr_exrcheck_fuzzer

Merged Pull Requests:

* [1015](https://github.com/AcademySoftwareFoundation/openexr/pull/1015)  Improvements for Bazel build support                            
* [1011](https://github.com/AcademySoftwareFoundation/openexr/pull/1011)  Restore fix to macOS universal 2 build lost from #854           
* [1009](https://github.com/AcademySoftwareFoundation/openexr/pull/1009)  Remove test/warning about CMake version < 3.11                  
* [1008](https://github.com/AcademySoftwareFoundation/openexr/pull/1008)  Clean up setting of OpenEXR version                             
* [1007](https://github.com/AcademySoftwareFoundation/openexr/pull/1007)  Fix TimeCode.frame max value to be 29 instead of 59             
* [1003](https://github.com/AcademySoftwareFoundation/openexr/pull/1003)  Prevent overflow in getScanlineChunkOffsetTableSize             
* [1001](https://github.com/AcademySoftwareFoundation/openexr/pull/1001)  Update CHANGES and SECURITY with recent CVE's                   
* [995](https://github.com/AcademySoftwareFoundation/openexr/pull/995)   Don't impose C++14 on downstream projects                       
* [993](https://github.com/AcademySoftwareFoundation/openexr/pull/993)   Add STATUS message showing Imath_DIR                            
* [992](https://github.com/AcademySoftwareFoundation/openexr/pull/992)   exrcheck -v prints OpenEXR and Imath versions and lib versions  
* [991](https://github.com/AcademySoftwareFoundation/openexr/pull/991)   exrcheck: make readDeepTile allocate memory for just one tile   

## Version 3.0.1 (April 1, 2021)

Major release with major build restructing, security improvements, and
new features:

* Restructuring:
  - The IlmBase/PyIlmBase submodules have been separated into the
    Imath project, now included by OpenEXR via a CMake submodule
    dependency, fetched automatically via CMake's FetchContent if
    necessary.
  - The library is now called ``libOpenEXR`` (instead of
    ``libIlmImf``).  No header files have been renamed, they retain
    the ``Imf`` prefix.
  - Symbol linkage visibility is limited to specific public symbols.

* Build improvements:
  - No more simultaneous static/shared build option.
  - Community-provided support for bazel.

* New Features:
  - ID Manifest Attributes, as described in ["A Scheme for Storing
    Object ID Manifests in OpenEXR
    Images"](https://doi.org/10.1145/3233085.3233086), Peter Hillman,
    DigiPro 18: Proceedings of the 8th Annual Digital Production
    Symposium, August 2018.
  - New program: exrcheck validates the contents of an EXR file.

* Changes:
  - EXR files with no channels are no longer allowed.
  - Hard limit on the size of deep tile sizes; tiles must be less than
    2^30 pixels.
  - Tiled DWAB files used STATIC_HUFFMAN compression.
  - ``Int64`` and ``SInt64`` types are deprecated in favor of
    ``uint64_t`` and ``int64_t``.
  - Header files have been pruned of extraneous ``#include``'s
    ("Include What You Use"), which may generate compiler errors in
    application source code from undefined symbols or
    partially-defined types. These can be resolved by identifying and
    including the appropriate header.
  - See the [porting
    guide](https://github.com/AcademySoftwareFoundation/Imath/blob/master/docs/PortingGuide2-3.md)
    for details about differences from previous releases and how to
    address them.
  - Also refer to the porting guide for details about changes to
    Imath.

Contains all changes in [3.0.1-beta](#version-301-beta-march-28-2021) and [3.0.0-beta](#version-300-beta-march-16-2021).

## Version 3.0.1-beta (March 28, 2021)

Beta patch release:

* OSS-fuzz [32370](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=32370) Out-of-memory in openexr_exrcheck_fuzzer	([987](https://github.com/AcademySoftwareFoundation/openexr/pull/987))
* OSS-fuzz [32067](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=32067) Out-of-memory in openexr_exrcheck_fuzzer	([966](https://github.com/AcademySoftwareFoundation/openexr/pull/966))
* OSS-fuzz [31172](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31172) Timeout in openexr_exrcheck_fuzzer	([987](https://github.com/AcademySoftwareFoundation/openexr/pull/987))

Merged Pull Requests:

* [989](https://github.com/AcademySoftwareFoundation/openexr/pull/989) Release notes for 3.0.1-beta
* [988](https://github.com/AcademySoftwareFoundation/openexr/pull/988) Remove deprecated argument to getChunkOffsetTableSize()
* [987](https://github.com/AcademySoftwareFoundation/openexr/pull/987) exrcheck: reduceMemory now checks pixel size and scanline compression mode 
* [983](https://github.com/AcademySoftwareFoundation/openexr/pull/983) Reduce warnigns reported in #982
* [980](https://github.com/AcademySoftwareFoundation/openexr/pull/980) Bazel cherry picks
* [979](https://github.com/AcademySoftwareFoundation/openexr/pull/979) Pin Imath version to 3.0.0-beta on RB-3.0  
* [968](https://github.com/AcademySoftwareFoundation/openexr/pull/968) Fix typos in Int64/SInt64 deprecation warnings
* [966](https://github.com/AcademySoftwareFoundation/openexr/pull/966) exrcheck: account for size of pixels when estimating memory

## Version 3.0.0-beta (March 16, 2021)

Major release with major build restructing, security improvements, and
new features:

* Restructuring:
  - The IlmBase/PyIlmBase submodules have been separated into the
    Imath project, now included by OpenEXR via a CMake submodule
    dependency, fetched automatically via CMake's FetchContent if
    necessary.
  - The library is now called ``libOpenEXR`` (instead of
    ``libIlmImf``).  No header files have been renamed, they retain
    the ``Imf`` prefix.
  - Symbol linkage visibility is limited to specific public
    symbols. See [SymbolVisibility.md](docs/SymbolVisibility.md) for more
    details.

* Build improvements:
  - No more simultaneous static/shared build option.
  - Community-provided support for bazel.

* New Features:
  - ID Manifest Attributes, as described in ["A Scheme for Storing
    Object ID Manifests in OpenEXR
    Images"](https://doi.org/10.1145/3233085.3233086), Peter Hillman,
    DigiPro 18: Proceedings of the 8th Annual Digital Production
    Symposium, August 2018.
  - New program: exrcheck validates the contents of an EXR file.

* Changes:
  - EXR files with no channels are no longer allowed.
  - Hard limit on the size of deep tile sizes; tiles must be less than
    2^30 pixels.
  - Tiled DWAB files used STATIC_HUFFMAN compression.
  - ``Int64`` and ``SInt64`` types are deprecated in favor of
    ``uint64_t`` and ``int64_t``.
  - Header files have been pruned of extraneous ``#include``'s
    ("Include What You Use"), which may generate compiler errors in
    application source code from undefined symbols or
    partially-defined types. These can be resolved by identifying and
    including the appropriate header.
  - See the [porting
    guide](https://github.com/AcademySoftwareFoundation/Imath/blob/master/docs/PortingGuide2-3.md)
    for details about differences from previous releases and how to
    address them.
  - Also refer to the porting guide for details about changes to
    Imath.

Specific OSS-fuzz issues addressed include:

* OSS-fuzz [31539](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31539) Out-of-memory in openexr_exrcheck_fuzzer	([946](https://github.com/AcademySoftwareFoundation/openexr/pull/946))
* OSS-fuzz [31390](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31390) Out-of-memory in openexr_exrcheck_fuzzer	([939](https://github.com/AcademySoftwareFoundation/openexr/pull/939))
* OSS-fuzz [31293](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31293) Segv on unknown address in Imf_2_5::copyIntoFrameBuffer	([932](https://github.com/AcademySoftwareFoundation/openexr/pull/932))
* OSS-fuzz [31291](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31291) Sanitizer CHECK failure in "((0 && "Address is not in memory and not in shadow?")) != (0)" (0x0, 0x0)	([932](https://github.com/AcademySoftwareFoundation/openexr/pull/932))
* OSS-fuzz [31228](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31228) Integer-overflow in bool Imf_2_5::readDeepTile<Imf_2_5::DeepTiledInputFile>	([930](https://github.com/AcademySoftwareFoundation/openexr/pull/930))
* OSS-fuzz [31221](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31221) Integer-overflow in bool Imf_2_5::readDeepTile<Imf_2_5::DeepTiledInputPart>	([930](https://github.com/AcademySoftwareFoundation/openexr/pull/930))
* OSS-fuzz [31072](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31072) Out-of-memory in openexr_exrcheck_fuzzer	([928](https://github.com/AcademySoftwareFoundation/openexr/pull/928))
* OSS-fuzz [31044](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31044) Timeout in openexr_exrcheck_fuzzer	([926](https://github.com/AcademySoftwareFoundation/openexr/pull/926))
* OSS-fuzz [31015](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=31015) Direct-leak in Imf_2_5::TypedAttribute<Imf_2_5::CompressedIDManifest>::readValueFrom	([925](https://github.com/AcademySoftwareFoundation/openexr/pull/925))
* OSS-fuzz [30969](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=30969) Direct-leak in Imf_2_5::DwaCompressor::LossyDctDecoderBase::execute	([923](https://github.com/AcademySoftwareFoundation/openexr/pull/923))
* OSS-fuzz [30616](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=30616) Timeout in openexr_exrcheck_fuzzer	([919](https://github.com/AcademySoftwareFoundation/openexr/pull/919))
* OSS-fuzz [30605](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=30605) Out-of-memory in openexr_exrcheck_fuzzer	([920](https://github.com/AcademySoftwareFoundation/openexr/pull/920))
* OSS-fuzz [30249](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=30249) Out-of-memory in openexr_exrcheck_fuzzer	([915](https://github.com/AcademySoftwareFoundation/openexr/pull/915))
* OSS-fuzz [29682](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=29682) Out-of-memory in openexr_exrcheck_fuzzer	([902](https://github.com/AcademySoftwareFoundation/openexr/pull/902))
* OSS-fuzz [29393](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=29393) Timeout in openexr_exrcheck_fuzzer	([902](https://github.com/AcademySoftwareFoundation/openexr/pull/902))
* OSS-fuzz [28419](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28419) Out-of-memory in openexr_exrcheck_fuzzer	([895](https://github.com/AcademySoftwareFoundation/openexr/pull/895))
* OSS-fuzz [28155](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28155) Crash in Imf_2_5::PtrIStream::read	([872](https://github.com/AcademySoftwareFoundation/openexr/pull/872))
* OSS-fuzz [28051](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28051) Heap-buffer-overflow in Imf_2_5::copyIntoFrameBuffer	([872](https://github.com/AcademySoftwareFoundation/openexr/pull/872))
* OSS-fuzz [27409](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=27409) Out-of-memory in openexr_exrcheck_fuzzer	([863](https://github.com/AcademySoftwareFoundation/openexr/pull/863))
* OSS-fuzz [26641](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=26641) Invalid-enum-value in readSingleImage	([859](https://github.com/AcademySoftwareFoundation/openexr/pull/859))
* OSS-fuzz [25648](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25648) Out-of-memory in openexr_scanlines_fuzzer	([839](https://github.com/AcademySoftwareFoundation/openexr/pull/839))
* OSS-fuzz [25156](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25156) Out-of-memory in openexr_scanlines_fuzzer	([824](https://github.com/AcademySoftwareFoundation/openexr/pull/824))
* OSS-fuzz [25002](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25002) Out-of-memory in openexr_deepscanlines_fuzzer	([824](https://github.com/AcademySoftwareFoundation/openexr/pull/824))
* OSS-fuzz [24959](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=24959) Integer-overflow in Imf_2_5::cachePadding	([824](https://github.com/AcademySoftwareFoundation/openexr/pull/824))
* OSS-fuzz [24857](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=24857) Out-of-memory in openexr_exrheader_fuzzer	([824](https://github.com/AcademySoftwareFoundation/openexr/pull/824))
* OSS-fuzz [24573](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=24573)	Out-of-memory in openexr_exrenvmap_fuzzer	([824](https://github.com/AcademySoftwareFoundation/openexr/pull/824))

### Merged Pull Requests

* [971](https://github.com/AcademySoftwareFoundation/openexr/pull/971)  Add missing #includes in OpenEXRFuzzTest
* [967](https://github.com/AcademySoftwareFoundation/openexr/pull/967)  3.0.0-beta release notes
* [965](https://github.com/AcademySoftwareFoundation/openexr/pull/965)  Bump version to 3.0.0
* [964](https://github.com/AcademySoftwareFoundation/openexr/pull/964)  Bazel-Support: Update Bazel build files to reflect CMake state
* [963](https://github.com/AcademySoftwareFoundation/openexr/pull/963)  Properly expose header files for float exceptions
* [962](https://github.com/AcademySoftwareFoundation/openexr/pull/962)  Remove IexMath as a library
* [961](https://github.com/AcademySoftwareFoundation/openexr/pull/961)  Enable policy 77 if possible.
* [960](https://github.com/AcademySoftwareFoundation/openexr/pull/960)  Still needed to push the OPENEXR_INSTALL definition higher
* [959](https://github.com/AcademySoftwareFoundation/openexr/pull/959)  The OPENEXR_INSTALL option needs to be defined before it's used
* [956](https://github.com/AcademySoftwareFoundation/openexr/pull/956)  Replace stray Imath:: with IMATH_NAMESPACE::
* [955](https://github.com/AcademySoftwareFoundation/openexr/pull/955)  Usability improvements for submodule use.
* [953](https://github.com/AcademySoftwareFoundation/openexr/pull/953)  Add GLOBAL to add_library(zlib)
* [952](https://github.com/AcademySoftwareFoundation/openexr/pull/952)  Remove 'long' overloads for Xdr::read and Xdr::write functions
* [951](https://github.com/AcademySoftwareFoundation/openexr/pull/951)  Change copyright notices to standard SPDX format
* [950](https://github.com/AcademySoftwareFoundation/openexr/pull/950)  Don't install ImfB44Compressor.h
* [949](https://github.com/AcademySoftwareFoundation/openexr/pull/949)  Bazel build: Bump Imath version to current master
* [948](https://github.com/AcademySoftwareFoundation/openexr/pull/948)  Replace Int64/SInt64 with uint64_t/int64_t
* [946](https://github.com/AcademySoftwareFoundation/openexr/pull/946)  better flag/type verification in deep input files
* [945](https://github.com/AcademySoftwareFoundation/openexr/pull/945)  Fix sign-compare warning
* [944](https://github.com/AcademySoftwareFoundation/openexr/pull/944)  Build-time options for where to get Imath
* [943](https://github.com/AcademySoftwareFoundation/openexr/pull/943)  Add include/OpenEXR to -I and OpenEXRUtil,Iex,IlmThread to -L
* [942](https://github.com/AcademySoftwareFoundation/openexr/pull/942)  Resolve #882 static/shared dual build
* [939](https://github.com/AcademySoftwareFoundation/openexr/pull/939)  enforce limit on area of deep tiles to prevent excessive memory use
* [938](https://github.com/AcademySoftwareFoundation/openexr/pull/938)  Replace UINT_MAX with explicit cast
* [937](https://github.com/AcademySoftwareFoundation/openexr/pull/937)  Add #include <limits> to fix Windows compile error
* [936](https://github.com/AcademySoftwareFoundation/openexr/pull/936)  Incorporate recent config changes into BUILD.bazel
* [932](https://github.com/AcademySoftwareFoundation/openexr/pull/932)  exrcheck: fix handling xSampling when computating slice base
* [930](https://github.com/AcademySoftwareFoundation/openexr/pull/930)  exrcheck: use 64 bit integer math to prevent pointer overflows
* [929](https://github.com/AcademySoftwareFoundation/openexr/pull/929)  Remove all references to "IlmBase"
* [928](https://github.com/AcademySoftwareFoundation/openexr/pull/928)  exrcheck: better tile checks in reduceMemory mode
* [926](https://github.com/AcademySoftwareFoundation/openexr/pull/926)  exrcheck: Revert to using 'getStep' for Rgba interfaces
* [925](https://github.com/AcademySoftwareFoundation/openexr/pull/925)  handle reallocation of idmanifest attributes
* [923](https://github.com/AcademySoftwareFoundation/openexr/pull/923)  free up memory if DWA unRle throws
* [921](https://github.com/AcademySoftwareFoundation/openexr/pull/921)  Only wait for and join joinable threads
* [920](https://github.com/AcademySoftwareFoundation/openexr/pull/920)  exrcheck: check for tilesize in reduceMemory mode
* [919](https://github.com/AcademySoftwareFoundation/openexr/pull/919)  validate size of DWA RLE buffer in decompress
* [916](https://github.com/AcademySoftwareFoundation/openexr/pull/916)  use NO_COMPRESSION in OpenEXRTest/testBackwardCompatibility
* [915](https://github.com/AcademySoftwareFoundation/openexr/pull/915)  exrcheck: assume lots of memory required whenever MultiPart ctor throws
* [913](https://github.com/AcademySoftwareFoundation/openexr/pull/913)  Fixes for recent Imath deprecations
* [911](https://github.com/AcademySoftwareFoundation/openexr/pull/911)  Prevent reading or writing OpenEXR images with no channels
* [909](https://github.com/AcademySoftwareFoundation/openexr/pull/909)  Add idmanifest attribute support
* [906](https://github.com/AcademySoftwareFoundation/openexr/pull/906)  expand testCompression to better test DWAA, DWAB and tiled images
* [902](https://github.com/AcademySoftwareFoundation/openexr/pull/902)  exrcheck: rework 'reduceMemory' and 'reduceTime' modes
* [899](https://github.com/AcademySoftwareFoundation/openexr/pull/899)  Change NOTICE to STATUS to address #891
* [898](https://github.com/AcademySoftwareFoundation/openexr/pull/898)  Add support for Bazel
* [895](https://github.com/AcademySoftwareFoundation/openexr/pull/895)  exrcheck: make reduced memory/time modes more sensitive
* [893](https://github.com/AcademySoftwareFoundation/openexr/pull/893)  Include <limits> where required by newer compilers
* [877](https://github.com/AcademySoftwareFoundation/openexr/pull/877)  ImfCompressor: use STATIC_HUFFMAN for tiled DWAB files (fix #344)
* [874](https://github.com/AcademySoftwareFoundation/openexr/pull/874)  Fix missing header for Visual Studio
* [872](https://github.com/AcademySoftwareFoundation/openexr/pull/872)  Handle xsampling and bad seekg() calls in exrcheck
* [869](https://github.com/AcademySoftwareFoundation/openexr/pull/869)  Enable extra version tag
* [868](https://github.com/AcademySoftwareFoundation/openexr/pull/868)  Make the default symbol visibility hidden for unixen builds
* [864](https://github.com/AcademySoftwareFoundation/openexr/pull/864)  Remove legacy throw() specifications, conform to c++11
* [862](https://github.com/AcademySoftwareFoundation/openexr/pull/862)  E2K: added initial support of MCST Elbrus 2000 CPU architecture
* [859](https://github.com/AcademySoftwareFoundation/openexr/pull/859)  Invalidenum workaround
* [858](https://github.com/AcademySoftwareFoundation/openexr/pull/858)  Merge RC-3 to master
* [848](https://github.com/AcademySoftwareFoundation/openexr/pull/848)  Validate reconstructed chunk sizes
* [846](https://github.com/AcademySoftwareFoundation/openexr/pull/846)  fix extra byte in DWA compressed data
* [839](https://github.com/AcademySoftwareFoundation/openexr/pull/839)  Validate tileoffset table size
* [828](https://github.com/AcademySoftwareFoundation/openexr/pull/828)  Address issues reported by Undefined Behavior Sanitizer running IlmImfTest
* [824](https://github.com/AcademySoftwareFoundation/openexr/pull/824)  reduce size limit for scanline files; prevent large chunkoffset allocations
* [819](https://github.com/AcademySoftwareFoundation/openexr/pull/819)  re-order shift/compare in FastHuf to prevent undefined shift overflow
* [815](https://github.com/AcademySoftwareFoundation/openexr/pull/815)  cmake: Fix paths in .pc files
* [802](https://github.com/AcademySoftwareFoundation/openexr/pull/802)  Modernize mutex
* [796](https://github.com/AcademySoftwareFoundation/openexr/pull/796)  Initial rename of OpenEXR and IlmBase directories and seperation of Test
* [791](https://github.com/AcademySoftwareFoundation/openexr/pull/791)  Initial removal of all Imath source files and minimal cmake adjustments
* [769](https://github.com/AcademySoftwareFoundation/openexr/pull/769)  Bugfix/arkellr remove cvsignore files

## Version 2.5.7 (June 16, 2021)

Patch release with security and build fixes:

* OSS-fuzz [28051](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28051) Heap-buffer-overflow in Imf_2_5::copyIntoFrameBuffer
* OSS-fuzz [28155](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28155) Crash in Imf_2_5::PtrIStream::read 
* Fix pkg-config lib suffix for cmake debug builds

### Merged Pull Requests

* [#1037](https://github.com/AcademySoftwareFoundation/openexr/pull/1037) verify data size in deepscanlines which are not compressed
* [#1036](https://github.com/AcademySoftwareFoundation/openexr/pull/1036) detect buffer overflows in RleUncompress
* [#1032](https://github.com/AcademySoftwareFoundation/openexr/pull/1032) Fix pkg-config lib suffix for cmake debug builds
* [#872](https://github.com/AcademySoftwareFoundation/openexr/pull/872) Handle xsampling and bad seekg() calls in exrcheck

## Version 2.5.6 (May 17, 2021)

Patch release that fixes a regression in Imath::succf()/Imath::predf():

* [#1013](https://github.com/AcademySoftwareFoundation/openexr/pull/1013)
Fixed regression in Imath::succf() and Imath::predf() when negative values are given

## Version 2.5.5 (February 12, 2021)

Patch release with various bug/sanitizer/security fixes, primarily
related to reading corrupted input files, but also a fix for universal
build support on macOS.

Specific OSS-fuzz issues include:

* OSS-fuzz [30291](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=30291) Timeout in openexr_exrcheck_fuzzer
* OSS-fuzz [29106](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=29106) Heap-buffer-overflow in Imf_2_5::FastHufDecoder::decode
* OSS-fuzz [28971](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28971) Undefined-shift in Imf_2_5::cachePadding
* OSS-fuzz [29829](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=29829) Integer-overflow in Imf_2_5::DwaCompressor::initializeBuffers
* OSS-fuzz [30121](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=30121) Out-of-memory in openexr_exrcheck_fuzzer

### Merged Pull Requests

* [914](https://github.com/AcademySoftwareFoundation/openexr/pull/914) additional verification of DWA data sizes
* [910](https://github.com/AcademySoftwareFoundation/openexr/pull/910) update tileoffset sanitycheck to handle ripmaps
* [903](https://github.com/AcademySoftwareFoundation/openexr/pull/903) prevent overflows by using Int64 for all vars in DWA initialize
* [901](https://github.com/AcademySoftwareFoundation/openexr/pull/901) Use size_t for DWA buffersize calculation
* [897](https://github.com/AcademySoftwareFoundation/openexr/pull/897) prevent overflow in RgbaFile cachePadding
* [896](https://github.com/AcademySoftwareFoundation/openexr/pull/896) add buffer size validation to FastHuf decode
* [893](https://github.com/AcademySoftwareFoundation/openexr/pull/893) Include <limits> where required by newer compilers
* [889](https://github.com/AcademySoftwareFoundation/openexr/pull/889) Add explicit #include <limits> for numeric_limits
* [854](https://github.com/AcademySoftwareFoundation/openexr/pull/854) Fix Apple Universal 2 (arm64/x86_64) builds
 
## Version 2.5.4 (December 31, 2020)
 
Patch release with various bug/sanitizer/security fixes, primarily
related to reading corrupted input files.

Security vulnerabilities fixed:

* [CVE-2021-20296](https://nvd.nist.gov/vuln/detail/CVE-2021-20296) Segv on unknown address in Imf_2_5::hufUncompress - Null Pointer dereference
* [CVE-2021-3479](https://nvd.nist.gov/vuln/detail/CVE-2021-3479) Out-of-memory in openexr_exrenvmap_fuzzer
* [CVE-2021-3478](https://nvd.nist.gov/vuln/detail/CVE-2021-3478) Out-of-memory in openexr_exrcheck_fuzzer
* [CVE-2021-3477](https://nvd.nist.gov/vuln/detail/CVE-2021-3477) Heap-buffer-overflow in Imf_2_5::DeepTiledInputFile::readPixelSampleCounts
* [CVE-2021-3476](https://nvd.nist.gov/vuln/detail/CVE-2021-3476) Undefined-shift in Imf_2_5::unpack14
* [CVE-2021-3475](https://nvd.nist.gov/vuln/detail/CVE-2021-3475) Integer-overflow in Imf_2_5::calculateNumTiles
* [CVE-2021-3474](https://nvd.nist.gov/vuln/detail/CVE-2021-3474) Undefined-shift in Imf_2_5::FastHufDecoder::FastHufDecoder

Specific OSS-fuzz issues include:

* OSS-fuzz [24854](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=24854) Segv on unknown address in Imf_2_5::hufUncompress
* OSS-fuzz [24831](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=24831) Undefined-shift in Imf_2_5::FastHufDecoder::FastHufDecoder
* OSS-fuzz [24969](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=24969) Invalid-enum-value in Imf_2_5::TypedAttribute<Imf_2_5::Envmap>::writeValueTo
* OSS-fuzz [25297](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25297) Integer-overflow in Imf_2_5::calculateNumTiles
* OSS-fuzz [24787](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=24787) Undefined-shift in Imf_2_5::unpack14
* OSS-fuzz [25326](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25326) Out-of-memory in openexr_scanlines_fuzzer
* OSS-fuzz [25399](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25399) Heap-buffer-overflow in Imf_2_5::FastHufDecoder::FastHufDecoder
* OSS-fuzz [25415](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25415) Abrt in __cxxabiv1::failed_throw
* OSS-fuzz [25370](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25370) Out-of-memory in openexr_exrenvmap_fuzzer
* OSS-fuzz [25501](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25501) Out-of-memory in openexr_scanlines_fuzzer
* OSS-fuzz [25505](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25505) Heap-buffer-overflow in Imf_2_5::copyIntoFrameBuffer
* OSS-fuzz [25562](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25562) Integer-overflow in Imf_2_5::hufUncompress
* OSS-fuzz [25740](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25740) Null-dereference READ in Imf_2_5::Header::operator
* OSS-fuzz [25743](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25743) Null-dereference in Imf_2_5::MultiPartInputFile::header
* OSS-fuzz [25913](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25913) Out-of-memory in openexr_exrenvmap_fuzzer
* OSS-fuzz [26229](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=26229) Undefined-shift in Imf_2_5::hufDecode
* OSS-fuzz [26658](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=26658) Out-of-memory in openexr_scanlines_fuzzer
* OSS-fuzz [26956](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=26956) Heap-buffer-overflow in Imf_2_5::DeepTiledInputFile::readPixelSampleCounts
* OSS-fuzz [27409](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=27409) Out-of-memory in openexr_exrcheck_fuzzer
* OSS-fuzz [25892](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25892) Divide-by-zero in Imf_2_5::calculateNumTiles
* OSS-fuzz [25894](https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=25894) Floating-point-exception in Imf_2_5::precalculateTileInfo

### Merged Pull Requests

* [817](https://github.com/AcademySoftwareFoundation/openexr/pull/817) double-check unpackedBuffer created in DWA uncompress (OSS-fuzz 24854)
* [818](https://github.com/AcademySoftwareFoundation/openexr/pull/818) compute Huf codelengths using 64 bit to prevent shift overrflow (OSS-fuzz 24831)
* [820](https://github.com/AcademySoftwareFoundation/openexr/pull/820) suppress sanitizer warnings when writing invalid enums (OSS-fuzz 24969)
* [825](https://github.com/AcademySoftwareFoundation/openexr/pull/825) Avoid overflow in calculateNumTiles when size=MAX_INT (OSS-fuzz 25297)
* [826](https://github.com/AcademySoftwareFoundation/openexr/pull/826) restrict maximum tile size to INT_MAX byte limit (OSS-fuzz 25297)
* [832](https://github.com/AcademySoftwareFoundation/openexr/pull/832) ignore unused bits in B44 mode detection (OSS-fuzz 24787)
* [827](https://github.com/AcademySoftwareFoundation/openexr/pull/827) lighter weight reading of Luma-only images via RgbaInputFile (OSS-fuzz 25326)
* [829](https://github.com/AcademySoftwareFoundation/openexr/pull/829) fix buffer overflow check in PIZ decompression (OSS-fuzz 25399, OSS-fuzz 25415)
* [830](https://github.com/AcademySoftwareFoundation/openexr/pull/830) refactor channel filling in InputFile API with tiled source (OSS-fuzz 25370 , OSS-fuzz 25501)
* [831](https://github.com/AcademySoftwareFoundation/openexr/pull/831) Use Int64 in dataWindowForTile to prevent integer overflow (OSS-fuzz 25505)
* [836](https://github.com/AcademySoftwareFoundation/openexr/pull/836) prevent overflow in hufUncompress if nBits is large (OSS-fuzz 25562)
* [840](https://github.com/AcademySoftwareFoundation/openexr/pull/840) add sanity check for reading multipart files with no parts (OSS-fuzz 25740 , OSS-fuzz 25743)
* [841](https://github.com/AcademySoftwareFoundation/openexr/pull/841) more elegant exception handling in exrmaketiled (ZhiWei Sun from Topsec Alpha Lab)
* [843](https://github.com/AcademySoftwareFoundation/openexr/pull/843) reduce B44 _tmpBufferSize (was allocating two bytes per byte) (OSS-fuzz 25913)
* [844](https://github.com/AcademySoftwareFoundation/openexr/pull/844) check EXRAllocAligned succeeded to allocate ScanlineInputFile lineBuffers (ZhiWei Sun from Topsec Alpha Lab)
* [845](https://github.com/AcademySoftwareFoundation/openexr/pull/845) test channels are DCT compressed before DWA decompression (ZhiWei Sun from Topsec Alpha Lab)
* [849](https://github.com/AcademySoftwareFoundation/openexr/pull/849) check for valid Huf code lengths (OSS-fuzz 26229)
* [860](https://github.com/AcademySoftwareFoundation/openexr/pull/860) check 1 part files with 'nonimage' bit have type attribute (OSS-fuzz 26658)
* [861](https://github.com/AcademySoftwareFoundation/openexr/pull/861) Fix overflow computing deeptile sample table size (OSS-fuzz 26956)
* [863](https://github.com/AcademySoftwareFoundation/openexr/pull/863) re-order shift/compare in FastHuf to prevent undefined shift overflow (OSS-fuzz 27409)
* Also, partial fixes from [842](https://github.com/AcademySoftwareFoundation/openexr/pull/842) which do not change the ABI: (OSS-fuzz 25892 , OSS-fuzz 25894)

### Commits \[ git log v2.5.3...v2.5.4\]

* [0c2b46f6](https://github.com/AcademySoftwareFoundation/openexr/commit/0c2b46f630a3b5f2f561c2849d047ee39f899179) Cherry-pick PRs from master branch which fix issues reported by fuzz tests (#875) ([peterhillman](@peterh@wetafx.co.nz) 2020-12-31)

## Version 2.5.3 (August 12, 2020)

Patch release with various bug/security fixes and build/install fixes, plus a performance optimization:

### Summary

* Various sanitizer/fuzz-identified issues related to handling of invalid input
* Fixes to misc compiler warnings
* Cmake fix for building on arm64 macOS (#772)
* Read performance optimization (#782)
* Fix for building on non-glibc (#798)
* Fixes to tests

### Merged Pull Requests

* [812](https://github.com/AcademySoftwareFoundation/openexr/pull/812) free memory if precalculateTileInfo throws
* [809](https://github.com/AcademySoftwareFoundation/openexr/pull/809) Avoid integer overflow in calculateNumTiles()
* [806](https://github.com/AcademySoftwareFoundation/openexr/pull/806) suppress clang undefined behavior sanitizer in EnvmapAttribute::copyValuesFrom()
* [805](https://github.com/AcademySoftwareFoundation/openexr/pull/805) remove extraneous vector allocation in getScanlineChunkOffsetTableSize 
* [804](https://github.com/AcademySoftwareFoundation/openexr/pull/804) prevent invalid tile description enums
* [803](https://github.com/AcademySoftwareFoundation/openexr/pull/803) Fix stack corruption in Matrix tests
* [801](https://github.com/AcademySoftwareFoundation/openexr/pull/801) prevent invalid Compression enum values being read from file
* [798](https://github.com/AcademySoftwareFoundation/openexr/pull/798) IexMathFpu.cpp: Fix build on non-glibc (e.g. musl libc)
* [795](https://github.com/AcademySoftwareFoundation/openexr/pull/795) prevent invalid values in LineOrder enum
* [794](https://github.com/AcademySoftwareFoundation/openexr/pull/794) suppress clang undefined behavior sanitizer in DeepImageStateAttribute::copyValuesFrom()
* [793](https://github.com/AcademySoftwareFoundation/openexr/pull/793) sanityCheckDisplayWindow() ensures that width and height don't cause integer overflow
* [792](https://github.com/AcademySoftwareFoundation/openexr/pull/792) cast signed chars to unsigned longs before left shift in Xdr::read of signed long
* [788](https://github.com/AcademySoftwareFoundation/openexr/pull/788) use 64 bit computation in chunk offset table reconstruction
* [787](https://github.com/AcademySoftwareFoundation/openexr/pull/787) change sanity check in stringvectorattribute to prevent overflow
* [785](https://github.com/AcademySoftwareFoundation/openexr/pull/785) prevent invalid values in Channel's PixelType enum
* [784](https://github.com/AcademySoftwareFoundation/openexr/pull/784) sanity check preview attribute sizes
* [783](https://github.com/AcademySoftwareFoundation/openexr/pull/783) explicitly cast signed chars to unsigned before bitwise left shift in Xdr::read()
* [782](https://github.com/AcademySoftwareFoundation/openexr/pull/782) refactor: use local loop variable in copyFromFrameBuffer
* [778](https://github.com/AcademySoftwareFoundation/openexr/pull/778) Sanity check stringvector size fields on read
* [777](https://github.com/AcademySoftwareFoundation/openexr/pull/777) IlmImfFuzzTest reports incorrect test names and missing files as errors
* [775](https://github.com/AcademySoftwareFoundation/openexr/pull/775) Removes overridden find_package in CMakeLists.txt
* [772](https://github.com/AcademySoftwareFoundation/openexr/pull/772) Disable OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX when building on arm64 macOS
* [770](https://github.com/AcademySoftwareFoundation/openexr/pull/770) IlmImf: Fix clang compiler warnings
* [738](https://github.com/AcademySoftwareFoundation/openexr/pull/738) always ignore chunkCount attribute unless it cannot be computed #738

### Commits \[ git log v2.5.2...v2.5.3\]

* [``425c104f``](https://github.com/AcademySoftwareFoundation/openexr/commit/425c104f7ae9e8e17cc3d9d120d684b93195c402) free memory if precalculateTileInfo throws ([Peter Hillman](@peterh@wetafx.co.nz) 2020-08-10)
* [``7212e337``](https://github.com/AcademySoftwareFoundation/openexr/commit/7212e33729e036d16fb5fd3494af815869771963) Set LIBTOOL_VERSION to 25:2:0 for 2.5.3 ([Cary Phillips](@cary@ilm.com) 2020-08-09)
* [``0b6d5185``](https://github.com/AcademySoftwareFoundation/openexr/commit/0b6d5185d99bff1c4ab7b2fe00d297ef2fcd46e8) Release notes for 2.5.3 ([Cary Phillips](@cary@ilm.com) 2020-08-09)
* [``6b55722b``](https://github.com/AcademySoftwareFoundation/openexr/commit/6b55722b4477e8d4aed04fbeb6b9f5b4226d2bbd) Bump version to 2.5.3 and LIBTOOL_CURRENT to 26 ([Cary Phillips](@cary@ilm.com) 2020-08-09)
* [``40a7ed76``](https://github.com/AcademySoftwareFoundation/openexr/commit/40a7ed76cde1427aa6c935565de96f7ee10d9f76) Change >= to > in overflow calculation ([Cary Phillips](@cary@ilm.com) 2020-08-08)
* [``b10412d5``](https://github.com/AcademySoftwareFoundation/openexr/commit/b10412d55964459e04ff95f982fd8ce2ded4ea43) Avoid integer overflow in calculateNumTiles() ([Cary Phillips](@cary@ilm.com) 2020-08-08)
* [``ed469311``](https://github.com/AcademySoftwareFoundation/openexr/commit/ed469311ac17a8912e2c4cb14856aa2b7f228fac) reformatted references to CVEs in CHANGES.md ([Cary Phillips](@cary@ilm.com) 2020-07-07)
* [``f7c8a7a1``](https://github.com/AcademySoftwareFoundation/openexr/commit/f7c8a7a11a69579d8618f31d0e4a1b7bcc20e939) Add references to CVE-2020-15304, CVE-2020-15305, CVE-2020-15306 to SECURITY.md and CHANGES.md ([Cary Phillips](@cary@ilm.com) 2020-07-07)
* [``0d226001``](https://github.com/AcademySoftwareFoundation/openexr/commit/0d22600163f58c4e3ca20b9f67bd2fe7866e9201) Add #755 to 2.4.2 release notes ([Cary Phillips](@cary@ilm.com) 2020-06-13)
* [``4a4a4f4a``](https://github.com/AcademySoftwareFoundation/openexr/commit/4a4a4f4a58a5d34a132655cc82116a383d787e5d) Improved formatting of commits in release notes ([Cary Phillips](@cary@ilm.com) 2020-06-11)
* [``9c42766b``](https://github.com/AcademySoftwareFoundation/openexr/commit/9c42766bd0347dccb84a68977d11fab8cc83ae3c) added merged PR's to v2.4.2 release notes. ([Cary Phillips](@cary@ilm.com) 2020-06-11)
* [``cc1809ed``](https://github.com/AcademySoftwareFoundation/openexr/commit/cc1809ed27aed48c54cfb730c90bdf570bb18551) Release notes for v2.4.2 ([Cary Phillips](@cary@ilm.com) 2020-06-11)
* [``7fe8d40d``](https://github.com/AcademySoftwareFoundation/openexr/commit/7fe8d40db0c2c02da5f7d2a602fb87a630c3c70d) Remove non-code-related PR's and commits from v2.5.2 release notes. ([Cary Phillips](@cary@ilm.com) 2020-06-11)
* [``bc0b229c``](https://github.com/AcademySoftwareFoundation/openexr/commit/bc0b229c5618ffdc6337817898e3d145b6854194) add commit history to release notes for v2.5.1 and v2.5.2 ([Cary Phillips](@cary@ilm.com) 2020-06-11)
* [``ba76b8ca``](https://github.com/AcademySoftwareFoundation/openexr/commit/ba76b8ca62c2f1d4ccabd2887dc8d09c69102c2f) always ignore chunkCount attribute unless it cannot be computed (#738) ([peterhillman](@peterh@wetafx.co.nz) 2020-05-27)
* [``81818f2a``](https://github.com/AcademySoftwareFoundation/openexr/commit/81818f2a9c9336d71b65b194aaecdef493e9122b) suppress clang undefined behavior sanitizer in EnvmapAttribute::copyValuesFrom() ([Peter Hillman](@peterh@wetafx.co.nz) 2020-08-07)
* [``2f83442f``](https://github.com/AcademySoftwareFoundation/openexr/commit/2f83442f067788751ce857effa3472bf4f79f743) allow undefined EnvMap enum values for future proofing ([Peter Hillman](@peterh@wetafx.co.nz) 2020-08-07)
* [``485b5fe4``](https://github.com/AcademySoftwareFoundation/openexr/commit/485b5fe4d6e575b4af389af98d7a3a2104ce828b) remove extraneous vector allocation in getScanlineChunkOffsetTableSize ([Peter Hillman](@peterh@wetafx.co.nz) 2020-08-06)
* [``7da32d3c``](https://github.com/AcademySoftwareFoundation/openexr/commit/7da32d3ccf6d4eace88ffad093f692a4287b2fbf) refactor: use local loop variable in copyFromFrameBuffer ([Gyula Gubacsi](@gyula.gubacsi@foundry.com) 2020-07-14)
* [``1ecaf4bd``](https://github.com/AcademySoftwareFoundation/openexr/commit/1ecaf4bdfa00204e17aa2a0f51d1ca7d672a9303) prevent invalid tile description enums ([Peter Hillman](@peterh@wetafx.co.nz) 2020-08-05)
* [``88420f93``](https://github.com/AcademySoftwareFoundation/openexr/commit/88420f93857eb2a892683a8a212472883abc8476) prevent invalid Compression enum values being read from file ([Peter Hillman](@peterh@wetafx.co.nz) 2020-07-31)
* [``90736089``](https://github.com/AcademySoftwareFoundation/openexr/commit/90736089eb2c51cfdc311de9b5acc337e4a4c49a) Fix out of bounds assignments ([Darby Johnston](@darbyjohnston@yahoo.com) 2020-08-01)
* [``9752e70d``](https://github.com/AcademySoftwareFoundation/openexr/commit/9752e70d31193f649eb5286bb649916ecfcc51ea) IexMathFpu.cpp: Fix build on non-glibc (e.g. musl libc). ([Niklas Hambüchen](@mail@nh2.me) 2020-07-30)
* [``37e16a88``](https://github.com/AcademySoftwareFoundation/openexr/commit/37e16a88db863da9feeadc721d8df86118c5aab5) cast signed chars to unsigned longs before left shift in read of signed long ([Cary Phillips](@cary@ilm.com) 2020-07-17)
* [``02e1ac54``](https://github.com/AcademySoftwareFoundation/openexr/commit/02e1ac54368ef40e493a67d6804bc706e1bd52db) suppress clang undefined behavior sanitizer in DeepImageStateAttribute::copyValuesFrom() ([Cary Phillips](@cary@ilm.com) 2020-07-22)
* [``bf3edf27``](https://github.com/AcademySoftwareFoundation/openexr/commit/bf3edf271a638e95120c83cbd794502b55f1c64e) fixed CI and Analysis badges in README.md ([Cary Phillips](@cary@ilm.com) 2020-07-16)
* [``93e9f2ac``](https://github.com/AcademySoftwareFoundation/openexr/commit/93e9f2ac3212353414a4e65eb359bcd6dbe7fe6f) prevent invalid values in LineOrder enum ([Cary Phillips](@cary@ilm.com) 2020-07-22)
* [``6bb6257f``](https://github.com/AcademySoftwareFoundation/openexr/commit/6bb6257ffb24f375dfcc40568bfd6357dd6028f8) fixed comment ([Cary Phillips](@cary@ilm.com) 2020-07-20)
* [``1a1e13fd``](https://github.com/AcademySoftwareFoundation/openexr/commit/1a1e13fd8579900ee9f05c3c12bdf2b2aa994593) sanityCheckDisplayWindow() ensures that width and height don't cause integer overflow ([Cary Phillips](@cary@ilm.com) 2020-07-20)
* [``45e14fdf``](https://github.com/AcademySoftwareFoundation/openexr/commit/45e14fdf0700b7afdb94ea7bb788ba9a162d04d7) IlmImfFuzzTest reports incorrect test names and missing files as errors rather than silently succeeding. ([Cary Phillips](@cary@ilm.com) 2020-07-09)
* [``a6bc10f5``](https://github.com/AcademySoftwareFoundation/openexr/commit/a6bc10f5f28c19b8338eb2c6c7226bb6408554f7) use ll in chunk size computation ([Peter Hillman](@peterh@wetafx.co.nz) 2020-07-17)
* [``c6058144``](https://github.com/AcademySoftwareFoundation/openexr/commit/c6058144b653c8ded2e8c0cf0709186486b2453d) use 64 bit computation in chunkoffsettable reconstruction ([Peter Hillman](@peterh@wetafx.co.nz) 2020-07-17)
* [``b33b1187``](https://github.com/AcademySoftwareFoundation/openexr/commit/b33b1187342ff76da08fc7a3ef848b937d7374a3) prevent invalid values in Channel's PixelType enum ([Peter Hillman](@peterh@wetafx.co.nz) 2020-07-16)
* [``b7b8a568``](https://github.com/AcademySoftwareFoundation/openexr/commit/b7b8a5685c0db270b4671ef78c388e3a89605e85) change sanity check in stringvectorattribute to prevent overflow (#787) ([peterhillman](@peterh@wetafx.co.nz) 2020-07-17)
* [``09eadd12``](https://github.com/AcademySoftwareFoundation/openexr/commit/09eadd12d86763fda854b36524ae37680d8ff4c5) cast signed chars to unsigned before bitwise left shift in Xdr::read() ([Cary Phillips](@cary@ilm.com) 2020-07-14)
* [``3cf874cb``](https://github.com/AcademySoftwareFoundation/openexr/commit/3cf874cbbd23d945a0057f10145bd5f3ce2be679) sanity check preview attribute sizes ([Peter Hillman](@peterh@wetafx.co.nz) 2020-07-15)
* [``849c6776``](https://github.com/AcademySoftwareFoundation/openexr/commit/849c6776f6627a11710227c026dd4aa6de8f7738) Tidy whitespace in ImfStringVectorAttribute.cpp ([peterhillman](@peterh@wetafx.co.nz) 2020-07-10)
* [``fcaa1691``](https://github.com/AcademySoftwareFoundation/openexr/commit/fcaa1691071f90df9202818315f4f9d1bc13c54e) sanity check string vectors on read ([Peter Hillman](@peterh@wetafx.co.nz) 2020-07-10)
* [``0d13c74a``](https://github.com/AcademySoftwareFoundation/openexr/commit/0d13c74a3bfa497465c3e42847b9c62089f0454b) Removes overridden find_package in CMakeLists.txt in favor of reusing the generated config files and setting (IlmBase/OpenEXR)_DIR variables Overriding a cmake function is undocumented functionallity and only works one time. Better to avoid if possible. ([Peter Steneteg](@peter@steneteg.se) 2020-06-17)
* [``1343c08a``](https://github.com/AcademySoftwareFoundation/openexr/commit/1343c08a7eb13764bbb6c21db22e5a78169754db) Cast to uintptr_t instead of size_t for mask ops on ptrs. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-06-25)
* [``72de4c9e``](https://github.com/AcademySoftwareFoundation/openexr/commit/72de4c9ef32e2e9eb4e6d9499a0fadb96ae28796) Switching to current c++ casting style. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-06-24)
* [``9534e36d``](https://github.com/AcademySoftwareFoundation/openexr/commit/9534e36d1d1993db7a7cc3ba4c58ec4d7a4a8dd5) IlmImf: Fix misc compiler warnings. ([Arkell Rasiah](@arkellrasiah@gmail.com) 2020-06-23)
* [``8e53ab8d``](https://github.com/AcademySoftwareFoundation/openexr/commit/8e53ab8d13b1b6c14c716573e6f16d079e799ab4) Disable OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX when building on arm64 macOS ([Yining Karl Li](@betajippity@gmail.com) 2020-07-03)
* [``67b1b88d``](https://github.com/AcademySoftwareFoundation/openexr/commit/67b1b88de6ad454a1b267ee9a4e19b4efbdbe19d) Addresses PR#767: Removal of legacy .cvsignore files. ([Arkell Rasiah](@arkellrasiah@gmail.com) 2020-06-19)
* [``801e5d87``](https://github.com/AcademySoftwareFoundation/openexr/commit/801e5d8750dd8b8a6e25c131899136c575b20d07) Fix typo in README ([cia-rana](@kiwamura0314@gmail.com) 2020-06-15)

## Version 2.5.2 (June 15, 2020)

Patch release with various bug/security fixes and build/install fixes.

### Summary

* [CVE-2020-15305](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15305) Invalid input could cause a heap-use-after-free error in DeepScanLineInputFile::DeepScanLineInputFile() 
* [CVE-2020-15306](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15306) Invalid chunkCount attributes could cause heap buffer overflow in getChunkOffsetTableSize() 
* [CVE-2020-15304](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15304) Invalid tiled input file could cause invalid memory access TiledInputFile::TiledInputFile() 
* OpenEXRConfig.h now correctly sets OPENEXR_PACKAGE_STRING to "OpenEXR" (rather than "IlmBase") 
* Various Windows build fixes

### Merged Pull Requests

* [755](https://github.com/AcademySoftwareFoundation/openexr/pull/755) Fix OPENEXR_PACKAGE_NAME
* [747](https://github.com/AcademySoftwareFoundation/openexr/pull/747) Fix the PyIlmBase tests for the autotools build
* [743](https://github.com/AcademySoftwareFoundation/openexr/pull/743) Applies OPENEXR_DLL only to shared libraries and no longer to static libraries
* [742](https://github.com/AcademySoftwareFoundation/openexr/pull/742) Removes symlink creation under Windows
* [738](https://github.com/AcademySoftwareFoundation/openexr/pull/738) always ignore chunkCount attribute unless it cannot be computed
* [733](https://github.com/AcademySoftwareFoundation/openexr/pull/733) added missing PyImathMatrix22.cpp to Makefile.am, for the autotools build
* [730](https://github.com/AcademySoftwareFoundation/openexr/pull/730) fix #728 - missing 'throw' in deepscanline error handling
* [727](https://github.com/AcademySoftwareFoundation/openexr/pull/727) check null pointer in broken tiled file handling

### Commits \[ git log v2.5.1...v2.5.2\]

* [``6f0d14d5``](https://github.com/AcademySoftwareFoundation/openexr/commit/6f0d14d576b6c2d3931f2c444b994207bc4bbc44) bump version to 2.5.2 ([Cary Phillips](@cary@ilm.com) 2020-06-11)
* [``162fe820``](https://github.com/AcademySoftwareFoundation/openexr/commit/162fe82092188fd172cba26af0deec3b0a95a4fa) Fix OPENXR_PACKAGE_NAME ([Cary Phillips](@cary@ilm.com) 2020-06-10)
* [``58e96f53``](https://github.com/AcademySoftwareFoundation/openexr/commit/58e96f534cd42bf9ee59725aadbf5d3b25d041fe) removed commented out lines in Makefile.am ([Cary Phillips](@cary@ilm.com) 2020-05-28)
* [``6c82409e``](https://github.com/AcademySoftwareFoundation/openexr/commit/6c82409e14f415d460a3318dc9848f0c266f1215) Fix PyImathTest, PyImathNumpyTest, PyIexTest to work in the autotools build. ([Cary Phillips](@cary@ilm.com) 2020-05-28)
* [``248abc23``](https://github.com/AcademySoftwareFoundation/openexr/commit/248abc23d134705bc41d167dcc04fafae231409c) Fix OPENEXR_DLL on test applications ([Transporter](@ogre.transporter@gmail.com) 2020-05-29)
* [``ccf91b95``](https://github.com/AcademySoftwareFoundation/openexr/commit/ccf91b95b662d97b0cd8b9d87fd3eb4f5d70e880) Applies OPENEXR_DLL only to shared libraries and no longer to static libraries ([Transporter](@ogre.transporter@gmail.com) 2020-05-26)
* [``c8f2463a``](https://github.com/AcademySoftwareFoundation/openexr/commit/c8f2463a910da90640d13d325ef689160d17ffe9) Removes symlink creation under Windows ([Transporter](@ogre.transporter@gmail.com) 2020-05-26)
* [``78274436``](https://github.com/AcademySoftwareFoundation/openexr/commit/782744364aa92d02add62f38bc29ae2ce2c743e9) added missing PyImathMatrix22.cpp to Makefile.am, for the autotools build. ([Cary Phillips](@cary@ilm.com) 2020-05-19)
* [``30349291``](https://github.com/AcademySoftwareFoundation/openexr/commit/303492919aa1cc39fb0c3d34d96b9f0090d3bdac) fix memory leak in deep scanline IlmImfFuzzTest ([Peter Hillman](@peterh@wetafx.co.nz) 2020-05-16)
* [``21014549``](https://github.com/AcademySoftwareFoundation/openexr/commit/21014549c2728049abe273a719c3fca074234799) fix memory leak in deep scanline IlmImfFuzzTest ([Peter Hillman](@peterh@wetafx.co.nz) 2020-05-16)
* [``07e93a3d``](https://github.com/AcademySoftwareFoundation/openexr/commit/07e93a3de1200355c1c32e2c4cc15ec87e312602) check null pointer in broken tiled file handling ([Peter Hillman](@peterh@wetafx.co.nz) 2020-05-16)
* [``d8741bcc``](https://github.com/AcademySoftwareFoundation/openexr/commit/d8741bccbcc5e68cc4fac3cb11f08c655e6553fc) fix #728 - missing 'throw' in deepscanline error handling ([Peter Hillman](@peterh@wetafx.co.nz) 2020-05-19)

## Version 2.5.1 (May 11, 2020)

A patch release that corrects the SO version for the v2.5 release,
which missed getting bumped in v2.5.0.

This release also fixes an improper failure in IlmImfTest when running
on ARMv7 and AAarch64.

### Merged Pull Requests

* [717](https://github.com/AcademySoftwareFoundation/openexr/pull/717) Fix #713: cast to unsigned in testHuf checksum
* [714](https://github.com/AcademySoftwareFoundation/openexr/pull/714) Bump the SO version to 25 for the 2.5 release

### Commits \[ git log v2.5.0...v2.5.1\]

* [``e823bf28``](https://github.com/AcademySoftwareFoundation/openexr/commit/e823bf282882d02e8ad1c7f6ca867807e1fd6044) Fix #713: cast to unsigned in testHuf checksum (#717) ([peterhillman](@peterh@wetafx.co.nz) 2020-05-10)
* [``5670325e``](https://github.com/AcademySoftwareFoundation/openexr/commit/5670325ea2f44c2b4d4764b151b7570181bd280b) Bump the version to 2.5.1 on the RB-2.5 branch ([Cary Phillips](@cary@ilm.com) 2020-05-09)
* [``d944ffac``](https://github.com/AcademySoftwareFoundation/openexr/commit/d944fface105c79fc4a34a4aa4bc1be39d5aabbc) set LIBTOOL_REVISION=0 for the 2.5 release. ([Cary Phillips](@cary@ilm.com) 2020-05-09)
* [``3ec82975``](https://github.com/AcademySoftwareFoundation/openexr/commit/3ec82975243d83e2732da7b7cbdc08f87f30609c) Bump SO version to 25 for the 2.5 release. ([Cary Phillips](@cary@ilm.com) 2020-05-09)

## Version 2.5.0 (May 6, 2020)

Minor release with miscellaneous bug fixes and small features

### Summary

* No more build-time header generation: toFloat.h, eLut.h,
  b44ExpLogTable.h, and dwaLookups.h are now ordinary header files, no
  longer generated on the fly.
* New StdISSTream class, an "input" stringstream version of StdOSStream
* New Matrix22 class in Imath
* Chromaticity comparison operator now includes white (formerly ignored)
* Various cmake fixes
* Bug fixes for various memory leaks
* Bug fixes for various invalid memory accesses
* New checks to detect damaged input files
* OpenEXR_Viewers has been deprecated, removed from the top-level
  cmake build and documentation.

### Merged Pull Requests

* [712](https://github.com/AcademySoftwareFoundation/openexr/pull/712) Removed #include PyIlmBaseConfigInternal.h from all public .h files.
* [711](https://github.com/AcademySoftwareFoundation/openexr/pull/711) Rewrote testToFloat(). 
* [709](https://github.com/AcademySoftwareFoundation/openexr/pull/709) Fix clean pthreads strikes back
* [708](https://github.com/AcademySoftwareFoundation/openexr/pull/708) Fix clean pthreads
* [707](https://github.com/AcademySoftwareFoundation/openexr/pull/707) A clean version of #673: Allow the use of Pthreads with WIN32/64 
* [705](https://github.com/AcademySoftwareFoundation/openexr/pull/705) added recent commits and PR's to 2.5.0 release notes 
* [704](https://github.com/AcademySoftwareFoundation/openexr/pull/704) fixed typos in README.md
* [703](https://github.com/AcademySoftwareFoundation/openexr/pull/703) Release notes for 2.2.2
* [702](https://github.com/AcademySoftwareFoundation/openexr/pull/702) bump version on the 2.2 branch to 2.2.2
* [700](https://github.com/AcademySoftwareFoundation/openexr/pull/700) Patch fixes for CVE-2020-* from commit e79d229 into release/2.2
* [699](https://github.com/AcademySoftwareFoundation/openexr/pull/699) Bump version to 2.5.0
* [698](https://github.com/AcademySoftwareFoundation/openexr/pull/698) Fix to make Boost_NO_BOOST_CMAKE a cache variable
* [697](https://github.com/AcademySoftwareFoundation/openexr/pull/697) Fix python module install on macOS
* [696](https://github.com/AcademySoftwareFoundation/openexr/pull/696) TSC meeting notes 4-23-20
* [694](https://github.com/AcademySoftwareFoundation/openexr/pull/694) TSC meeting notes 4-16-20
* [693](https://github.com/AcademySoftwareFoundation/openexr/pull/693) Update the release notes and security notices with 2020 CVE's
* [692](https://github.com/AcademySoftwareFoundation/openexr/pull/692) Meetings 4.2.20
* [690](https://github.com/AcademySoftwareFoundation/openexr/pull/690) Implementation of Matrix22
* [685](https://github.com/AcademySoftwareFoundation/openexr/pull/685) Fix libc++abi.dylib: Pure virtual function called!
* [683](https://github.com/AcademySoftwareFoundation/openexr/pull/683) Typo in INSTALL.md: cmake now builds three modules, not four.
* [682](https://github.com/AcademySoftwareFoundation/openexr/pull/682) TSC meeting notes 2020 03 05
* [680](https://github.com/AcademySoftwareFoundation/openexr/pull/680) fixed a/an use in Matrix33 and Matrix44
* [676](https://github.com/AcademySoftwareFoundation/openexr/pull/676) Remove OpenEXR_Viewers from the cmake build files and from INSTALL.md
* [675](https://github.com/AcademySoftwareFoundation/openexr/pull/675) TSC meeting notes for 2/27/2020
* [672](https://github.com/AcademySoftwareFoundation/openexr/pull/672) Fix cmake install failing when install dir contains spaces, fix symlinks for shared libraries on DLL platforms
* [669](https://github.com/AcademySoftwareFoundation/openexr/pull/669) CLA docs moved from "docs" to "contributors"
* [667](https://github.com/AcademySoftwareFoundation/openexr/pull/667) TSC meeting notes 2/20/2020
* [666](https://github.com/AcademySoftwareFoundation/openexr/pull/666) ImfChromaticities.cpp: Add back white to equality ops.
* [662](https://github.com/AcademySoftwareFoundation/openexr/pull/662) support reloading broken files with IlmImfFuzzTest
* [661](https://github.com/AcademySoftwareFoundation/openexr/pull/661) fix unitialized pointer and double-delete bugs
* [660](https://github.com/AcademySoftwareFoundation/openexr/pull/660) exrdisplay: limit maximum image size based on available screen res
* [659](https://github.com/AcademySoftwareFoundation/openexr/pull/659) fix memory leaks and invalid memory accesses
* [658](https://github.com/AcademySoftwareFoundation/openexr/pull/658) Fix yet more memory leaks from constructor exceptions
* [657](https://github.com/AcademySoftwareFoundation/openexr/pull/657) Release notes for 2.4.1 patch release.
* [656](https://github.com/AcademySoftwareFoundation/openexr/pull/656) fix crash with damaged EXR files
* [655](https://github.com/AcademySoftwareFoundation/openexr/pull/655) Notes 2020 02 06
* [653](https://github.com/AcademySoftwareFoundation/openexr/pull/653) fix memory leak from constructor exceptions
* [650](https://github.com/AcademySoftwareFoundation/openexr/pull/650) ImfAcesFile.cpp: Fix primary conversion edge case.
* [649](https://github.com/AcademySoftwareFoundation/openexr/pull/649) ImfChromaticities.h/cpp: Typo and pass by ref change.
* [647](https://github.com/AcademySoftwareFoundation/openexr/pull/647) fix typo and ref to theory document in InterpretingDeepPixels
* [645](https://github.com/AcademySoftwareFoundation/openexr/pull/645) Various CI Updates:
* [643](https://github.com/AcademySoftwareFoundation/openexr/pull/643) Various checks to improve handling of damaged input files
* [642](https://github.com/AcademySoftwareFoundation/openexr/pull/642) Fixed security email address to security@openexr.org
* [641](https://github.com/AcademySoftwareFoundation/openexr/pull/641) Updates to fix broken Windows build.
* [640](https://github.com/AcademySoftwareFoundation/openexr/pull/640) notes from 1/9/20 and 1/16/20
* [639](https://github.com/AcademySoftwareFoundation/openexr/pull/639) Split Targets and Config and add find_package
* [638](https://github.com/AcademySoftwareFoundation/openexr/pull/638) OpenEXR/ImfStdIO.[cpp h]: Added StdISStream.
* [637](https://github.com/AcademySoftwareFoundation/openexr/pull/637) OpenEXR/IlmImf/ImfHuf.cpp: Fix OS dependent exr binary data for piz.
* [635](https://github.com/AcademySoftwareFoundation/openexr/pull/635) Make docs install optional in CMake
* [634](https://github.com/AcademySoftwareFoundation/openexr/pull/634) Add interface includes to IlmBase and OpenEXR
* [631](https://github.com/AcademySoftwareFoundation/openexr/pull/631) add missing IMF_EXPORT to ImfOpenInputFile for dllexport
* [629](https://github.com/AcademySoftwareFoundation/openexr/pull/629) Fixed some typos
* [627](https://github.com/AcademySoftwareFoundation/openexr/pull/627) possible fix for #570: avoid writing NaNs into test images
* [626](https://github.com/AcademySoftwareFoundation/openexr/pull/626) fix testB44ExpLogTable and testDwaLookups, and Makefile.am
* [622](https://github.com/AcademySoftwareFoundation/openexr/pull/622) Azure and SonarCloud updates.
* [620](https://github.com/AcademySoftwareFoundation/openexr/pull/620) Switch from int to size_t to prevent overflow with huge images in exrdisplay (Fix for #610)
* [619](https://github.com/AcademySoftwareFoundation/openexr/pull/619) TSC meeting notes November 21, 2019
* [618](https://github.com/AcademySoftwareFoundation/openexr/pull/618) change URL to https://github.com/AcademySoftwareFoundation/openexr
* [616](https://github.com/AcademySoftwareFoundation/openexr/pull/616) Fix issue 289, C++17 compatibility
* [615](https://github.com/AcademySoftwareFoundation/openexr/pull/615) Add a missing break statement when determining compression in exr2aces
* [613](https://github.com/AcademySoftwareFoundation/openexr/pull/613) Notes 2019 11 15
* [612](https://github.com/AcademySoftwareFoundation/openexr/pull/612) Add a 'clang-format' build target
* [608](https://github.com/AcademySoftwareFoundation/openexr/pull/608) Fix #595 and others, issue with pkgconfig generation under cmake
* [606](https://github.com/AcademySoftwareFoundation/openexr/pull/606) Remove all build-time header generation
* [601](https://github.com/AcademySoftwareFoundation/openexr/pull/601) "Rule of 5" copy/assign/move declarations
* [600](https://github.com/AcademySoftwareFoundation/openexr/pull/600) TSC Meeting Notes 2019/10/24
* [599](https://github.com/AcademySoftwareFoundation/openexr/pull/599) Prepare 2.4 release branch
* [598](https://github.com/AcademySoftwareFoundation/openexr/pull/598) Fix for #571: keep all values word-aligned in IlmImfTest testLargeDataWindowOffsets
* [590](https://github.com/AcademySoftwareFoundation/openexr/pull/590) TSC Meeting notes for 2019-10-17
* [588](https://github.com/AcademySoftwareFoundation/openexr/pull/588) Gitignore
* [587](https://github.com/AcademySoftwareFoundation/openexr/pull/587) removed unnecessary .m4 files
* [586](https://github.com/AcademySoftwareFoundation/openexr/pull/586) TSC Meeting Notes 2019-10-3
* [585](https://github.com/AcademySoftwareFoundation/openexr/pull/585) Added mission statement to README.md
* [582](https://github.com/AcademySoftwareFoundation/openexr/pull/582) Azure macOS job fixes
* [580](https://github.com/AcademySoftwareFoundation/openexr/pull/580) More SonarCloud "bug" fixes in Imath
* [578](https://github.com/AcademySoftwareFoundation/openexr/pull/578) More fixes for SonarCloud bugs
* [577](https://github.com/AcademySoftwareFoundation/openexr/pull/577) Sonar fixes in IlmImf and IlmImfTest
* [576](https://github.com/AcademySoftwareFoundation/openexr/pull/576) TSC meeting notes
* [575](https://github.com/AcademySoftwareFoundation/openexr/pull/575) Sonar fixes for Iex, IexTest, and ImathTest
* [574](https://github.com/AcademySoftwareFoundation/openexr/pull/574) Change Azure SonarCloud job to run weekly.
* [569](https://github.com/AcademySoftwareFoundation/openexr/pull/569) TSC meeting notes for the last 3 weeks
* [562](https://github.com/AcademySoftwareFoundation/openexr/pull/562) CVE listing in SECURITY.md
* [561](https://github.com/AcademySoftwareFoundation/openexr/pull/561) A few more NOSONAR suppressions in PyImath
* [560](https://github.com/AcademySoftwareFoundation/openexr/pull/560) Clean up small number of sonarcloud bugs / warnings
* [559](https://github.com/AcademySoftwareFoundation/openexr/pull/559) Change Azure SonarCloud job to run for every PR, rather than only for…
* [558](https://github.com/AcademySoftwareFoundation/openexr/pull/558) Add NOSONAR comments to PyImath to suppress "self==self" bug reports.
* [557](https://github.com/AcademySoftwareFoundation/openexr/pull/557) Fix warnings when _FORTIFY_SOURCE set
* [556](https://github.com/AcademySoftwareFoundation/openexr/pull/556) Fix #555
* [554](https://github.com/AcademySoftwareFoundation/openexr/pull/554) Replace use of rand()/srand() with the C++11 <random>
* [553](https://github.com/AcademySoftwareFoundation/openexr/pull/553) Increase Azure timeout for SonarCloud, Linux and MacOS jobs.

### Closed Issues

* [689](https://github.com/AcademySoftwareFoundation/openexr/issues/689) I was able to get OpenEXR to install by adding `-std=c++11` to the `extra_compile_flags` in setup.py, as lgritz  and peterhillman suggested. Here's the file with it added:
* [688](https://github.com/AcademySoftwareFoundation/openexr/issues/688) Invalid shift (141647077)
* [687](https://github.com/AcademySoftwareFoundation/openexr/issues/687) ZLIB not found
* [686](https://github.com/AcademySoftwareFoundation/openexr/issues/686) Using the example Chromacity files - issue with chromaticities
* [679](https://github.com/AcademySoftwareFoundation/openexr/issues/679) mipmap / ripmap question
* [674](https://github.com/AcademySoftwareFoundation/openexr/issues/674) OpenEXR_Viewers  / libCg aliasing error
* [671](https://github.com/AcademySoftwareFoundation/openexr/issues/671) Tiles/Ocean.exr core dumps on latest code built from git
* [668](https://github.com/AcademySoftwareFoundation/openexr/issues/668) 2.4.1: test suite build is failing
* [665](https://github.com/AcademySoftwareFoundation/openexr/issues/665) openexr/OpenEXR_Viewers/config/LocateCg.cmake issue on Linux
* [663](https://github.com/AcademySoftwareFoundation/openexr/issues/663) 2.4.1: missing dist tar balls
* [654](https://github.com/AcademySoftwareFoundation/openexr/issues/654) build breaks backwards compatibility
* [651](https://github.com/AcademySoftwareFoundation/openexr/issues/651) ImfChromaticities.cpp: Revisiting the == and != operators
* [648](https://github.com/AcademySoftwareFoundation/openexr/issues/648) find_package macro redefinition conflicts with vcpkg macro on Windows.
* [633](https://github.com/AcademySoftwareFoundation/openexr/issues/633) ImfStdIO.[cpp h]:  Missing StdISStream class
* [632](https://github.com/AcademySoftwareFoundation/openexr/issues/632) Platform/OS dependent piz compressed binary data
* [630](https://github.com/AcademySoftwareFoundation/openexr/issues/630) OpenEXR loading not making use of multiple threads
* [628](https://github.com/AcademySoftwareFoundation/openexr/issues/628) Missing C++11 type traits for half
* [625](https://github.com/AcademySoftwareFoundation/openexr/issues/625) OPENEXR_DLL issues
* [623](https://github.com/AcademySoftwareFoundation/openexr/issues/623) Documentation : Typo in "Interpreting Deep Pixels"
* [617](https://github.com/AcademySoftwareFoundation/openexr/issues/617) Move openexr repo to AcademySoftwareFoundation organization
* [611](https://github.com/AcademySoftwareFoundation/openexr/issues/611) How to force Python3 build on Linux
* [610](https://github.com/AcademySoftwareFoundation/openexr/issues/610) huge images cause bad_array_new_length exception in exrdisplay
* [607](https://github.com/AcademySoftwareFoundation/openexr/issues/607) Getting started... build & hello world.
* [604](https://github.com/AcademySoftwareFoundation/openexr/issues/604) PyIlmBaseConfig not in export set?
* [595](https://github.com/AcademySoftwareFoundation/openexr/issues/595) Broken pkgconfig files when building with cmake
* [594](https://github.com/AcademySoftwareFoundation/openexr/issues/594) Python modules are not linked to Boost::python
* [593](https://github.com/AcademySoftwareFoundation/openexr/issues/593) Python modules are not installed with cmake
* [584](https://github.com/AcademySoftwareFoundation/openexr/issues/584) OpenEXR CLAs
* [581](https://github.com/AcademySoftwareFoundation/openexr/issues/581) ImfCompressor.h  is not installed
* [579](https://github.com/AcademySoftwareFoundation/openexr/issues/579) ImfFrameBuffer.h not compatible with C++98 code
* [573](https://github.com/AcademySoftwareFoundation/openexr/issues/573) Linker cannot find "boost_python-vc140-mt-x64-1_66.lib" on Windows.
* [572](https://github.com/AcademySoftwareFoundation/openexr/issues/572) "IlmImf-2_3.dll" shared library is not built anymore in 2.3.0.
* [571](https://github.com/AcademySoftwareFoundation/openexr/issues/571) Test failure on ARMv7
* [570](https://github.com/AcademySoftwareFoundation/openexr/issues/570) Test failure on i686
* [567](https://github.com/AcademySoftwareFoundation/openexr/issues/567) CMake builds produce invalid pkg-config files
* [566](https://github.com/AcademySoftwareFoundation/openexr/issues/566) throwErrno symbols missing when Iex is linked before other libraries statically
* [565](https://github.com/AcademySoftwareFoundation/openexr/issues/565) 2.4.0 tarball signature missing
* [564](https://github.com/AcademySoftwareFoundation/openexr/issues/564) CVE-2006-2277
* [563](https://github.com/AcademySoftwareFoundation/openexr/issues/563) CVE-2016-4629 and CVE-2016-4630
* [555](https://github.com/AcademySoftwareFoundation/openexr/issues/555) cmake errors when used as sub-project via add_subdirectory()

### Commits \[ git log v2.4.0...v2.5.0\]

* [``b12ea7f3``](https://github.com/AcademySoftwareFoundation/openexr/commit/b12ea7f30d624d51f1b69a2ffa9159a4f07a7974) Pthreads: Some stuff @meshula overlooked. ([Gregorio Litenstein](@g.litenstein@gmail.com) 2020-05-04)
* [``af8864d2``](https://github.com/AcademySoftwareFoundation/openexr/commit/af8864d259d3ef523fc75eaab4f9a74b0f3b092c) pthreads: Fix CMake/Autotools to check for them ([Gregorio Litenstein](@g.litenstein@gmail.com) 2020-05-04)
* [``2ef3d626``](https://github.com/AcademySoftwareFoundation/openexr/commit/2ef3d6265a56cd1ca7c4112a616db6987f134c4a) Pthreads: Some stuff @meshula overlooked. ([Gregorio Litenstein](@g.litenstein@gmail.com) 2020-05-04)
* [``3ab677bd``](https://github.com/AcademySoftwareFoundation/openexr/commit/3ab677bd375db896215459a49de77ac87fbbb19c) A clean version of #673: Allow the use of Pthreads with WIN32/64 builds under MinGW ([Cary Phillips](@cary@ilm.com) 2020-05-01)
* [``4bb99704``](https://github.com/AcademySoftwareFoundation/openexr/commit/4bb99704799830f1be1fa8cde559e3f2f63068a1) added recent commits and PR's to 2.5.0 release notes ([Cary Phillips](@cary@ilm.com) 2020-04-30)
* [``ac4fb158``](https://github.com/AcademySoftwareFoundation/openexr/commit/ac4fb15895447ce042528cc965ce2b242d130311) fixed wording of OpenEXR_Viewers in 2.5.0 release notes. ([Cary Phillips](@cary@ilm.com) 2020-04-29)
* [``c0542060``](https://github.com/AcademySoftwareFoundation/openexr/commit/c0542060df75e5726e0b51ecc1de01aa29b3b448) fixed spacing in 2.5.0 release notes ([Cary Phillips](@cary@ilm.com) 2020-04-28)
* [``c65d0d87``](https://github.com/AcademySoftwareFoundation/openexr/commit/c65d0d8708e99758a7f10fb1b4596d53298a52bb) Added summary of changes to 2.5.0 release notes. ([Cary Phillips](@cary@ilm.com) 2020-04-28)
* [``275ab234``](https://github.com/AcademySoftwareFoundation/openexr/commit/275ab2341081d32160298b01ef903eb4befce8c9) added merged PR's and closed issues to 2.5.0 release notes ([Cary Phillips](@cary@ilm.com) 2020-04-27)
* [``99bcaf5a``](https://github.com/AcademySoftwareFoundation/openexr/commit/99bcaf5aba87eb0c987f5e43e279d1f72ad8b953) added 2.5.0 commits to release notes ([Cary Phillips](@cary@ilm.com) 2020-04-27)
* [``70202128``](https://github.com/AcademySoftwareFoundation/openexr/commit/70202128342b5daf0e97b227c923cb573edecff5) Update SECURITY.md to note that CVE-2020-* are not in v2.2.2 ([Cary Phillips](@cary@ilm.com) 2020-04-29)
* [``8ab7adbc``](https://github.com/AcademySoftwareFoundation/openexr/commit/8ab7adbc2b4c0fca637a7115344f5f34c6f26139) Release notes for 2.2.2 ([Cary Phillips](@cary@ilm.com) 2020-04-29)
* [``72e9ff25``](https://github.com/AcademySoftwareFoundation/openexr/commit/72e9ff25d8e843f6a475a2e81b8aae0df04d3a25) TSC meeting notes 4-23-20 ([Cary Phillips](@cary@ilm.com) 2020-04-23)
* [``656a3a5c``](https://github.com/AcademySoftwareFoundation/openexr/commit/656a3a5c60394880b50081c95c4ca0ab2cf4143e) Add cmake option to install PyIlmBase pkg-config file ([Cary Phillips](@cary@ilm.com) 2020-03-05)
* [``2a82f18c``](https://github.com/AcademySoftwareFoundation/openexr/commit/2a82f18c83a17cf1e98d6d9349779a150f5bdc3c) fixed typos in README.md ([Cary Phillips](@cary@ilm.com) 2020-04-29)
* [``27f45978``](https://github.com/AcademySoftwareFoundation/openexr/commit/27f459781b6a3cb69727397bb989e8dc2aa8850c) Bump version to 2.5.0 ([Cary Phillips](@cary@ilm.com) 2020-04-28)
* [``72cc6e02``](https://github.com/AcademySoftwareFoundation/openexr/commit/72cc6e02e991771db075d5c5e6a184325be47b0a) Fix to make Boost_NO_BOOST_CMAKE a cache variable ([Mark Sisson](@5761292+marksisson@users.noreply.github.com) 2020-04-26)
* [``b9199b51``](https://github.com/AcademySoftwareFoundation/openexr/commit/b9199b5155ab8c1d245ddeb61006b1bf2de66d84) Fix python module install on macos ([Mark Sisson](@5761292+marksisson@users.noreply.github.com) 2020-04-26)
* [``0b26caf6``](https://github.com/AcademySoftwareFoundation/openexr/commit/0b26caf6c33656d38cf10f7a090d3713ac4ee291) TSC meeting notes 4-16-20 ([Cary Phillips](@cary@ilm.com) 2020-04-16)
* [``9d8bb109``](https://github.com/AcademySoftwareFoundation/openexr/commit/9d8bb109968cd169765f1da7b2022a6b6b3a93f0) Implemented all tests transferable to the 2x2 matrix case from 3x3. Added needed functionality to ensure boost::python worked for testing. ([Owen Thompson](@oxt3479@rit.edu) 2020-04-09)
* [``713e6ce5``](https://github.com/AcademySoftwareFoundation/openexr/commit/713e6ce54babdd4181c23d7d0e6c8bb00164a953) Implemented additional C++ functionality needed to pass 2x2 testing parameters: extracting euler angles and overloaded vector multiplication. ([Owen Thompson](@oxt3479@rit.edu) 2020-04-09)
* [``1b20f7bd``](https://github.com/AcademySoftwareFoundation/openexr/commit/1b20f7bd7dc1a8bf37200d46f84645c613513c4a) Wrote tests transferable to the 2x2 cases in C++ ([Owen Thompson](@oxt3479@rit.edu) 2020-04-09)
* [``d404df49``](https://github.com/AcademySoftwareFoundation/openexr/commit/d404df499a32b63ab57f48177d275806ea8addd5) Matrix22 template constructor and make identity no longer use memset. ([Owen Thompson](@oxt3479@rit.edu) 2020-03-30)
* [``f20e1602``](https://github.com/AcademySoftwareFoundation/openexr/commit/f20e1602d64b03397bc54425ff7f5be2e3214aff) Implementation of operator << on Matrix22 for stream output. ([Owen Thompson](@oxt3479@rit.edu) 2020-03-30)
* [``c5a10a77``](https://github.com/AcademySoftwareFoundation/openexr/commit/c5a10a776c8655dbcbe241a3496952c4c3787071) Implementation of arbitrarily transferable functions from 3x3 to 2x2. Removed gaus-jordan and other problematic operations (doesn't work on 2x2) ([Owen Thompson](@oxt3479@rit.edu) 2020-03-24)
* [``308f1076``](https://github.com/AcademySoftwareFoundation/openexr/commit/308f1076d001e573a9fbf240bb85995aaea1ce2c) fixed spacing ([Cary Phillips](@cary@ilm.com) 2020-04-16)
* [``a2392101``](https://github.com/AcademySoftwareFoundation/openexr/commit/a2392101e8101bed90fc3370e8840d208dd88c02) update with new CVE's ([Cary Phillips](@cary@ilm.com) 2020-04-16)
* [``d7da549e``](https://github.com/AcademySoftwareFoundation/openexr/commit/d7da549e1561002d9d278960fdc537f7c535376c) edited GSoC discussion ([Cary Phillips](@cary@ilm.com) 2020-04-03)
* [``c4d27400``](https://github.com/AcademySoftwareFoundation/openexr/commit/c4d27400c51db899b35b4fb729815e17054391ea) typo ([Cary Phillips](@cary@ilm.com) 2020-04-02)
* [``c76f4c8d``](https://github.com/AcademySoftwareFoundation/openexr/commit/c76f4c8d3feb670c14a4320f6c171deea4c750da) added John ([Cary Phillips](@cary@ilm.com) 2020-04-02)
* [``e9ff88bb``](https://github.com/AcademySoftwareFoundation/openexr/commit/e9ff88bb1d3de0d19029d159ac9d1414b790b88c) typo. ([Cary Phillips](@cary@ilm.com) 2020-04-02)
* [``b0f4dc48``](https://github.com/AcademySoftwareFoundation/openexr/commit/b0f4dc4849910819b4d54a895823591057e9d2a5) TSC meeting notes 4/2/2020 ([Cary Phillips](@cary@ilm.com) 2020-04-02)
* [``fa435e2a``](https://github.com/AcademySoftwareFoundation/openexr/commit/fa435e2afe5fce3f5e26220bc46474b8775c6716) Fix libc++abi.dylib: Pure virtual function called! ([dgmzc](@dorian.gmz@hotmail.com) 2020-03-10)
* [``e23fdf6e``](https://github.com/AcademySoftwareFoundation/openexr/commit/e23fdf6e02dbd8157b1d468143a82f6632781dee) Typo in INSTALL.md: cmake now builds three modules, not four. ([Cary Phillips](@cary@ilm.com) 2020-03-05)
* [``0132627f``](https://github.com/AcademySoftwareFoundation/openexr/commit/0132627f3e46fd376c785e223abbc3f5e418ae5e) added some details. ([Cary Phillips](@cary@ilm.com) 2020-03-05)
* [``23c7e72c``](https://github.com/AcademySoftwareFoundation/openexr/commit/23c7e72cb25daa3820d9745d9f49c86320316082) TSC Meeting notes 3/5/2020 ([Cary Phillips](@cary@ilm.com) 2020-03-05)
* [``6780843d``](https://github.com/AcademySoftwareFoundation/openexr/commit/6780843d9da05e5f7bebab2bda9dd437cc4a1909) fixed a/an use in Matrix33 and Matrix44 ([Phyrexian](@jarko.paska@gmail.com) 2020-03-05)
* [``560f7c2e``](https://github.com/AcademySoftwareFoundation/openexr/commit/560f7c2e0eb1e4adec40884eb6126585d08e70f5) Remove OpenEXR_Viewers from the cmake build files and from the INSTALL.md instructions. ([Cary Phillips](@cary@ilm.com) 2020-02-27)
* [``01fa5a20``](https://github.com/AcademySoftwareFoundation/openexr/commit/01fa5a20dde82849203117bfe5de2b2cb21d84a4) TSC meeting notes for 2/27/2020 ([Cary Phillips](@cary@ilm.com) 2020-02-27)
* [``d2639ab3``](https://github.com/AcademySoftwareFoundation/openexr/commit/d2639ab3bb60b7b316d6f7893446e38591bd9f3e) Add interface includes to IlmBase and OpenEXR ([Harry Mallon](@hjmallon@gmail.com) 2020-01-07)
* [``6da250f6``](https://github.com/AcademySoftwareFoundation/openexr/commit/6da250f63d8460788a8b6bb2a642d9c981ab2bb8) Fix cmake install failing when install dir contains spaces, fix symlinks for shared libraries on DLL platforms ([Simon Boorer](@sboorer@ilm.com) 2020-02-26)
* [``6d26cbfc``](https://github.com/AcademySoftwareFoundation/openexr/commit/6d26cbfc1c453c79513b0dad5704fed13e76feda) Split Targets and Config and add find_package ([Harry Mallon](@hjmallon@gmail.com) 2020-01-07)
* [``2f92fcbb``](https://github.com/AcademySoftwareFoundation/openexr/commit/2f92fcbb4fb25caafe4358d1c4a4c7b940016af4) ImfChromaticities.cpp: Add back white to equality ops. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-02-20)
* [``d2deb6d8``](https://github.com/AcademySoftwareFoundation/openexr/commit/d2deb6d8844814615fed247048d36898e7f4c407) IlmImfTest/testExistingStreams.cpp: Test for Imf::StdOSStream/StdISStream. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-02-26)
* [``9a172a22``](https://github.com/AcademySoftwareFoundation/openexr/commit/9a172a220d59dd4363f4c07bac59facae9a1ae6f) OpenEXR/ImfStdIO.[cpp h]: Added StdISStream. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-01-10)
* [``d9019d81``](https://github.com/AcademySoftwareFoundation/openexr/commit/d9019d81a74ac7cf8710bb72590200369686856d) CLA docs moved from "docs" to "contributors" ([Cary Phillips](@cary@ilm.com) 2020-02-24)
* [``48c21063``](https://github.com/AcademySoftwareFoundation/openexr/commit/48c2106310c8edefc7c1387cffc466665e4f38d2) ImfAcesFile.cpp: Remove redundant equality check. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-02-07)
* [``adc21e52``](https://github.com/AcademySoftwareFoundation/openexr/commit/adc21e5250cf938ecd6bf6fcbcfb4e7da7382671) ImfAcesFile.cpp: Fix primary conversion edge case. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-01-31)
* [``3576a8bd``](https://github.com/AcademySoftwareFoundation/openexr/commit/3576a8bd16ba36ad20832b5727d3fe9ff210dd0d) TSC meeting notes 2/20/2020 ([Cary Phillips](@cary@ilm.com) 2020-02-20)
* [``817faac5``](https://github.com/AcademySoftwareFoundation/openexr/commit/817faac5c18a7c9e66ae2adb9e3f312ff3e7f631) fix unitialised pointer and double-delete bugs (#661) ([peterhillman](@peterh@wetafx.co.nz) 2020-02-24)
* [``a0e84f62``](https://github.com/AcademySoftwareFoundation/openexr/commit/a0e84f62def6276f38e518a2724d9f7ac4daa9ad) add help and options information IlmImfTest and IlmImfFuzzTest ([Peter Hillman](@peterh@wetafx.co.nz) 2020-02-13)
* [``608b4938``](https://github.com/AcademySoftwareFoundation/openexr/commit/608b4938fb23861cf6e6792ecf8373e0c9a8ebb5) support reloading broken files with IlmImfFuzzTest ([Peter Hillman](@peterh@wetafx.co.nz) 2020-02-11)
* [``d129fae9``](https://github.com/AcademySoftwareFoundation/openexr/commit/d129fae907a2928f675b1c654f9c8a32a84103b2) Added #659 and associated commit to 2.4.1 release notes. ([Cary Phillips](@cary@ilm.com) 2020-02-10)
* [``b6bef538``](https://github.com/AcademySoftwareFoundation/openexr/commit/b6bef53821384c82ebd0912513be3e4579d1c176) Release notes for 2.4.1 patch release. ([Cary Phillips](@cary@ilm.com) 2020-02-06)
* [``0ca9b6e8``](https://github.com/AcademySoftwareFoundation/openexr/commit/0ca9b6e855d32a794874b1624581d68fcc3f87c0) Added #659 and associated commit to 2.4.1 release notes. ([Cary Phillips](@cary@ilm.com) 2020-02-10)
* [``a966db03``](https://github.com/AcademySoftwareFoundation/openexr/commit/a966db0341369108c6f85bdd92b44ef26265f43b) Release notes for 2.4.1 patch release. ([Cary Phillips](@cary@ilm.com) 2020-02-06)
* [``d06c223f``](https://github.com/AcademySoftwareFoundation/openexr/commit/d06c223f2e9f36766ef7dbec89954393a3b0ba0b) exrdisplay: limit maximum image size based on available screen resolution ([Peter Hillman](@peterh@wetafx.co.nz) 2020-02-10)
* [``e79d2296``](https://github.com/AcademySoftwareFoundation/openexr/commit/e79d2296496a50826a15c667bf92bdc5a05518b4) fix memory leaks and invalid memory accesses ([Peter Hillman](@peterh@wetafx.co.nz) 2020-02-08)
* [``2c37c4bd``](https://github.com/AcademySoftwareFoundation/openexr/commit/2c37c4bd39d3b03248cca42c63d0adbc40827c58) Fix yet more memory leaks from constructor exceptions (#658) ([peterhillman](@peterh@wetafx.co.nz) 2020-02-09)
* [``3422b344``](https://github.com/AcademySoftwareFoundation/openexr/commit/3422b344f6189e499fe4c00f11491843a23d24a4) fix crash with damaged EXR files (#656) ([peterhillman](@peterh@wetafx.co.nz) 2020-02-08)
* [``5754217f``](https://github.com/AcademySoftwareFoundation/openexr/commit/5754217fc506efad20f0ccb509ac447c7c68b671) typo in date. ([Cary Phillips](@cary@ilm.com) 2020-02-06)
* [``7f183953``](https://github.com/AcademySoftwareFoundation/openexr/commit/7f18395337d9a439246370245eb585e5c2efaa6c) TSC meeting notes for Feb 6, 2020 ([Cary Phillips](@cary@ilm.com) 2020-02-06)
* [``cdc70f60``](https://github.com/AcademySoftwareFoundation/openexr/commit/cdc70f60c525c533aefa2b0663b9e0b723cad463) ImfChromaticities.h/cpp: Typo and pass by ref change. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-01-31)
* [``51bd0ff5``](https://github.com/AcademySoftwareFoundation/openexr/commit/51bd0ff530fb20586c4bf95241d035f237792989) fix memory leak from constructor exceptions (#653) ([peterhillman](@peterh@wetafx.co.nz) 2020-02-07)
* [``88246d99``](https://github.com/AcademySoftwareFoundation/openexr/commit/88246d991e0318c043e6f584f7493da08a31f9f8) OpenEXR/IlmImfTest/testHuf.cpp: Do the compressVerify() on deterministic data sets. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-01-17)
* [``0042c451``](https://github.com/AcademySoftwareFoundation/openexr/commit/0042c45171aeff6ec2d165c4c2621514f055d380) OpenEXR/IlmImf/ImfHuf.cpp: Fix OS dependent exr binary data for piz. ([Arkell Rasiah](@arasiah@pixsystem.com) 2020-01-10)
* [``89ce46f3``](https://github.com/AcademySoftwareFoundation/openexr/commit/89ce46f38c5e658d21df9179c1641c496cab7396) force x/y Sampling to 1 for Deep Scanline Images ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-28)
* [``0a1aa55e``](https://github.com/AcademySoftwareFoundation/openexr/commit/0a1aa55ef108169c933ddaa631c1f6cb02b69050) minor tweaks and typo fixes ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-28)
* [``acad98d6``](https://github.com/AcademySoftwareFoundation/openexr/commit/acad98d6d3e787f36012a3737c23c42c7f43a00f) missing header for ptrdiff_t ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-24)
* [``c14af4bb``](https://github.com/AcademySoftwareFoundation/openexr/commit/c14af4bb58c8748cfe2f132147ba38abd0845812) fix test suite memory leak from testDeepTiledBasic ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-24)
* [``a8508ab0``](https://github.com/AcademySoftwareFoundation/openexr/commit/a8508ab05ffeedba394e646506030f94769e0f15) test for multipart threading was leaking memory ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-24)
* [``b673e6ad``](https://github.com/AcademySoftwareFoundation/openexr/commit/b673e6ad0ec6cef94d86b9586244d26088a3d792) Fix cleanup when DeepScanLineInputFile constructor throws ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-24)
* [``53a06468``](https://github.com/AcademySoftwareFoundation/openexr/commit/53a06468ef5a08f4f2beb2d264a20547d7a78753) fixes to memory leak when constructors throw exceptions ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-23)
* [``d4fbaad4``](https://github.com/AcademySoftwareFoundation/openexr/commit/d4fbaad4efe5d0ddf325da44ecbab105ebb2954e) fix memory leak in test suite ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-23)
* [``dea0ef1e``](https://github.com/AcademySoftwareFoundation/openexr/commit/dea0ef1ee7b2f4d2aa42ffba7b442e5d8051222b) fix memory leak on DeepTiledInput files: compressor for sample count table wasn't deleted ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-23)
* [``2ae5f837``](https://github.com/AcademySoftwareFoundation/openexr/commit/2ae5f8376b0a6c3e2bb100042f5de79503ba837a) fix check for valid ruleSize ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-23)
* [``a6408c90``](https://github.com/AcademySoftwareFoundation/openexr/commit/a6408c90339bdf19f89476578d7f936b741be9b2) avoid creating compression object just to compute numLinesInBuffer ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-23)
* [``e7c26f6e``](https://github.com/AcademySoftwareFoundation/openexr/commit/e7c26f6ef5bf7ae8ea21ecf19963186cd1391720) abort when file claims to have excessive scanline data requirements ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-22)
* [``b1c34c49``](https://github.com/AcademySoftwareFoundation/openexr/commit/b1c34c496b62117115b1089b18a44e0031800a09) fix memory leak when reading damaged PIZ files ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-22)
* [``ea334989``](https://github.com/AcademySoftwareFoundation/openexr/commit/ea3349896d4a8a3b523e8f3b830334a85240b1e6) sanity check data reads from PIZ data ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-21)
* [``43cd3ad4``](https://github.com/AcademySoftwareFoundation/openexr/commit/43cd3ad47d53356da6ae2e983e47c8313aebf72e) improve bad count detection in huf decompress ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-21)
* [``801272c9``](https://github.com/AcademySoftwareFoundation/openexr/commit/801272c9bf8b84a66c62f1e8a4490ece81da6a56) check for bad bit counts in Huff encoded data ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-21)
* [``7a52d40a``](https://github.com/AcademySoftwareFoundation/openexr/commit/7a52d40ae23c148f27116cb1f6e897b9143b372c) bypass SSE optimization when skipping subsampled channels ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-21)
* [``b9997d0c``](https://github.com/AcademySoftwareFoundation/openexr/commit/b9997d0c045fa01af3d2e46e1a74b07cc4519446) prevent int overflow when calculating buffer offsets ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-20)
* [``822e218c``](https://github.com/AcademySoftwareFoundation/openexr/commit/822e218c795e989abdf74112b924d0da8acc967b) exrmakepreview: switch preview-to-full scaling vars from floats to doubles to prevent rounding causing overflows ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-20)
* [``3eda5d70``](https://github.com/AcademySoftwareFoundation/openexr/commit/3eda5d70aba127bae9bd6bae9956fcf024b64031) fixes for DWA uncompress: sanity check unknown data reading, off-by-one error on max suffix string length ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-20)
* [``37750013``](https://github.com/AcademySoftwareFoundation/openexr/commit/37750013830def57f19f3c3b7faaa9fc1dae81b3) Sanity check for input buffer overruns in RLE uncompress ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-20)
* [``df987cab``](https://github.com/AcademySoftwareFoundation/openexr/commit/df987cabc20c90803692022fd232def837cb88cc) validate tiles have valid headers when raw reading tiles ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-20)
* [``6bad53af``](https://github.com/AcademySoftwareFoundation/openexr/commit/6bad53af7eebed507564dd5fc90320e4c6a6c0bc) Force tile sizes to be less than INT_MAX bytes, in line with the maximum dimensions of data windows ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-20)
* [``1cbf6b9a``](https://github.com/AcademySoftwareFoundation/openexr/commit/1cbf6b9a4497f71586ac11dc60ed21cf0cc529bd) fix typo and ref to theory document in InterpretingDeepPixels ([Peter Hillman](@peterh@wetafx.co.nz) 2020-01-28)
* [``6546ff20``](https://github.com/AcademySoftwareFoundation/openexr/commit/6546ff20961003825f86662efe16842ff6a64f32) Various CI Updates: - fix python warnings. - fix Cmake include(clang_format) error - added Linux VFX 2020 builds - removed MacOS 10.13 due to Azure ending support - temporarily disable gcov in Sonar, due to SC regression CPP-2395 ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2020-01-25)
* [``f9229e98``](https://github.com/AcademySoftwareFoundation/openexr/commit/f9229e98c93b4bc0179bb12904d03071cc5a8718) Updates to fix broken Windows build. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2020-01-20)
* [``fce8c17b``](https://github.com/AcademySoftwareFoundation/openexr/commit/fce8c17b3731fd3212c8fba52fecfea597eb98fa) possible fix for #570: avoid writing NaNs into test images ([Peter Hillman](@peterh@wetafx.co.nz) 2019-11-29)
* [``9750a1db``](https://github.com/AcademySoftwareFoundation/openexr/commit/9750a1db7f92be3def678977eb741b6762316580) fix testB44ExpLogTable and testDwaLookups, and Makefile.am ([Peter Hillman](@peterh@wetafx.co.nz) 2019-11-29)
* [``bd6ab91f``](https://github.com/AcademySoftwareFoundation/openexr/commit/bd6ab91f6002e90c17c385391e17c06c7ea1dfb8) use Int64 types for width*height calculations in exrdisplay ([Peter Hillman](@peterh@wetafx.co.nz) 2019-11-25)
* [``5adac705``](https://github.com/AcademySoftwareFoundation/openexr/commit/5adac705e18de43008ec0ecb562969ede2a7a927) Switch from int to size_t to prevent overflow with huge images ([Peter Hillman](@peterh@wetafx.co.nz) 2019-11-25)
* [``b1477e0e``](https://github.com/AcademySoftwareFoundation/openexr/commit/b1477e0eea3d9e901012795bed2f499c96e028dc) added Rod to TSC notes ([Cary Phillips](@cary@ilm.com) 2019-11-14)
* [``220f9d4b``](https://github.com/AcademySoftwareFoundation/openexr/commit/220f9d4b2a36c994b9043aa785b1970ad652b8f1) TSC notes 2019-11-15 ([Cary Phillips](@cary@ilm.com) 2019-11-14)
* [``93a4c794``](https://github.com/AcademySoftwareFoundation/openexr/commit/93a4c794950c042ee025f8c4250e1c5b34c18af8) Don't change CMAKE_MODULE_PATH ([Larry Gritz](@lg@larrygritz.com) 2019-11-24)
* [``18d7b6a1``](https://github.com/AcademySoftwareFoundation/openexr/commit/18d7b6a184718a6bb7a0583ae072f507b83bab66) typo ([Larry Gritz](@lg@larrygritz.com) 2019-11-14)
* [``66f48992``](https://github.com/AcademySoftwareFoundation/openexr/commit/66f48992ddd1401f8e9f4f876a737c2c62c209f8) Add a 'clang-format' build target ([Larry Gritz](@lg@larrygritz.com) 2019-11-12)
* [``63fdd366``](https://github.com/AcademySoftwareFoundation/openexr/commit/63fdd36686baf1fd69990309ae43128fb2ab3f16) notes from 1/9/20 and 1/16/20 ([Cary Phillips](@cary@ilm.com) 2020-01-16)
* [``56b248ec``](https://github.com/AcademySoftwareFoundation/openexr/commit/56b248ec86499992488b549863d4ef1bc6eb459f) fixed typo in CONTRIBUTING.md ([Cary Phillips](@cary@ilm.com) 2020-01-24)
* [``7e6e6f0b``](https://github.com/AcademySoftwareFoundation/openexr/commit/7e6e6f0b1244450c5da2a4e8caed8febd19e1d95) Reference SECURITY.md in CONTRIBUTING.md ([John Mertic](@jmertic@linuxfoundation.org) 2020-01-21)
* [``fbe08034``](https://github.com/AcademySoftwareFoundation/openexr/commit/fbe08034eebf9eff192d4068ffbdb807351a3c46) Fixed security email address to security@openexr.org ([John Mertic](@jmertic@linuxfoundation.org) 2020-01-21)
* [``8f43dd55``](https://github.com/AcademySoftwareFoundation/openexr/commit/8f43dd559609a2b1f8787b922c1e5a87a8057838) TSC meeting notes November 21, 2019 ([Cary Phillips](@cary@ilm.com) 2019-11-21)
* [``767d497c``](https://github.com/AcademySoftwareFoundation/openexr/commit/767d497c09d9a20dea4c510fc997b6393d52c33d) add missing IMF_EXPORT to ImfOpenInputFile in order to be able to use it from a windows dll ([Laurens Voerman](@l.voerman@rug.nl) 2019-12-12)
* [``7bd899ac``](https://github.com/AcademySoftwareFoundation/openexr/commit/7bd899ac6f90efe5348518389d944856b5a73c7e) Make docs install optional in CMake ([Harry Mallon](@hjmallon@gmail.com) 2020-01-08)
* [``afa84f87``](https://github.com/AcademySoftwareFoundation/openexr/commit/afa84f87ae24546f71a0e9ffac400e92a8da8b99) Fixed typos ([John Mertic](@jmertic@linuxfoundation.org) 2019-12-04)
* [``b65a275f``](https://github.com/AcademySoftwareFoundation/openexr/commit/b65a275f189ee679c1e252c60085e8ceadce929f) Fixed some typos ([John Mertic](@jmertic@linuxfoundation.org) 2019-12-04)
* [``824ed557``](https://github.com/AcademySoftwareFoundation/openexr/commit/824ed557b3c59288a685356c708e5806b1122fe1) Updated SonarCloud properties/token and README status widgets. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-11-26)
* [``c02742f2``](https://github.com/AcademySoftwareFoundation/openexr/commit/c02742f28c23226352bd5d2050a282fbe9b868e1) change URL to https://github.com/AcademySoftwareFoundation/openexr ([Cary Phillips](@cary@ilm.com) 2019-11-21)
* [``2ae3d476``](https://github.com/AcademySoftwareFoundation/openexr/commit/2ae3d476ac19f6aa9950bb7beffdb10dbd120308) change URL to https://github.com/AcademySoftwareFoundation/openexr ([Cary Phillips](@cary@ilm.com) 2019-11-21)
* [``1296f73c``](https://github.com/AcademySoftwareFoundation/openexr/commit/1296f73cac143eaa50ee87ae1947129e4ce740cf) Add a missing break statement when determining compression in exr2aces ([karlhendrikse](@karlhendrikse@gmail.com) 2019-11-20)
* [``780c2230``](https://github.com/AcademySoftwareFoundation/openexr/commit/780c22304e2b1419d5d5267183e4dfc39dcd7373) Fix issue 289, C++17 compatibility ([Nick Porcino](@nporcino@pixar.com) 2019-11-20)
* [``d778a0b4``](https://github.com/AcademySoftwareFoundation/openexr/commit/d778a0b47b18fe1ede3824fe013cea9dd7404360) add toFloat.h and eLut.h to SOURCES ([Cary Phillips](@cary@ilm.com) 2019-11-02)
* [``d80927be``](https://github.com/AcademySoftwareFoundation/openexr/commit/d80927bebec8c38147c3614f1d3c7df898beebbf) move dwaLookups.h declarations to the OPENEXR_IMF_INTERNAL_NAMESPACE ([Cary Phillips](@cary@ilm.com) 2019-11-02)
* [``36edde92``](https://github.com/AcademySoftwareFoundation/openexr/commit/36edde927a831686040f6c97a8f080ff13aa4759) add b44ExpLogTable.h and dwaLookups.h as official headers ([Cary Phillips](@cary@ilm.com) 2019-11-02)
* [``00bf05cd``](https://github.com/AcademySoftwareFoundation/openexr/commit/00bf05cd090e4ac4a79877962abf26869c6c1672) add toFloat.h and eLut.h as source files ([Cary Phillips](@cary@ilm.com) 2019-11-01)
* [``861aad16``](https://github.com/AcademySoftwareFoundation/openexr/commit/861aad165e78c4281dae8306f108ede2ae15322c) typo from previous commit: operator= wasn't returning a value ([Cary Phillips](@cary@ilm.com) 2019-11-03)
* [``03b464a0``](https://github.com/AcademySoftwareFoundation/openexr/commit/03b464a01aedbc6607f0f3ca0cb5a61cfac78e12) mvoe TestType and TestTypedAttribute to OPENEXR_IMF_INTERNAL_NAMESPACE ([Cary Phillips](@cary@ilm.com) 2019-11-03)
* [``9a5c8d4f``](https://github.com/AcademySoftwareFoundation/openexr/commit/9a5c8d4f41dd6e972f9a03860b0a42d136609364) remove const from arg declaration in move-constructors/move-assignments ([Cary Phillips](@cary@ilm.com) 2019-11-03)
* [``b7857b96``](https://github.com/AcademySoftwareFoundation/openexr/commit/b7857b96aed4f29ee1605caf1e56e60fa2dd4389) =default copy/move/assign for TypedAttribute ([Cary Phillips](@cary@ilm.com) 2019-11-03)
* [``fa2e4585``](https://github.com/AcademySoftwareFoundation/openexr/commit/fa2e4585ea960d77ec220c5f13de00da7edbdcb6) SonarCloud-inspired fixes ([Cary Phillips](@cary@ilm.com) 2019-10-25)
* [``19cd1014``](https://github.com/AcademySoftwareFoundation/openexr/commit/19cd10142a399fc9ddce863acc3dc46ec2b703b5) SonarCloud-inspired bug fixes: ([Cary Phillips](@cary@ilm.com) 2019-10-24)
* [``64f145a0``](https://github.com/AcademySoftwareFoundation/openexr/commit/64f145a05135aefaac3e9e467be80869ffa276fe) More SonarCloud-inspired fixes: ([Cary Phillips](@cary@ilm.com) 2019-10-24)
* [``5c985fcf``](https://github.com/AcademySoftwareFoundation/openexr/commit/5c985fcf79d38188caae4ccb75b2f77718a44298) SonarCloud-inspired fixes ([Cary Phillips](@cary@ilm.com) 2019-10-24)
* [``8e7ba0fa``](https://github.com/AcademySoftwareFoundation/openexr/commit/8e7ba0fafa53ba91d9aa8382af4652c905d5cea0) sonar fixes ([Cary Phillips](@cary@ilm.com) 2019-10-21)
* [``ba3d5efb``](https://github.com/AcademySoftwareFoundation/openexr/commit/ba3d5efb6bfdadb4d0c489dac89ef1d0aa3996b8) sonar fixes ([Cary Phillips](@cary@ilm.com) 2019-10-21)
* [``031199cd``](https://github.com/AcademySoftwareFoundation/openexr/commit/031199cd4fc062dd7bfe902c6552cf22f6bfbbdb) Fix overzealous removal of if statements breaking all builds except win32 ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``8228578d``](https://github.com/AcademySoftwareFoundation/openexr/commit/8228578da6f86d17b9a2a3f8c6053f8b4ee3fb71) Handle python2 not being installed, but python3 being present ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``d10895ef``](https://github.com/AcademySoftwareFoundation/openexr/commit/d10895ef0ad25dd60e68a2ab00bab7c0592f8c5b) Fix issue with defines not being set correctly for win32 ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``b303f678``](https://github.com/AcademySoftwareFoundation/openexr/commit/b303f6788a434fd61e52c1bacb93a96c4c3440ea) Re-enable Boost_NO_BOOST_CMAKE by default, document, clean up status messages ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``8ec1440c``](https://github.com/AcademySoftwareFoundation/openexr/commit/8ec1440cbd999f17457be605150bc53395fbb334) Set CMP0074 such that people who set Boost_ROOT won't get warnings ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``28d1cb25``](https://github.com/AcademySoftwareFoundation/openexr/commit/28d1cb256f1b46f120adb131e606b2699acc72d7) ensure paths are canonicalized by get_filename_component prior to comparing ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``34ce16c2``](https://github.com/AcademySoftwareFoundation/openexr/commit/34ce16c2653d02fcef6a297a2a61112dbf693922) Fix issue with drive letter under windows ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-06)
* [``650da0d6``](https://github.com/AcademySoftwareFoundation/openexr/commit/650da0d63410d863c4a0aed15a6bee1b46b559cb) Extract to function, protect against infinite loop ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-06)
* [``df768ec8``](https://github.com/AcademySoftwareFoundation/openexr/commit/df768ec8a97adb82947fc4b92a199db9a38c044c) Fixes #593, others - issues with pyilmbase install ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-05)
* [``ed4807b9``](https://github.com/AcademySoftwareFoundation/openexr/commit/ed4807b9e4dc8d94ce79d0b2ed36acc548bee57e) Take DESTDIR into account when creating library symlinks ([Antonio Rojas](@arojas@archlinux.org) 2019-10-19)
* [``f1b017c8``](https://github.com/AcademySoftwareFoundation/openexr/commit/f1b017c8029b529c5c5ed01b6ad1b10a0e48036c) No longer install ImfMisc.h ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``a571bdfe``](https://github.com/AcademySoftwareFoundation/openexr/commit/a571bdfe42866a1f1c579114e2fcae8318172c21) add boost to python module link library ([Jens Lindgren](@lindgren_jens@hotmail.com) 2019-10-22)
* [``cf8b35c9``](https://github.com/AcademySoftwareFoundation/openexr/commit/cf8b35c9bbde9ea78036af2fda04a7c6e9c9a399) Fix overzealous removal of if statements breaking all builds except win32 ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``44266538``](https://github.com/AcademySoftwareFoundation/openexr/commit/442665384d44e464c68381d560f08bea295b9e04) Handle python2 not being installed, but python3 being present ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``1eb2df5a``](https://github.com/AcademySoftwareFoundation/openexr/commit/1eb2df5aa219a819153bb891dc4488875259fb28) Fix issue with defines not being set correctly for win32 ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``8a05994a``](https://github.com/AcademySoftwareFoundation/openexr/commit/8a05994a86fabf39f18890928ee5cef1913fa85a) Re-enable Boost_NO_BOOST_CMAKE by default, document, clean up status messages ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``40e68bb9``](https://github.com/AcademySoftwareFoundation/openexr/commit/40e68bb9f38bf791594ccbaf1320ec520f58180b) Set CMP0074 such that people who set Boost_ROOT won't get warnings ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``b021db40``](https://github.com/AcademySoftwareFoundation/openexr/commit/b021db409cfe52a9f28ad432897552bee735aeee) ensure paths are canonicalized by get_filename_component prior to comparing ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)
* [``7e0714f2``](https://github.com/AcademySoftwareFoundation/openexr/commit/7e0714f279fdb42956235bf4141c59f382b6c3a1) Fix issue with drive letter under windows ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-06)
* [``89dceca8``](https://github.com/AcademySoftwareFoundation/openexr/commit/89dceca80dc28fbabf262e38c9e1acf4863d97f6) Extract to function, protect against infinite loop ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-06)
* [``599e5211``](https://github.com/AcademySoftwareFoundation/openexr/commit/599e52119d01004d6c5252f1070073fbd1518bfa) Fixes #593, others - issues with pyilmbase install ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-05)
* [``0b26a9de``](https://github.com/AcademySoftwareFoundation/openexr/commit/0b26a9dedda4924841323677f1ce0bce37bfbeb4) Fix #595 and others, issue with pkgconfig generation under cmake ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-05)
* [``4e54bde7``](https://github.com/AcademySoftwareFoundation/openexr/commit/4e54bde78f65c0fef8a9f794aaacea07813fba09) Take DESTDIR into account when creating library symlinks ([Antonio Rojas](@arojas@archlinux.org) 2019-10-19)
* [``a2c12ec3``](https://github.com/AcademySoftwareFoundation/openexr/commit/a2c12ec3619de1923de86436c134be458523e5fd) No longer install ImfMisc.h ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``04aa9d33``](https://github.com/AcademySoftwareFoundation/openexr/commit/04aa9d332718748da0afa30dbb66e03b9ea789ab) formatting tweaks ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``29af8e8b``](https://github.com/AcademySoftwareFoundation/openexr/commit/29af8e8b50373d3bb8de38486ac3973f9758575d) formatting tweaks ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``9c876646``](https://github.com/AcademySoftwareFoundation/openexr/commit/9c8766467bb738787dd2bdde527f3391d2da7058) formatting tweaks ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``b79e44c6``](https://github.com/AcademySoftwareFoundation/openexr/commit/b79e44c6e2c41b2e7362f0d7b5517ea1ce4b56e8) formatting tweaks ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``d31c84e3``](https://github.com/AcademySoftwareFoundation/openexr/commit/d31c84e3f2db70dd247578ea2cdbd3d3ae3c4157) formatting tweaks ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``b459acdc``](https://github.com/AcademySoftwareFoundation/openexr/commit/b459acdc485e8f1cc280700157642a607637eb4d) README formatting tweaks ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``c5931e54``](https://github.com/AcademySoftwareFoundation/openexr/commit/c5931e548d354c45a9107f690bc81a9b8400ea76) image tweak ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``c0469c8c``](https://github.com/AcademySoftwareFoundation/openexr/commit/c0469c8c44e59eb33f51db4c1480415b5713fa40) tweak image ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``15d8706b``](https://github.com/AcademySoftwareFoundation/openexr/commit/15d8706bc2692b05a5818de142644dfa3dca26d9) tweak to image in README.md ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``e993b8f4``](https://github.com/AcademySoftwareFoundation/openexr/commit/e993b8f434c8f663a0b095ba819f9f77e9f7e888) tweaks to the mission statement ([Cary Phillips](@cary@ilm.com) 2019-10-31)
* [``df4315a4``](https://github.com/AcademySoftwareFoundation/openexr/commit/df4315a4ecafd2190387cdcc73e3ba9caaec296f) updated mission statement in README.md ([Cary Phillips](@cary@ilm.com) 2019-10-21)
* [``5147f255``](https://github.com/AcademySoftwareFoundation/openexr/commit/5147f255c43049e2ff38dba903b8db4c350a6a35) Added mission statement to README.md ([Cary Phillips](@cary@ilm.com) 2019-10-17)
* [``4527b6f3``](https://github.com/AcademySoftwareFoundation/openexr/commit/4527b6f351bca040a70470b93d67704e5b30d5f3) typo ([Cary Phillips](@cary@ilm.com) 2019-10-24)
* [``ca31d92e``](https://github.com/AcademySoftwareFoundation/openexr/commit/ca31d92e5a8f0dbd19a1cbf428432adc4a67e63e) TSC Meeting notes 2019-10-24 ([Cary Phillips](@cary@ilm.com) 2019-10-24)
* [``4273e84f``](https://github.com/AcademySoftwareFoundation/openexr/commit/4273e84f86fe27392dec53a5cef900caf6727154) Update Azure build to work with new branch. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-26)
* [``e53ebd3e``](https://github.com/AcademySoftwareFoundation/openexr/commit/e53ebd3ef677ab983f83f927f6525efcb5dcb995) Fix int32 overflow bugs with deep images ([Larry Gritz](@lg@larrygritz.com) 2019-10-17)
* [``486ff105``](https://github.com/AcademySoftwareFoundation/openexr/commit/486ff10547d034530c5190bbef6181324b42c209) Prepare 2.4 release branch ([Larry Gritz](@lg@larrygritz.com) 2019-10-24)
* [``c1c82f7d``](https://github.com/AcademySoftwareFoundation/openexr/commit/c1c82f7d2582fb74ad53e5cd1b6525e8dcdfa492) add boost to python module link library ([Jens Lindgren](@lindgren_jens@hotmail.com) 2019-10-22)
* [``a3c246b3``](https://github.com/AcademySoftwareFoundation/openexr/commit/a3c246b335d44fef35a66f6da36444d9f812bbf5) keep all values word-aligned in IlmImfTest testLargeDataWindowOffsets ([Peter Hillman](@peterh@wetafx.co.nz) 2019-10-24)
* [``5db03642``](https://github.com/AcademySoftwareFoundation/openexr/commit/5db0364244d0e27a44dc245f8a0c686d76471e91) fix Contrib/DtexToExr/DtexToExr in .gitignore ([Cary Phillips](@cary@ilm.com) 2019-10-17)
* [``a4b69af6``](https://github.com/AcademySoftwareFoundation/openexr/commit/a4b69af6a714f7a267da90d2cd934c2bb89dc56e) add PyIlmBaseConfigInternal.h and Contrib/DtexToExr to .gitignore ([Cary Phillips](@cary@ilm.com) 2019-10-17)
* [``eadfbf82``](https://github.com/AcademySoftwareFoundation/openexr/commit/eadfbf82875cce8106047c933c0b053809a8ff74) Fix int32 overflow bugs with deep images ([Larry Gritz](@lg@larrygritz.com) 2019-10-17)
* [``eef4c99d``](https://github.com/AcademySoftwareFoundation/openexr/commit/eef4c99d08f7b31a5d392024031a4e4b447df1b5) TSC Meeting notes for 2019-10-17 ([Cary Phillips](@cary@ilm.com) 2019-10-17)
* [``046b2f75``](https://github.com/AcademySoftwareFoundation/openexr/commit/046b2f75700044e6b581cba437e0f86a6f9d625c) TSC Meeting Notes 2019-10-3 ([Cary Phillips](@cary@ilm.com) 2019-10-17)
* [``e65b3890``](https://github.com/AcademySoftwareFoundation/openexr/commit/e65b38903bc4259295f042b4f3f442ba2aca7deb) removed unnecessary .m4 files ([Cary Phillips](@cary@ilm.com) 2019-10-17)
* [``cb162323``](https://github.com/AcademySoftwareFoundation/openexr/commit/cb16232387a8dabf75797ff8d3015594a7a87abe) Fixed various MacOS Azure pipeline issues, all tests run now. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-09)
* [``aef48d68``](https://github.com/AcademySoftwareFoundation/openexr/commit/aef48d6811df5d1ba1a446df0c4d039444d18b32) Fix links ([John Mertic](@jmertic@linuxfoundation.org) 2019-10-14)
* [``7e7e0d47``](https://github.com/AcademySoftwareFoundation/openexr/commit/7e7e0d476e3ab0a38df50c387964ead1f8896433) Explicitly define destructors. Suppress SonarCloud bug reports for array index operators. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-07)
* [``d8fc18e8``](https://github.com/AcademySoftwareFoundation/openexr/commit/d8fc18e8edd6d7db055975f6ad0a02d185c188eb) Removed unreachable return statement. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-07)
* [``a2a133ad``](https://github.com/AcademySoftwareFoundation/openexr/commit/a2a133ad266a1d65ad5410f43f2949a43834a8f6) More NOSONAR suppressions in PyImath. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-07)
* [``2b64316c``](https://github.com/AcademySoftwareFoundation/openexr/commit/2b64316c8272111120b628d1395200b4107c7d64) Change 'a!=a' to std::isnan() to fix Sonar "bug", added infinity checks ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-07)
* [``bf1288de``](https://github.com/AcademySoftwareFoundation/openexr/commit/bf1288def9c09176cdf6658a58934ec018e33d24) Fix static analysis warning re: potential null pointer dereference. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-03)
* [``247dbacd``](https://github.com/AcademySoftwareFoundation/openexr/commit/247dbacd5ddde6766f6362a3109ea721f378fc4a) Edit macro to use only a single instance of '#'. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-03)
* [``e2860cec``](https://github.com/AcademySoftwareFoundation/openexr/commit/e2860cec59853ba5552f4dc39e55b341f362e54e) Remove unreached 'return'. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-03)
* [``6337096e``](https://github.com/AcademySoftwareFoundation/openexr/commit/6337096e825036c2da04a3bca76c506610bfb21b) Remove unnecessary break statements. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-03)
* [``e1ff97f1``](https://github.com/AcademySoftwareFoundation/openexr/commit/e1ff97f15755963b4dd19aba052d4321af0c58f1) TSC meeting notes for the last 3 weeks ([Cary Phillips](@cary@ilm.com) 2019-09-24)
* [``2f4937ba``](https://github.com/AcademySoftwareFoundation/openexr/commit/2f4937baf455feabe1bb0c837c7aa776aaa60bd3) fixed date of last TSC meeting ([Cary Phillips](@cary@ilm.com) 2019-10-02)
* [``9a4a0c56``](https://github.com/AcademySoftwareFoundation/openexr/commit/9a4a0c567be8dd9e8d237ea7e8409041070e0e2b) TSC meeting notes from 9/26/2019 ([Cary Phillips](@cary@ilm.com) 2019-10-02)
* [``4dec0313``](https://github.com/AcademySoftwareFoundation/openexr/commit/4dec0313570f021661302ae776d25edb1950ba97) TSC meeting notes for the last 3 weeks ([Cary Phillips](@cary@ilm.com) 2019-09-24)
* [``f82e1989``](https://github.com/AcademySoftwareFoundation/openexr/commit/f82e1989f462e535e571aca2bf3f78edf9dde28e) Added tests for all exception types derived from BaseExc. ([Cary Phillips](@cary@ilm.com) 2019-09-22)
* [``a82c4c23``](https://github.com/AcademySoftwareFoundation/openexr/commit/a82c4c23d4b3db281db3bba109b3ec272dccb109) operator = (const BaseExc& be) throw () = delete; ([Cary Phillips](@cary@ilm.com) 2019-09-22)
* [``09a14a9e``](https://github.com/AcademySoftwareFoundation/openexr/commit/09a14a9ee3ec9ee2d030e7da3d5b36c01c7cc303) change floating-point loop variables to iterate on a fixed-size array. ([Cary Phillips](@cary@ilm.com) 2019-09-22)
* [``bd7a04f7``](https://github.com/AcademySoftwareFoundation/openexr/commit/bd7a04f7c75e6392595e00895c720524aae82ec3) Change Azure SonarCloud job to run weekly. ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-09-29)
* [``8dd91127``](https://github.com/AcademySoftwareFoundation/openexr/commit/8dd9112733ae15f1c108b64124e6c77a11f3eb83) removed references to the CVE's that are not specific to OpenEXR ([Cary Phillips](@cary@ilm.com) 2019-09-20)
* [``33d1ac61``](https://github.com/AcademySoftwareFoundation/openexr/commit/33d1ac61d46c075171cb37cccc21736ab4cf03d8) CVE listing in SECURITY.md ([Cary Phillips](@cary@ilm.com) 2019-09-19)

## Version 2.4.3 (May 17, 2021)

Patch release that addresses the following security vulnerabilities:

* [CVE-2021-20296](https://nvd.nist.gov/vuln/detail/CVE-2021-20296) Segv on unknown address in Imf_2_5::hufUncompress - Null Pointer dereference ([817](https://github.com/AcademySoftwareFoundation/openexr/pull/817))
* [CVE-2021-3479](https://nvd.nist.gov/vuln/detail/CVE-2021-3479) Out-of-memory in openexr_exrenvmap_fuzzer ([830](https://github.com/AcademySoftwareFoundation/openexr/pull/830))
* [CVE-2021-3478](https://nvd.nist.gov/vuln/detail/CVE-2021-3478) Out-of-memory in openexr_exrcheck_fuzzer ([863](https://github.com/AcademySoftwareFoundation/openexr/pull/863))
* [CVE-2021-3477](https://nvd.nist.gov/vuln/detail/CVE-2021-3477) Heap-buffer-overflow in Imf_2_5::DeepTiledInputFile::readPixelSampleCounts ([861](https://github.com/AcademySoftwareFoundation/openexr/pull/861))
* [CVE-2021-3476](https://nvd.nist.gov/vuln/detail/CVE-2021-3476) Undefined-shift in Imf_2_5::unpack14 ([832](https://github.com/AcademySoftwareFoundation/openexr/pull/832))
* [CVE-2021-3475](https://nvd.nist.gov/vuln/detail/CVE-2021-3475) Integer-overflow in Imf_2_5::calculateNumTiles ([825](https://github.com/AcademySoftwareFoundation/openexr/pull/825))
* [CVE-2021-3474](https://nvd.nist.gov/vuln/detail/CVE-2021-3474) Undefined-shift in Imf_2_5::FastHufDecoder::FastHufDecoder ([818](https://github.com/AcademySoftwareFoundation/openexr/pull/818))

Also:

* [1013](https://github.com/AcademySoftwareFoundation/openexr/pull/1013) Fixed regression in Imath::succf() and Imath::predf() when negative values are given

## Version 2.4.2 (June 15, 2020)

This is a patch release that includes fixes for the following security vulnerabilities:

* [CVE-2020-15305](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15305) Invalid input could cause a heap-use-after-free error in DeepScanLineInputFile::DeepScanLineInputFile() 
* [CVE-2020-15306](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15306) Invalid chunkCount attributes could cause heap buffer overflow in getChunkOffsetTableSize() 
* [CVE-2020-15304](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15304) Invalid tiled input file could cause invalid memory access TiledInputFile::TiledInputFile() 
* OpenEXRConfig.h now correctly sets OPENEXR_PACKAGE_STRING to "OpenEXR" (rather than "IlmBase")

### Merged Pull Requests

* [755](https://github.com/AcademySoftwareFoundation/openexr/pull/755) Fix OPENEXR_PACKAGE_NAME
* [738](https://github.com/AcademySoftwareFoundation/openexr/pull/738) always ignore chunkCount attribute unless it cannot be computed
* [730](https://github.com/AcademySoftwareFoundation/openexr/pull/730) fix #728 - missing 'throw' in deepscanline error handling
* [727](https://github.com/AcademySoftwareFoundation/openexr/pull/727) check null pointer in broken tiled file handling

## Version 2.4.1 (February 11, 2020)

Patch release with minor bug fixes.

### Summary

* Various fixes for memory leaks and invalid memory accesses
* Various fixes for integer overflow with large images.
* Various cmake fixes for build/install of python modules.
* ImfMisc.h is no longer installed, since it's a private header.

### Security Vulnerabilities

This version fixes the following security vulnerabilities:

* [CVE-2020-11765](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11765) There is an off-by-one error in use of the ImfXdr.h read function by DwaCompressor::Classifier::ClasGsifier, leading to an out-of-bounds read.
* [CVE-2020-11764](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11764) There is an out-of-bounds write in copyIntoFrameBuffer in ImfMisc.cpp.
* [CVE-2020-11763](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11763) There is an std::vector out-of-bounds read and write, as demonstrated by ImfTileOffsets.cpp.
* [CVE-2020-11762](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11762) There is an out-of-bounds read and write in DwaCompressor::uncompress in ImfDwaCompressor.cpp when handling the UNKNOWN compression case.
* [CVE-2020-11761](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11761) There is an out-of-bounds read during Huffman uncompression, as demonstrated by FastHufDecoder::refill in ImfFastHuf.cpp.
* [CVE-2020-11760](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11760) There is an out-of-bounds read during RLE uncompression in rleUncompress in ImfRle.cpp.
* [CVE-2020-11759](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11759) Because of integer overflows in CompositeDeepScanLine::Data::handleDeepFrameBuffer and readSampleCountForLineBlock, an attacker can write to an out-of-bounds pointer.
* [CVE-2020-11758](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11758) There is an out-of-bounds read in ImfOptimizedPixelReading.h.


### Merged Pull Requests

* [659](https://github.com/AcademySoftwareFoundation/openexr/pull/659) fix memory leaks and invalid memory accesses
* [609](https://github.com/AcademySoftwareFoundation/openexr/pull/609) Fixes #593, others - issues with pyilmbase install 
* [605](https://github.com/AcademySoftwareFoundation/openexr/pull/605) No longer install ImfMisc.h 
* [603](https://github.com/AcademySoftwareFoundation/openexr/pull/603) Update Azure build to work with new RB-2.4 branch. 
* [596](https://github.com/AcademySoftwareFoundation/openexr/pull/596) Add Boost::Python to Python modules link libraries
* [592](https://github.com/AcademySoftwareFoundation/openexr/pull/592) Take DESTDIR into account when creating library symlinks
* [589](https://github.com/AcademySoftwareFoundation/openexr/pull/589) Fix int32 overflow bugs with deep images 

### Commits \[ git log v2.4.0...v2.4.1\]

* [fix memory leaks and invalid memory accesses](https://github.com/AcademySoftwareFoundation/openexr/commit/e79d2296496a50826a15c667bf92bdc5a05518b4) ([Peter Hillman](@peterh@wetafx.co.nz) 2020-02-08)

* [Fix overzealous removal of if statements breaking all builds except win32](https://github.com/AcademySoftwareFoundation/openexr/commit/031199cd4fc062dd7bfe902c6552cf22f6bfbbdb) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)

* [Handle python2 not being installed, but python3 being present](https://github.com/AcademySoftwareFoundation/openexr/commit/8228578da6f86d17b9a2a3f8c6053f8b4ee3fb71) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)

* [Fix issue with defines not being set correctly for win32](https://github.com/AcademySoftwareFoundation/openexr/commit/d10895ef0ad25dd60e68a2ab00bab7c0592f8c5b) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)

* [Re-enable Boost_NO_BOOST_CMAKE by default, document, clean up status messages](https://github.com/AcademySoftwareFoundation/openexr/commit/b303f6788a434fd61e52c1bacb93a96c4c3440ea) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)

* [Set CMP0074 such that people who set Boost_ROOT won't get warnings](https://github.com/AcademySoftwareFoundation/openexr/commit/8ec1440cbd999f17457be605150bc53395fbb334) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)

* [ensure paths are canonicalized by get_filename_component prior to comparing](https://github.com/AcademySoftwareFoundation/openexr/commit/28d1cb256f1b46f120adb131e606b2699acc72d7) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-07)

* [Fix issue with drive letter under windows](https://github.com/AcademySoftwareFoundation/openexr/commit/34ce16c2653d02fcef6a297a2a61112dbf693922) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-06)

* [Extract to function, protect against infinite loop](https://github.com/AcademySoftwareFoundation/openexr/commit/650da0d63410d863c4a0aed15a6bee1b46b559cb) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-06)

* [Fixes #593, others - issues with pyilmbase install](https://github.com/AcademySoftwareFoundation/openexr/commit/df768ec8a97adb82947fc4b92a199db9a38c044c) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-11-05)

* [Take DESTDIR into account when creating library symlinks](https://github.com/AcademySoftwareFoundation/openexr/commit/ed4807b9e4dc8d94ce79d0b2ed36acc548bee57e) ([Antonio Rojas](@arojas@archlinux.org) 2019-10-19)

* [No longer install ImfMisc.h](https://github.com/AcademySoftwareFoundation/openexr/commit/f1b017c8029b529c5c5ed01b6ad1b10a0e48036c) ([Cary Phillips](@cary@ilm.com) 2019-10-31)

* [add boost to python module link library](https://github.com/AcademySoftwareFoundation/openexr/commit/a571bdfe42866a1f1c579114e2fcae8318172c21) ([Jens Lindgren](@lindgren_jens@hotmail.com) 2019-10-22)

* [Update Azure build to work with new branch.](https://github.com/AcademySoftwareFoundation/openexr/commit/4273e84f86fe27392dec53a5cef900caf6727154) ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-10-26)

* [Fix int32 overflow bugs with deep images](https://github.com/AcademySoftwareFoundation/openexr/commit/e53ebd3ef677ab983f83f927f6525efcb5dcb995) ([Larry Gritz](@lg@larrygritz.com) 2019-10-17)

* [Prepare 2.4 release branch](https://github.com/AcademySoftwareFoundation/openexr/commit/486ff10547d034530c5190bbef6181324b42c209) ([Larry Gritz](@lg@larrygritz.com) 2019-10-24)

## Version 2.4.0 (September 19, 2019)

### Summary

* Completely re-written CMake configuration files
* Improved support for building on Windows, via CMake
* Improved support for building on macOS, via CMake
* All code compiles without warnings on gcc, clang, msvc
* Cleanup of license and copyright notices
* floating-point exception handling is disabled by default
* New Slice::Make method to reliably compute base pointer for a slice.
* Miscellaneous bug fixes

### Security Vulnerabilities

This version fixes the following security vulnerabilities:

* [CVE-2020-16589](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-16589) A head-based buffer overflow exists in Academy Software Foundation OpenEXR 2.3.0 in writeTileData in ImfTiledOutputFile.cpp that can cause a denial of service via a crafted EXR file.
* [CVE-2020-16588](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-16588) A Null Pointer Deference issue exists in Academy Software Foundation OpenEXR 2.3.0 in generatePreview in makePreview.cpp that can cause a denial of service via a crafted EXR file.
* [CVE-2020-16587](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-16587) A heap-based buffer overflow vulnerability exists in Academy Software Foundation OpenEXR 2.3.0 in chunkOffsetReconstruction in ImfMultiPartInputFile.cpp that can cause a denial of service via a crafted EXR file.
* [CVE-2018-18444](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2018-18444) makeMultiView.cpp in exrmultiview in OpenEXR 2.3.0 has an out-of-bounds write, leading to an assertion failure or possibly unspecified other impact.
* [CVE-2018-18443](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2018-18443) OpenEXR 2.3.0 has a memory leak in ThreadPool in IlmBase/IlmThread/IlmThreadPool.cpp, as demonstrated by exrmultiview.

### Closed Issues

* [529](https://github.com/AcademySoftwareFoundation/openexr/issues/529) The OpenEXR_viewer can't be installed successfully due to the Cg support
* [511](https://github.com/AcademySoftwareFoundation/openexr/issues/511) A confused problem in the EXR to JPEG
* [494](https://github.com/AcademySoftwareFoundation/openexr/issues/494) SEGV exrmakepreview in ImfTiledOutputFile.cpp:458
* [493](https://github.com/AcademySoftwareFoundation/openexr/issues/493) SEGV exrmakepreview in makePreview.cpp:132
* [491](https://github.com/AcademySoftwareFoundation/openexr/issues/491) SEGV exrheader in ImfMultiPartInputFile.cpp:579
* [488](https://github.com/AcademySoftwareFoundation/openexr/issues/488) Wiki has outdated info
* [462](https://github.com/AcademySoftwareFoundation/openexr/issues/462) Inconsistent line terminators (CRLF)
* [461](https://github.com/AcademySoftwareFoundation/openexr/issues/461) Wrong LC_RPATH after make install (cmake setup on macos)
* [457](https://github.com/AcademySoftwareFoundation/openexr/issues/457) New CMake setup fails on cmake 3.12
* [455](https://github.com/AcademySoftwareFoundation/openexr/issues/455) Build for mac using cmake to Xcode fails to compile
* [449](https://github.com/AcademySoftwareFoundation/openexr/issues/449) OpenEXR.cpp:36:10: fatal error: 'ImathBox.h' file not found
* [424](https://github.com/AcademySoftwareFoundation/openexr/issues/424) Integrating with OSS-Fuzz
* [421](https://github.com/AcademySoftwareFoundation/openexr/issues/421) How to normalize multi-channel exr image?
* [400](https://github.com/AcademySoftwareFoundation/openexr/issues/400) Create security@openexr.com and info@openexr.com addresses
* [398](https://github.com/AcademySoftwareFoundation/openexr/issues/398) Document CVE's in CHANGES.md release notes file
* [396](https://github.com/AcademySoftwareFoundation/openexr/issues/396) Set up a CREDITS.md file
* [395](https://github.com/AcademySoftwareFoundation/openexr/issues/395) Migrate CLA's from openexr.com to the GitHub repo
* [394](https://github.com/AcademySoftwareFoundation/openexr/issues/394) Properly document the OpenEXR coding style
* [393](https://github.com/AcademySoftwareFoundation/openexr/issues/393) Set up CODEOWNERS file
* [389](https://github.com/AcademySoftwareFoundation/openexr/issues/389) fix -Wall compiler warnings
* [388](https://github.com/AcademySoftwareFoundation/openexr/issues/388) OpenEXR build fails with multiple errors
* [381](https://github.com/AcademySoftwareFoundation/openexr/issues/381) Replace deprecated FindPythonLibs in CMakeLists.txt
* [380](https://github.com/AcademySoftwareFoundation/openexr/issues/380) undefined symbol: _ZTIN7Iex_2_27BaseExcE
* [379](https://github.com/AcademySoftwareFoundation/openexr/issues/379) ZLIB_LIBRARY ZLIB_INCLUDE_DIR being ignored (LNK2019 errors) in OpenEXR\IlmImf\IlmImf.vcxproj
* [377](https://github.com/AcademySoftwareFoundation/openexr/issues/377) 2.3.0: test suite is failing
* [364](https://github.com/AcademySoftwareFoundation/openexr/issues/364) Standalone build of openexr on windows - (with already installed ilmbase)
* [363](https://github.com/AcademySoftwareFoundation/openexr/issues/363) `OpenEXRSettings.cmake` is missing from the release tarball
* [362](https://github.com/AcademySoftwareFoundation/openexr/issues/362) Cmake macro `SET_ILMBASE_INCLUDE_DIRS` assumes
* [360](https://github.com/AcademySoftwareFoundation/openexr/issues/360) Specified Boost.Python not found on Boost versions < 1.67
* [359](https://github.com/AcademySoftwareFoundation/openexr/issues/359) [VS2015] Compile error C2782: 'ssize_t' in PyImathFixedMatrix
* [357](https://github.com/AcademySoftwareFoundation/openexr/issues/357) Move ILMBASE_HAVE_CONTROL_REGISTER_SUPPORT to a private header
* [353](https://github.com/AcademySoftwareFoundation/openexr/issues/353) Add --with-cg-libdir option to support arch dependant Cg library paths
* [352](https://github.com/AcademySoftwareFoundation/openexr/issues/352) buffer-overflow
* [351](https://github.com/AcademySoftwareFoundation/openexr/issues/351) Out of Memory 
* [350](https://github.com/AcademySoftwareFoundation/openexr/issues/350) heap-buffer-overflow
* [348](https://github.com/AcademySoftwareFoundation/openexr/issues/348) Possible compile/install issues in PyIlmBase with multiple jobs
* [343](https://github.com/AcademySoftwareFoundation/openexr/issues/343) CMake issues on Windows
* [342](https://github.com/AcademySoftwareFoundation/openexr/issues/342) IlmImf CMake dependency issue
* [340](https://github.com/AcademySoftwareFoundation/openexr/issues/340) Cannot figure out how to build OpenEXR under mingw64 with v2.3.0
* [333](https://github.com/AcademySoftwareFoundation/openexr/issues/333) openexr 2.3.0 static cmake build broken.
* [302](https://github.com/AcademySoftwareFoundation/openexr/issues/302) Error when linking Half project: unresolved external symbol "private: static union half::uif const * const half::_toFloat" (?_toFloat@half@@0QBTuif@1@B)
* [301](https://github.com/AcademySoftwareFoundation/openexr/issues/301) How to link different IlmBase library names according to Debug/Release configuration, when building OpenEXR with CMake + VS2015?
* [294](https://github.com/AcademySoftwareFoundation/openexr/issues/294) Problem building OpenEXR-2.2.1 in Visual Studio 2015 x64
* [290](https://github.com/AcademySoftwareFoundation/openexr/issues/290) Out Of Memory in Pxr24Compressor (79678745)
* [288](https://github.com/AcademySoftwareFoundation/openexr/issues/288) Out of Memory in B44Compressor (79258415)
* [282](https://github.com/AcademySoftwareFoundation/openexr/issues/282) IlmBase should link pthread
* [281](https://github.com/AcademySoftwareFoundation/openexr/issues/281) Error in installing OpenEXR
* [276](https://github.com/AcademySoftwareFoundation/openexr/issues/276) The savanah.nongnu.org tar.gz hosting
* [274](https://github.com/AcademySoftwareFoundation/openexr/issues/274) Cmake installation of ilmbase places .dll files under `/lib` instead of `/bin`
* [271](https://github.com/AcademySoftwareFoundation/openexr/issues/271) heap-buffer-overflow
* [270](https://github.com/AcademySoftwareFoundation/openexr/issues/270) Out of Memory in TileOffsets (73566621)
* [268](https://github.com/AcademySoftwareFoundation/openexr/issues/268) Invalid Shift at FastHufDecoder (72367575)
* [267](https://github.com/AcademySoftwareFoundation/openexr/issues/267) Cast Overflow at FastHufDecoder (72375479)
* [266](https://github.com/AcademySoftwareFoundation/openexr/issues/266) Divide by Zero at calculateNumTiles (72239767)
* [265](https://github.com/AcademySoftwareFoundation/openexr/issues/265) Signed Integer Overflow in getTiledChunkOffsetTableSize (72377177)
* [264](https://github.com/AcademySoftwareFoundation/openexr/issues/264) Signed Integer Overflow in calculateNumTiles (73181093)
* [263](https://github.com/AcademySoftwareFoundation/openexr/issues/263) Signed Integer Overflow in chunkOffsetReconstruction (72873449, 73090589)
* [262](https://github.com/AcademySoftwareFoundation/openexr/issues/262) Heap Out-of-Bounds write in Imf_2_2::copyIntoFrameBuffer (72940266)
* [261](https://github.com/AcademySoftwareFoundation/openexr/issues/261) Heap Out of Bounds Read in TiledInputFile (72228841)
* [259](https://github.com/AcademySoftwareFoundation/openexr/issues/259) Heap Out of Bounds Access (72839282)
* [257](https://github.com/AcademySoftwareFoundation/openexr/issues/257) Out of Memory / Invalid allocation in lmfArray resizeErase (72828572, 72837441)
* [255](https://github.com/AcademySoftwareFoundation/openexr/issues/255) Process for reporting security bugs
* [254](https://github.com/AcademySoftwareFoundation/openexr/issues/254) [VS 2015] Can't run tests and OpenVDB compile errors
* [253](https://github.com/AcademySoftwareFoundation/openexr/issues/253) C++11-style compile-time type information for `half`.
* [252](https://github.com/AcademySoftwareFoundation/openexr/issues/252) `std::numeric_limits<half>::digits10` value is wrong.
* [250](https://github.com/AcademySoftwareFoundation/openexr/issues/250) SO version change in 2.2.1
* [246](https://github.com/AcademySoftwareFoundation/openexr/issues/246) half.h default user-provided constructor breaks c++ semantics (value/zero initialization vs default initialization)
* [244](https://github.com/AcademySoftwareFoundation/openexr/issues/244) Cannot write to Z channel
* [240](https://github.com/AcademySoftwareFoundation/openexr/issues/240) CpuId' was not declared in this scope
* [239](https://github.com/AcademySoftwareFoundation/openexr/issues/239) pyilmbase error vs2015 with boost1.61 and python27 please help ，alse error
* [238](https://github.com/AcademySoftwareFoundation/openexr/issues/238)  heap-based buffer overflow in exrmaketiled 
* [237](https://github.com/AcademySoftwareFoundation/openexr/issues/237) Can RgbaOutputFile use 32-bit float?
* [234](https://github.com/AcademySoftwareFoundation/openexr/issues/234) How to link compress2, uncompress and compress on 64 bit Windows 7 & Visual Studio 2015 when building openexr?
* [232](https://github.com/AcademySoftwareFoundation/openexr/issues/232) Multiple segmentation faults CVE-2017-9110 to CVE-2017-9116
* [231](https://github.com/AcademySoftwareFoundation/openexr/issues/231) Half.h stops OpenEXR from compiling
* [230](https://github.com/AcademySoftwareFoundation/openexr/issues/230) Imf::OutputFile Produce binary different files
* [226](https://github.com/AcademySoftwareFoundation/openexr/issues/226) IMathExc - multiple definitions on linking.
* [224](https://github.com/AcademySoftwareFoundation/openexr/issues/224) Make PyIlmBase compatible with Python 3.x
* [217](https://github.com/AcademySoftwareFoundation/openexr/issues/217) Issue with optimized build compiled with Intel C/C++ compiler (ICC)
* [213](https://github.com/AcademySoftwareFoundation/openexr/issues/213) AddressSanitizer CHECK failed in ImageMagick fuzz test.  
* [208](https://github.com/AcademySoftwareFoundation/openexr/issues/208) build issues on OSX: ImfDwaCompressorSimd.h:483:no such instruction: `vmovaps (%rsi), %ymm0'
* [205](https://github.com/AcademySoftwareFoundation/openexr/issues/205) Building with VS 2015
* [202](https://github.com/AcademySoftwareFoundation/openexr/issues/202) Documentation error: File Layout "Verson Field" lists wrong bits
* [199](https://github.com/AcademySoftwareFoundation/openexr/issues/199) Unexpected rpaths on macOS
* [194](https://github.com/AcademySoftwareFoundation/openexr/issues/194) RLE Broken for 32-bit formats
* [191](https://github.com/AcademySoftwareFoundation/openexr/issues/191) PyIlmBase Cmake unable to find Boost
* [189](https://github.com/AcademySoftwareFoundation/openexr/issues/189) store to misaligned address / for type 'int64_t', which requires 8 byte alignment
* [188](https://github.com/AcademySoftwareFoundation/openexr/issues/188) iex_debugTrap link error
* [182](https://github.com/AcademySoftwareFoundation/openexr/issues/182) Many C4275 warning compiling on Windows
* [176](https://github.com/AcademySoftwareFoundation/openexr/issues/176) Implement a canonical FindIlmbase.cmake
* [166](https://github.com/AcademySoftwareFoundation/openexr/issues/166) CMake static build of OpenEXR 2.2 fails to link dwaLookups on Linux
* [165](https://github.com/AcademySoftwareFoundation/openexr/issues/165) Clang compilation warnings
* [164](https://github.com/AcademySoftwareFoundation/openexr/issues/164) OpenEXR.pc is not created during "configure" stage.
* [163](https://github.com/AcademySoftwareFoundation/openexr/issues/163) Problems building the OpenEXR-2.2.0
* [160](https://github.com/AcademySoftwareFoundation/openexr/issues/160) Visual Studio 2013 not linking properly with IlmThread
* [158](https://github.com/AcademySoftwareFoundation/openexr/issues/158) Python3 support
* [150](https://github.com/AcademySoftwareFoundation/openexr/issues/150) build issue, debian 7.0 x64
* [139](https://github.com/AcademySoftwareFoundation/openexr/issues/139) configure scripts contain bashisms
* [134](https://github.com/AcademySoftwareFoundation/openexr/issues/134) DWA compressor fails to compile on Win/Mac for some compiler versions
* [132](https://github.com/AcademySoftwareFoundation/openexr/issues/132) Wrong namespaces used in DWA Compressor.
* [125](https://github.com/AcademySoftwareFoundation/openexr/issues/125) cmake: cannot link against static ilmbase libraries
* [123](https://github.com/AcademySoftwareFoundation/openexr/issues/123) cmake: allow building of static and dynamic libs at the same time
* [105](https://github.com/AcademySoftwareFoundation/openexr/issues/105) Building pyilmbase 1.0.0 issues
* [098](https://github.com/AcademySoftwareFoundation/openexr/issues/98) Race condition in creation of LockedTypeMap and registerAttributeTypes 
* [095](https://github.com/AcademySoftwareFoundation/openexr/issues/95) Compile fail with MinGW-w64 on Windows
* [094](https://github.com/AcademySoftwareFoundation/openexr/issues/94) CMake does not generate "toFloat.h" with Ninja
* [092](https://github.com/AcademySoftwareFoundation/openexr/issues/92) MultiPartOutputFile API fails when single part has no type
* [089](https://github.com/AcademySoftwareFoundation/openexr/issues/89) gcc 4.8 compilation issues
* [086](https://github.com/AcademySoftwareFoundation/openexr/issues/86) VS 2010 broken: exporting std::string subclass crashes
* [079](https://github.com/AcademySoftwareFoundation/openexr/issues/79) compile openexr with mingw 64 bit
* [067](https://github.com/AcademySoftwareFoundation/openexr/issues/67) testBox failure on i386
* [050](https://github.com/AcademySoftwareFoundation/openexr/issues/50) Recommended way of opening an EXR file in python?
* [015](https://github.com/AcademySoftwareFoundation/openexr/issues/15) IlmImf Thread should report an 'optimal' number ofthreads to use.

### Merged Pull Requests

* [541](https://github.com/AcademySoftwareFoundation/openexr/pull/541) TSC meeting notes Aug 22, 2019
* [540](https://github.com/AcademySoftwareFoundation/openexr/pull/540) Fix exports when compiling DLLs enabled with mingw
* [539](https://github.com/AcademySoftwareFoundation/openexr/pull/539) Force exception handling / unwind disposition under msvc
* [538](https://github.com/AcademySoftwareFoundation/openexr/pull/538) Add option to control whether pyimath uses the fp exception mechanism
* [537](https://github.com/AcademySoftwareFoundation/openexr/pull/537) Set default value for buildSharedLibs
* [536](https://github.com/AcademySoftwareFoundation/openexr/pull/536) Force the python binding libraries to shared
* [535](https://github.com/AcademySoftwareFoundation/openexr/pull/535) Fix cmake warnings, fix check for numpy
* [534](https://github.com/AcademySoftwareFoundation/openexr/pull/534) Create a "holder" object to fix stale reference to array
* [533](https://github.com/AcademySoftwareFoundation/openexr/pull/533) Disable the debug postfix for the python modules
* [532](https://github.com/AcademySoftwareFoundation/openexr/pull/532) explicitly add the boost includes to the target
* [531](https://github.com/AcademySoftwareFoundation/openexr/pull/531) Update license for DreamWorks Lossy Compression
* [530](https://github.com/AcademySoftwareFoundation/openexr/pull/530) Azure updates for MacOS/Windows/Linux
* [528](https://github.com/AcademySoftwareFoundation/openexr/pull/528) brief notes of TSC meeting 2019-08-16
* [526](https://github.com/AcademySoftwareFoundation/openexr/pull/526) Fix compile warnings from the latest merges
* [525](https://github.com/AcademySoftwareFoundation/openexr/pull/525) Rework boost python search logic to be simpler and more robust
* [524](https://github.com/AcademySoftwareFoundation/openexr/pull/524) Fix #268, issue with right shift in fast huf decoder
* [523](https://github.com/AcademySoftwareFoundation/openexr/pull/523) Address issues with mingw and win32 wide filenames
* [522](https://github.com/AcademySoftwareFoundation/openexr/pull/522) 2.4.0 release notes
* [520](https://github.com/AcademySoftwareFoundation/openexr/pull/520) Add missing symbol export to Slice::Make
* [519](https://github.com/AcademySoftwareFoundation/openexr/pull/519) TSC meeting notes August 8, 2019
* [518](https://github.com/AcademySoftwareFoundation/openexr/pull/518) Makes building of fuzz test optional
* [517](https://github.com/AcademySoftwareFoundation/openexr/pull/517) Added defines for DWAA and DWAB compression.
* [516](https://github.com/AcademySoftwareFoundation/openexr/pull/516) changed AP_CPPFLAGS to AM_CPPFLAGS in PyImathNumpy/Makefile.am.
* [515](https://github.com/AcademySoftwareFoundation/openexr/pull/515) add the files generated by bootstrap/configure to .gitignore.
* [514](https://github.com/AcademySoftwareFoundation/openexr/pull/514) suppress SonarCloud warnings about unhandled exceptions
* [512](https://github.com/AcademySoftwareFoundation/openexr/pull/512) Project documentation edits
* [510](https://github.com/AcademySoftwareFoundation/openexr/pull/510) Added MacOS jobs to Azure pipeline
* [509](https://github.com/AcademySoftwareFoundation/openexr/pull/509) Contrib cleanup
* [503](https://github.com/AcademySoftwareFoundation/openexr/pull/503) TSC meeting notes from 7/25/2019
* [501](https://github.com/AcademySoftwareFoundation/openexr/pull/501) license and copyright fixes
* [500](https://github.com/AcademySoftwareFoundation/openexr/pull/500) Fix another set of warnings that crept in during previous fix merges
* [498](https://github.com/AcademySoftwareFoundation/openexr/pull/498) Fix #491, issue with part number range check reconstructing chunk off…
* [497](https://github.com/AcademySoftwareFoundation/openexr/pull/497) Fix logic for 1 pixel high/wide preview images (Fixes #493)
* [495](https://github.com/AcademySoftwareFoundation/openexr/pull/495) Fix for #494: validate tile coordinates when doing copyPixels
* [490](https://github.com/AcademySoftwareFoundation/openexr/pull/490) Normalize library naming between cmake and autoconf
* [489](https://github.com/AcademySoftwareFoundation/openexr/pull/489) Refresh of README's
* [487](https://github.com/AcademySoftwareFoundation/openexr/pull/487) Azure: updated docker containers, added windows install scripts.
* [486](https://github.com/AcademySoftwareFoundation/openexr/pull/486) Fix #246, add type traits check
* [483](https://github.com/AcademySoftwareFoundation/openexr/pull/483) Large dataWindow Offset test: for discussion
* [482](https://github.com/AcademySoftwareFoundation/openexr/pull/482) Update Azure Linux/SonarCloud jobs to work with new build
* [481](https://github.com/AcademySoftwareFoundation/openexr/pull/481) rewrite of build and installation documentation in INSTALL.md
* [480](https://github.com/AcademySoftwareFoundation/openexr/pull/480) Put all runtime artefacts in a single folder to help win32 find dlls
* [479](https://github.com/AcademySoftwareFoundation/openexr/pull/479) Fix compile warnings
* [478](https://github.com/AcademySoftwareFoundation/openexr/pull/478) Fixes #353, support for overriding Cg libdir
* [477](https://github.com/AcademySoftwareFoundation/openexr/pull/477) Fix #224, imath python code such that tests pass under python3
* [476](https://github.com/AcademySoftwareFoundation/openexr/pull/476) Fix dos files to unix, part of #462
* [475](https://github.com/AcademySoftwareFoundation/openexr/pull/475) Fixes #252, incorrect math computing half digits
* [474](https://github.com/AcademySoftwareFoundation/openexr/pull/474) Fixes #139
* [473](https://github.com/AcademySoftwareFoundation/openexr/pull/473) Fix missing #include <cmath> for std::isnormal
* [472](https://github.com/AcademySoftwareFoundation/openexr/pull/472) Add viewers library to default build
* [471](https://github.com/AcademySoftwareFoundation/openexr/pull/471) Warn the user, but make PyIlmBase not fail a build by default
* [470](https://github.com/AcademySoftwareFoundation/openexr/pull/470) Fix #352, issue with aspect ratio
* [468](https://github.com/AcademySoftwareFoundation/openexr/pull/468) Fix #455 by not using object libraries under apple
* [467](https://github.com/AcademySoftwareFoundation/openexr/pull/467) NumPy lookup logic is only in newer versions of cmake than our minimum
* [466](https://github.com/AcademySoftwareFoundation/openexr/pull/466) Remove last vestiges of old ifdef for windows
* [465](https://github.com/AcademySoftwareFoundation/openexr/pull/465) Fix #461, issue with macos rpath support
* [463](https://github.com/AcademySoftwareFoundation/openexr/pull/463) Fix #457, (unused) policy tag only in 3.13+ of cmake, no longer needed
* [460](https://github.com/AcademySoftwareFoundation/openexr/pull/460) TSC meeting notes 7/18/2019
* [459](https://github.com/AcademySoftwareFoundation/openexr/pull/459) added missing copyright notices
* [458](https://github.com/AcademySoftwareFoundation/openexr/pull/458) fix for failing PyIlmBase/configure because it can't run the IlmBase test program.
* [456](https://github.com/AcademySoftwareFoundation/openexr/pull/456) fix incorrect license identifier
* [450](https://github.com/AcademySoftwareFoundation/openexr/pull/450) change INCLUDES to AM_CPPFLAGS, upon the recommendation of automake warnings
* [448](https://github.com/AcademySoftwareFoundation/openexr/pull/448) Fixes #95, compilation issue with mingw
* [447](https://github.com/AcademySoftwareFoundation/openexr/pull/447) Implements #15, request for hardware concurrency utility function
* [446](https://github.com/AcademySoftwareFoundation/openexr/pull/446) Fixes #282, missing link against pthread
* [444](https://github.com/AcademySoftwareFoundation/openexr/pull/444) added missing files in autoconf setup
* [443](https://github.com/AcademySoftwareFoundation/openexr/pull/443) don't index empty array in testMultiPartSharedAttributes
* [442](https://github.com/AcademySoftwareFoundation/openexr/pull/442) TiledInputFile only supports regular TILEDIMAGE types, not DEEPTILE...
* [441](https://github.com/AcademySoftwareFoundation/openexr/pull/441) TSC meeting notes, July 7, 2019
* [440](https://github.com/AcademySoftwareFoundation/openexr/pull/440) security policy
* [439](https://github.com/AcademySoftwareFoundation/openexr/pull/439) code of conduct
* [438](https://github.com/AcademySoftwareFoundation/openexr/pull/438) Azure and SonarCloud setup
* [437](https://github.com/AcademySoftwareFoundation/openexr/pull/437) address #271: catch scanlines with negative sizes
* [436](https://github.com/AcademySoftwareFoundation/openexr/pull/436) specific check for bad size field in header attributes (related to #248)
* [435](https://github.com/AcademySoftwareFoundation/openexr/pull/435) Refactor cmake
* [434](https://github.com/AcademySoftwareFoundation/openexr/pull/434) Issue #262
* [433](https://github.com/AcademySoftwareFoundation/openexr/pull/433) Fix for #263: prevent overflow in multipart chunk offset reconstruction
* [432](https://github.com/AcademySoftwareFoundation/openexr/pull/432) Fix for #378, bswap on read on big-endian architectures
* [431](https://github.com/AcademySoftwareFoundation/openexr/pull/431) Fixed column labels in OpenEXRFileLayout document
* [429](https://github.com/AcademySoftwareFoundation/openexr/pull/429) change OpaqueAttribute's _typeName field to be std::string
* [428](https://github.com/AcademySoftwareFoundation/openexr/pull/428) Added Coding Style section on Type Casting.
* [427](https://github.com/AcademySoftwareFoundation/openexr/pull/427) adding source .odt files for the .pdf's on the documentation page
* [425](https://github.com/AcademySoftwareFoundation/openexr/pull/425) Handle exceptions, per SonarCloud rules
* [423](https://github.com/AcademySoftwareFoundation/openexr/pull/423) Address #270: limit Tiled images to INT_MAX total number of tiles
* [422](https://github.com/AcademySoftwareFoundation/openexr/pull/422) Add exr2aces to autoconf build script
* [420](https://github.com/AcademySoftwareFoundation/openexr/pull/420) updated references to CVE's in release notes.
* [417](https://github.com/AcademySoftwareFoundation/openexr/pull/417) TSC meeting notes June 27, 2019
* [416](https://github.com/AcademySoftwareFoundation/openexr/pull/416) Fix #342, copy paste bug with dependencies
* [415](https://github.com/AcademySoftwareFoundation/openexr/pull/415) convert_index returns Py_ssize_t
* [414](https://github.com/AcademySoftwareFoundation/openexr/pull/414) Fix part of #232, issue with pointer overflows
* [413](https://github.com/AcademySoftwareFoundation/openexr/pull/413) Fix library suffix issue in cmake file for exr2aces
* [412](https://github.com/AcademySoftwareFoundation/openexr/pull/412) Fix #350 - memory leak on exit
* [411](https://github.com/AcademySoftwareFoundation/openexr/pull/411) Fixes the rpath setting to have the correct variable name
* [410](https://github.com/AcademySoftwareFoundation/openexr/pull/410) Fixed the 2.3.0 release notes to mention that CVE-2017-12596 is fixed.
* [409](https://github.com/AcademySoftwareFoundation/openexr/pull/409) Add initial rules for running clang-format on the code base
* [408](https://github.com/AcademySoftwareFoundation/openexr/pull/408) Add ImfFloatVectorAttribute.h to the automake install
* [406](https://github.com/AcademySoftwareFoundation/openexr/pull/406) New CI with aswfstaging/ci-base image
* [405](https://github.com/AcademySoftwareFoundation/openexr/pull/405) June 20, 2019 TSC meeting notes
* [404](https://github.com/AcademySoftwareFoundation/openexr/pull/404) Miscellaneous documentation improvements
* [403](https://github.com/AcademySoftwareFoundation/openexr/pull/403) Added CLA forms
* [402](https://github.com/AcademySoftwareFoundation/openexr/pull/402) TSC Meeting notes June 13, 2019
* [397](https://github.com/AcademySoftwareFoundation/openexr/pull/397) Updates to README.md, and initial CONTRIBUTING.md, GOVERNANCE.md, INSTALL.md
* [383](https://github.com/AcademySoftwareFoundation/openexr/pull/383) Fixed formatting
* [382](https://github.com/AcademySoftwareFoundation/openexr/pull/382) TSC meeting notes 2019-5-2
* [339](https://github.com/AcademySoftwareFoundation/openexr/pull/339) fix standalone and combined cmake

### Commits \[ git log v2.3.0...v2.4.0\]

* [Add missing include](https://github.com/AcademySoftwareFoundation/openexr/commit/cd1b068ab1d2e2b40cb81c79e997fecfe31dfa11) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Add option to control whether pyimath uses the fp exception mechanism](https://github.com/AcademySoftwareFoundation/openexr/commit/be0df7b76106ba4b33efca289641fdeb59adb3a2) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Update license for DreamWorks Lossy Compression](https://github.com/AcademySoftwareFoundation/openexr/commit/5b64c63cef71f4542ef4e2452077f62755b66252) ([``jbradley``](@jbradley@dreamworks.com) 2019-08-19)

* [Added defines for DWAA and DWAB compression.](https://github.com/AcademySoftwareFoundation/openexr/commit/1b88251b8d955124d7a5da9716ec287ef78440e5) ([Dirk Lemstra](@dirk@lemstra.org) 2019-08-08)

* [TSC meeting notes Aug 22, 2019](https://github.com/AcademySoftwareFoundation/openexr/commit/9307279963b44d31152441bbe771de044329f356) ([Cary Phillips](@cary@ilm.com) 2019-08-26)

* [2.4.0 release notes * Added commit history * Added table of contents Signed-off-by: Cary Phillips <cary@ilm.com>](https://github.com/AcademySoftwareFoundation/openexr/commit/9fe66510bb5c353bb855b6a5bdbb6be8d3762778) ([Cary Phillips](@cary@ilm.com) 2019-08-10)

* [Fix vtable insertion for win32, use new macro everywhere](https://github.com/AcademySoftwareFoundation/openexr/commit/54d46dacb88fbfa41608c7e347cffa5552742bc4) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-25)

* [Use unique id, not typeid reference which may differ](https://github.com/AcademySoftwareFoundation/openexr/commit/728c26ccbd9f0700633c89c94b8328ee78f40cec) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-25)

* [Force vtable into a translation unit](https://github.com/AcademySoftwareFoundation/openexr/commit/7678a9d09c45cc9ae2b9f591f3565d10a503aadd) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-25)

* [Fix exports when compiling DLLs enabled with mingw](https://github.com/AcademySoftwareFoundation/openexr/commit/3674dd27ce45c1f2cc11993957dccee4bdd840dd) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-25)

* [Force exception handling / unwind disposition under msvc](https://github.com/AcademySoftwareFoundation/openexr/commit/b4d5d867a49029e93b4b3aa6708d1fc0093613cc) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-25)

* [Force the python binding libraries to shared](https://github.com/AcademySoftwareFoundation/openexr/commit/39c17b9ceef2ec05b1ebd25a9ee3f15e5fe17181) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Fix cmake warnings, fix check for numpy](https://github.com/AcademySoftwareFoundation/openexr/commit/85bde2ea9afbddffc6ffbfa597f8bb1d25b42859) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Remove unused typedef from previous failed attempt at boost python usage](https://github.com/AcademySoftwareFoundation/openexr/commit/6d5b23a258b562c29012953e13d67012a66322f0) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Create a "holder" object to fix stale reference to array](https://github.com/AcademySoftwareFoundation/openexr/commit/d2a9dec4d37143feb3b9daeb646b9e93632c5d8a) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Disable the debug postfix for the python modules](https://github.com/AcademySoftwareFoundation/openexr/commit/311ebb0485a253445c7324b3d42eaadd01ceb8b4) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [explicitly add the boost includes to the target as Boost::headers does not seem to](https://github.com/AcademySoftwareFoundation/openexr/commit/bdedcc6361da71e7512f978d4017a1fbb25ace92) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Set default value for buildSharedLibs](https://github.com/AcademySoftwareFoundation/openexr/commit/62427d2dc3d3ee147e01e6d0e3b2119f37dfa689) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-24)

* [Azure updates for MacOS/Windows/Linux](https://github.com/AcademySoftwareFoundation/openexr/commit/3a49e9fe3f3d586a57d25265335752380cbe1b31) ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-08-18)

* [brief notes of TSC meeting 2019-08-16](https://github.com/AcademySoftwareFoundation/openexr/commit/36fb144da1110232bf416d5e1c4abde263056d17) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-16)

* [Fix compile warnings from the latest merges](https://github.com/AcademySoftwareFoundation/openexr/commit/181add33e9391372e76abb6bfc654f37d3788e4a) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-14)

* [Fix boost checks when a versioned python is not found](https://github.com/AcademySoftwareFoundation/openexr/commit/d6c176718595415e7b17e7a6c77af0df75cc36de) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-14)

* [Rework boost python search logic to be simpler and more robust](https://github.com/AcademySoftwareFoundation/openexr/commit/c21272230b30562d219d41d00cdcbc98be602c37) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-14)

* [Fix spacing](https://github.com/AcademySoftwareFoundation/openexr/commit/4f8137070fa257557f7b474c41b9b9c260b7f3cd) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-14)

* [Fix #268, issue with right shift in fast huf decoder](https://github.com/AcademySoftwareFoundation/openexr/commit/2f33f0ff08cf66286fda5cf60ee6f995821bde0d) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-14)

* [Add mechanism for test programs to use win32 wide filename fix when manually creating std::fstreams](https://github.com/AcademySoftwareFoundation/openexr/commit/e0ac10e045b6d932c221c9223d88940b14e12b8b) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-12)

* [Use temp directory for tests under win32, properly cleanup files from util tests](https://github.com/AcademySoftwareFoundation/openexr/commit/1d0b240557a230cf704c8797f97ce373a3ca5474) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-12)

* [Fix issue with mingw gcc and wide / utf8 filenames](https://github.com/AcademySoftwareFoundation/openexr/commit/02fbde4e1942e2ffcf652eb99e32fb15530cc93d) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-12)

* [Remove unused using statements](https://github.com/AcademySoftwareFoundation/openexr/commit/ce09ee004050ec2c1c0fff72b28d1d69a98dfaea) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-12)

* [Add missing exports for ImfAcesFile](https://github.com/AcademySoftwareFoundation/openexr/commit/631d5d49bab5ef0194983a0e15471102b5acacd9) ([Nick Porcino](@meshula@hotmail.com) 2019-08-10)

* [Add missing symbol export to Slice::Make](https://github.com/AcademySoftwareFoundation/openexr/commit/efb5d10f6001e165149bf0dc17f96b4671d213c3) ([Nick Porcino](@meshula@hotmail.com) 2019-08-09)

* [TSC meeting notes August 8, 2019](https://github.com/AcademySoftwareFoundation/openexr/commit/ee8830f108e7a930f6326175f444ed026e504f27) ([Cary Phillips](@cary@ilm.com) 2019-08-08) Signed-off-by: Cary Phillips <cary@ilm.com>

* [changed AP_CPPFLAGS to AM_CPPFLAGS in PyImathNumpy/Makefile.am.](https://github.com/AcademySoftwareFoundation/openexr/commit/859017261d4401ebdb965f268d88b10455984719) ([Cary Phillips](@cary@ilm.com) 2019-08-07) What this a typo? The automake-generated Makefiles expect 'AM', which
was leading to a failure to find PyImath.h. Signed-off-by: Cary Phillips <cary@ilm.com>

* [Removed the d_exr Renderman plugin from Contrib. It was hopelessly outdated, not updated since 2003, and no longer of benefit.](https://github.com/AcademySoftwareFoundation/openexr/commit/6999eb39465d99d5fbb01eff9f1acfdb424d9f82) ([Cary Phillips](@cary@ilm.com) 2019-07-27) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Removed the Photoshop plugin from Contrib. It was hopelessly outdated and no longer of benefit.](https://github.com/AcademySoftwareFoundation/openexr/commit/e84040bde6259777035b3032337aee4a24f34548) ([Cary Phillips](@cary@ilm.com) 2019-07-27) Signed-off-by: Cary Phillips <cary@ilm.com>

* [added SPDX license identifier.](https://github.com/AcademySoftwareFoundation/openexr/commit/e9e4f34616460b3a3c179a7bcc2be2e8f4e79ae8) ([Cary Phillips](@cary@ilm.com) 2019-07-27) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Upon the request of the ASWF Governing Board and the advice of Pixar/Lucasfilm attorney Richard Guo, changed the license on the DtexToExr source code to BSD-3-Clause, to bring in line with the standard OpenEXR license. Also, removed COPYING, as it only contained license info; remoted INSTALL because it was only a copy of the boilerplate bootstrap/config documentation; remove NEWS because we're not using that file any more.](https://github.com/AcademySoftwareFoundation/openexr/commit/a73956bfd4809769bcb8fe2229f7d888c7deccff) ([Cary Phillips](@cary@ilm.com) 2019-07-27) Signed-off-by: Cary Phillips <cary@ilm.com>

* [TSC meeting notes from 7/25/2019](https://github.com/AcademySoftwareFoundation/openexr/commit/2ebd7ade2f392fc3da50c0227e3ff11a7a2f4d8e) ([Cary Phillips](@cary@ilm.com) 2019-07-26) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Match variable style syntax per Cary](https://github.com/AcademySoftwareFoundation/openexr/commit/f5ab8176637d8ea1decc83929950aa3864c87141) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-10) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Add headers to build so programs that can parse and display that will do so](https://github.com/AcademySoftwareFoundation/openexr/commit/19557bfaf1b6b38a2407a6a261ee8f3b376c0bd6) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-25) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [First pass of describing versioning and naming of library names](https://github.com/AcademySoftwareFoundation/openexr/commit/eeae20a72f596589b6429ba43bff69281b801015) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-25) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Normalize library naming between cmake and autoconf](https://github.com/AcademySoftwareFoundation/openexr/commit/c3ebd44bdb64c5bfe0065f3d0ac898387a0fbb63) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-25) installed libraries should follow the following basic pattern: (-> indicates a symlink)

    libFoo.so -> libFoo-LIB_SUFFIX.so
    libFoo-LIB_SUFFIX.so -> libFoo-LIB_SUFFIX.so.MAJ_SO_VERSION
    libFoo-LIB_SUFFIX.so.MAJ_SO_VERSION ->
    libFoo-LIB_SUFFIX.so.FULL_SO_VERSION

    so with a concrete example of 2.3 lib w/ so version of 24

    libFoo.so -> libFoo-2_3.so
    libFoo-2_3.so -> libFoo-2_3.so.24
    libFoo-2_3.so.24 -> libFoo-2_3.so.24.0.0
    libFoo-2_3.so.24.0.0.0 <--- actual file

    (there may be slight variations in the link destinations based on
    differences in libtool and cmake, but the file names available should
    all be there) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [only perform check in c++14 to avoid old c++11 standards deficient compilers](https://github.com/AcademySoftwareFoundation/openexr/commit/1aeba79984bef35cead1da540550441f2b8244af) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-25) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix #246, add type traits check](https://github.com/AcademySoftwareFoundation/openexr/commit/5323c345361dcf01d012fd8f40e8c6c975b9cb83) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) previous cleanup did most of the work, but add an explicit test that
half is now trivial and default constructible.  Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [remove sanityCheck for 32 bit overflow. Add test for large offsets](https://github.com/AcademySoftwareFoundation/openexr/commit/b0acdd7bcbd006ff93972cc3c6d66c617280c557) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-23) 

* [Makes building of fuzz test optional](https://github.com/AcademySoftwareFoundation/openexr/commit/73d5676079d77b4241719f57d0219a3287503b8b) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-08-09) This further makes the fuzz test compilation dependent on whether you
want to include the fuzz test in the ctest "make test" rule. This is
mostly for sonar cloud such that it doesn't complain that the fuzz test
code isn't being run as a false positive (because it isn't included in
the test) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Added MacOS jobs to Azure pipeline](https://github.com/AcademySoftwareFoundation/openexr/commit/29eab92cdee9130b7d1cc6adb801966d0bc87c94) ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-07-27) 

* [initial draft of release notes for 2.3.1](https://github.com/AcademySoftwareFoundation/openexr/commit/4fa4251dc1cce417a7832478f6d05421561e2fd2) ([Cary Phillips](@cary@ilm.com) 2019-08-06) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Add //NOSONAR to the "unhandled exception" catches that SonarCloud identifies as vulnerabilities, to suppress the warning. In each of these cases, a comment explains that no action is called for in the catch, so it should not, in fact, be regarded as a bug or vulnerability.](https://github.com/AcademySoftwareFoundation/openexr/commit/c46428acaca50e824403403ebdaec45b97d92bca) ([Cary Phillips](@cary@ilm.com) 2019-07-28) Signed-off-by: Cary Phillips <cary@ilm.com>

* [explicitly name the path for the autoconf-generated files in .gitignore.](https://github.com/AcademySoftwareFoundation/openexr/commit/220cfcdd7e08d28098bf13c992d48df4b0ab191d) ([Cary Phillips](@cary@ilm.com) 2019-08-04) 

* [add the file generated by bootstrap/configure to .gitignore.](https://github.com/AcademySoftwareFoundation/openexr/commit/81af15fd5ea58c33cfa18c60797daaba55126c1b) ([Cary Phillips](@cary@ilm.com) 2019-08-04) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Fixes #353, support for overriding Cg libdir](https://github.com/AcademySoftwareFoundation/openexr/commit/63924fd0f47e428b63c82579e8b03a1eeb4e4ca1) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-22) to handle systems where it isn't lib, but lib64, as needed
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [more documentation tweaks](https://github.com/AcademySoftwareFoundation/openexr/commit/b6c006aafc500816e42909491437bf9af79bb03c) ([Cary Phillips](@cary@ilm.com) 2019-07-28) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Updates to README, CONTRIBUTING, GOVERNANCE: better introduction, removed some of the TSC process descriptions that are redudant in the charter.](https://github.com/AcademySoftwareFoundation/openexr/commit/1cd03756bbf22a65f84eb42c9d83b78be2902c02) ([Cary Phillips](@cary@ilm.com) 2019-07-28) Signed-off-by: Cary Phillips <cary@ilm.com>

* [update to the template copyright notice.](https://github.com/AcademySoftwareFoundation/openexr/commit/21c307aaf054f304f52bb488258f81d68e38385f) ([Cary Phillips](@cary@ilm.com) 2019-07-25) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Updates to LICENSE and CONTRIBUTORS.](https://github.com/AcademySoftwareFoundation/openexr/commit/559186e6c638190ec1db122ec5f1a0890c056a16) ([Cary Phillips](@cary@ilm.com) 2019-07-25) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Fix another set of warnings that crept in during previous fix merges](https://github.com/AcademySoftwareFoundation/openexr/commit/e07ef34af508b7ce9115ebc5454edeaacb35fb8c) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-25) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix logic for 1 pixel high/wide preview images (Fixes #493)](https://github.com/AcademySoftwareFoundation/openexr/commit/74504503cff86e986bac441213c403b0ba28d58f) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-25) 

* [Fix for #494: validate tile coordinates when doing copyPixels](https://github.com/AcademySoftwareFoundation/openexr/commit/6bb36714528a9563dd3b92720c5063a1284b86f8) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-25) 

* [add test for filled channels in DeepScanlines](https://github.com/AcademySoftwareFoundation/openexr/commit/c04673810a86ba050d809da42339aeb7129fc910) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-18) 

* [add test for skipped and filled channels in DeepTiles](https://github.com/AcademySoftwareFoundation/openexr/commit/b1a5c8ca1921a3fc573952c8034fddd8fdac214b) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-18) 

* [slightly rearrange test for filled channels](https://github.com/AcademySoftwareFoundation/openexr/commit/3c9d0b244ec31ab5e5849e1b6020c55096707ab5) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-18) 

* [Make sure to skip over slices that will only be filled when computing the uncompressed pixel size. Otherwise chunks that compressed to larger sizes than the original will fail to load.](https://github.com/AcademySoftwareFoundation/openexr/commit/14905ee6d802b27752890d39880cd05338337e39) ([Halfdan Ingvarsson](@halfdan@sidefx.com) 2013-04-25) 

* [Fix #491, issue with part number range check reconstructing chunk offset table](https://github.com/AcademySoftwareFoundation/openexr/commit/8b5370c688a7362673c3a5256d93695617a4cd9a) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-25) The chunk offset was incorrectly testing for a part number that was the
same size (i.e. an invalid index)
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [removed logo, that didn't work.](https://github.com/AcademySoftwareFoundation/openexr/commit/d5800c14296527b3540da7aefd28b5937158d2cc) ([Cary Phillips](@cary@ilm.com) 2019-07-23) Signed-off-by: Cary Phillips <cary@ilm.com>

* [added logo](https://github.com/AcademySoftwareFoundation/openexr/commit/70435d286a0fe1a022ba26f00a1fd6eb37505a32) ([Cary Phillips](@cary@ilm.com) 2019-07-23) Signed-off-by: Cary Phillips <cary@ilm.com>

* [OpenEXR logo](https://github.com/AcademySoftwareFoundation/openexr/commit/d6eeb1432bc626709f934da7428561d4aeb8c5a5) ([Cary Phillips](@cary@ilm.com) 2019-07-23) Signed-off-by: Cary Phillips <cary@ilm.com>

* [smaller window image](https://github.com/AcademySoftwareFoundation/openexr/commit/fcedcad366988a24fb9c756510488f8fb83dc2ac) ([Cary Phillips](@cary@ilm.com) 2019-07-23) Signed-off-by: Cary Phillips <cary@ilm.com>

* [fixed image references in README.md](https://github.com/AcademySoftwareFoundation/openexr/commit/6def338579442d0fe1e3fbed0d458db3c5cf2a42) ([Cary Phillips](@cary@ilm.com) 2019-07-23) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Revised the overview information in README.md, and condensed the information in the module README.md's, and removed the local AUTHORS, NEWS, ChangeLog files.](https://github.com/AcademySoftwareFoundation/openexr/commit/0c04c734d1a7ba3f3f85577ec56388238c9202c6) ([Cary Phillips](@cary@ilm.com) 2019-07-23) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Azure: updated docker containers, added windows install scripts.](https://github.com/AcademySoftwareFoundation/openexr/commit/941082379a49a1aecafe2b9e84f3403314d910a9) ([Christina Tempelaar-Lietz](@xlietz@gmail.com) 2019-07-22) 

* [rewrite of build and installation documentation in INSTALL.md](https://github.com/AcademySoftwareFoundation/openexr/commit/591b671ba549bccca1e41ad457f569107242565d) ([Cary Phillips](@cary@ilm.com) 2019-07-22) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Convert constructor casts to static_cast](https://github.com/AcademySoftwareFoundation/openexr/commit/625b95fa026c3b78e537e9bb6a39fcd51920ad13) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Convert constructor casts to static_cast, remove dead code](https://github.com/AcademySoftwareFoundation/openexr/commit/5cbf3cb368cd7013a119c3f08555a69fe33a932b) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix issues and warnings compiling in optimized using gcc -Wall](https://github.com/AcademySoftwareFoundation/openexr/commit/6d4e118cebbb7adf8ed29d846bb6f7fb0fb198eb) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Ensure tests have assert when building in a release mode](https://github.com/AcademySoftwareFoundation/openexr/commit/fe93c2c1ade319a7bc9a733cbeaad3c625a31d0d) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) Fixes warnings and makes sure tests are ... testing
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Cleanup warnings for clang -Wall](https://github.com/AcademySoftwareFoundation/openexr/commit/a5fbf7d669ca6b2b402f4fdf9022b43e5eea616f) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [First pass of warning cleanup](https://github.com/AcademySoftwareFoundation/openexr/commit/c1501ec2b29c95501c8fc324f4ec91bd93f0c1d3) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) This fixes g++ -Wall to compile warning free
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Update Azure Linux/SonarCloud jobs to work with new build](https://github.com/AcademySoftwareFoundation/openexr/commit/b19c8d221976bc6c0debc77431b0fe40dfeb8887) ([¨Christina Tempelaar-Lietz¨](@xlietz@gmail.com) 2019-07-21) Signed-off-by: Christina Tempelaar-Lietz <xlietz@gmail.com>

* [Fix dos files to unix, part of #462](https://github.com/AcademySoftwareFoundation/openexr/commit/0f97a86349b377e0f380d2782326844bef652820) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-22) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Put all runtime artefacts in a single folder to help win32 find dlls](https://github.com/AcademySoftwareFoundation/openexr/commit/e2e8b53e267c373971f3e6da700670679a46403d) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-23) This will (hopefully) fix issues with compiling ilmbase as a dll and
using that to generate and compile openexr
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix #224, imath python code such that tests pass under python3](https://github.com/AcademySoftwareFoundation/openexr/commit/ab50d774e91a6448443e6cdb303bd040105cfaf8) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-22) Previously had fixed print functions, this pass converts the following:
- integer division changed in python3 3/2 -> 1.5, have to use 3//2 to
get an int
- xrange is no more, just use range
- integer type coersion for division not working, force type constructor
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fixes #252, incorrect math computing half digits](https://github.com/AcademySoftwareFoundation/openexr/commit/bca0bc002b222d64712b748a733d9c9a0701f834) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-22) Based on float / double math for base 10 digits, with 1 bit of rounding
error, the equation should be floor( mantissa_digits - 1 ) * log10(2) ),
which in the case of half becomes floor( 10 * log10(2) ) or 3
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fixes #139](https://github.com/AcademySoftwareFoundation/openexr/commit/ba329cba788d4f320e6fc455919233222c27a0dd) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) Removes bash-isms from the autoconf bootstrap / configure.ac files
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Add viewers library to default build](https://github.com/AcademySoftwareFoundation/openexr/commit/f52164dcc92c98775c3503aa9827fbd5d1e69b63) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) If libraries can't be found, will warn and not build
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Warn the user, but make PyIlmBase not fail a build by default](https://github.com/AcademySoftwareFoundation/openexr/commit/a0dcd35c51fc7811bc17b766ded17622f91e3fd0) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) By default, many people won't have the dependencies to build PyIlmBase.
Make it such that the build will warn, but continue to build without the
python extension
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix missing #include <cmath> for std::isnormal](https://github.com/AcademySoftwareFoundation/openexr/commit/9aa10cfac3209ac398b12c14eec2611420f20985) ([Axel Waggershauser](@awagger@gmail.com) 2019-07-21) fixes compile regression on macos + clang-6

* [further cleanup and remove old mworks checks that had been copied around](https://github.com/AcademySoftwareFoundation/openexr/commit/351ad1897e3b84bd5b1e29835c7e68bb09f1f914) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Remove last vestiges of old ifdef for windows previously removed elsewhere](https://github.com/AcademySoftwareFoundation/openexr/commit/b3651854491afa8b6c98e9078a5f4a33178c1a66) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) Previously PLATFORM_WINDOWS was used to conditionally include things,
but that had been removed elsewhere, and a few spots missed.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix #352, issue with aspect ratio](https://github.com/AcademySoftwareFoundation/openexr/commit/34e2e78f205c49eafb49b7589701746f748194ad) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) If a file is contructed with an abnormal aspect ratio, tools like make
preview will fail. This adds an extra check to the creation / reading of
ImfHeader to avoid this issue
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix #455 by not using object libraries under apple](https://github.com/AcademySoftwareFoundation/openexr/commit/0451df8f7986ff5ab37c26d2aa6a7aeb115c8948) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) Per the docs, add_library calls with only object library dependencies
are not yet handled properly by Xcode and similar. Disable the use of
object libraries as a compilation speedup mechanism as a result.
Similarly, disable under win32 when building both types of libs to avoid
exported symbols in the static libs. Finally, use same mechanism to
avoid extra layer of libs in generated exports when only building one
config on all platforms
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [NumPy lookup logic is only in newer versions of cmake than our minimum](https://github.com/AcademySoftwareFoundation/openexr/commit/5b4b23d1cf49ee89132251bc7987d65b7a11efe6) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) We are doing the numpy lookup manually for now
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix #461, issue with macos rpath support, remove half-baked framework support](https://github.com/AcademySoftwareFoundation/openexr/commit/9aa52c8c0c96b24c8d645d7850dae77f4bf64620) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Refactor origin function to a Slice factory and Rgba custom utility](https://github.com/AcademySoftwareFoundation/openexr/commit/119eb2d4672e5c77a79929758f7e4c566f47c794) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) Instead of a general templated routine, have a Slice factory function
and then a custom Rgba utility function to clarify and avoid missing
strides, etc. when dealing with slices
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [merges common fixes and move bounds check to central location](https://github.com/AcademySoftwareFoundation/openexr/commit/6a41400b47d574a5fc6133b9a7139bcd7b59d585) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-01) PR #401 had conflicts, and some of the checks were not in a central
location. This incorporates those changes, moving the extra range checks
to the central sanityCheck already in ImfHeader. Then adds a new utility
function for computing the pointer offsets that can prevent simple
overflow when there are large offsets from origin or widths with
subsampling.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>
Co-Authored-By: pgajdos <pgajdos@suse.cz>

* [Fix part of #232, issue with pointer overflows](https://github.com/AcademySoftwareFoundation/openexr/commit/4aa6a4e0fcd52b220c71807307b9139966c3644c) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-27) This addresses pointer overflow in exr2aces with large datawindow
offsets. It also fixes similar issues in exrenvmap and exrmakepreview.
This addresses the crashes in CVE-2017-9111, CVE-2017-9113,
CVE-2017-9115
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix portion of #462](https://github.com/AcademySoftwareFoundation/openexr/commit/2309b42be084939e8593e036b814049f98eb7888) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-21) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix copyright notice, clarify version requirement comment](https://github.com/AcademySoftwareFoundation/openexr/commit/688b50d1982854b1a2be63160eae03472cf4820e) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-20) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix copyright notice, clarify version requirement comment](https://github.com/AcademySoftwareFoundation/openexr/commit/bbf1f5ed9814f35f953c5b28349ca8dd59a3ed87) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-20) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix #457, (unused) policy tag only in 3.13+ of cmake, no longer needed](https://github.com/AcademySoftwareFoundation/openexr/commit/e69dc2131791a42d5e0618506a4846ec7d53b997) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-20) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [TSC meeting notes 7/18/2019](https://github.com/AcademySoftwareFoundation/openexr/commit/04e21585d01c36790dad186a34c4c64c8e0a1dae) ([Cary Phillips](@cary@ilm.com) 2019-07-18) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Typo in Makefile.am, AM_CPPFLAGS should append to the previous value.](https://github.com/AcademySoftwareFoundation/openexr/commit/97626390f86007fcff2d33c68919389e211983e1) ([Cary Phillips](@cary@ilm.com) 2019-07-18) Signed-off-by: Cary Phillips <cary@ilm.com>

* [changed INCLUDE to AM_CPPFLAGS, upon the recommendation of automake warnings.](https://github.com/AcademySoftwareFoundation/openexr/commit/f91edef414e319235959a537e0ef62c49dddcde3) ([Cary Phillips](@cary@ilm.com) 2019-07-17) Signed-off-by: Cary Phillips <cary@ilm.com>

* [added missing copyright notices](https://github.com/AcademySoftwareFoundation/openexr/commit/76cb1ef869a23ab49f4313fee16a4d5750e91485) ([Cary Phillips](@cary@ilm.com) 2019-07-18) Signed-off-by: Cary Phillips <cary@ilm.com>

* [in PyIlmBase/configure.ac, set LD_LIBRARY_PATH explicitly for the ilmbase test program,so that it finds the libraries when it executes.](https://github.com/AcademySoftwareFoundation/openexr/commit/0bd322d424781f20750141ddc829fc9e16f7e305) ([Cary Phillips](@cary@ilm.com) 2019-07-18) Signed-off-by: Cary Phillips <cary@ilm.com>

* [remove the reference to the LICENSE file in the copyright notice template.](https://github.com/AcademySoftwareFoundation/openexr/commit/1aedb3ceec973e9bc0bad88fc151b2504884e84c) ([Cary Phillips](@cary@ilm.com) 2019-07-18) Signed-off-by: Cary Phillips <cary@ilm.com>

* [fix incorrect license identifier](https://github.com/AcademySoftwareFoundation/openexr/commit/02f1e3d876a784cfd0ab8d0581bafe1fd0d98df2) ([Cary Phillips](@cary@ilm.com) 2019-07-18) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Rename new function and clarify purpose](https://github.com/AcademySoftwareFoundation/openexr/commit/e8dc4326383540ef4a4e2a388cb176da72c120fb) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-18) After discussion with phillman, renamed to give this routine a purpose
beyond some soon to be deleted legacy support, and clarified this in the
comment documenting the function.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Implements #15, request for hardware concurrency utility function](https://github.com/AcademySoftwareFoundation/openexr/commit/23eaf0f45ff531ba0ab3fb1540d5c7d31b4bfe94) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-18) new static member of ThreadPool, call as
ThreadPool::hardwareConcurrency, so no abi breakage or api change
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [use headers.data() instead of &headers[0]](https://github.com/AcademySoftwareFoundation/openexr/commit/42665b55f4062f1492156c7bc9482318c7b49cda) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-17) 

* [don't index empty array in testMultiPartSharedAttributes](https://github.com/AcademySoftwareFoundation/openexr/commit/bb5aad9b793b1113cae42d80fea8925503607de1) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-16) 

* [Added IlmThreadSemaphoreOSX to IlmBase/IlmThread/Makefile.am and added PyIlmBase/PyIlmBase.pc.in back in, looks like it got inadvertently removed by a previous commit.](https://github.com/AcademySoftwareFoundation/openexr/commit/c580d3531c36ed1de35fbfe359eed5f74c2de6dc) ([Cary Phillips](@cary@ilm.com) 2019-07-16) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Azure and SonarCloud setup](https://github.com/AcademySoftwareFoundation/openexr/commit/9d053e4871e721144ad25ac04437646cf4f16d66) ([¨Christina Tempelaar-Lietz¨](@xlietz@gmail.com) 2019-07-12) Signed-off-by: ¨Christina Tempelaar-Lietz¨ <xlietz@gmail.com>

* [Fixes #95, compilation issue with mingw](https://github.com/AcademySoftwareFoundation/openexr/commit/2cf0560dd8eb469680d2281e6d80348dad9ad500) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-18) The tree now compiles using mingw to compile, tested by cross compiling
for windows from linux
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fixes #282, missing link against pthread](https://github.com/AcademySoftwareFoundation/openexr/commit/e90f1b0ed19cb05821c7351ce8d5d9a22fb094eb) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-18) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Remove removed file, add CMakeLists.txt file](https://github.com/AcademySoftwareFoundation/openexr/commit/9683c48479ed2372d26eb51ed91d89b01c495dfd) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-18) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [PyIlmBase finished refactor, misc cleanup](https://github.com/AcademySoftwareFoundation/openexr/commit/4d97270c6ce0916483c1aff5b1f77846cfff11a0) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-18) - add extra dist to automake for make dist
- finish numpy lookup
- add sample vfx 15 toolchain file for doc purposes
- merge cxx standard, pay attention to global setting if set
- merge clang tidy option
- add default build type if not set
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Remove un-needed files now that cmake can provide correct values](https://github.com/AcademySoftwareFoundation/openexr/commit/08332041bb46b45e93855c9843a2aa916ec4ebef) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-18) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix issues with rpath, message cleanup, checkpoint better python layer](https://github.com/AcademySoftwareFoundation/openexr/commit/0eff97241f495027021b54978028475f0b2459dd) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-17) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Start to integrate python module using FindPython and FindBoost from modern cmake](https://github.com/AcademySoftwareFoundation/openexr/commit/c236ed81b7146947999b75fd93aedc5d54d78f64) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-16) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Attempt to set rpath for more common scenarios when people are building custom versions](https://github.com/AcademySoftwareFoundation/openexr/commit/10adf360120898c6ad3a0be2838056948bf22233) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-16) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Documentation pass](https://github.com/AcademySoftwareFoundation/openexr/commit/ba22a8e0a366c87677c53bab72af72dbc378b0dd) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-16) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Enable custom install subfolder for headers](https://github.com/AcademySoftwareFoundation/openexr/commit/9067b792c6f178bd2ff1d15e7b4d898fc1677495) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-13) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Generate version file to ensure proper version check](https://github.com/AcademySoftwareFoundation/openexr/commit/edb6938738462009990086fb7081a860412ec0d4) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-13) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Properly include additional cmake files in "make dist" under autoconf](https://github.com/AcademySoftwareFoundation/openexr/commit/ae54f3d656f8c6336c22385ee5d5ab1f35324c37) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-13) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [First pass updating the documentation for cmake builds](https://github.com/AcademySoftwareFoundation/openexr/commit/120b93ecf33c45284dff68eaf0ee779fa1cb6747) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-12) 

* [Switch testing control to use standard ctest setting option](https://github.com/AcademySoftwareFoundation/openexr/commit/fe6bf4c585723ff8851dfe965343a2adb0f1c1f4) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-12) 

* [First pass making cross compile work, cross compiling windows using mingw on linux](https://github.com/AcademySoftwareFoundation/openexr/commit/f44721e0c504b0b400a71513600295fc5e00f014) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-12) This currently works for building using static libraries, but not yet
tested with dlls.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix new (WIP) cmake setup to work on OS/X](https://github.com/AcademySoftwareFoundation/openexr/commit/2fe5a26d7ef36276ba4aa354178b81fc6612868d) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-12) This includes a fix for the semaphore configure check as well as a
couple of compile warnings
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Add missing file, remove unused exrbuild tool cmake](https://github.com/AcademySoftwareFoundation/openexr/commit/9a1ca7579b1ac793ae2d7bbee667e498d9bc8322) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-12) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Refactor cmake](https://github.com/AcademySoftwareFoundation/openexr/commit/df41027db50bd52a0b797444f02d5907b756652e) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-07-12) This refactors the cmake setup, modernizing it to a current flavor of
cmake and cleaning up the definitions. This also makes the top level
folder a "super project", meaning it is including what should be
distinct / standalone sub-projects with their own finds that should
work.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [TiledInputFile only supports regular TILEDIMAGE types, not DEEPTILE or unknown tiled types. Enforce for both InputFile and InputPart API. Fixes #266, Related to #70](https://github.com/AcademySoftwareFoundation/openexr/commit/ece555214a63aaf0917ad9df26be7e17451fefb9) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-15) 

* [address #271: catch scanlines with negative sizes](https://github.com/AcademySoftwareFoundation/openexr/commit/849c616e0c96665559341451a08fe730534d3cec) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-12) 

* [TSC meeting notes, July 7, 2019](https://github.com/AcademySoftwareFoundation/openexr/commit/960a56f58da13be6c97c59eae1f57bd8882c4588) ([Cary Phillips](@cary@ilm.com) 2019-07-12) Signed-off-by: Cary Phillips <cary@ilm.com>

* [securty policy](https://github.com/AcademySoftwareFoundation/openexr/commit/8f483c2552070f3d9dd2df98f6500dfa1c051dcc) ([Cary Phillips](@cary@ilm.com) 2019-07-12) Signed-off-by: Cary Phillips <cary@ilm.com>

* [code of conduct](https://github.com/AcademySoftwareFoundation/openexr/commit/f31407518aa361263c77eae13f1eef46999ca01f) ([Cary Phillips](@cary@ilm.com) 2019-07-12) Signed-off-by: Cary Phillips <cary@ilm.com>

* [bswap_32 to correct endianness on read, to address #81.](https://github.com/AcademySoftwareFoundation/openexr/commit/225ddb8777e75978b88c2d6311bb0cf94c0b6f22) ([Cary Phillips](@cary@ilm.com) 2019-07-02) Signed-off-by: Cary Phillips <cary@ilm.com>

* [fix reading files](https://github.com/AcademySoftwareFoundation/openexr/commit/5350d10ffc03c774e5cd574062297fc91001064d) ([Dan Horák](@dan@danny.cz) 2019-04-15) testFutureProofing and testMultiPartFileMixingBasic both use fread(&length,4,f) to get a 4 byte
integer value from input file. The value read is not converted from the little endian format to
the machine format causing problems (eg. test didn't finish after 24 hours).
fixes issue #81

* [SonarCloud considers strcpy() a vulernability. It was used only in OpaqueAttribute, whose type name was stored as Array<char>.  I changed the type to std::string. I suspect this simply dates to a time before std::string was commonly used.](https://github.com/AcademySoftwareFoundation/openexr/commit/29d18b70bf542ef9ec6e8861c015d2e7b3d3ec58) ([Cary Phillips](@cary@ilm.com) 2019-07-09) Also, it appears that nothing in the test suite validated opaque attributes, which hold values read from a file when the attribute type is not known. I added a test to validate the behavior, which also validates that the typeName() works when implemented with std::string instead of Array<char>.
Signed-off-by: Cary Phillips <cary@ilm.com>

* [Updated pdf with fixes for file version bits on page 7.](https://github.com/AcademySoftwareFoundation/openexr/commit/8da36708caaf0591f72538bfa414d8af20af90e9) ([Cary Phillips](@cary@ilm.com) 2019-07-11) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Fixed column labels in table on page 7; bit 11 is "deep", bit 12 is "multi-part". Bit 9 is long names, and is not in the table.](https://github.com/AcademySoftwareFoundation/openexr/commit/a3198419f7593564747337e763083492c0470f45) ([Cary Phillips](@cary@ilm.com) 2019-07-09) Signed-off-by: Cary Phillips <cary@ilm.com>

* [New CI with aswfstaging/ci-base image](https://github.com/AcademySoftwareFoundation/openexr/commit/5e7cde5c082881009516aa57a711a19e3eb92f64) ([aloysb](@aloysb@al.com.au) 2019-06-17) Signed-off-by: Aloys Baillet <aloys.baillet@gmail.com>
Conflicts:
	azure-pipelines.yml

* [use static_cast in error test](https://github.com/AcademySoftwareFoundation/openexr/commit/700e4996ce619743d5bebe07b4158ccc4547e9ad) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-11) 

* [throw better exceptions in multipart chunk reconstruction](https://github.com/AcademySoftwareFoundation/openexr/commit/001a852cca078c23d98c6a550c65268cc160042a) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-11) 

* [Fix for #263: prevent overflow in multipart chunk offset table reconstruction](https://github.com/AcademySoftwareFoundation/openexr/commit/6e4b6ac0b5223f6e813e025532b3f0fc4e02f541) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-09) 

* [protect against negative sized tiles](https://github.com/AcademySoftwareFoundation/openexr/commit/395aa4cbcaf91ce37aeb5e9876c44291bed4d1f9) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-11) 

* [apply suggested for for #262](https://github.com/AcademySoftwareFoundation/openexr/commit/9e9e4616f60891a8b27ee9cdeac930e5686dca4f) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-10) 

* [specific check for bad size field in header attributes (related to #248)](https://github.com/AcademySoftwareFoundation/openexr/commit/4c146c50e952655bc193567224c2a081c7da5e98) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-12) 

* [use static_cast and numeric_limits as suggested](https://github.com/AcademySoftwareFoundation/openexr/commit/eda733c5880e226873116ba66ce9069dbc844bdd) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-09) 

* [Address #270: limit to INT_MAX tiles total](https://github.com/AcademySoftwareFoundation/openexr/commit/7f438ffac4f6feb46383f66cb7e83ab41074943d) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-05) 

* [exr2aces wasn't built via the configure script](https://github.com/AcademySoftwareFoundation/openexr/commit/1959f74ee7f47948038a1ecb16c8ba8b84d4eb89) ([Peter Hillman](@peterh@wetafx.co.nz) 2019-07-05) 

* [added links for CVE's](https://github.com/AcademySoftwareFoundation/openexr/commit/afd9beac8b7e114def78793b6810cbad8764a477) ([Cary Phillips](@cary@ilm.com) 2019-07-02) Signed-off-by: Cary Phillips <cary@ilm.com>

* [added "Test Policy" section to CONTRIBUTING.](https://github.com/AcademySoftwareFoundation/openexr/commit/695019e4b98b55ed583d1455a9219e55fc777d1a) ([Cary Phillips](@cary@ilm.com) 2019-07-02) Signed-off-by: Cary Phillips <cary@ilm.com>

* [updated references to CVE's in release notes.](https://github.com/AcademySoftwareFoundation/openexr/commit/2a0226b4c99c057ab7f3b038dafd92543ade3e6f) ([Cary Phillips](@cary@ilm.com) 2019-07-02) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Fixed the 2.3.0 release notes to mention that CVE-2017-12596 is fixed.](https://github.com/AcademySoftwareFoundation/openexr/commit/9da28302194b413b57da757ab69eb33373407f51) ([Cary Phillips](@cary@ilm.com) 2019-06-26) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Added Coding Style section on Type Casting.](https://github.com/AcademySoftwareFoundation/openexr/commit/7790ad78bb4e2b6f4bf22a7c1703af1e352004a4) ([Cary Phillips](@cary@ilm.com) 2019-07-08) Signed-off-by: Cary Phillips <cary@ilm.com>

* [adding source .odt files for the .pdf's on the documention page on openexr.com](https://github.com/AcademySoftwareFoundation/openexr/commit/2f7847e3faf7146f2be8c1c0c3053c50b7ee9d97) ([Cary Phillips](@cary@ilm.com) 2019-07-03) Signed-off-by: Cary Phillips <cary@ilm.com>

* [fix readme typo](https://github.com/AcademySoftwareFoundation/openexr/commit/67c1d4d2fc62f1bbc94202e49e65bd92de2e580f) ([Nick Porcino](@meshula@hotmail.com) 2019-07-08) 

* [Handle exceptions, per SonarCloud rules; all catch blocks must do something to indicate the exception isn't ignored.](https://github.com/AcademySoftwareFoundation/openexr/commit/fbce9002eff631b3feeeb18d45419c1fba4204ea) ([Cary Phillips](@cary@ilm.com) 2019-07-07) Signed-off-by: Cary Phillips <cary@ilm.com>

* [TSC meeting notes June 27, 2019](https://github.com/AcademySoftwareFoundation/openexr/commit/4093d0fbb16ad687779ec6cc7b44308596d5579f) ([Cary Phillips](@cary@ilm.com) 2019-06-28) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Implement semaphore for osx](https://github.com/AcademySoftwareFoundation/openexr/commit/fbb912c3c8b13a9581ffde445e390c1603bae35d) ([oleksii.vorobiov](@oleksii.vorobiov@globallogic.com) 2018-11-01) 

* [Various fixes to address compiler warnings: - removed unused variables and functions - added default cases to switch statements - member initialization order in class constructors - lots of signed/unsigned comparisons fixed either by changing a loop iterator from int to size_t, or by selective type casting.](https://github.com/AcademySoftwareFoundation/openexr/commit/c8a7f6a5ebce9a6d5bd9a3320bc746221789f407) ([Cary Phillips](@cary@ilm.com) 2019-06-24) Signed-off-by: Cary Phillips <cary@ilm.com>

* [convert_index returns Py_ssize_t](https://github.com/AcademySoftwareFoundation/openexr/commit/ce886b87336ba04a12eb631ecfcc71da0c9b74bf) ([Cary Phillips](@cary@ilm.com) 2019-06-27) Signed-off-by: Cary Phillips <cary@ilm.com>

* [Fix #342, copy paste bug with dependencies](https://github.com/AcademySoftwareFoundation/openexr/commit/2b28d90bc5e329c989dc44c1d5fdcdf715d225d7) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-28) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fixes the rpath setting to have the correct variable name](https://github.com/AcademySoftwareFoundation/openexr/commit/5093aaa05278030d07304588fa52466538794fe7) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-27) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Add ImfFloatVectorAttribute.h to the automake install](https://github.com/AcademySoftwareFoundation/openexr/commit/d61c0967cb7cd8fa255de64e4e79894d59c0f82d) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-26) The CMake file was previously updated to include this file on install,
but was missing from the automake side.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix #350 - memory leak on exit](https://github.com/AcademySoftwareFoundation/openexr/commit/adbc1900cb9d25fcc4df008d4008b781cf2fa4f8) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-27) This fixes CVE-2018-18443, the last thread pool provider set into the
pool was not being correctly cleaned up at shutdown of the thread pool.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Fix library suffix issue in cmake file for exr2aces](https://github.com/AcademySoftwareFoundation/openexr/commit/e4099a673e3348d4836c79a760e07b28b1912083) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-27) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Iterate on formatting, add script to run the formatting](https://github.com/AcademySoftwareFoundation/openexr/commit/969305c5731aef054e170e776086e3747eb20ee0) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-27) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [Add initial rules for running clang-format on the code base](https://github.com/AcademySoftwareFoundation/openexr/commit/6513fcf2e25ebd92c8f80f18e8cd7718ba7c4a41) ([Kimball Thurston](@kdt3rd@gmail.com) 2019-06-27) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [find Boost.Python 3 on older Boost versions](https://github.com/AcademySoftwareFoundation/openexr/commit/9b58cf0fc197947dc5798854de639233bb35c6cb) ([Jens Lindgren](@lindgren_jens@hotmail.com) 2018-11-19) 

* [MSYS support](https://github.com/AcademySoftwareFoundation/openexr/commit/a19c806a7b52cdf74bfa6966b720efd8b24a2590) ([Harry Mallon](@hjmallon@gmail.com) 2019-01-30) 

* [Only find_package ZLIB when required](https://github.com/AcademySoftwareFoundation/openexr/commit/ab357b0a7a6d7e0ee761bf8ee5846688626d9236) ([Harry Mallon](@hjmallon@gmail.com) 2019-02-06) 

* [Remove unused headers](https://github.com/AcademySoftwareFoundation/openexr/commit/db9fcdc9c448a9f0d0da78010492398a394c87e7) ([Grant Kim](@6302240+enpinion@users.noreply.github.com) 2019-06-13) 

* [WIN32 to _WIN32 for Compiler portability](https://github.com/AcademySoftwareFoundation/openexr/commit/6e2a73ed8721da899a5bd844397444d5b15a5c71) ([Grant Kim](@6302240+enpinion@users.noreply.github.com) 2019-06-11) https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=vs-2019
_WIN32 is the standard according to the official documentation from Microsoft and also this fixes MinGW compile error.

* [Update README.md](https://github.com/AcademySoftwareFoundation/openexr/commit/45e9910be6009ac4ddf4db51c3c505daafc942a3) ([Huibean Luo](@huibean.luo@gmail.com) 2019-04-08) 

* [Added a few people to CREDITS.](https://github.com/AcademySoftwareFoundation/openexr/commit/db512f5de8f4cc0f6ff81a67bf1bb7e8e7f0cc53) ([Cary Phillips](@cary@ilm.com) 2019-06-20) Signed-off-by: Cary Phillips <cary@ilm.com>

* [added release note summary information for all old releases from the "Announcements" section of openexr.com to CHANGES.md, so the repo's release notes are complete.](https://github.com/AcademySoftwareFoundation/openexr/commit/61bbd0df59494cc2fa0e508506f32526acf2bf51) ([Cary Phillips](@cary@ilm.com) 2019-06-20) Signed-off-by: Cary Phillips <cary@ilm.com>

* [first real draft of coding style, and steps in the release process.](https://github.com/AcademySoftwareFoundation/openexr/commit/1d514e66313cac0440b80c290b35cfa6b8f89b51) ([Cary Phillips](@cary@ilm.com) 2019-06-20) Signed-off-by: Cary Phillips <cary@ilm.com>

* [- added CREDITS.md (generated from "git log") - added CODEOWNERS (mostly a placeholder, everything is currently owned by TSC members) - the Release Process section of CONTRIBUTING gives the git log arguments to generate release notes. - remove stray meeting minutes file at the root level.](https://github.com/AcademySoftwareFoundation/openexr/commit/050048c72ef4c32119d21cdb499e23418429f529) ([Cary Phillips](@cary@ilm.com) 2019-06-19) Signed-off-by: Cary Phillips <cary@ilm.com>

* [fixed references to renamed ASWF folder](https://github.com/AcademySoftwareFoundation/openexr/commit/bd4c36cf07db310bb8350a4e5f575d86f1c7f8cb) ([Cary Phillips](@cary@ilm.com) 2019-06-19) Signed-off-by: Cary Phillips <cary@ilm.com>

* [June 20, 2019 TSC meeting notes](https://github.com/AcademySoftwareFoundation/openexr/commit/82134840a001c2692ee762b0a767ab1b43cb64db) ([Cary Phillips](@cary@ilm.com) 2019-06-20) Signed-off-by: Cary Phillips <cary@ilm.com>

* [CLA's Renamed aswf-tsc to ASWF](https://github.com/AcademySoftwareFoundation/openexr/commit/7ebb766d7540ae9a2caea80b9f1c9799d7c8d8af) ([Cary Phillips](@cary@ilm.com) 2019-06-15) Signed-off-by: Cary Phillips <cary@ilm.com>

* [2019-06-13.md](https://github.com/AcademySoftwareFoundation/openexr/commit/9b2719c68635879421805ed3b602ea19aae68a77) ([seabeepea](@seabeepea@gmail.com) 2019-06-14) Signed-off-by: seabeepea <seabeepea@gmail.com>

* [Missed John on the attendee list.](https://github.com/AcademySoftwareFoundation/openexr/commit/0035649cc6d7f4d86be8609758b927b01b8c110c) ([Cary Phillips](@cary@ilm.com) 2019-06-13) Signed-off-by: Cary Phillips <cary@ilm.com>

* [TSC Meeting notes June 13, 2019](https://github.com/AcademySoftwareFoundation/openexr/commit/79857214aec3d81f73f2e9613a4b44caa21751c8) ([Cary Phillips](@cary@ilm.com) 2019-06-13) Signed-off-by: Cary Phillips <cary@ilm.com>

* [- Formatting section is TBD - fixed references to license - removed references to CI - added section on GitHub labels](https://github.com/AcademySoftwareFoundation/openexr/commit/0045a12d20112b253895d88b4e2bce3ffcff0d90) ([Cary Phillips](@cary@ilm.com) 2019-06-14) Signed-off-by: Cary Phillips <cary@ilm.com>

* [fixing minor typos](https://github.com/AcademySoftwareFoundation/openexr/commit/f62e9c0f9903e03c1d0d80e68e29ffba573c7f8d) ([xlietz](@31363633+xlietz@users.noreply.github.com) 2019-06-12) 

* [Edits to README.md and CONTRIBUTING.md](https://github.com/AcademySoftwareFoundation/openexr/commit/55a674bde7ee63c1badacbe061d3cb222927c68e) ([Cary Phillips](@cary@ilm.com) 2019-06-11) 

* [Add initial Azure pipeline setup file](https://github.com/AcademySoftwareFoundation/openexr/commit/9ed83bd964008c4ff19958b0e2824e08bdf6e610) ([seabeepea](@seabeepea@gmail.com) 2019-06-12) 

* [typos](https://github.com/AcademySoftwareFoundation/openexr/commit/10e33e334df9202cd8c8a940c7cd3ec36548d7d8) ([seabeepea](@seabeepea@gmail.com) 2019-06-09) 

* [Contributing and Goverance sections](https://github.com/AcademySoftwareFoundation/openexr/commit/ce9f05fbcc4c47330c43815cc40fc164e2ad53d3) ([seabeepea](@seabeepea@gmail.com) 2019-06-09) 

* [meeting notes](https://github.com/AcademySoftwareFoundation/openexr/commit/eed7c0aa972cf8b5f5641ca9946b27a3a054155f) ([Cary Phillips](@cary@ilm.com) 2019-05-09) 

* [Fixed formatting](https://github.com/AcademySoftwareFoundation/openexr/commit/b10e1015e349313b589f4c0b5b4bddefd3da64f7) ([John Mertic](@jmertic@linuxfoundation.org) 2019-05-08) Signed-off-by: John Mertic <jmertic@linuxfoundation.org>

* [moved charter to charter subfolder.](https://github.com/AcademySoftwareFoundation/openexr/commit/db49dcfdfcfaca5a60a84f65ced11df97d0df1ec) ([Cary Phillips](@cary@ilm.com) 2019-05-08) 

* [OpenEXR-Technical-Charter.md](https://github.com/AcademySoftwareFoundation/openexr/commit/2a33b9a4ca520490c5f368d6028decb9c76f8837) ([Cary Phillips](@cary@ilm.com) 2019-05-08) 

* [OpenEXR-Adoption-Proposal.md](https://github.com/AcademySoftwareFoundation/openexr/commit/3e22cab39663b5c97ba3fd20df02ae634e21fc84) ([Cary Phillips](@cary@ilm.com) 2019-05-08) 

* [Meeting notes 2019-5-2](https://github.com/AcademySoftwareFoundation/openexr/commit/c33d52f6c5a7d453d4b969224ab33852e47fe084) ([Cary Phillips](@cary@ilm.com) 2019-05-05) 

* [Remove unused cmake variable](https://github.com/AcademySoftwareFoundation/openexr/commit/c3a1da6f47279d34c23d29f6e2f264cf2126a4f8) ([Nick Porcino](@nick.porcino@oculus.com) 2019-03-29) 

* [add build-win/, build-nuget/, and *~ to .gitignore.](https://github.com/AcademySoftwareFoundation/openexr/commit/94ab55d8d4103881324ec15b8a41b3298ca7e467) ([Cary Phillips](@cary@ilm.com) 2018-09-22) 

* [Update the README files with instructions for building on Windows, specifically calling out the proper Visual Studio version.](https://github.com/AcademySoftwareFoundation/openexr/commit/ab742b86a37a7eb93f0312d98fc47f7526ddd65a) ([Cary Phillips](@cary@ilm.com) 2018-09-22) 

* [Removed OpenEXRViewers.pc.in and PyIlmBase.pc.in. Since these modules are binaries, not libraries, there is no need to support pkgconfig for them.](https://github.com/AcademySoftwareFoundation/openexr/commit/999a49d721604bb88178b596675deda4dc25cf1b) ([Cary Phillips](@cary@ilm.com) 2018-09-22) 

* [Rebuild OpenEXR NuGet with 2.3 source and enable exrviewer for testing purposes](https://github.com/AcademySoftwareFoundation/openexr/commit/c0d0a637a25e1741f528999a2556eda39102ddac) ([mancoast](@RobertPancoast77@gmail.com) 2018-09-15) 

* [fix standalone and combined cmake](https://github.com/AcademySoftwareFoundation/openexr/commit/017d027cc27ac0a7b2af90196fe3e49c4afe1aab) ([Kimball Thurston](@kdt3rd@gmail.com) 2018-09-08) This puts the version numbers into one file, and the settings and
variables for building into another, that is then replicated and
conditionally included when building a standalone package.
Signed-off-by: Kimball Thurston <kdt3rd@gmail.com>

* [CONTRIBUTING.md, INSTALL.md, and changes README.md and INSTALL.md](https://github.com/AcademySoftwareFoundation/openexr/commit/d1d9f19475c858e66c1260fcc2be9e26dcddfc03) ([seabeepea](@seabeepea@gmail.com) 2019-06-09) 

* [added GOVERNANCE.md](https://github.com/AcademySoftwareFoundation/openexr/commit/09a11a92b149f0e7d51a62086572050ad4fdc4fe) ([seabeepea](@seabeepea@gmail.com) 2019-06-09) 



## Version 2.3.0 (August 13, 2018)

### Features/Improvements:

* ThreadPool overhead improvements, enable custom thread pool to be registered via ThreadPoolProvider class
* Fixes to enable custom namespaces for Iex, Imf
* Improve read performance for deep/zipped data, and SIMD-accelerated uncompress support
* Added rawPixelDataToBuffer() function for access to compressed scanlines
* Iex::BaseExc no longer derived from std::string.
* Imath throw() specifiers removed
* Initial Support for Python 3

### Bugs:

* 25+ various bug fixes (see detailed Release Notes for the full list)

* This release addresses vulnerability [CVE-2017-12596](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-12596).

### Build Fixes:

* Various fixes to the cmake and autoconf build infrastructures
* Various changes to support compiling for C++11 / C++14 / C++17 and GCC 6.3.1
* Various fixes to address Windows build issues
* 60+ total build-related fixes (see detailed Release Notes for the full list)

### Diff Stats \[git diff --stat v2.2.1\]

    CHANGES.md                                         |  1487 +++
    CMakeLists.txt                                     |   194 +
    Contrib/DtexToExr/bootstrap                        |     2 +-
    Contrib/DtexToExr/configure.ac                     |     2 +-
    IlmBase/CMakeLists.txt                             |   214 +-
    IlmBase/COPYING                                    |    34 -
    IlmBase/Half/CMakeLists.txt                        |   107 +-
    IlmBase/Half/half.cpp                              |     6 +-
    IlmBase/Half/half.h                                |     8 +-
    IlmBase/Half/halfExport.h                          |    44 +-
    IlmBase/Half/halfLimits.h                          |     9 +
    IlmBase/HalfTest/CMakeLists.txt                    |     4 +-
    IlmBase/HalfTest/testLimits.cpp                    |    13 +-
    IlmBase/INSTALL                                    |     2 -
    IlmBase/Iex/CMakeLists.txt                         |    81 +-
    IlmBase/Iex/IexBaseExc.cpp                         |    71 +-
    IlmBase/Iex/IexBaseExc.h                           |    87 +-
    IlmBase/Iex/IexMacros.h                            |    62 +-
    IlmBase/IexMath/CMakeLists.txt                     |    76 +-
    IlmBase/IexMath/IexMathFloatExc.cpp                |    18 +
    IlmBase/IexMath/IexMathFloatExc.h                  |    36 +-
    IlmBase/IexTest/CMakeLists.txt                     |     4 +-
    IlmBase/IexTest/testBaseExc.cpp                    |     2 +-
    IlmBase/IlmThread/CMakeLists.txt                   |    78 +-
    IlmBase/IlmThread/IlmThread.cpp                    |    48 +-
    IlmBase/IlmThread/IlmThread.h                      |    48 +-
    IlmBase/IlmThread/IlmThreadForward.h               |     8 +
    IlmBase/IlmThread/IlmThreadMutex.cpp               |     7 +-
    IlmBase/IlmThread/IlmThreadMutex.h                 |    48 +-
    IlmBase/IlmThread/IlmThreadMutexPosix.cpp          |    10 +-
    IlmBase/IlmThread/IlmThreadMutexWin32.cpp          |     9 +-
    IlmBase/IlmThread/IlmThreadPool.cpp                |   720 +-
    IlmBase/IlmThread/IlmThreadPool.h                  |    64 +-
    IlmBase/IlmThread/IlmThreadPosix.cpp               |     2 +
    IlmBase/IlmThread/IlmThreadSemaphore.h             |    49 +-
    .../IlmThread/IlmThreadSemaphorePosixCompat.cpp    |    78 +-
    IlmBase/IlmThread/IlmThreadWin32.cpp               |     6 +
    IlmBase/Imath/CMakeLists.txt                       |   130 +-
    IlmBase/Imath/ImathBox.cpp                         |    37 -
    IlmBase/Imath/ImathEuler.h                         |     7 +-
    IlmBase/Imath/ImathInt64.h                         |     3 +
    IlmBase/Imath/ImathMatrix.h                        |    56 +-
    IlmBase/Imath/ImathShear.cpp                       |    54 -
    IlmBase/Imath/ImathVec.cpp                         |    24 +-
    IlmBase/Imath/ImathVec.h                           |    48 +-
    IlmBase/Imath/Makefile.am                          |     4 +-
    IlmBase/ImathTest/CMakeLists.txt                   |     6 +-
    IlmBase/Makefile.am                                |     5 +-
    IlmBase/README                                     |    70 -
    IlmBase/README.CVS                                 |    16 -
    IlmBase/README.OSX                                 |   101 -
    IlmBase/README.cmake.txt                           |    65 -
    IlmBase/README.git                                 |    16 -
    IlmBase/README.md                                  |   277 +
    IlmBase/README.namespacing                         |    83 -
    IlmBase/bootstrap                                  |     4 +-
    IlmBase/config.windows/IlmBaseConfig.h             |     1 +
    IlmBase/config/IlmBaseConfig.h.in                  |     7 +
    IlmBase/configure.ac                               |    50 +-
    IlmBase/m4/ax_cxx_compile_stdcxx.m4                |   982 ++
    LICENSE                                            |    34 +
    OpenEXR/AUTHORS                                    |     2 +
    OpenEXR/CMakeLists.txt                             |   272 +-
    OpenEXR/COPYING                                    |    34 -
    OpenEXR/INSTALL                                    |     2 -
    OpenEXR/IlmImf/CMakeLists.txt                      |   396 +-
    OpenEXR/IlmImf/ImfAcesFile.h                       |    38 +-
    OpenEXR/IlmImf/ImfAttribute.cpp                    |     6 +
    OpenEXR/IlmImf/ImfAttribute.h                      |     8 +-
    OpenEXR/IlmImf/ImfB44Compressor.h                  |    10 +-
    OpenEXR/IlmImf/ImfCRgbaFile.h                      |     2 +-
    OpenEXR/IlmImf/ImfChannelList.h                    |    45 +-
    OpenEXR/IlmImf/ImfChromaticities.h                 |     5 +-
    OpenEXR/IlmImf/ImfCompositeDeepScanLine.h          |    14 +-
    OpenEXR/IlmImf/ImfCompressionAttribute.h           |     6 +-
    OpenEXR/IlmImf/ImfCompressor.h                     |    14 +-
    OpenEXR/IlmImf/ImfDeepCompositing.h                |     6 +-
    OpenEXR/IlmImf/ImfDeepFrameBuffer.h                |    38 +-
    OpenEXR/IlmImf/ImfDeepScanLineInputFile.cpp        |     6 +-
    OpenEXR/IlmImf/ImfDeepScanLineInputFile.h          |    20 +-
    OpenEXR/IlmImf/ImfDeepScanLineInputPart.h          |    18 +-
    OpenEXR/IlmImf/ImfDeepScanLineOutputFile.cpp       |    14 +-
    OpenEXR/IlmImf/ImfDeepScanLineOutputFile.h         |    14 +-
    OpenEXR/IlmImf/ImfDeepScanLineOutputPart.h         |    12 +-
    OpenEXR/IlmImf/ImfDeepTiledInputFile.cpp           |    16 +-
    OpenEXR/IlmImf/ImfDeepTiledInputFile.h             |    37 +-
    OpenEXR/IlmImf/ImfDeepTiledInputPart.cpp           |     2 +-
    OpenEXR/IlmImf/ImfDeepTiledInputPart.h             |    34 +-
    OpenEXR/IlmImf/ImfDeepTiledOutputFile.cpp          |    18 +-
    OpenEXR/IlmImf/ImfDeepTiledOutputFile.h            |    33 +-
    OpenEXR/IlmImf/ImfDeepTiledOutputPart.h            |    31 +-
    OpenEXR/IlmImf/ImfDwaCompressor.cpp                |   232 +-
    OpenEXR/IlmImf/ImfDwaCompressor.h                  |    43 +-
    OpenEXR/IlmImf/ImfDwaCompressorSimd.h              |    67 +-
    OpenEXR/IlmImf/ImfFastHuf.cpp                      |    62 +-
    OpenEXR/IlmImf/ImfFastHuf.h                        |     5 +
    OpenEXR/IlmImf/ImfFrameBuffer.h                    |    36 +-
    OpenEXR/IlmImf/ImfGenericInputFile.h               |     5 +-
    OpenEXR/IlmImf/ImfGenericOutputFile.h              |     6 +-
    OpenEXR/IlmImf/ImfHeader.h                         |    90 +-
    OpenEXR/IlmImf/ImfIO.h                             |    13 +-
    OpenEXR/IlmImf/ImfInputFile.cpp                    |    41 +-
    OpenEXR/IlmImf/ImfInputFile.h                      |    42 +-
    OpenEXR/IlmImf/ImfInputPart.cpp                    |     8 +
    OpenEXR/IlmImf/ImfInputPart.h                      |    22 +-
    OpenEXR/IlmImf/ImfInputPartData.h                  |     1 +
    OpenEXR/IlmImf/ImfInt64.h                          |     1 +
    OpenEXR/IlmImf/ImfKeyCode.h                        |    19 +-
    OpenEXR/IlmImf/ImfLut.h                            |     8 +-
    OpenEXR/IlmImf/ImfMisc.cpp                         |    55 +-
    OpenEXR/IlmImf/ImfMisc.h                           |    20 +-
    OpenEXR/IlmImf/ImfMultiPartInputFile.cpp           |     4 +-
    OpenEXR/IlmImf/ImfMultiPartInputFile.h             |    10 +-
    OpenEXR/IlmImf/ImfMultiPartOutputFile.cpp          |     4 +-
    OpenEXR/IlmImf/ImfMultiPartOutputFile.h            |    10 +-
    OpenEXR/IlmImf/ImfName.h                           |     9 +
    OpenEXR/IlmImf/ImfOpaqueAttribute.h                |    10 +-
    OpenEXR/IlmImf/ImfOptimizedPixelReading.h          |     4 +-
    OpenEXR/IlmImf/ImfOutputFile.cpp                   |    95 +-
    OpenEXR/IlmImf/ImfOutputFile.h                     |    15 +-
    OpenEXR/IlmImf/ImfOutputPart.h                     |    13 +-
    OpenEXR/IlmImf/ImfOutputPartData.h                 |    23 +-
    OpenEXR/IlmImf/ImfPizCompressor.h                  |    10 +-
    OpenEXR/IlmImf/ImfPreviewImage.h                   |    14 +-
    OpenEXR/IlmImf/ImfPxr24Compressor.h                |    10 +-
    OpenEXR/IlmImf/ImfRational.h                       |     3 +-
    OpenEXR/IlmImf/ImfRgbaFile.h                       |    47 +-
    OpenEXR/IlmImf/ImfRleCompressor.h                  |     7 +-
    OpenEXR/IlmImf/ImfScanLineInputFile.cpp            |    42 +-
    OpenEXR/IlmImf/ImfScanLineInputFile.h              |    37 +-
    OpenEXR/IlmImf/ImfSimd.h                           |    11 +-
    OpenEXR/IlmImf/ImfStdIO.cpp                        |    36 +-
    OpenEXR/IlmImf/ImfStdIO.h                          |    24 +-
    OpenEXR/IlmImf/ImfSystemSpecific.h                 |    15 +-
    OpenEXR/IlmImf/ImfTileOffsets.h                    |    16 +-
    OpenEXR/IlmImf/ImfTiledInputFile.cpp               |    16 +-
    OpenEXR/IlmImf/ImfTiledInputFile.h                 |    32 +-
    OpenEXR/IlmImf/ImfTiledInputPart.h                 |    30 +-
    OpenEXR/IlmImf/ImfTiledOutputFile.cpp              |    66 +-
    OpenEXR/IlmImf/ImfTiledOutputFile.h                |    39 +-
    OpenEXR/IlmImf/ImfTiledOutputPart.h                |    33 +-
    OpenEXR/IlmImf/ImfTiledRgbaFile.h                  |    83 +-
    OpenEXR/IlmImf/ImfTimeCode.h                       |    35 +-
    OpenEXR/IlmImf/ImfVersion.h                        |     4 +-
    OpenEXR/IlmImf/ImfZip.cpp                          |   191 +-
    OpenEXR/IlmImf/ImfZip.h                            |     8 +
    OpenEXR/IlmImf/ImfZipCompressor.h                  |     5 +
    OpenEXR/IlmImf/Makefile.am                         |    12 +-
    OpenEXR/IlmImf/dwaLookups.cpp                      |    10 +-
    OpenEXR/IlmImfExamples/CMakeLists.txt              |    18 +-
    OpenEXR/IlmImfExamples/Makefile.am                 |     8 +-
    OpenEXR/IlmImfExamples/previewImageExamples.cpp    |     6 +-
    OpenEXR/IlmImfFuzzTest/CMakeLists.txt              |    27 +-
    OpenEXR/IlmImfFuzzTest/Makefile.am                 |     6 +-
    OpenEXR/IlmImfTest/CMakeLists.txt                  |    18 +-
    OpenEXR/IlmImfTest/Makefile.am                     |     6 +-
    OpenEXR/IlmImfTest/compareDwa.h                    |     4 +-
    OpenEXR/IlmImfTest/testDwaCompressorSimd.cpp       |    47 +-
    OpenEXR/IlmImfUtil/CMakeLists.txt                  |   113 +-
    OpenEXR/IlmImfUtil/ImfDeepImage.h                  |    33 +-
    OpenEXR/IlmImfUtil/ImfDeepImageChannel.h           |    35 +-
    OpenEXR/IlmImfUtil/ImfDeepImageIO.h                |    26 +-
    OpenEXR/IlmImfUtil/ImfDeepImageLevel.cpp           |     2 +-
    OpenEXR/IlmImfUtil/ImfDeepImageLevel.h             |    44 +-
    OpenEXR/IlmImfUtil/ImfFlatImage.h                  |    29 +-
    OpenEXR/IlmImfUtil/ImfFlatImageChannel.h           |    10 +-
    OpenEXR/IlmImfUtil/ImfFlatImageIO.h                |    26 +-
    OpenEXR/IlmImfUtil/ImfFlatImageLevel.cpp           |     2 +-
    OpenEXR/IlmImfUtil/ImfFlatImageLevel.h             |    31 +-
    OpenEXR/IlmImfUtil/ImfImage.cpp                    |     4 +-
    OpenEXR/IlmImfUtil/ImfImage.h                      |    31 +-
    OpenEXR/IlmImfUtil/ImfImageChannel.h               |    10 +-
    OpenEXR/IlmImfUtil/ImfImageDataWindow.cpp          |     3 +-
    OpenEXR/IlmImfUtil/ImfImageDataWindow.h            |     2 +
    OpenEXR/IlmImfUtil/ImfImageIO.h                    |    10 +-
    OpenEXR/IlmImfUtil/ImfImageLevel.cpp               |     2 +-
    OpenEXR/IlmImfUtil/ImfImageLevel.h                 |    20 +-
    OpenEXR/IlmImfUtil/ImfSampleCountChannel.h         |    23 +-
    OpenEXR/IlmImfUtil/ImfUtilExport.h                 |    46 +
    OpenEXR/IlmImfUtil/Makefile.am                     |    16 +-
    OpenEXR/IlmImfUtilTest/CMakeLists.txt              |    20 +-
    OpenEXR/IlmImfUtilTest/Makefile.am                 |     6 +-
    OpenEXR/Makefile.am                                |     5 +-
    OpenEXR/README                                     |    77 -
    OpenEXR/README.CVS                                 |    16 -
    OpenEXR/README.OSX                                 |    57 -
    OpenEXR/README.cmake.txt                           |    54 -
    OpenEXR/README.git                                 |    16 -
    OpenEXR/README.md                                  |   132 +
    OpenEXR/README.namespacing                         |    83 -
    OpenEXR/bootstrap                                  |     4 +-
    OpenEXR/build.log                                  | 11993 -------------------
    OpenEXR/configure.ac                               |   284 +-
    OpenEXR/doc/Makefile.am                            |     1 -
    OpenEXR/doc/TheoryDeepPixels.pdf                   |   Bin 331719 -> 334777 bytes
    OpenEXR/exr2aces/CMakeLists.txt                    |    10 +-
    OpenEXR/exrbuild/CMakeLists.txt                    |    13 +-
    OpenEXR/exrenvmap/CMakeLists.txt                   |    10 +-
    OpenEXR/exrenvmap/Makefile.am                      |     6 +-
    OpenEXR/exrheader/CMakeLists.txt                   |    15 +-
    OpenEXR/exrheader/Makefile.am                      |     6 +-
    OpenEXR/exrmakepreview/CMakeLists.txt              |    10 +-
    OpenEXR/exrmakepreview/Makefile.am                 |     6 +-
    OpenEXR/exrmakepreview/makePreview.cpp             |     6 +-
    OpenEXR/exrmaketiled/CMakeLists.txt                |     9 +-
    OpenEXR/exrmaketiled/Makefile.am                   |     6 +-
    OpenEXR/exrmaketiled/makeTiled.cpp                 |     8 +-
    OpenEXR/exrmultipart/CMakeLists.txt                |    13 +-
    OpenEXR/exrmultipart/Makefile.am                   |     8 +-
    OpenEXR/exrmultiview/CMakeLists.txt                |    12 +-
    OpenEXR/exrmultiview/Makefile.am                   |     6 +-
    OpenEXR/exrstdattr/CMakeLists.txt                  |    13 +-
    OpenEXR/exrstdattr/Makefile.am                     |     6 +-
    OpenEXR/m4/ax_cxx_compile_stdcxx.m4                |   982 ++
    OpenEXR/m4/path.pkgconfig.m4                       |    63 +-
    OpenEXR_Viewers/AUTHORS                            |    12 -
    OpenEXR_Viewers/CMakeLists.txt                     |    71 +-
    OpenEXR_Viewers/COPYING                            |    34 -
    OpenEXR_Viewers/INSTALL                            |     2 -
    OpenEXR_Viewers/Makefile.am                        |     6 +-
    OpenEXR_Viewers/NEWS                               |     2 -
    OpenEXR_Viewers/README                             |    95 -
    OpenEXR_Viewers/README.CVS                         |    16 -
    OpenEXR_Viewers/README.OSX                         |    18 -
    OpenEXR_Viewers/README.md                          |   278 +
    OpenEXR_Viewers/README.win32                       |   196 -
    OpenEXR_Viewers/bootstrap                          |     4 +-
    OpenEXR_Viewers/configure.ac                       |    47 +-
    OpenEXR_Viewers/exrdisplay/CMakeLists.txt          |    15 +-
    OpenEXR_Viewers/exrdisplay/GlWindow3d.h            |     5 +
    OpenEXR_Viewers/m4/ax_cxx_compile_stdcxx.m4        |   982 ++
    OpenEXR_Viewers/playexr/CMakeLists.txt             |     8 +-
    PyIlmBase/AUTHORS                                  |    10 -
    PyIlmBase/CMakeLists.txt                           |   128 +-
    PyIlmBase/COPYING                                  |    34 -
    PyIlmBase/INSTALL                                  |     2 -
    PyIlmBase/Makefile.am                              |     7 +-
    PyIlmBase/NEWS                                     |     2 -
    PyIlmBase/PyIex/CMakeLists.txt                     |    52 +-
    PyIlmBase/PyIex/PyIex.cpp                          |     4 +-
    PyIlmBase/PyIex/PyIex.h                            |     4 +-
    PyIlmBase/PyIex/PyIexExport.h                      |    45 +-
    PyIlmBase/PyIex/iexmodule.cpp                      |     5 +-
    PyIlmBase/PyIexTest/CMakeLists.txt                 |     4 +-
    PyIlmBase/PyImath/CMakeLists.txt                   |    53 +-
    PyIlmBase/PyImath/PyImath.cpp                      |     5 +-
    PyIlmBase/PyImath/PyImath.h                        |     8 +-
    PyIlmBase/PyImath/PyImathAutovectorize.cpp         |     2 +-
    PyIlmBase/PyImath/PyImathAutovectorize.h           |     6 +-
    PyIlmBase/PyImath/PyImathBasicTypes.cpp            |     9 +-
    PyIlmBase/PyImath/PyImathBasicTypes.h              |     4 +-
    PyIlmBase/PyImath/PyImathBox.cpp                   |    18 +-
    PyIlmBase/PyImath/PyImathBox.h                     |     4 +-
    PyIlmBase/PyImath/PyImathBox2Array.cpp             |     4 +-
    PyIlmBase/PyImath/PyImathBox3Array.cpp             |     4 +-
    PyIlmBase/PyImath/PyImathBoxArrayImpl.h            |    10 +-
    PyIlmBase/PyImath/PyImathColor.h                   |     3 +-
    PyIlmBase/PyImath/PyImathColor3.cpp                |     8 +-
    PyIlmBase/PyImath/PyImathColor3ArrayImpl.h         |     4 +-
    PyIlmBase/PyImath/PyImathColor4.cpp                |     6 +-
    PyIlmBase/PyImath/PyImathColor4Array2DImpl.h       |     7 +-
    PyIlmBase/PyImath/PyImathColor4ArrayImpl.h         |     4 +-
    PyIlmBase/PyImath/PyImathEuler.cpp                 |     8 +-
    PyIlmBase/PyImath/PyImathEuler.h                   |     3 +-
    PyIlmBase/PyImath/PyImathExport.h                  |    52 +-
    PyIlmBase/PyImath/PyImathFixedArray.cpp            |     2 +-
    PyIlmBase/PyImath/PyImathFixedArray.h              |    11 +-
    PyIlmBase/PyImath/PyImathFixedArray2D.h            |     9 +
    PyIlmBase/PyImath/PyImathFixedMatrix.h             |     9 +
    PyIlmBase/PyImath/PyImathFixedVArray.cpp           |    14 +-
    PyIlmBase/PyImath/PyImathFixedVArray.h             |     2 +-
    PyIlmBase/PyImath/PyImathFrustum.cpp               |     8 +-
    PyIlmBase/PyImath/PyImathFrustum.h                 |     3 +-
    PyIlmBase/PyImath/PyImathFun.cpp                   |     8 +-
    PyIlmBase/PyImath/PyImathFun.h                     |     2 +-
    PyIlmBase/PyImath/PyImathLine.cpp                  |    16 +-
    PyIlmBase/PyImath/PyImathLine.h                    |     2 +-
    PyIlmBase/PyImath/PyImathM44Array.cpp              |     6 +-
    PyIlmBase/PyImath/PyImathM44Array.h                |     2 +-
    PyIlmBase/PyImath/PyImathMatrix.h                  |     3 +-
    PyIlmBase/PyImath/PyImathMatrix33.cpp              |     8 +-
    PyIlmBase/PyImath/PyImathMatrix44.cpp              |    10 +-
    PyIlmBase/PyImath/PyImathOperators.h               |     4 +-
    PyIlmBase/PyImath/PyImathPlane.cpp                 |    20 +-
    PyIlmBase/PyImath/PyImathPlane.h                   |     2 +-
    PyIlmBase/PyImath/PyImathQuat.cpp                  |    10 +-
    PyIlmBase/PyImath/PyImathQuat.h                    |     3 +-
    PyIlmBase/PyImath/PyImathRandom.cpp                |    10 +-
    PyIlmBase/PyImath/PyImathShear.cpp                 |     8 +-
    PyIlmBase/PyImath/PyImathStringArray.cpp           |     6 +-
    PyIlmBase/PyImath/PyImathStringArray.h             |     4 +-
    PyIlmBase/PyImath/PyImathStringArrayRegister.h     |     2 +-
    PyIlmBase/PyImath/PyImathStringTable.cpp           |     4 +-
    PyIlmBase/PyImath/PyImathTask.cpp                  |    10 +-
    PyIlmBase/PyImath/PyImathTask.h                    |    34 +-
    PyIlmBase/PyImath/PyImathUtil.cpp                  |     6 +-
    PyIlmBase/PyImath/PyImathUtil.h                    |    14 +-
    PyIlmBase/PyImath/PyImathVec.h                     |     4 +-
    PyIlmBase/PyImath/PyImathVec2Impl.h                |    12 +-
    PyIlmBase/PyImath/PyImathVec3ArrayImpl.h           |    12 +-
    PyIlmBase/PyImath/PyImathVec3Impl.h                |     6 +-
    PyIlmBase/PyImath/PyImathVec4ArrayImpl.h           |    10 +-
    PyIlmBase/PyImath/PyImathVec4Impl.h                |     6 +-
    PyIlmBase/PyImath/imathmodule.cpp                  |    38 +-
    PyIlmBase/PyImathNumpy/CMakeLists.txt              |    25 +-
    PyIlmBase/PyImathNumpy/imathnumpymodule.cpp        |    14 +-
    PyIlmBase/PyImathNumpyTest/CMakeLists.txt          |     6 +-
    PyIlmBase/PyImathNumpyTest/pyImathNumpyTest.in     |    81 +-
    PyIlmBase/PyImathTest/CMakeLists.txt               |     2 +
    PyIlmBase/PyImathTest/pyImathTest.in               |  1090 +-
    PyIlmBase/README                                   |    51 -
    PyIlmBase/README.OSX                               |    21 -
    PyIlmBase/README.md                                |    99 +
    PyIlmBase/bootstrap                                |     4 +-
    PyIlmBase/configure.ac                             |    64 +-
    PyIlmBase/m4/ax_cxx_compile_stdcxx.m4              |   982 ++
    README                                             |    68 -
    README.md                                          |   202 +
    cmake/FindIlmBase.cmake                            |   192 +
    cmake/FindNumPy.cmake                              |    51 +
    cmake/FindOpenEXR.cmake                            |   198 +
    321 files changed, 12796 insertions(+), 16398 deletions(-)
   
### Commits \[ git log v2.2.1...v.2.3.0\]

*  [Reverted python library -l line logic to go back to the old PYTHON_VERSION based logic.](https://github.com/AcademySoftwareFoundation/openexr/commit/02310c624547fd765cd6e08abe459755d4ecebcc) ([Nick Rasmussen](@nick@ilm.com), 2018-08-09) 

*  [Updated build system to use local copies of the ax_cxx_copmile_stdcxx.m4 macro.](https://github.com/AcademySoftwareFoundation/openexr/commit/3d6c9302b3d7f394a90ac3c95d12b1db1c183812) ([Nick Rasmussen](@nick@ilm.com), 2018-08-09) 

*  [accidentally commited Makefile instead of Makefile.am](https://github.com/AcademySoftwareFoundation/openexr/commit/46dda162ef2b3defceaa25e6bdd2b71b98844685) ([Cary Phillips](@cary@ilm.com), 2018-08-09) 

*  [update CHANGES.md](https://github.com/AcademySoftwareFoundation/openexr/commit/ea46c15be9572f81549eaa76a1bdf8dbe364f780) ([Cary Phillips](@cary@ilm.com), 2018-08-08) 

*  [Added FindNumPy.cmake](https://github.com/AcademySoftwareFoundation/openexr/commit/63870bb10415ca7ea76ecfdafdfe70f5894f66f2) ([Nick Porcino](@meshula@hotmail.com), 2018-08-08) 

*  [Add PyImathNumpyTest to Makefile and configure.ac](https://github.com/AcademySoftwareFoundation/openexr/commit/36abd2b728e8759b010ceffe94363d5f473fe6dc) ([Cary Phillips](@cary@ilm.com), 2018-08-08) 

*  [Add ImfExportUtil.h to Makefile.am](https://github.com/AcademySoftwareFoundation/openexr/commit/82f78f4a895e29b42d2ccc0d66be08948203f507) ([Cary Phillips](@cary@ilm.com), 2018-08-08) 

*  [fix pyilmbase tests, static compilation](https://github.com/AcademySoftwareFoundation/openexr/commit/75c918b65c2394c7f7a9f769fee87572d06e81b5) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-09) - python extensions must be shared, so can not follow the overall lib type for the library. - the code should be compiled fPIC when building a static library such that it can be linked into a .so - remove the dependency on the particle python extension in the numpy test - add environment variables such that the python tests will work in the build tree without a "make install" (win32 doesn't neede ld_library_path, but it doesn't hurt, but may need path?) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [fix OPENEXR_VERSION and OPENEXR_SOVERSION](https://github.com/AcademySoftwareFoundation/openexr/commit/4481442b467e492a3a515b0992391dc160282786) ([Cary Phillips](@cary@ilm.com), 2018-08-08) 

*  [update readme documentation for new cmake option](https://github.com/AcademySoftwareFoundation/openexr/commit/081c9f9f9f26afc6943f1b2e63d171802895bee5) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-08) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [fix compile errors under c++17](https://github.com/AcademySoftwareFoundation/openexr/commit/6d9e3f6e2a9545e9d060f599967868d228d9a56a) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-08) Fixes errors with collisions due to the addition of clamp to the std namespace Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [add last ditch effort for numpy](https://github.com/AcademySoftwareFoundation/openexr/commit/af5fa2d84acf74e411d6592201890b1e489978c4) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-08) Apparently not all distributions include a FindNumPy.cmake or similar, even if numpy is indeed installed. This makes a second effort to find using python itself Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [make pyilmbase tests conditional](https://github.com/AcademySoftwareFoundation/openexr/commit/07951c8bdf6164e34f37c3d88799e4e98e46d1ee) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-08) This makes the PyIlmBase tests conditional in the same manner as OpenEXR and IlmBase Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [optimize regeneration of config files](https://github.com/AcademySoftwareFoundation/openexr/commit/b610ff33e827c38ac3693d3e43ad973c891d808c) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-08) This makes the config files write to a temporary file, then use cmake's configure_file command with copyonly to compare the contents and not copy if they are the same. Incremental builds are much faster as a result when working on new features and adding files to the cmakelists.txt Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [make fuzz test optional like autoconf](https://github.com/AcademySoftwareFoundation/openexr/commit/79a50ea7eb869a94bb226841aebad9d46ecc3836) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-08) This makes running the fuzz tests as part of the "make test" rule optional. Even with this off by default, if building tests is enabled, the fuzz test will still be compiled, and is available to run via "make fuzz". This should enable a weekly jenkins build config to run the fuzz tests, given that it takes a long time to run. Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [Fix SO version](https://github.com/AcademySoftwareFoundation/openexr/commit/f4055c33bb128bd4544d265b167337c584364716) ([Nick Porcino](@meshula@hotmail.com), 2018-08-07) 

*  [CHANGES.md formatting](https://github.com/AcademySoftwareFoundation/openexr/commit/8cd1b9210855fa4f6923c1b94df8a86166be19b1) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [format old release notes](https://github.com/AcademySoftwareFoundation/openexr/commit/3c5b5f894def68cf5240e8f427147c867f745912) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [release notes upates](https://github.com/AcademySoftwareFoundation/openexr/commit/534e4bcde71ce34b9f8fa9fc39e9df1a58aa3f80) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [CHANGES.md](https://github.com/AcademySoftwareFoundation/openexr/commit/471d7bd1c558c54ecc3cbbb2a65932f1e448a370) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [OpenEXR_Viewers/README.md formatting](https://github.com/AcademySoftwareFoundation/openexr/commit/806db743cf0bcb7710d08f56ee6f2ece10e31367) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [more README fixes.](https://github.com/AcademySoftwareFoundation/openexr/commit/82bc701e605e092ae5f31d142450d921c293ded1) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [README.md cleanup](https://github.com/AcademySoftwareFoundation/openexr/commit/d1d9760b084f460cf21de2b8e273e8d6adcfb4f6) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [fix dependencies when building static](https://github.com/AcademySoftwareFoundation/openexr/commit/03329c8d34c93ecafb4a35a8cc645cd3bea14217) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-08) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [fix exrdisplay compile under cmake](https://github.com/AcademySoftwareFoundation/openexr/commit/a617dc1a9cc8c7b85df040f5587f1727dec31caf) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-07) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [PyIlmBase README.md cleanup](https://github.com/AcademySoftwareFoundation/openexr/commit/a385fd4f09ab5dd1163fab6870393f1b71e163eb) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [Updates to README's](https://github.com/AcademySoftwareFoundation/openexr/commit/0690e762bb45afadd89e94838270080447998a48) ([Cary Phillips](@cary@ilm.com), 2018-08-07) 

*  [added --foreign to automake in bootstrap](https://github.com/AcademySoftwareFoundation/openexr/commit/4a74696f2066dd4bb58433bbcb706fdf526a7770) ([Cary Phillips](@cary@ilm.com), 2018-08-06) 

*  [Remove obsolete README files from Makefile.am](https://github.com/AcademySoftwareFoundation/openexr/commit/57259b7811f3adce23a1e4c99411d686c55fefed) ([Cary Phillips](@cary@ilm.com), 2018-08-06) 

*  [Removed COPYING, INSTALL, README.cmake.txt](https://github.com/AcademySoftwareFoundation/openexr/commit/54d3bbcfef10a367591cced99f759b89e8478b07) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [cleaned up README files for root and IlmBase](https://github.com/AcademySoftwareFoundation/openexr/commit/54e6ae149addd5b9673d1ee0f2954759b5ed073d) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [LIBTOOL_CURRENT=24](https://github.com/AcademySoftwareFoundation/openexr/commit/7b7ea9c86bbf8744cb41df6fa7e5f7dd270294a5) ([Cary Phillips](@cary@ilm.com), 2018-08-06) 

*  [bump version to 2.3](https://github.com/AcademySoftwareFoundation/openexr/commit/8a7b4ad263103e725fda4e624962cc0f559c4faa) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [folding in internal ILM changes - conditional delete in exception catch block.](https://github.com/AcademySoftwareFoundation/openexr/commit/656f898dff3ab7d06c4d35219385251f7948437b) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [Removed COPYING, INSTALL, README.cmake.txt](https://github.com/AcademySoftwareFoundation/openexr/commit/94ece7ca86ffccb3ec2bf4138f4ad47e3f496167) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [edits to READMEs](https://github.com/AcademySoftwareFoundation/openexr/commit/405fa911ad974eeaf3c3769820b7c4a0c59f0099) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [README fixes.](https://github.com/AcademySoftwareFoundation/openexr/commit/c612d8276a5d9e28ae6bdc39b770cbc083e21cf4) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [cleaned up README files for root and IlmBase](https://github.com/AcademySoftwareFoundation/openexr/commit/cda04c6451b0b196c887b03e68d8a80863f58832) ([Cary Phillips](@cary@ilm.com), 2018-08-05) 

*  [Fallback default system provided Boost Python](https://github.com/AcademySoftwareFoundation/openexr/commit/a174497d1fd84378423f733053f1a058608d81f0) ([Thanh Ha](@thanh.ha@linuxfoundation.org), 2018-08-03) User provided Python version via OPENEXR_PYTHON_MAJOR and OPENEXR_PYTHON_MINOR parameters, failing that fallback onto the system's default "python" whichever that may be. Signed-off-by: Thanh Ha <thanh.ha@linuxfoundation.org> 

*  [fix double delete in multipart files, check logic in others](https://github.com/AcademySoftwareFoundation/openexr/commit/da96e3759758c1fcac5963e07eab8e1f58a674e7) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-04) Multipart files have a Data object that automatically cleans up it's stream if appropriate, the other file objects have the destructor of the file object perform the delete (instead of Data). This causes a double delete to happen in MultiPart objects when unable to open a stream. Additionally, fix tabs / spaces to just be spaces Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [fix scenario where ilmimf is being compiled from parent directory](https://github.com/AcademySoftwareFoundation/openexr/commit/c246315fe392815399aee224f38bafd01585594b) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-04) need to use current source dir so test images can be found Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [Fix logic errors with BUILD_DWALOOKUPS](https://github.com/AcademySoftwareFoundation/openexr/commit/dc7cb41c4e8a3abd60dec46d0bcb6a1c9ef31452) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-04) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [remove debug print](https://github.com/AcademySoftwareFoundation/openexr/commit/8e16aa8930a85f1ef3f1f6ba454af275aabc205d) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-04) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [fully set variables as pkg config does not seem to by default](https://github.com/AcademySoftwareFoundation/openexr/commit/f478511f796e5d05dada28f9841dcf9ebd9730ac) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-04) 

*  [add check with error message for zlib, fix defaults, restore old thread check](https://github.com/AcademySoftwareFoundation/openexr/commit/788956537282cfcca712c1e9690d72cd19978ce0) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-04) 

*  [PR #187 CMake enhancements to speed up dependency builds of OpenEXR.](https://github.com/AcademySoftwareFoundation/openexr/commit/17e10ab10ddf937bc2809bda858bf17af6fb3448) ([Nick Porcino](@meshula@hotmail.com), 2018-08-02) 

*  [restore prefix, update to use PKG_CHECK_MODULES](https://github.com/AcademySoftwareFoundation/openexr/commit/fb9d1be5c07779c90e7744ccbf27201fcafcdfdb) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-03) previous commit from dracwyrm had made it such that pkg-config must be used and ilmbase must be installed in the default pkg-config path by default. restore the original behaviour by which a prefix could be provided, yet still retain use of PKG_CHECK_MODULES to find IlmBase if the prefix is not specified, and continue to use pkg-config to find zlib instead of assuming -lz Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [restore original API for Lock since we can't use typedef to unique_lock](https://github.com/AcademySoftwareFoundation/openexr/commit/e7fc2258a16ab7fe17d24855d16d4e56b80c172e) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-02) 

*  [fixes #292, issue with utf-8 filenames](https://github.com/AcademySoftwareFoundation/openexr/commit/846fe64c584ebb89434aaa02f5d431fbd3ca6165) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-01) windows needs to widen the string to properly open files, this implements a solution for compiling with MSVC anyway using the extension for fstream to take a wchar Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [fix maintainer mode issue, extra line in paste](https://github.com/AcademySoftwareFoundation/openexr/commit/772ff9ad045032fc338af1b684cb50983191bc0d) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-08-02) 

*  [Default the python bindings to on](https://github.com/AcademySoftwareFoundation/openexr/commit/dc5e26136b1c5edce911ff0eccc17cda40388b54) ([Nick Porcino](@meshula@hotmail.com), 2018-08-01) 

*  [Add Find scripts, and ability to build OpenEXR with pre-existing IlmBase](https://github.com/AcademySoftwareFoundation/openexr/commit/34ee51e9118097f784653f08c9482c886f83d2ef) ([Nick Porcino](@meshula@hotmail.com), 2018-08-01) 

*  [fix names, disable rules when not building shared](https://github.com/AcademySoftwareFoundation/openexr/commit/dbd3b34baf4104e844c273b682e7b133304294f2) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-31) 

*  [add suffix variable for target names to enable static-only build](https://github.com/AcademySoftwareFoundation/openexr/commit/7b1ed10e241e793db9d8933df30dd305a93835dd) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-31) 

*  [The string field is now called _message.](https://github.com/AcademySoftwareFoundation/openexr/commit/bd32e84632da4754cfe6db47f2e72c29f4d7df27) ([Cary Phillips](@cary@ilm.com), 2018-08-01) 

*  [C++11 support for numeric_limits<Half>::max_digits10() and lowest()](https://github.com/AcademySoftwareFoundation/openexr/commit/2d931bab38840ab3cdf9c6322767a862aae4037d) ([Cary Phillips](@cary@ilm.com), 2018-07-31) 

*  [fixes for GCC 6.3.1 (courtesy of Will Harrower): - renamed local variables in THROW macros to avoid warnings - cast to bool](https://github.com/AcademySoftwareFoundation/openexr/commit/7fda69a377ee41979284137795cb338bb3c6d147) ([Cary Phillips](@cary@rnd-build7-sf-38.lucasfilm.com), 2018-07-31) 

*  [renames name to message and removes implicit cast](https://github.com/AcademySoftwareFoundation/openexr/commit/54105e3c292c6884e7870ecfddb561deda7a3458) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-31) This removes the implicit cast, which is arguably more "standard", and also less surprising. Further, renames the name function to message to match internal ILM changes, and message makes more sense as a function name than ... name. 

*  [Remove IEX_THROW_SPEC](https://github.com/AcademySoftwareFoundation/openexr/commit/02c896501da244ec6345d7ee5ef825d71ba1f0a2) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-31) This removes the macro and uses therein. We changed the API with removing the subclass from std::string of Iex::BaseExc, so there is no reason to retain this compatibility as well, especially since it isn't really meaningful anyway in (modern) C++ 

*  [CMake3 port. Various Windows fixes](https://github.com/AcademySoftwareFoundation/openexr/commit/b2d37be8b874b300be1907f10339cac47e39170b) ([Nick Porcino](@meshula@hotmail.com), 2018-07-29) 

*  [changes to enable custom namespace defines to be used](https://github.com/AcademySoftwareFoundation/openexr/commit/acd76e16276b54186096b04b06bd118eb32a1bcf) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-29) 

*  [fix extraneous unsigned compare accidentally merged](https://github.com/AcademySoftwareFoundation/openexr/commit/a56773bd7a1f9a8bb10afe5fb36c4e03f622eff6) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-29) 

*  [Use proper definition of namespaces instead of default values.](https://github.com/AcademySoftwareFoundation/openexr/commit/c6978f9fd998df32b2c56a7b25bbbd52005bbf9e) ([Juri Abramov](@gabramov@nvidia.com), 2014-08-18) 

*  [fixes #260, out of bounds vector access](https://github.com/AcademySoftwareFoundation/openexr/commit/efc360fc17935453e95f62939dd5d7caacce4bf7) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-29) noticed by Google Autofuzz Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [fix potential io streams leak in case of exceptions during 'initialize' function](https://github.com/AcademySoftwareFoundation/openexr/commit/19bac86f27bab8649858ef79658224e9a54cb4cf) ([CAHEK7](@ghosts.in.a.box@gmail.com), 2016-02-12) 

*  [OpenEXR: Fix build system and change doc install](https://github.com/AcademySoftwareFoundation/openexr/commit/60cc8b711ab402c5526ca1f872de5209ad15ec7d) ([dracwyrm](@j.scruggs@gmail.com), 2017-08-11) The build sysem for the OpenEXR sub-module is has issues. This patch is being used on Gentoo Linux with great success. It also adresses the issue of linking to previously installed versions. Signed-off by: Jonathan Scruggs (j.scruggs@gmail.com) Signed-off by: David Seifert (soap@gentoo.org) 

*  [Note that numpy is required to build PyIlmBase](https://github.com/AcademySoftwareFoundation/openexr/commit/76935a912a8e365ed4fe8c7a54b60561790dafd5) ([Thanh Ha](@thanh.ha@linuxfoundation.org), 2018-07-20) Signed-off-by: Thanh Ha <thanh.ha@linuxfoundation.org> 

*  [Fixed exports on DeepImageChannel and FlatImageChannel. If the whole class isn't exported, the typeinfo doesn't get exported, and so dynamic casting into those classes will not work.](https://github.com/AcademySoftwareFoundation/openexr/commit/942ff971d30cba1b237c91e9f448376d279dc5ee) ([Halfdan Ingvarsson](@halfdan@sidefx.com), 2014-10-06) Also fixed angle-bracket include to a quoted include. 

*  [Fixed angle bracket vs quote includes.](https://github.com/AcademySoftwareFoundation/openexr/commit/fd8570927a7124ff2990f5f38556b2ec03d77a44) ([Halfdan Ingvarsson](@halfdan@sidefx.com), 2014-03-18) 

*  [Change IexBaseExc to no longer derive from std::string, but instead include it as a member variable. This resolves a problem with MSVC 2012 and dllexport-ing template classes.](https://github.com/AcademySoftwareFoundation/openexr/commit/fa59776fd83a8f35ed5418b83bbc9975ba0ef3bc) ([Halfdan Ingvarsson](@halfdan@sidefx.com), 2014-03-03) 

*  [make code more amenable to compiling with mingw for cross-compiling](https://github.com/AcademySoftwareFoundation/openexr/commit/dd867668c4c63d23c034cc2ea8f2352451e8554d) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-07-29) Signed-off-by: Kimball Thurston <kdt3rd@gmail.com> 

*  [Fix shebang line to bash](https://github.com/AcademySoftwareFoundation/openexr/commit/d3512e07a5af5053397ed62bd0d306b10357358c) ([Thanh Ha](@thanh.ha@linuxfoundation.org), 2018-07-19) Depending on the distro running the script the following error might appear: ./bootstrap: 4: [: Linux: unexpected operator This is because #!/bin/sh is not the same on every distro and this script is actually expecting bash. So update the shebang line to be bash. Signed-off-by: Thanh Ha <thanh.ha@linuxfoundation.org> 

*  [Visual Studio and Windows fixes](https://github.com/AcademySoftwareFoundation/openexr/commit/4cfefeab4be94b8c46d604075367b6496d29dcb5) ([Liam Fernandez](@liam@utexas.edu), 2018-06-20) IlmBase: Fix IF/ELSEIF clause (WIN32 only) PyImath: Install *.h in 'include' dir PyImathNumpy: Change python library filename to 'imathnumpy.pyd' (WIN32 only) 

*  [Fix probable typo for static builds.](https://github.com/AcademySoftwareFoundation/openexr/commit/31e1ae8acad3126a63044dfb8518d70390131c7b) ([Simon Otter](@skurmedel@gmail.com), 2018-06-18) 

*  [Must also export protected methods](https://github.com/AcademySoftwareFoundation/openexr/commit/17384ee01e5fa842f282c833ab2bc2aa33e07125) ([Nick Porcino](@meshula@hotmail.com), 2018-06-10) 

*  [IlmImfUtilTest compiles successfully](https://github.com/AcademySoftwareFoundation/openexr/commit/6093789bc7b7c543f128ab2b055987808ec15167) ([Nick Porcino](@meshula@hotmail.com), 2018-06-09) 

*  [IlmImfUtil now builds on Windows](https://github.com/AcademySoftwareFoundation/openexr/commit/d7328287d1ea363ab7839201e90d7d7f4deb635f) ([Nick Porcino](@meshula@hotmail.com), 2018-06-09) 

*  [Set python module suffix per platform](https://github.com/AcademySoftwareFoundation/openexr/commit/39b9edfdfcad5e77601d4462a6f9ba93bef83835) ([Nick Porcino](@meshula@hotmail.com), 2018-06-05) 

*  [fix include ifdef](https://github.com/AcademySoftwareFoundation/openexr/commit/32723d8112d1addf0064e8295b824faab60f0162) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-05-26) 

*  [switch from shared pointer to a manually counted object as gcc 4.8 and 4.9 do not provide proper shared_ptr atomic functions](https://github.com/AcademySoftwareFoundation/openexr/commit/3f532a7ab81c33f61dc6786a8c7ce6e0c09acc07) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-05-26) 

*  [Fix typos to TheoryDeepPixels document](https://github.com/AcademySoftwareFoundation/openexr/commit/655f96032e0eddd868a122fee80bd558e0cbf17d) ([peterhillman](@peter@peterhillman.org.uk), 2018-05-17) Equations 6 and 7 were incorrect. 

*  [initial port of PyIlmBase to python 3](https://github.com/AcademySoftwareFoundation/openexr/commit/84dbf637c5c3ac4296181dd93de4fb5ffdc4b582) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-05-04) 

*  [replicate configure / cmake changes from ilmbase](https://github.com/AcademySoftwareFoundation/openexr/commit/00df2c72ca1b7cb148e19a9bdc44651a6c74c9e4) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-05-04) This propagates the same chnages to configure.ac and cmakelists.txt to enable compiling with c++11/14. Additionally, adds some minor changes to configure to enable python 3 to be configured (source code changes tbd) 

*  [add move constructor and assignment operator](https://github.com/AcademySoftwareFoundation/openexr/commit/cfebcc24e1a1cc307678ea757ec38bff02a5dc51) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-05-03) 

*  [Fix Windows Python binding builds. Does not address PyImath runtime issues, but does allow build to succeed](https://github.com/AcademySoftwareFoundation/openexr/commit/15ce54ca02fdfa16c4a99f45a30c7a54826c6ac3) ([Nick Porcino](@meshula@hotmail.com), 2018-04-30) 

*  [Fix c++11 detection issue on windows. Fix ilmbase DLL export warnings](https://github.com/AcademySoftwareFoundation/openexr/commit/7376f9b736f9503a9d34b67c99bc48ce826a6334) ([Nick Porcino](@meshula@hotmail.com), 2018-04-27) 

*  [enable different c++ standards to be selected instead of just c++14](https://github.com/AcademySoftwareFoundation/openexr/commit/99ecfcabbc2b95acb40283f04ab358b3db9cc0f9) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-15) 

*  [Incorporate review feedback](https://github.com/AcademySoftwareFoundation/openexr/commit/99b367d963ba0892e7ab830458b6a990aa3033ce) ([Nick Porcino](@meshula@hotmail.com), 2018-04-04) 

*  [add compatibility std::condition_variable semaphore when posix semaphores not available](https://github.com/AcademySoftwareFoundation/openexr/commit/b6dc2a6b71f9373640d988979f9ae1929640397a) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [fix error overwriting beginning of config file](https://github.com/AcademySoftwareFoundation/openexr/commit/01680dc4d90c9f7fd64e498e57588f630a52a214) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [remove the dynamic exception for all versions of c++ unless FORCE_CXX03 is on](https://github.com/AcademySoftwareFoundation/openexr/commit/45cb2c8fb2418afaa3900c553e26ad3886cd5acf) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [ThreadPool improvements](https://github.com/AcademySoftwareFoundation/openexr/commit/bf0cb8cdce32fce36017107c9982e1e5db2fb3fa) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) - switch to use c++11 features - Add API to enable replacement of the thread pool - Add custom, low-latency handling when threads is 0 - Lower lock boundary when adding tasks (or eliminate in c++11 mode) 

*  [switch mutex to be based on std::mutex when available](https://github.com/AcademySoftwareFoundation/openexr/commit/848c8c329b16aeee0d3773e827d506a2a53d4840) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [switch IlmThread to use c++11 threads when enabled](https://github.com/AcademySoftwareFoundation/openexr/commit/eea1e607177e339e05daa1a2ec969a9dd12f2497) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [use dynamic exception macro to avoid warnings in c++14 mode](https://github.com/AcademySoftwareFoundation/openexr/commit/610179cbe3ffc2db206252343e75a16221d162b4) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [add #define to manage dynamic exception deprecation in c++11/14](https://github.com/AcademySoftwareFoundation/openexr/commit/b133b769aaee98566e695191476f59f32eece591) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [configuration changes to enable c++14](https://github.com/AcademySoftwareFoundation/openexr/commit/5f58c94aea83d44e27afd1f65e4defc0f523f6be) ([Kimball Thurston](@kdt3rd@gmail.com), 2018-04-04) 

*  [Cmake now building OpenEXR successfully for Windows](https://github.com/AcademySoftwareFoundation/openexr/commit/ac055a9e50c974f4cd58c28a5a0bb96011812072) ([Nick Porcino](@meshula@hotmail.com), 2018-03-28) 

*  [Missing symbols on Windows due to missing IMF_EXPORT](https://github.com/AcademySoftwareFoundation/openexr/commit/965c1eb6513ad80c71b425c8a1b04a70b3bae291) ([Ibraheem Alhashim](@ibraheem.alhashim@gmail.com), 2018-03-05) 

*  [Implement SIMD-accelerated ImfZip::uncompress](https://github.com/AcademySoftwareFoundation/openexr/commit/32f2aa58fe4f6f6691eef322fdfbbc9aa8363f80) ([John Loy](@jloy@pixar.com), 2017-04-12) The main bottleneck in ImfZip::uncompress appears not to be zlib but the predictor & interleaving loops that run after zlib's decompression. Fortunately, throughput in both of these loops can be improved with SIMD operations. Even though each trip of the predictor loop has data dependencies on all previous values, the usual SIMD prefix-sum construction is able to provide a significant speedup. While the uses of SSSE3 and SSE4.1 are minor in this change and could maybe be replaced with some slightly more complicated SSE2, SSE4.1 was released in 2007, so it doesn't seem unreasonable to require it in 2017. 

*  [Compute sample locations directly in Imf::readPerDeepLineTable.](https://github.com/AcademySoftwareFoundation/openexr/commit/e64095257a29f9bc423298ee8dbc09a317f22046) ([John Loy](@jloy@pixar.com), 2017-04-06) By changing the function to iterate over sample locations directly instead of discarding unsampled pixel positions, we can avoid computing a lot of modulos (more than one per pixel.) Even on modern x86 processors, idiv is a relatively expensive instruction. Though it may appear like this optimization could be performed by a sufficiently sophisticated compiler, gcc 4.8 does not get there (even at -O3.) 

*  [Manually hoist loop invariants in Imf::bytesPerDeepLineTable.](https://github.com/AcademySoftwareFoundation/openexr/commit/71b8109a4ad123ef0d5783f01922463a16d2ca59) ([John Loy](@jloy@pixar.com), 2017-04-05) This is primarily done to avoid a call to pixelTypeSize within the inner loop. In particular, gcc makes the call to pixelTypeSize via PLT indirection so it may have arbitrary side-effects (i.e. ELF symbol interposition strikes again) and may not be moved out of the loop by the compiler. 

*  [Inline Imf::sampleCount; this is an ABI-breaking change.](https://github.com/AcademySoftwareFoundation/openexr/commit/5aa0afd5a4f8df9e09d6461f115e6e0cec5cbe46) ([John Loy](@jloy@pixar.com), 2017-03-29) gcc generates calls to sampleCount via PLT indirection even within libIlmImf. As such, they are not inlined and must be treated as having arbitrary side effects (because of ELF symbol interposition.) Making addressing computations visible at call sites allows a much wider range of optimizations by the compiler beyond simply eliminating the function call overhead. 

*  [Delete build.log](https://github.com/AcademySoftwareFoundation/openexr/commit/148f1c230b5ecd94d795ca172a8246785c7caca7) ([Arkady Shapkin](@arkady.shapkin@gmail.com), 2017-02-18) 

*  [fix defect in semaphore implementation which caused application hang at exit time, because not all worker threads get woken up when task semaphore is repeatedly posted (to wake them up) after setting the stopping flag in the thread pool](https://github.com/AcademySoftwareFoundation/openexr/commit/4706d615e942462a532381a8a86bc5fe820c6816) ([Richard Goedeken](@Richard@fascinationsoftware.com), 2016-11-22) 

*  [fix comparison of unsigned expression < 0 (Issue #165)](https://github.com/AcademySoftwareFoundation/openexr/commit/9e3913c94c55549640c732f549d2912fbd85c336) ([CAHEK7](@ghosts.in.a.box@gmail.com), 2016-02-15) 

*  [Added Iex library once more for linker dependency](https://github.com/AcademySoftwareFoundation/openexr/commit/b0b50791b5b36fddb010b5ad630dd429f947a080) ([Eric Sommerlade](@es0m@users.noreply.github.com), 2015-02-20) 

*  [windows/cmake: Commands depend on Half.dll which needs to be in path. Running commands in Half.dll's directory addresses this and the commands run on first invocation](https://github.com/AcademySoftwareFoundation/openexr/commit/1a23716fd7e9ae167f53c7f2099651ede1279fbb) ([E Sommerlade](@es0m@users.noreply.github.com), 2015-02-10) 

*  [Fixed memory corruption / actual crashes on Window](https://github.com/AcademySoftwareFoundation/openexr/commit/c330c40e1962257b0e59328fdceaa9cdcde3041b) ([JuriAbramov](@openexr@dr-abramov.de), 2015-01-19) Fixed memory corruption caused by missing assignment operator with non-trivial copy constructor logic. FIxes crashes on Windows when "dwaa" or "dwab" codecs are used for saving files. 

*  [std namespace should be specified for transform](https://github.com/AcademySoftwareFoundation/openexr/commit/4a00a9bc6c92b20443c61f5e9877123e7fef16e6) ([JuriAbramov](@openexr@dr-abramov.de), 2014-08-20) Fixes build with some VS and clang version. 

*  [m4/path.pkgconfig.m4: use PKG_PROG_PKG_CONFIG to find correct pkg-config](https://github.com/AcademySoftwareFoundation/openexr/commit/056cb9f09efa9116c7f5fb8bc0717a260ad23744) ([Michael Thomas (malinka)](@malinka@entropy-development.com), 2016-05-24) pkg-config supplies this macro and prefers it to be used to allow for cross-compilation scenarios where target-prefixed binaries are prefered to pkg-config 

*  [Updated list of EXTRA_DIST files to reflect the updated test images and prior removal of README.win32](https://github.com/AcademySoftwareFoundation/openexr/commit/165dceaeee86e0f8ce1ed1db3e3030c609a49f17) ([Nick Rasmussen](@nick@ilm.com), 2017-11-17) 

*  [Updated list of EXTRA_DIST files to reflect the updated test images and prior removal of README.win32](https://github.com/AcademySoftwareFoundation/openexr/commit/dcaf5fdb4d1244d8e60a58832cfe9c54734a2257) ([Nick Rasmussen](@nick@ilm.com), 2017-11-17) 

*  [Updated openexr version to 2.2.1, resynced the .so version number to 23 across all projects.](https://github.com/AcademySoftwareFoundation/openexr/commit/e69de40ddbb6bd58341618a506b2e913e5ac1797) ([Nick Rasmussen](@nick@ilm.com), 2017-11-17) 

*  [Add additional input validation in an attempt to resolve issue #232](https://github.com/AcademySoftwareFoundation/openexr/commit/49db4a4192482eec9c27669f75db144cf5434804) ([Shawn Walker-Salas](@shawn.walker@oracle.com), 2017-05-30) 

*  [Add additional input validation in an attempt to resolve issue #232](https://github.com/AcademySoftwareFoundation/openexr/commit/f09f5f26c1924c4f7e183428ca79c9881afaf53c) ([Shawn Walker-Salas](@shawn.walker@oracle.com), 2017-05-30) 

*  [root level LICENSE](https://github.com/AcademySoftwareFoundation/openexr/commit/a774d643b566d56314f26695f2bf9b75f88e64f6) ([cary-ilm](@cary@ilm.com), 2017-10-23) 

*  [Fix copyright/license notice in halfExport.h](https://github.com/AcademySoftwareFoundation/openexr/commit/20d043d017d4b752356bb76946ffdffaa9c15c72) ([Ed Hanway](@ehanway@ilm.com), 2017-01-09) 

*  [Merge branch 'jkingsman-cleanup-readme' into develop](https://github.com/AcademySoftwareFoundation/openexr/commit/6f6d9cea513ea409d4b65da40ac096eab9a549b0) ([Ed Hanway](@ehanway@ilm.com), 2016-10-28) 

*  [README edits.](https://github.com/AcademySoftwareFoundation/openexr/commit/098a4893910d522b867082ed38d7388e6265bee0) ([Ed Hanway](@ehanway@ilm.com), 2016-10-28) 

*  [Merge branch 'cleanup-readme' of https://github.com/jkingsman/openexr into jkingsman-cleanup-readme](https://github.com/AcademySoftwareFoundation/openexr/commit/43e50ed5dca1ddfb3ca2cb4c38c7752497db6e50) ([Ed Hanway](@ehanway@ilm.com), 2016-10-28) 

*  [Install ImfStdIO.h](https://github.com/AcademySoftwareFoundation/openexr/commit/2872d3b230a7920696510f80a50d9ce36b6cc94e) ([Ed Hanway](@ehanway@ilm.com), 2016-10-28) This was originally intended to be an internal class only, but its use has become the de facto way to handle UTF-8 filenames on Windows. 

*  [Merge pull request #204 from dlemstra/IMF_HAVE_SSE2](https://github.com/AcademySoftwareFoundation/openexr/commit/cbb01bf286a2e04df95fb51458d1c2cbdc08935b) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2016-10-19) Consistent check for IMF_HAVE_SSE2. 

*  [Remove fixed-length line breaks](https://github.com/AcademySoftwareFoundation/openexr/commit/0ea6b8c7d077a18fb849c2b2ff532cd952d06a38) ([Jack Kingsman](@jack.kingsman@gmail.com), 2016-10-19) 

*  [Update README to markdown](https://github.com/AcademySoftwareFoundation/openexr/commit/9c6d22e23a25d761f5456e08623b8d77c0f8930a) ([Jack Kingsman](@jack.kingsman@gmail.com), 2016-10-18) 

*  [Merge pull request #206 from lgritz/lg-register](https://github.com/AcademySoftwareFoundation/openexr/commit/6788745398594d479e8cf91a6c301fea0537108b) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2016-09-30) Remove 'register' keyword. 

*  [Remove 'register' keyword.](https://github.com/AcademySoftwareFoundation/openexr/commit/6d297f35c5dbfacc8a5e94f33b986db7ab468db9) ([Larry Gritz](@lg@larrygritz.com), 2016-09-30) 'register' is a relic of K&R-era C, it's utterly useless in modern compilers. It's been deprecated in C++11, and therefore will generate warnings when encountered -- and many packages that use OpenEXR's public headers use -Werr to turn warnings into errors. Starting in C++17, the keyword is removed entirely, and thus will certainly be a build break for that version of the standard. So it's time for it to go. 

*  [Consistent check for IMF_HAVE_SSE2.](https://github.com/AcademySoftwareFoundation/openexr/commit/7403524c8fed971383c724d85913b2d52672caf3) ([dirk](@dirk@git.imagemagick.org), 2016-09-17) 

*  [Merge pull request #141 from lucywilkes/develop](https://github.com/AcademySoftwareFoundation/openexr/commit/c23f5345a6cc89627cc416b3e0e6b182cd427479) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2016-09-16) Adding rawPixelDataToBuffer() function for access to compressed scanlines 

*  [Merge pull request #198 from ZeroCrunch/develop](https://github.com/AcademySoftwareFoundation/openexr/commit/891437f74805f6c8ebc897932091cbe0bb7e1163) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2016-08-02) Windows compile fix 

*  [Windows compile fix](https://github.com/AcademySoftwareFoundation/openexr/commit/77faf005b50e8f77a8080676738ef9b9c807bf53) ([Jamie Kenyon](@jamie.kenyon@thefoundry.co.uk), 2016-07-29) std::min wasn't found due to <algorithm> not being included. 

*  [Merge pull request #179 from CAHEK7/NullptrBug](https://github.com/AcademySoftwareFoundation/openexr/commit/a0a68393a4d3b622251fb7c490ee9d59e080b776) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2016-07-26) fix potential memory leak 

*  [Merge branch 'develop' of https://github.com/r-potter/openexr into r-potter-develop](https://github.com/AcademySoftwareFoundation/openexr/commit/b206a243a03724650b04efcdf863c7761d5d5d5b) ([Ed Hanway](@ehanway@ilm.com), 2016-07-26) 

*  [Merge pull request #154 into develop](https://github.com/AcademySoftwareFoundation/openexr/commit/bc372d47186db31d104e84e4eb9e84850819db8d) ([Ed Hanway](@ehanway@ilm.com), 2016-07-25) 

*  [Merge pull request #168 into develop](https://github.com/AcademySoftwareFoundation/openexr/commit/44d077672f558bc63d907891bb88d741b334d807) ([Ed Hanway](@ehanway@ilm.com), 2016-07-25) 

*  [Merge pull request #175 into develop](https://github.com/AcademySoftwareFoundation/openexr/commit/7513fd847cf38af89572cc209b03e5b548e6bfc8) ([Ed Hanway](@ehanway@ilm.com), 2016-07-25) 

*  [Merge pull request #174 into develop](https://github.com/AcademySoftwareFoundation/openexr/commit/b16664a2ee4627c235b9ce798f4fc911e9c5694f) ([Ed Hanway](@ehanway@ilm.com), 2016-07-25) 

*  [Merge branch pull request 172 into develop: fix copy and paste bug in ImfDeepTiledInputPart.cpp](https://github.com/AcademySoftwareFoundation/openexr/commit/ef7b78d5988d37dbbc74c21ad245ed5c80927223) ([Ed Hanway](@ehanway@ilm.com), 2016-07-25) 

*  [Merge pull request #195 from openexr/master](https://github.com/AcademySoftwareFoundation/openexr/commit/bc234de193bd9cd32d94648e2936270aa4406e91) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2016-07-25) Catch develop branch up with commits in master. 

*  [fix potential memory leak](https://github.com/AcademySoftwareFoundation/openexr/commit/d2f10c784d52f841b85e382620100cdbf0d3b1e5) ([CAHEK7](@ghosts.in.a.box@gmail.com), 2016-02-05) 

*  [Fix warnings when compiled with MSVC 2013.](https://github.com/AcademySoftwareFoundation/openexr/commit/3aabef263083024db9e563007d0d76609ac8d585) ([Xo Wang](@xow@google.com), 2016-01-06) Similar fix to that from a27e048451ba3084559634e5e045a92a613b1455. 

*  [Fix typo in C bindings (Close #140)](https://github.com/AcademySoftwareFoundation/openexr/commit/c229dfe63380f41dfae1e977b10dfc7c49c7efc7) ([Edward Kmett](@ekmett@gmail.com), 2015-12-09) IMF_RAMDOM_Y should be IMF_RANDOM_Y 

*  [Fix copy and paste bug](https://github.com/AcademySoftwareFoundation/openexr/commit/501b654d851e2da1d9e5ca010a1e13fe34ae24ab) ([Christopher Kulla](@fpsunflower@users.noreply.github.com), 2015-11-19) The implementation of DeepTiledInputPart::tileXSize was copy and pasted from the function above but not changed. This causes it tor return incorrect values. 

*  [Switch AVX detection asm to not use an empty clobber list for use with older gcc versions](https://github.com/AcademySoftwareFoundation/openexr/commit/51073d1aa8f96963fc6a3ecad8f844ce70c90991) ([Kevin Wheatley](@kevin.wheatley@framestore.com), 2015-10-14) 

*  [Merge pull request #145 from karlrasche/DWAx_clamp_float32](https://github.com/AcademySoftwareFoundation/openexr/commit/521b25df787b460e57d5c1e831b232152b93a6ee) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2015-10-23) Clamp, don't cast, float inputs with DWAx compression 

*  [Merge pull request #143 from karlrasche/DWAx_bad_zigzag_order](https://github.com/AcademySoftwareFoundation/openexr/commit/9547d38199f5db2712c06ccdda9195badbecccaa) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2015-10-23) Wrong zig-zag ordering used for DWAx decode optimization 

*  [Merge pull request #157 from karlrasche/DWAx_compress_bound](https://github.com/AcademySoftwareFoundation/openexr/commit/de27156b77896aeef5b1c99edbca2bc4fa784b51) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2015-10-23) Switch over to use compressBound() instead of manually computing headroom for compress() 

*  [Switch over to use compressBound() instead of manually computing headroom for compress()](https://github.com/AcademySoftwareFoundation/openexr/commit/c9a2e193ce243c66177ddec6be43bc6f655ff78a) ([Karl Rasche](@karl.rasche@dreamworks.com), 2015-02-18) 

*  [Fix a linker error when compiling OpenEXR statically on Linux](https://github.com/AcademySoftwareFoundation/openexr/commit/caa09c1b361e2b152786d9e8b2b90261c9d9a3aa) ([Wenzel Jakob](@wenzel@inf.ethz.ch), 2015-02-02) Linking OpenEXR and IlmBase statically on Linux failed due to interdependencies between Iex and IlmThread. Simply reversing their order in CMakeLists.txt fixes the issue (which only arises on Linux since the GNU linker is particularly sensitive to the order of static libraries) 

*  [Clamp incoming float values to half, instead of simply casting, on encode.](https://github.com/AcademySoftwareFoundation/openexr/commit/cb172eea58b8be078b88eca35f246e12df2de620) ([Karl Rasche](@karl.rasche@dreamworks.com), 2014-11-24) Casting can introduce Infs, which are zero'ed later on, prior to the forward DCT step. This can have the nasty side effect of forcing bright values to zero, instead of clamping them to 65k. 

*  [Remove errant whitespace](https://github.com/AcademySoftwareFoundation/openexr/commit/fc67c8245dbff48e546abae027cc9c80c98b3db1) ([Karl Rasche](@karl.rasche@dreamworks.com), 2014-11-20) 

*  [Use the correct zig-zag ordering when finding choosing between fast-path inverse DCT versions (computing which rows are all zero)](https://github.com/AcademySoftwareFoundation/openexr/commit/b0d0d47b65c5ebcb8c6493aa2238b9f890c4d7fe) ([Karl Rasche](@karl.rasche@dreamworks.com), 2014-11-19) 

*  [Resolve dependency issue building eLut.h/toFloat.h with CMake/Ninja.](https://github.com/AcademySoftwareFoundation/openexr/commit/8eed7012c10f1a835385d750fd55f228d1d35df9) ([Ralph Potter](@r.potter@bath.ac.uk), 2014-11-05) 

*  [Adding rawPixelDataToBuffer() function for access to compressed data read from scanline input files.](https://github.com/AcademySoftwareFoundation/openexr/commit/1f6eddeea176ce773dacd5cdee0cbad0ab549bae) ([Lucy Wilkes](@lucywilkes@users.noreply.github.com), 2014-10-22) Changes from The Foundry to add rawPixelDataToBuffer(...) function to the OpenEXR library. This allows you to read raw scan lines into an external buffer. It's similar to the existing function rawPixelData, but unlike this existing function it allows the user to control where the data will be stored instead of reading it into a local buffer. This means you can store multiple raw scan lines at once and enables the decompression of these scan lines to be done in parallel using an application's own threads. (cherry picked from commit ca76ebb40a3c5a5c8e055f0c8d8be03ca52e91c8) 

*  [Merge pull request #137 from karlrasche/interleaveByte2_sse_bug](https://github.com/AcademySoftwareFoundation/openexr/commit/f4a6d3b9fabd82a11b63abf938e9e32f42d2d6d7) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2014-10-15) Fixing SSE2 byte interleaving path to work with short runs 

*  [Fixing SSE2 byte interleaving path to work with short runs](https://github.com/AcademySoftwareFoundation/openexr/commit/da28ad8cd54dfa3becfdac33872c5b1401a9cc3c) ([Karl Rasche](@karl.rasche@dreamworks.com), 2014-09-08) 

*  [Merge pull request #126 from fnordware/LL_literal](https://github.com/AcademySoftwareFoundation/openexr/commit/91015147e5a6a1914bcb16b12886aede9e1ed065) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2014-08-14) Use LL for 64-bit literals 

*  [Change suffixes to ULL because Int64 is unsigned](https://github.com/AcademySoftwareFoundation/openexr/commit/353cbc2e89c582e07796f01bce8f203e84c8ae46) ([Brendan Bolles](@brendan@fnordware.com), 2014-08-14) As discusses in pull request #126 

*  [Merge pull request #127 from openexr/tarball_contents_fix](https://github.com/AcademySoftwareFoundation/openexr/commit/699b4a62d5de9592d26f581a9cade89fdada7e6a) ([Ed Hanway](@ehanway-ilm@users.noreply.github.com), 2014-08-14) Tarball contents fix 

*  [Add dwa test images to dist (tarball) manifest. Also drop README.win32 from tarball. (Already removed from repo.)](https://github.com/AcademySoftwareFoundation/openexr/commit/cbac202a84b0b0bac0fcd92e5b5c8d634085329e) ([Ed Hanway](@ehanway@ilm.com), 2014-08-14) [New Cmake-centric instructions covering builds for Windows and other platforms to follow.] 

*  [Use LL for 64-bit literals](https://github.com/AcademySoftwareFoundation/openexr/commit/57ecf581d053f5cacf2e8fc3c024490e0bbe536f) ([Brendan Bolles](@brendan@fnordware.com), 2014-08-13) On a 32-bit architecture, these literals are too big for just a long, they need to be long long, otherwise I get an error in GCC.

## Version 2.2.2 (April 30, 2020)

This is a patch release that includes fixes for the following security vulnerabilities:

* [CVE-2020-11765](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11765) There is an off-by-one error in use of the ImfXdr.h read function by DwaCompressor::Classifier::ClasGsifier, leading to an out-of-bounds read.
* [CVE-2020-11764](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11764) There is an out-of-bounds write in copyIntoFrameBuffer in ImfMisc.cpp.
* [CVE-2020-11763](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11763) There is an std::vector out-of-bounds read and write, as demonstrated by ImfTileOffsets.cpp.
* [CVE-2020-11762](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11762) There is an out-of-bounds read and write in DwaCompressor::uncompress in ImfDwaCompressor.cpp when handling the UNKNOWN compression case.
* [CVE-2020-11761](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11761) There is an out-of-bounds read during Huffman uncompression, as demonstrated by FastHufDecoder::refill in ImfFastHuf.cpp.
* [CVE-2020-11760](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11760) There is an out-of-bounds read during RLE uncompression in rleUncompress in ImfRle.cpp.
* [CVE-2020-11759](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11759) Because of integer overflows in CompositeDeepScanLine::Data::handleDeepFrameBuffer and readSampleCountForLineBlock, an attacker can write to an out-of-bounds pointer.
* [CVE-2020-11758](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11758) There is an out-of-bounds read in ImfOptimizedPixelReading.h.

## Version 2.2.1 (November 30, 2017)

This maintenance release addresses the reported OpenEXR security
vulnerabilities, specifically:

* [CVE-2017-9110](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9110)
* [CVE-2017-9111](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9111)
* [CVE-2017-9112](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9112)
* [CVE-2017-9113](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9113)
* [CVE-2017-9114](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9114)
* [CVE-2017-9115](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9115)
* [CVE-2017-9116](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-9116)

## Version 2.2.0 (August 10, 2014)

This release includes the following components:

* OpenEXR: v2.2.0
* IlmBase: v2.2.0
* PyIlmBase: v2.2.0
* OpenEXR_Viewers: v2.2.0

This significant new features of this release include:

* **DreamWorks Lossy Compression** A new high quality, high performance
  lossy compression codec contributed by DreamWorks Animation. This
  codec allows control over variable lossiness to balance visual
  quality and file size. This contribution also includes performance
  improvements that speed up the PIZ codec.

* **IlmImfUtil** A new library intended to aid in development of image
  file manipulation utilities that support the many types of OpenEXR
  images.

This release also includes improvements to cross-platform build
support using CMake.

## Version 2.1.0 (November 25, 2013)

This release includes the following components (version locked):
* OpenEXR: v2.1.0
* IlmBase: v2.1.0
* PyIlmBase: v2.1.0
* OpenEXR_Viewers: v2.1.0

This release includes a refactoring of the optimised read paths for
RGBA data, optimisations for some of the python bindings to Imath,
improvements to the cmake build environment as well as additional
documentation describing deep data in more detail.

## Version 2.0.1 (July 11, 2013)

### Detailed Changes:

* Temporarily turning off optimisation code path (Piotr Stanczyk)
          
* Added additional tests for future optimisation refactoring (Piotr
	  Stanczyk / Peter Hillman)

* Fixes for StringVectors (Peter Hillman)

* Additional checks for type mismatches (Peter Hillman)
          
* Fix for Composite Deep Scanline (Brendan Bolles)

## Version 2.0 (April 9, 2013)

Industrial Light & Magic (ILM) and Weta Digital announce the release
of OpenEXR 2.0, the major version update of the open source high
dynamic range file format first introduced by ILM and maintained and
expanded by a number of key industry leaders including Weta Digital,
Pixar Animation Studios, Autodesk and others.

The release includes a number of new features that align with the
major version number increase. Amongst the major improvements are:

* **Deep Data support** - Pixels can now store a variable-length list of
  samples. The main rationale behind deep images is to enable the
  storage of multiple values at different depths for each
  pixel. OpenEXR 2.0 supports both hard-surface and volumetric
  representations for Deep Compositing workflows.

* **Multi-part Image Files** - With OpenEXR 2.0, files can now contain
  a number of separate, but related, data parts in one file. Access to
  any part is independent of the others, pixels from parts that are
  not required in the current operation don't need to be accessed,
  resulting in quicker read times when accessing only a subset of
  channels. The multipart interface also incorporates support for
  Stereo images where views are stored in separate parts. This makes
  stereo OpenEXR 2.0 files significantly faster to work with than the
  previous multiview support in OpenEXR.

* **Optimized pixel reading** - decoding RGB(A) scanline images has
  been accelerated on SSE processors providing a significant speedup
  when reading both old and new format images, including multipart and
  multiview files.

* **Namespacing** - The library introduces versioned namespaces to
  avoid conflicts between packages compiled with different versions of
  the library.

Although OpenEXR 2.0 is a major version update, files created by the
new library that don't exercise the new feature set are completely
backwards compatible with previous versions of the library. By using
the OpenEXR 2.0 library, performance improvements, namespace versions
and basic multi-part/deep reading support should be available to
applications without code modifications.

This code is designed to support Deep Compositing - a revolutionary
compositing workflow developed at Weta Digital that detached the
rendering of different elements in scene. In particular, changes in
one layer could be rendered separately without the need to re-render
other layers that would be required to handle holdouts in a
traditional comp workflow or sorting of layers in complex scenes with
elements moving in depth. Deep Compositing became the primary
compositing workflow on Avatar and has seen wide industry
adoption. The technique allows depth and color value to be stored for
every pixel in a scene allowing for much more efficient handling of
large complex scenes and greater freedom for artists to iterate.

True to the open source ethos, a number of companies contributed to
support the format and encourage adoption. Amongst others, Pixar
Animation Studios has contributed its DtexToExr converter to the
OpenEXR repository under a Microsoft Public License, which clears any
concerns about existing patents in the area, and Autodesk provided
performance optimizations geared towards real-time post-production
workflows.

Extensive effort has been put in ensuring all requirements were met to
help a wide adoption, staying true to the wide success of
OpenEXR. Many software companies were involved in the beta cycle to
insure support amongst a number of industry leading
applications. Numerous packages like SideFX's Houdini, Autodesk's
Maya, Solid Angle's Arnold renderer, Sony Pictures Imageworks' Open
Image IO have already announced their support of the format.

Open EXR 2.0 is an important step in the adoption of deep compositing
as it provides a consistent file format for deep data that is easy to
read and work with throughout a visual effects pipeline. The Foundry
has build OpenEXR 2.0 support into its Nuke Compositing application as
the base for the Deep Compositing workflows.

OpenEXR 2.0 is already in use at both Weta Digital and Industrial
Light & Magic. ILM took advantage of the new format on Marvel's The
Avengers and two highly anticipated summer 2013 releases, Pacific Rim
and The Lone Ranger. Recent examples of Weta Digital's use of the
format also include Marvel's Avengers as well as Prometheus and The
Hobbit. In addition, a large number of visual effects studios have
already integrated a deep workflow into their compositing pipelines or
are in the process of doing so including:, Sony Pictures Imageworks,
Pixar Animation Studios, Rhythm & Hues, Fuel and MPC.

In addition to visual effects, the new additions to the format, means
that depth data can also be assigned to two-dimensional data for a use
in many design fields including, architecture, graphic design,
automotive and product prototyping.

### Detailed Changes:

* Updated Documentation
	   (Peter Hillman)
* Updated Namespacing mechanism
	   (Piotr Stanczyk)
* Fixes for succd & predd
	   (Peter Hillman)
* Fixes for FPE control registers
	   (Piotr Stanczyk)
* Additional checks and tests on DeepImages, scanlines and tiles
	   (Peter Hillman)
* Folded in Autodesk read optimisations for RGB(A) files
	  (Pascal Jette, Peter Hillman)
* Updated the bootstrap scripts to use libtoolize if glibtoolize isn't available on darwin. 
	  (Nick Rasmussen)
* Numerous minor fixes, missing includes etc

## Version 2.0.0.beta.1 (June 15, 2012)

Development of OpenEXR v2 has been undertaken in a collaborative
environment (cf. previous github announcement) comprised of Industrial
Light & Magic, Weta Digital as well as a number of other contributors.

Some of the new features included in the Beta.1 release of OpenEXR v2
are:

* **Deep Data** Pixels can now store a variable length list of
  samples. The main rationale behind deep-images is to have multiple
  values at different depths for each pixel. OpenEXR v2 supports both
  hard surface and volumetric representation requirements for deep
  compositing workflows.

* **Multi-part image files** With OpenEXR v2, files can now contain a
    number of separate, but related, images in one file. Access to any
    part is independent of the others; in particular, no access of
    data need take place for unrequested parts.

In addition, OpenEXR v2 also contains platform independent mechanisms
for handling co-existing library version conflicts in the same process
space. (Currently implemented in IlmImf)

Finally, a reminder that this is a Beta release and potentially
incompatible changes may be introduced in future releases prior to the
v2.0.0 production version.

Please read the separate file for v2 additions and changes.

### Detailed Changes:

* Added git specific files 
	  (Piotr Stanczyk)
* Updated the so verison to 20
	  (Piotr Stanczyk)
* Initial use of the CMake build system 
	  (Nicholas Yue)

## Version 1.7.1 (July 31, 2012)

This release includes the following components:

* OpenEXR: v1.7.1
* IlmBase: v1.0.3
* PyIlmBase: v1.0.0 (introduces a Boost dependency)
* OpenEXR_Viewers: v1.0.2

Of particular note is the introduction of PyIlmBase. This module forms
a comprehensive set of python bindings to the IlmBase module.

In addition, contained in this release is a number of additions to
Imath as well as a minor tweak to Imath::Frustrum (for better support
for Windows platforms) as well as other minor fixes, including
correction for soname version of IlmImf.

## Version 1.7.0 (July 23, 2010)

This release includes support for stereoscopic images, please see the
adjoining documentation in the ``MultiViewOpenEXR.pdf``. (Many thanks
to Weta Digital for their contribution.) In addition, we added support
for targeting 64 bit Windows, fixes for buffer overruns and a number
of other minor fixes, additions and optimisations. Please see the
Changelog files for more detailed information.

### Bugs

This release addresses the following security vulnerabilities:

* [CVE-2009-1720](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2009-1720)
* [CVE-2009-1721](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2009-1721)
* [CVE-2009-1722](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2009-1722)

### Detailed Changes:

* Added support for targetting builds on 64bit Windows and minimising
  number of compiler warnings on Windows. Thanks to Ger Hobbelt for
  his contributions to CreateDLL.  (Ji Hun Yu)
          
* Added new atttribute types (Florian Kainz):
  * **M33dAttribute** 3x3 double-precision matrix
  * **M44dAttribute** 4x4 double-precision matrix
  * **V2d** 2D double-precision vector
  * **V3d** 3D double-precision vector  
	  
* Bug fix: crash when reading a damaged image file (found by Apple).
  An exception thrown inside the PIZ Huffman decoder bypasses
  initialization of an array of pointers.  The uninitialized pointers
  are later passed to operator delete.  (Florian Kainz)

* Bug fix: crash when reading a damaged image file (found by Apple).
  Computing the size of input certain buffers may overflow and wrap
  around to a small number, later causing writes beyond the end of the
  buffer.  (Florian Kainz)

* In the "Technical Introduction" document, added Premultiplied
  vs. Un-Premulitiplied Color section: states explicitly that pixels
  with zero alpha and non-zero RGB are allowed, points out that
  preserving such a pixel can be a problem in application programs
  with un-premultiplied internal image representations.  (Florian
  Kainz)

* exrenvmap improvements:

  - New command line flags set the type of the input image to
    latitude-longitude map or cube-face map, overriding the envmap
    attribute in the input file header.

  - Cube-face maps can now be assembled from or split into six
    square sub-images.

  - Converting a cube-face map into a new cube-face map with the same
    face size copies the image instead of resampling it.  This avoids
    blurring when a cube-face map is assembled from or split into
    sub-images.  (Florian Kainz)

* Updated standard chromaticities in ImfAcesFile.cpp to match final
  ACES (Academy Color Encoding Specification) document.  (Florian
  Kainz)

* Added worldToCamera and worldToNDC matrices to
  ImfStandardAttributes.h (Florian Kainz)

* Increased the maximum length of attribute and channel names from 31
  to 255 characters.  For files that do contain names longer than 31
  characters, a new LONG_NAMES_FLAG in the fil version number is set.
  This flag causes older versions of the IlmImf library (1.6.1 and
  earlier) to reject files with long names.  Without the flag, older
  library versions would mis-interpret files with long names as
  broken.  (Florian Kainz)

* Reading luminance/chroma-encoded files via the RGBA interface is
  faster: buffer padding avoids cache thrashing for certain image
  sizes, redundant calls to saturation() have been eliminated.  (Mike
  Wall)

* Added "hemispherical blur" option to exrenvmap.  (Florian Kainz)

* Added experimental version of I/O classes for ACES file format
  (restricted OpenEXR format with special primaries and white point);
  added exr2aces file converter.  (Florian Kainz)

* Added new constructors to classes Imf::RgbaInputFile and
  Imf::TiledRgbaInputFile.  The new constructors have a layerName
  parameter, which allows the caller to specify which layer of a
  multi-layer or multi-view image will be read.  (Florian Kainz)

* A number of member functions in classes Imf::Header,
  Imf::ChannelList and Imf::FrameBuffer have parameters of type "const
  char *".  Added equivalent functions that take "const std::string &"
  parameters.  (Florian Kainz)

* Added library support for Weta Digital multi-view images:
  StringVector attribute type, multiView standard attribute of type
  StringVector, utility functions related to grouping channels into
  separate views.  (Peter Hillman, Florian Kainz)

## Version 1.6.1 (October 22, 2007)

This release fixes a buffer overrun in OpenEXR and a Windows build
problem in CTL, and it removes a few unnecessary files from the
.tar.gz packages.

### Detailed Changes:

* Removed Windows .suo files from distribution.  (Eric Wimmer)

* Bug fix: crashes, memory leaks and file descriptor leaks when
  reading damaged image files (some reported by Apple, others found by
  running IlmImfFuzzTest).  (Florian Kainz)
          
* Added new IlmImfFuzzTest program to test how resilient the IlmImf
  library is with respect broken input files: the program first
  damages OpenEXR files by partially overwriting them with random
  data; then it tries to read the damaged files.  If all goes well,
  the program doesn't crash.  (Florian Kainz)

## Version 1.6.0 (August 3, 2007)

OpenEXR 1.6.0:

* Reduced generational loss in B44- and B44A-compressed images.

* Added B44A compression. This is a variation of B44, but with a
  better compression ratio for images with large uniform areas, such
  as in an alpha channel.

* Bug fixes.

CTL 1.4.0:

* Added new functions to the CTL standard library: 3x3 matrix support,
  1D lookup tables with cubic interpolation.

* Added new "ctlversion" statement to the language.

* Bug fixes.

OpenEXR_CTL 1.0.0:

* Applying CTL transforms to a frame buffer is multi-threaded.
Bug fixes.

OpenEXR_Viewers 1.0.0:

* Implemented new naming conventions for CTL parameters.

IlmBase 1.0.0:

* Half now implements "round to nearest even" mode.

### Detailed Changes:

* Bumped DSO version number to 6.0 (Florian Kainz)

* Added new standard attributes related to color rendering with CTL
  (Color Transformation Language): renderingTransform,
  lookModTransform and adoptedNeutral.  (Florian Kainz)

* Bug fix: for pixels with luminance near HALF_MIN, conversion from
  RGB to luminance/chroma produces NaNs and infinities (Florian Kainz)
          
* Bug fix: excessive desaturation of small details with certain colors
  after repeatedly loading and saving luminance/chroma encoded images
  with B44 compression.  (Florian Kainz)

* Added B44A compression, a minor variation of B44: in most cases, the
  compression ratio is 2.28:1, the same as with B44, but in uniform
  image areas where all pixels have the same value, the compression
  ratio increases to 10.66:1.  Uniform areas occur, for example, in an
  image's alpha channel, which typically contains large patches that
  are solid black or white, or in computer- generated images with a
  black background.  (Florian Kainz)

* Added flag to configure.ac to enable or disable use of large auto
  arrays in the IlmImf library.  Default is "enable" for Linux,
  "disable" for everything else.  (Darby Johnston, Florian Kainz)

* corrected version number on dso's (libtool) - now 5.0

* Separated ILMBASE_LDFLAGS and ILMBASE_LIBS so that test programs can
  link with static libraries properly

* eliminated some warning messages during install (Andrew Kunz)
	
## Version 1.5.0 (December 15, 2006)

The new version includes several significant changes:

* OpenEXR supports a new image compression method, called B44. It has
  a fixed compression rate of 2.28:1, or 4.57:1 if used in combination
  with luminance/chroma encoding. B44-compressed images can be
  uncompressed fast enough to support real-time playback of image
  sequences.

* The new playexr program plays back moving image sequences. Playexr
  is multi-threaded and utilizes the threading capabilities of the
  IlmImf library that were introduced in OpenEXR 1.3.0. The program
  plays back B44-compressed images with fairly high-resolution in real
  time on commodity hardware.

* The playexr program and a new version of the existing exrdisplay
  image viewer both support color rendering via color transforms
  written in the new Color Transformation Language or CTL. CTL is not
  part of OpenEXR; it will be released separately. CTL support in
  playexr and exrdisplay is optional; the programs can be built and
  will run without CTL.

* In preparation for the release of CTL, OpenEXR has been split into
  three separate packages:

  * IlmBase 0.9.0 includes the Half, Iex, Imath and IlmThread libraries

  * OpenEXR 1.5.0 includes the IlmImf library, programming examples and utility programs such as exrheader or exrenvmap

  * OpenEXRViewers 0.9.0 includes the playexr and exrdisplay programs

* The "Technical Introduction to OpenEXR" document now includes a
  recommendation for storing CIE XYZ pixel data in OpenEXR files.

* A new "OpenEXR Image Viewing Software" document describes the
  playexr and exrdisplay programs. It briefly explains real-time
  playback and color rendering, and includes recommendations for
  testing if other image viewing software displays OpenEXR images
  correctly.

* The OpenEXR sample image set now includes B44-compressed files and
  files with CIE XYZ pixel data.

### Detailed Changes:  

* reorganized packaging of OpenEXR libraries to facilitate integration
  with CTL.  Now this library depends on the library IlmBase.  Some
  functionality has been moved into OpenEXR_Viewers, which depends on
  two other libraries, CTL and OpenEXR_CTL.  Note: previously there
  were separate releases of OpenEXR-related plugins for Renderman,
  Shake and Photoshop.  OpenEXR is supported natively by Rendermand
  and Photoshop, so these plugins will not be supported for this or
  future versions of OpenEXR.  (Andrew Kunz)

* New build scripts for Linux/Unix (Andrew Kunz)

* New Windows project files and build scripts (Kimball Thurston)

* float-to-half conversion now preserves the sign of float zeroes and
  of floats that are so small that they become half zeroes.  (Florian
  Kainz)

* Bug fix: Imath::Frustum<T>::planes() returns incorrect planes if the
  frustum is orthogonal.  (Philip Hubbard)

* added new framesPerSecond optional standard attribute (Florian
  Kainz)

* Imath cleanup:

  - Rewrote function Imath::Quat<T>::setRotation() to make it
    numerically more accurate, added confidence tests

  - Rewrote function Imath::Quat<T>::slerp() using Don Hatch's method,
    which is numerically more accurate, added confidence tests.

  - Rewrote functions Imath::closestPoints(), Imath::intersect(),
    added confidence tests.

  - Removed broken function Imath::nearestPointOnTriangle().

  - Rewrote Imath::drand48(), Imath::lrand48(), etc. to make them
    functionally identical with the Unix/Linux versions of drand48(),
    lrand48() and friends.

  - Replaced redundant definitions of Int64 in Imath and IlmImf with a
    single definition in ImathInt64.h.  (Florian Kainz)

* exrdisplay: if the file's and the display's RGB chromaticities
  differ, the pixels RGB values are transformed from the file's to the
  display's RGB space.  (Florian Kainz)

* Added new lossy B44 compression method.  HALF channels are
  compressed with a fixed ratio of 2.28:1.  UINT and FLOAT channels
  are stored verbatim, without compression.  (Florian Kainz)

## Version 1.4.0a (August 9, 2006)

* Fixed the ReleaseDLL targets for Visual Studio 2003.  (Barnaby Robson)
	
## Version 1.4.0 (August 2, 2006)	

 This is the next major production-ready release of OpenEXR and offers
 full compatibility with our last production release, which was
 1.2.2. This version obsoletes versions 1.3.x, which were test
 versions for 1.4.0. If you have been using 1.3.x, please upgrade to
 1.4.0.

* Production release.

* Bug Fix: calling setFrameBuffer() for every scan line while reading
  a tiled file through the scan line API returns bad pixel data. (Paul
  Schneider, Florian Kainz)

## Version 1.3.1 (June 14, 2006)

* Fixed the ReleaseDLL targets for Visual Studio 2005.  (Nick Porcino, Drew Hess)

* Fixes/enhancements for createDLL.  (Nick Porcino)
	
## Version 1.3.0 (June 8, 2006)

This is a test release. The major new feature in this version is
support for multithreaded file I/O. We've been testing the threaded
code internally at ILM for a few months, and we have not encountered
any bugs, but we'd like to get some feedback from others before we
release the production version.

Here's a summary of the changes since version 1.2.2:

* Support for multithreaded file reading and writing.

* Support for Intel-based OS X systems.

* Support for Visual Studio 2005.

* Better handling of **PLATFORM_** and **HAVE_** macros.

* Updated documentation.

* Bug fixes related to handling of incomplete and damaged files.

* Numerous bug fixes and cleanups to the autoconf-based build system.

* Removed support for the following configurations that were
  previously supported. Some of these configurations may happen to
  continue to function, but we can't help you if they don't, largely
  because we don't have any way to test them:

  * IRIX
  * OSF1
  * SunOS
  * OS X versions prior to 10.3.
  * gcc on any platform prior to version 3.3

### Detailed Changes:

* Removed openexr.spec file, it's out of date and broken to boot.
 (Drew Hess)
          
* Support for Visual Studio 2005.  (Drew Hess, Nick Porcino)

* When compiling against OpenEXR headers on Windows, you no longer
  need to define any **HAVE_** or **PLATFORM_** macros in your
  projects.  If you are using any OpenEXR DLLs, however, you must
  define OPENEXR_DLL in your project's preprocessor directives.  (Drew
  Hess)

* Many fixes to the Windows VC7 build system.  (Drew Hess, Nick
  Porcino)

* Support for building universal binaries on OS X 10.4.  (Drew Hess,
Paul Schneider)
          
* Minor configure.ac fix to accomodate OS X's automake.  (Drew Hess)
          
* Removed CPU-specific optimizations from configure.ac, autoconf's
	  guess at the CPU type isn't very useful, anyway.  Closes
	  #13429.  (Drew Hess)
          
* Fixed quoting for tests in configure.ac.  Closes #13428.  (Drew
  Hess)
          
* Use host specification instead of target in configure.ac.  Closes
  #13427.  (Drew Hess)

* Fix use of AC_ARG_ENABLE in configure.ac.  Closes #13426.  (Drew
Hess)

* Removed workaround for OS X istream::read bug.  (Drew Hess)
          
* Added pthread support to OpenEXR pkg-config file.  (Drew Hess)
          
* Added -no-undefined to LDFLAGS and required libs to LIBADD for
  library projects with other library dependencies, per Rex Dieter's
  patch.  (Drew Hess)
          
* **HAVE_** macros are now defined in the OpenEXRConfig.h header file
  instead of via compiler flags.  There are a handful of public
  headers which rely on the value of these macros, and projects
  including these headers have previously needed to define the same
  macros and values as used by OpenEXR's 'configure', which is bad
  form.  Now 'configure' writes these values to the OpenEXRConfig.h
  header file, which is included by any OpenEXR source files that need
  these macros.  This method of specifying **HAVE_** macros guarantees
  that projects will get the proper settings without needing to add
  compile- time flags to accomodate OpenEXR.  Note that this isn't
  implemented properly for Windows yet.  (Drew Hess)

* Platform cleanups:

  - No more support for IRIX or OSF1.

  - No more explicit support for SunOS, because we have no way to
    verify that it's working.  I suspect that newish versions of SunOS
    will just work out of the box, but let me know if not.

  - No more **PLATFORM_** macros (vestiges of the ILM internal build
    system).  PLATFORM_DARWIN_PPC is replaced by HAVE_DARWIN.
    PLATFORM_REDHAT_IA32 (which was only used in IlmImfTest) is
    replaced by HAVE_LINUX_PROCFS.

  - OS X 10.4, which is the minimum version we're going to support
    with this version, appears to have support for nrand48 and
    friends, so no need to use the Imath-supplied version of them
    anymore.  (Drew Hess)

* No more PLATFORM_WINDOWS or PLATFORM_WIN32, replace with proper
  standard Windows macros.  (Drew Hess)

* Remove support for gcc 2.95, no longer supported.  (Drew Hess)

* Eliminate HAVE_IOS_BASE macro, OpenEXR now requires support for
  ios_base.  (Drew Hess)

* Eliminate HAVE_STL_LIMITS macro, OpenEXR now requires the ISO C++
  <limits> header.  (Drew Hess)

* Use double quote-style include dirctives for OpenEXR
  includes.  (Drew Hess)

* Added a document that gives an overview of the on-disk
  layout of OpenEXR files (Florian Kainz)

* Added sections on layers and on memory-mapped file input
  to the documentation.  (Florian Kainz)

* Bug fix: reading an incomplete file causes a deadlock while
  waiting on a semaphore.  (Florian Kainz)

* Updated documentation (ReadingAndWritingImageFiles.sxw) and sample
  code (IlmImfExamples): Added a section about multi-threading,
  updated section on thread-safety, changed documentation and sample
  code to use readTiles()/writeTiles() instead of
  readTile()/writeTile() where possible, mentioned that environment
  maps contain redundant pixels, updated section on testing if a file
  is an OpenEXR file.  (Florian Kainz)

* Multi-threading bug fixes (exceptions could be thrown multiple
  times, some operations were not thread safe), updated some comments,
  added comments, more multithreaded testing.  (Florian Kainz)

* Added multi-threading support: multiple threads
  cooperate to read or write a single OpenEXR file.
  (Wojciech Jarosz)

* Added operator== and operator!= to Imath::Frustum. (Andre Mazzone)

* Bug fix: Reading a PIZ-compressed file with an invalid Huffman code
  table caused crashes by indexing off the end of an array.  (Florian
  Kainz)

## Version 1.2.2 (March 15, 2005)

This is a relatively minor update to the project, with the following changes:

* New build system for Windows; support for DLLs.

* Switched documentation from HTML to PDF format.

* IlmImf: support for image layers in ChannelList.

* IlmImf: added isComplete() method to file classes to check whether a file is complete.

* IlmImf: exposed staticInitialize() in ImfHeader.h in order to allow
  thread-safe library initialization in multithreaded applications.

* IlmImf: New "time code" standard attribute.

* exrdisplay: support for displaying wrap-around texture map images.

* exrmaketiled: can now specify wrap mode.

* IlmImf: New "wrapmodes" standard attribute to indicate extrapolation
  mode for mipmaps and ripmaps.

* IlmImf: New "key code" standard attribute to identify motion picture
  film frames.

* Imath: Removed TMatrix<T> classes; these classes are still under
  development and are too difficult to keep in sync with OpenEXR CVS.

### Detailed Changes:


* Updated README to remove option for building with Visual C++ 6.0.
	  (Drew Hess)

* Some older versions of gcc don't support a full iomanip
	  implemenation; check for this during configuration. 
	  (Drew Hess)

* Install PDF versions of documentation, remove old/out-of-date
	  HTML documentation.  (Florian Kainz)

* Removed vc/vc6 directory; Visual C++ 6.0 is no longer
	  supported.  (Drew Hess)

* Updated README.win32 with details of new build system.
	  (Florian Kainz, Drew Hess)

* New build system for Windows / Visual C++ 7 builds both
	  static libraries and DLLs.
	  (Nick Porcino)

* Removed Imath::TMatrix<T> and related classes, which are not
	  used anywhere in OpenEXR.
	  (Florian Kainz)

* Added minimal support for "image layers" to class Imf::ChannelList
	  (Florian Kainz)

* Added new isComplete() method to InputFile, TiledInputFile
	  etc., that checks if a file is complete or if any pixels
	  are missing (for example, because writing the file was
	  aborted prematurely).
	  (Florian Kainz)

* Exposed staticInitialize() function in ImfHeader.h in order
	  to allow thread-safe library initialization in multithreaded
	  programs.
	  (Florian Kainz)

* Added a new "time code" attribute
	  (Florian Kainz)

* exrmaketiled: when a MIPMAP_LEVELS or RIPMAP_LEVELS image
	  is produced, low-pass filtering takes samples outside the
	  image's data window.  This requires extrapolating the image.
	  The user can now specify how the image is extrapolated
	  horizontally and vertically (image is surrounded by black /
	  outermost row of pixels repeats / entire image repeats /
	  entire image repeats, every other copy is a mirror image).
	  exrdisplay: added option to swap the top and botton half,
	  and the left and right half of an image, so that the image's
	  four corners end up in the center.  This is useful for checking
	  the seams of wrap-around texture map images.
	  IlmImf library: Added new "wrapmodes" standard attribute
	  to indicate the extrapolation mode for MIPMAP_LEVELS and
	  RIPMAP_LEVELS images.
	  (Florian Kainz)

* Added a new "key code" attribute to identify motion picture
	  film frames.
	  (Florian Kainz)

* Removed #include <Iex.h> from ImfAttribute.h, ImfHeader.h
	  and ImfXdr.h so that including header files such as
	  ImfInputFile.h no longer defines ASSERT and THROW macros,
	  which may conflict with similar macros defined by
	  application programs.
	  (Florian Kainz)

* Converted HTML documentation to OpenOffice format to
	  make maintaining the documents easier:
	      api.html -> ReadingAndWritingImageFiles.sxw
	      details.html -> TechnicalIntroduction.sxw
	  (Florian Kainz)

## Version 1.2.1 (June 6, 2004)

This is a fairly minor release, mostly just a few tweaks, a few bug
fixes, and some new documentation. Here are the most important
changes:

* reduced memory footprint of exrenvmap and exrmaketiled utilities.

* IlmImf: new helper functions to determine whether a file is an OpenEXR file, and whether it's scanline- or tile-based.

* IlmImf: bug fix for PXR24 compression with ySampling != 1.

* Better support for gcc 3.4.

* Warning cleanups in Visual C++.

### Detailed Changes:

* exrenvmap and exrmaketiled use slightly less memory
	  (Florian Kainz)

* Added functions to IlmImf for quickly testing if a file
	  is an OpenEXR file, and whether the file is scan-line
	  based or tiled. (Florian Kainz)

* Added preview image examples to IlmImfExamples.  Added
	  description of preview images and environment maps to
	  docs/api.html (Florian Kainz)

* Bug fix: PXR24 compression did not work properly for channels
	  with ySampling != 1.
	  (Florian Kainz)

* Made ``template <class T>`` become ``template <class S, class T>`` for 
          the ``transform(ObjectS, ObjectT)`` methods. This was done to allow
          for differing templated objects to be passed in e.g.  say a 
          ``Box<Vec3<S>>`` and a ``Matrix44<T>``, where S=float and T=double.
          (Jeff Yost, Arkell Rasiah)

* New method Matrix44::setTheMatrix(). Used for assigning a 
          M44f to a M44d. (Jeff Yost, Arkell Rasiah)

* Added convenience Color typedefs for half versions of Color3
          and Color4. Note the Makefile.am for both Imath and ImathTest
          have been updated with -I and/or -L pathing to Half.
          (Max Chen, Arkell Rasiah)

* Methods equalWithAbsError() and equalWithRelError() are now
          declared as const. (Colette Mullenhoff, Arkell Rasiah)

* Fixes for gcc34. Mainly typename/template/using/this syntax
          correctness changes. (Nick Ramussen, Arkell Rasiah)

* Added Custom low-level file I/O examples to IlmImfExamples
	  and to the docs/api.html document.  (Florian Kainz)

* Eliminated most warnings messages when OpenEXR is compiled
	  with Visual C++.  The OpenEXR code uses lots of (intentional
	  and unintended) implicit type conversions.  By default, Visual
	  C++ warns about almost all of them.  Most implicit conversions
	  have been removed from the .h files, so that including them
	  should not generate warnings even at warning level 3.  Most
	  .cpp files are now compiled with warning level 1.
	  (Florian Kainz)

## Version 1.2.0 (May 11, 2004)

OpenEXR 1.2.0 is now available. This is the first official,
production-ready release since OpenEXR 1.0.7. If you have been using
the development 1.1 series, please switch to 1.2.0 as soon as
possible. We believe that OpenEXR 1.2.0 is ready for use in shipping
applications. We have been using it in production at ILM for several
months now with no problems. There are quite a few major new features
in the 1.2 series as compared to the original 1.0 series:

* Support for tiled images, including mipmaps and ripmaps. Note that
  software based on the 1.0 series cannot read or write tiled
  images. However, simply by recompiling your software against the 1.2
  release, any code that reads scanline images can read tiled images,
  too.

* A new Pxr24 compressor, contributed by Pixar Animation
  Studios. Values produced by the Pxr24 compressor provide the same
  range as 32-bit floating-point numbers with slightly less precision,
  and compress quite a bit better. The Pxr24 compressor stores UINT
  and HALF channels losslessly, and for these data types performs
  similarly to the ZIP compressor.

* OpenEXR now supports high dynamic-range YCA (luminance/chroma/alpha)
  images with subsampled chroma channels. These files are supported
  via the RGBA convenience interface, so that data is presented to the
  application as RGB(A) but stored in the file as YC(A). OpenEXR also
  supports Y and YA (black-and-white/black-and-white with alpha)
  images.

* An abstracted file I/O interface, so that you can use OpenEXR with
  interfaces other than C++'s iostreams.

* Several new utilities for manipulating tiled image files.

### Detailed Changes:

* Production-ready release.

* Disable long double warnings on OS X.  (Drew Hess)

* Add new source files to VC7 IlmImfDll target.  (Drew Hess)

* Iex: change the way that APPEND_EXC and REPLACE_EXC modify
	  their what() string to work around an issue with Visual C++
	  7.1.  (Florian Kainz, Nick Porcino)

* Bumped OpenEXR version to 1.2 and .so versions to 2.0.0 in
	  preparation for the release.  (Drew Hess)

* Imath: fixed ImathTMatrix.h to work with gcc 3.4.  (Drew Hess)

* Another quoting fix in openexr.m4.  (Drew Hess)

* Quoting fix in acinclude.m4 for automake 1.8.  (Brad Hards)

* Imath: put inline at beginning of declaration in ImathMatrix.h
	  to fix a warning.  (Ken McGaugh)

* Imath: made Vec equalWithError () methods const.

* Cleaned up compile-time Win32 support.  (Florian Kainz)

* Bug fix: Reading a particular broken PIZ-compressed file
	  caused crashes by indexing off the end of an array.
	  (Florian Kainz)

## Version 1.1.1 (March 27, 2004)

OpenEXR 1.1.1 is now available. This another development release. We
expect to release a stable version, 1.2, around the end of
April. Version 1.1.1 includes support for PXR24 compression, and for
high-dynamic-range luminance/chroma images with subsampled chroma
channels. Version 1.1.1 also fixes a bug in the 1.1.0 tiled file
format.

### Detailed Changes:

* Half: operator= and variants now return by reference rather
	  than by value.  This brings half into conformance with
	  built-in types.  (Drew Hess)

* Half: remove copy constructor, let compiler supply its
	  own.  This improves performance up to 25% on some
	  expressions using half.  (Drew Hess)

* configure: don't try to be fancy with CXXFLAGS, just use
	  what the user supplies or let configure choose a sensible
	  default if CXXFLAGS is not defined.

* IlmImf: fixed a bug in reading scanline files on big-endian
          architectures.  (Drew Hess)

* exrmaketiled: Added an option to select compression type.
	  (Florian Kainz)

* exrenvmap: Added an option to select compression type.
	  (Florian Kainz)

* exrdisplay: Added some new command-line options.  (Florian Kainz)

* IlmImf: Added Pixar's new "slightly lossy" image compression
	  method.  The new method, named PXR24, preserves HALF and
	  UINT data without loss, but FLOAT pixels are converted to
	  a 24-bit representation.  PXR24 appears to compress
	  FLOAT depth buffers very well without losing much accuracy.
	  (Loren Carpenter, Florian Kainz)

* Changed top-level LICENSE file to allow for other copyright
	  holders for individual files.

* IlmImf: TILED FILE FORMAT CHANGE.  TiledOutputFile was
	  incorrectly interleaving channels and scanlines before
	  passing pixel data to a compressor.  The lossless compressors
	  still work, but lossy compressors do not.  Fix the bug by
	  interleaving channels and scanlines in tiled files in the
	  same way as ScanLineOutputFile does.  Programs compiled with
	  the new version of IlmImf cannot read tiled images produced
	  with version 1.1.0.  (Florian Kainz)

* IlmImf: ImfXdr.h fix for 64-bit architectures.  (Florian Kainz)

* IlmImf: OpenEXR now supports YCA (luminance/chroma/alpha)
	  images with subsampled chroma channels.  When an image
	  is written with the RGBA convenience interface, selecting
	  WRITE_YCA instead of WRITE_RGBA causes the library to
	  convert the pixels to YCA format.  If WRITE_Y is selected,
	  only luminance is stored in the file (for black and white
	  images).  When an image file is read with the RGBA convenience
	  interface, YCA data are automatically converted back to RGBA.
	  (Florian Kainz)

* IlmImf: speed up reading tiled files as scan lines.
	  (Florian Kainz)

* Half:  Fixed subtle bug in Half where signaling float NaNs
	  were being converted to inf in half.  (Florian Kainz)

* gcc 3.3 compiler warning cleanups.  (various)

* Imath: ImathEuler.h fixes for gcc 3.4.  (Garrick Meeker)
	
## Version 1.1.0 (February 6, 2004)

 OpenEXR 1.1.0 is now available. This is a major new release with
 support for tiled images, multi-resolution files (mip/ripmaps),
 environment maps, and abstracted file I/O. We've also released a new
 set of images that demonstrate these features, and updated the
 CodeWarrior project and Photoshop plugins for this release. See the
 downloads section for the source code and the new images.

### Detailed Changes:

* Added new targets to Visual C++ .NET 2003 project
	  for exrmaketiled, exrenvmap, exrmakepreview, and exrstdattr.
	  (Drew Hess)

* A few assorted Win32 fixes for Imath.  (Drew Hess)

* GNU autoconf builds now produce versioned libraries.
	  This release is 1:0:0.  (Drew Hess)

* Fixes for Visual C++ .NET 2003.  (Paul Schneider)

* Updated Visual C++ zlib project file to zlib 1.2.1.
	  (Drew Hess)

* exrdisplay: Fixed fragment shader version.  (Drew Hess)

* *Test: Fixed some compiler issues.  (Drew Hess)

* Imath: Handle "restrict" keyword properly.  (Drew Hess)

* IlmImfExamples: Updated to latest versions of example
	  source code, includes tiling and multi-res images.
	  (Florian Kainz)

* exrmakepreview: A new utility to create preview images.
	  (Florian Kainz)

* exrenvmap: A new utility to create OpenEXR environment
	  maps.  (Florian Kainz)

* exrstdattr: A new utility to modify standard 
	  attributes.  (Florian Kainz)

* Updated exrheader to print level rounding mode and
	  preview image size.  (Florian Kainz)

* Updated exrmaketiled to use level rounding mode.
	  (Florian Kainz)

* IlmImf: Changed the orientation of lat-long envmaps to
	  match typical panoramic camera setups.  (Florian Kainz)

* IlmImf: Fixed a bug where partially-completed files with
	  DECREASING_Y could not be read.  (Florian Kainz)

* IlmImf: Added support for selectable rounding mode (up/down)
	  when generating multiresolution files.  (Florian Kainz)

* exrdisplay: Support for tiled images, mip/ripmaps, preview
	  images, and display windows.  (Florian Kainz, Drew Hess)

* exrmaketiled: A new utility which generates tiled
	  versions of OpenEXR images.  (Florian Kainz)

* IlmImf: Changed Imf::VERSION to Imf::EXR_VERSION to
	  work around problems with autoconf VERSION macro
	  conflict.  (Drew Hess)

* exrheader: Support for tiles, mipmaps, environment
	  maps.  (Florian Kainz)

* IlmImf: Environment map support.  (Florian Kainz)

* IlmImf: Abstracted stream I/O support.  (Florian Kainz)

* IlmImf: Support for tiled and mip/ripmapped files;
	  requires new file format.  (Wojciech Jarosz, Florian Kainz)

* Imath: **TMatrix**, generic 2D matricies and algorithms.
	  (Francesco Callari)

* Imath: major quaternions cleanup.  (Cary Phillips)

* Imath: added GLBegin, GLPushAttrib, GLPushMatrix objects
	  for automatic cleanup on exceptions.  (Cary Phillips)

* Imath: removed implicit scalar->vector promotions and vector
	  comparisons.  (Nick Rasmussen)
	
## Version 1.0.7 (January 7, 2004)

OpenEXR 1.0.7 is now available. In addition to some bug fixes, this
version adds support for some new standard attributes, such as primary
and white point chromaticities, lens aperture, film speed, image
acquisition time and place, and more. If you want to use these new
attributes in your applications, see the ImfStandardAttributes.h
header file for documentation.

Our project hosting site, Savannah, is still recovering from a
compromise last month, so in the meantime, we're hosting file
downloads here. Some of the files are not currently available, but
we're working to restore them.

### Detailed Changes:

* Fixed a typo in one of the IlmImfTest tests. (Paul Schneider)

* Fixed a bug in exrdisplay that causes the image to display
	  as all black if there's a NaN or infinity in an OpenEXR
	  image. (Florian Kainz)

* Updated exrheader per recent changes to IlmImf library.
	  (Florian Kainz)

* Changed an errant float to a T in ImathFrame.h nextFrame().
	  (Cary Phillips)

* Support for new "optional standard" attributes
	  (chromaticities, luminance, comments, etc.).
	  (Florian Kainz, Greg Ward, Joseph Goldstone)

* Fixed a buffer overrun in ImfOpaqueAttribute. (Paul Schneider)

* Added new function, isImfMagic (). (Florian Kainz)
	
## Version 1.0.6:

* Added README.win32 to disted files.

* Fixed OpenEXR.pc.in pkg-config file, OpenEXR now works
	  with pkg-config.

* Random fixes to readme files for new release.

* Fixed openexr.m4, now looks in /usr by default.

* Added Visual Studio .NET 2003 "solution."

* Fixes for Visual Studio .NET 2003 w/ Microsoft C++ compiler.
	  (Various)

* Random Imath fixes and enhancements.  Note that 
	  extractSHRT now takes an additional optional
          argument, see ImathMatrixAlgo.h for details.  (Various)

* Added Wojciech Jarosz to AUTHORS file.

* Added test cases for uncompressed case, preview images,
	  frame buffer type conversion.  (Wojciech Jarosz,
	  Florian Kainz)

* Fix a bug in IlmImf where uncompressed data doesn't get
	  read/written correctly.  (Wojciech Jarosz)

* Added support for preview images and preview image
	  attributes (thumbnail images) in IlmImf.  (Florian Kainz)

* Added support for automatic frame buffer type conversion
	  in IlmImf.  (Florian Kainz)

* Cleaned up some compile-time checks.

* Added HalfTest unit tests.

* [exrdisplay] Download half framebuffer to texture memory 
	  instead of converting to float first.  Requires latest
	  Nvidia drivers.

## Version 1.0.5 (April 3, 2003)

Industrial Light & Magic has released the source code for an OpenEXR
display driver for Pixar's Renderman. This display driver is covered
under the OpenEXR free software license. See the downloads section for
the source code.

### Detailed Changes:

* Fixed IlmImf.dll to use static runtime libs (Andreas).

* Added exrheader project to Visual Studio 6.0 workspace.

* Added some example code showing how to use the IlmImf library.
	  (Florian)

* Use DLL runtime libs for Win32 libraries rather than static
	  runtime libs.

* Add an exrdisplay_fragshader project to the Visual Studio 6.0
	  workspace to enable fragment shaders in Win32.

* Add an IlmImfDll project to the Visual Studio 6.0 workspace.

* In Win32, export the ImfCRgbaFile C interface via a DLL so
	  that Visual C++ 6.0 users can link against an Intel-compiled
	  IlmImf.  (Andreas Kahler)

* Use auto_ptr in ImfAutoArray on Win32, it doesn't like large 
	  automatic stacks.

* Performance improvements in PIZ decoding, between
	  20 and 60% speedup on Athlon and Pentium 4 systems.
          (Florian)

* Updated the README with various information, made
	  some cosmetic changes for readability.

* Added fragment shader support to exrdisplay.

* Bumped the version to 1.0.5 in prep for release.

* Updated README and README.OSX to talk about CodeWarrior 
          project files.

* Incorporated Rodrigo Damazio's patch for an openexr.m4
	  macro file and an openexr.spec file for building RPMs.

* Small change in ImfAttribute.h to make IlmImf compile with gcc 2.95.

* Updated ImfDoubleAttribute.h for Codewarrior on MacOS.

* Added exrheader utility.

* Update to AUTHORS file.

* Added a README.win32 file.

* Added project files for Visual Studio 6.0.

* Initial Win32 port.  Requires Visual Studio 6.0 and Intel C++
	  compiler version 7.0.

* Added new intersectT method in ImathSphere.h

* Fixed some bugs in ImathQuat.h

* Proper use of fltk-config to get platform-specific FLTK
	  compile- and link-time flags.

* exrdisplay uses Imath::Math<T>::pow instead of powf now.
	  powf is not availble on all platforms.

* Roll OS X "hack" into the source until Apple fixes their
	  istream implementation.
	
## Version 1.0.4

### Detailed Changes:

* OpenEXR is now covered by a modified BSD license.  See LICENSE
	  for the new terms.

## Version 1.0.3:

### Detailed Changes:

* OpenEXR is now in sf.net CVS.

* Imf::Xdr namespace cleanups.

* Some IlmImfTest cleanups for OS X.

* Use .cpp extension in exrdisplay sources.

* Iex cleanups.

* Make IlmImf compile with Metrowerks Codewarrior.

* Change large automatic stacks in ImfHuf.C to auto_ptrs allocated
	  off the heap.  MacOS X default stack size isn't large enough.

* std::ios fix for MacOS X in ImfInputFile.C.

* Added new FP predecessor/successor functions to Imath, added
	  tests to ImathTest

* Fixed a bug in Imath::extractSHRT for 3x3 matricies when
	  exactly one of the original scaling factors is negative, updated
	  ImathTest to check this case.

* Install include files when 'make install' is run.

* exrdisplay requires fltk 1.1+ now in an effort to support
	  a MacOS X display program (fltk 1.1 runs on OS X), though this
	  is untested.

* renamed configure.in to configure.ac

* Removed some tests from IexTest that are no longer used.

* Removed ImfHalfXdr.h, it's not used anymore.

* Revamped the autoconf system, added some compile-time 
          optimizations, a pkgconfig target, and some maintainer-specific
          stuff.

## Version 1.0.2

### Detailed Changes:


* More OS X fixes in Imath, IlmImf and IlmImfTest.

* Imath updates.

* Fixed a rotation bug in Imath

## Version 1.0.1

### Detailed Changes:

* Used autoconf 2.53 and automake 1.6 to generate build environment.

* Makefile.am cleanups.

* OS X fixes.

* removed images directory (now distributed separately).

## Version 1.0

### Detailed Changes:

* first official release.

* added some high-level documentation, removed the old OpenEXR.html
          documentation.

* fixed a few nagging build problems.

* bumped IMV_VERSION_NUMBER to 2

## Version 0.9

### Detailed Changes:

* added exrdisplay viewer application.

* cleanup _data in Imf::InputFile and Imf::OutputFile constructors.

* removed old ILM copyright notices.

## Version 0.8

### Detailed Changes:

* Initial release.
