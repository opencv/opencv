This folder contains libraries and headers of a few
very popular still image codecs used by highgui.
The libraries and headers are only to build Win32 and Win64 versions of OpenCV.
On UNIX systems all the libraries are automatically detected by configure script.

------------------------------------------------------------------------------------
libjpeg 6b (6.2) - The Independent JPEG Group's JPEG software.
             Copyright (C) 1994-1997, Thomas G. Lane.
             See IGJ home page http://www.ijg.org
             for details and links to the source code

             HAVE_JPEG preprocessor flag must be set to make highgui use libjpeg.
             On UNIX systems configure script takes care of it.

------------------------------------------------------------------------------------
libpng 1.4.3 - Portable Network Graphics library.
               Copyright (C) 1998-2010, Glenn Randers-Pehrson.
               See libpng home page http://www.libpng.org
               for details and links to the source code

               HAVE_PNG preprocessor flag must be set to make highgui use libpng.
               On UNIX systems configure script takes care of it.

------------------------------------------------------------------------------------
libtiff 3.9.4 - Tag Image File Format (TIFF) Software
                Copyright (c) 1988-1997 Sam Leffler
                Copyright (c) 1991-1997 Silicon Graphics, Inc.
                See libtiff home page http://www.libtiff.org
                for details and links to the source code


                HAVE_TIFF preprocessor flag must be set to make highgui use libtiff.
                On UNIX systems configure script takes care of it.

                In this build support for ZIP (LZ77 compression), JPEG and LZW
                are included.
------------------------------------------------------------------------------------
zlib 1.2.5 - General purpose LZ77 compression library
             Copyright (C) 1995-2010 Jean-loup Gailly and Mark Adler.
             See zlib home page http://www.gzip.org/zlib
             for details and links to the source code

             No preprocessor definition is needed to make highgui use this library -
             it is included automatically if either libpng or libtiff are used.

------------------------------------------------------------------------------------

jasper-1.900.1 - JasPer is a collection of software
             (i.e., a library and application programs) for the coding
             and manipulation of images.  This software can handle image data in a
             variety of formats.  One such format supported by JasPer is the JPEG-2000
             format defined in ISO/IEC 15444-1.
             
             Copyright (c) 1999-2000 Image Power, Inc.
             Copyright (c) 1999-2000 The University of British Columbia
             Copyright (c) 2001-2003 Michael David Adams

             The JasPer license can be found in src/libjasper.
             
             OpenCV on Windows uses pre-built libjasper library
             (lib/libjasper*). To get the latest source code,
             please, visit the project homepage:
             http://www.ece.uvic.ca/~mdadams/jasper/

------------------------------------------------------------------------------------

openexr-1.4.0 - OpenEXR is a high dynamic-range (HDR) image file format developed
                by Industrial Light & Magic for use in computer imaging applications.

             Copyright (c) 2004, Industrial Light & Magic, a division of Lucasfilm
             Entertainment Company Ltd.  Portions contributed and copyright held by
             others as indicated.  All rights reserved.

             The project homepage: http://www.openexr.com/

             OpenCV on Windows does not include openexr codec by default.
             To add it, you will need to recompile highgui with OpenEXR support
             using VS.NET2003 or VS.NET2005 (MSVC6 can not compile it):
             1) download binaries (e.g. openexr-1.4.0-vs2005.zip)
                from the official site.
             2) copy
                half.lib, iex.lib, ilmimf.lib ilmthread.lib imath.lib to
                _graphics/lib
             3) copy include/openexr/*.h to _graphics/include/openexr
             4) open _make/opencv.sln
             5) in highgui/_highgui.h uncomment
                #define HAVE_ILMIMF 1
             6) build debug/release configurations of highgui.

------------------------------------------------------------------------------------

ffmpeg-0.5.1 - FFmpeg is a complete, cross-platform solution to record,
             convert and stream audio and video. It includes libavcodec -
             the leading audio/video codec library, and also libavformat, libavutils and
             other helper libraries that are used by OpenCV (in highgui module) to
             read and write video files.

             The project homepage: http://ffmpeg.org/

------------------------------------------------------------------------------------

videoInput-0.1995 - Video capturing library for Windows using DirectShow as backend
             Written by Theodore Watson
             http://muonics.net/school/spring05/videoInput/

------------------------------------------------------------------------------------

clapack-3.2.1 - F2C translation of the Linear Algebra PACKage (LAPACK),
                Copyright (c) 1992-2010 The University of Tennessee. All rights reserved.
                http://www.netlib.org/lapack/
                http://www.netlib.org/clapack/

                Note, that only a subset of package is used in OpenCV.
                It can be extended and/or replaced with future upstream releases
                in the future.

------------------------------------------------------------------------------------
