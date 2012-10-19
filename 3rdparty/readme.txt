This folder contains libraries and headers of a few very popular still image codecs
used by highgui module.
The libraries and headers are preferably to build Win32 and Win64 versions of OpenCV.
On UNIX systems all the libraries are automatically detected by configure script.
In order to use these versions of libraries instead of system ones on UNIX systems you
should use BUILD_<library_name> CMake flags (for example, BUILD_PNG for the libpng library).

------------------------------------------------------------------------------------
libjpeg 8d (8.4)  -   The Independent JPEG Group's JPEG software.
                      Copyright (C) 1991-2012, Thomas G. Lane, Guido Vollbeding.
                      See IGJ home page http://www.ijg.org
                      for details and links to the source code

                      HAVE_JPEG preprocessor flag must be set to make highgui use libjpeg.
                      On UNIX systems configure script takes care of it.
------------------------------------------------------------------------------------
libpng 1.5.12     -   Portable Network Graphics library.
                      Copyright (c) 2004, 2006-2012 Glenn Randers-Pehrson.
                      See libpng home page http://www.libpng.org
                      for details and links to the source code

                      HAVE_PNG preprocessor flag must be set to make highgui use libpng.
                      On UNIX systems configure script takes care of it.
------------------------------------------------------------------------------------
libtiff 4.0.2     -   Tag Image File Format (TIFF) Software
                      Copyright (c) 1988-1997 Sam Leffler
                      Copyright (c) 1991-1997 Silicon Graphics, Inc.
                      See libtiff home page http://www.remotesensing.org/libtiff/
                      for details and links to the source code

                      HAVE_TIFF preprocessor flag must be set to make highgui use libtiff.
                      On UNIX systems configure script takes care of it.
                      In this build support for ZIP (LZ77 compression) is turned on.
------------------------------------------------------------------------------------
zlib 1.2.7        -   General purpose LZ77 compression library
                      Copyright (C) 1995-2012 Jean-loup Gailly and Mark Adler.
                      See zlib home page http://www.zlib.net
                      for details and links to the source code

                      No preprocessor definition is needed to make highgui use this library -
                      it is included automatically if either libpng or libtiff are used.
------------------------------------------------------------------------------------
jasper-1.900.1    -   JasPer is a collection of software
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
openexr-1.7.1     -   OpenEXR is a high dynamic-range (HDR) image file format developed
                      by Industrial Light & Magic for use in computer imaging applications.

                      Copyright (c) 2006, Industrial Light & Magic, a division of Lucasfilm
                      Entertainment Company Ltd. Portions contributed and copyright held by
                      others as indicated. All rights reserved.

                      The project homepage: http://www.openexr.com
------------------------------------------------------------------------------------
ffmpeg-0.8.0      -   FFmpeg is a complete, cross-platform solution to record,
                      convert and stream audio and video. It includes libavcodec -
                      the leading audio/video codec library, and also libavformat, libavutils and
                      other helper libraries that are used by OpenCV (in highgui module) to
                      read and write video files.

                      The project homepage: http://ffmpeg.org/
------------------------------------------------------------------------------------
