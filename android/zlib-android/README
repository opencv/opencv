ZLIB DATA COMPRESSION LIBRARY

zlib 1.2.3 is a general purpose data compression library.  All the code is
thread safe.  The data format used by the zlib library is described by RFCs
(Request for Comments) 1950 to 1952 in the files
http://www.ietf.org/rfc/rfc1950.txt (zlib format), rfc1951.txt (deflate format)
and rfc1952.txt (gzip format). These documents are also available in other
formats from ftp://ftp.uu.net/graphics/png/documents/zlib/zdoc-index.html

All functions of the compression library are documented in the file zlib.h
(volunteer to write man pages welcome, contact zlib@gzip.org). A usage example
of the library is given in the file example.c which also tests that the library
is working correctly. Another example is given in the file minigzip.c. The
compression library itself is composed of all source files except example.c and
minigzip.c.

To compile all files and run the test program, follow the instructions given at
the top of Makefile. In short "make test; make install" should work for most
machines. For Unix: "./configure; make test; make install". For MSDOS, use one
of the special makefiles such as Makefile.msc. For VMS, use make_vms.com.

Questions about zlib should be sent to <zlib@gzip.org>, or to Gilles Vollant
<info@winimage.com> for the Windows DLL version. The zlib home page is
http://www.zlib.org or http://www.gzip.org/zlib/ Before reporting a problem,
please check this site to verify that you have the latest version of zlib;
otherwise get the latest version and check whether the problem still exists or
not.

PLEASE read the zlib FAQ http://www.gzip.org/zlib/zlib_faq.html before asking
for help.

Mark Nelson <markn@ieee.org> wrote an article about zlib for the Jan. 1997
issue of  Dr. Dobb's Journal; a copy of the article is available in
http://dogma.net/markn/articles/zlibtool/zlibtool.htm

The changes made in version 1.2.3 are documented in the file ChangeLog.

Unsupported third party contributions are provided in directory "contrib".

A Java implementation of zlib is available in the Java Development Kit
http://java.sun.com/j2se/1.4.2/docs/api/java/util/zip/package-summary.html
See the zlib home page http://www.zlib.org for details.

A Perl interface to zlib written by Paul Marquess <pmqs@cpan.org> is in the
CPAN (Comprehensive Perl Archive Network) sites
http://www.cpan.org/modules/by-module/Compress/

A Python interface to zlib written by A.M. Kuchling <amk@amk.ca> is
available in Python 1.5 and later versions, see
http://www.python.org/doc/lib/module-zlib.html

A zlib binding for TCL written by Andreas Kupries <a.kupries@westend.com> is
availlable at http://www.oche.de/~akupries/soft/trf/trf_zip.html

An experimental package to read and write files in .zip format, written on top
of zlib by Gilles Vollant <info@winimage.com>, is available in the
contrib/minizip directory of zlib.


Notes for some targets:

- For Windows DLL versions, please see win32/DLL_FAQ.txt

- For 64-bit Irix, deflate.c must be compiled without any optimization. With
  -O, one libpng test fails. The test works in 32 bit mode (with the -n32
  compiler flag). The compiler bug has been reported to SGI.

- zlib doesn't work with gcc 2.6.3 on a DEC 3000/300LX under OSF/1 2.1 it works
  when compiled with cc.

- On Digital Unix 4.0D (formely OSF/1) on AlphaServer, the cc option -std1 is
  necessary to get gzprintf working correctly. This is done by configure.

- zlib doesn't work on HP-UX 9.05 with some versions of /bin/cc. It works with
  other compilers. Use "make test" to check your compiler.

- gzdopen is not supported on RISCOS, BEOS and by some Mac compilers.

- For PalmOs, see http://palmzlib.sourceforge.net/

- When building a shared, i.e. dynamic library on Mac OS X, the library must be
  installed before testing (do "make install" before "make test"), since the
  library location is specified in the library.


Acknowledgments:

  The deflate format used by zlib was defined by Phil Katz. The deflate
  and zlib specifications were written by L. Peter Deutsch. Thanks to all the
  people who reported problems and suggested various improvements in zlib;
  they are too numerous to cite here.

Copyright notice:

 (C) 1995-2004 Jean-loup Gailly and Mark Adler

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  Jean-loup Gailly        Mark Adler
  jloup@gzip.org          madler@alumni.caltech.edu

If you use the zlib library in a product, we would appreciate *not*
receiving lengthy legal documents to sign. The sources are provided
for free but without warranty of any kind.  The library has been
entirely written by Jean-loup Gailly and Mark Adler; it does not
include third-party code.

If you redistribute modified sources, we would appreciate that you include
in the file ChangeLog history information documenting your changes. Please
read the FAQ for more information on the distribution of modified source
versions.
