README for libpng version 1.6.37 - April 14, 2019
=================================================

See the note about version numbers near the top of png.h.
See INSTALL for instructions on how to install libpng.

Libpng comes in several distribution formats.  Get libpng-*.tar.gz or
libpng-*.tar.xz or if you want UNIX-style line endings in the text
files, or lpng*.7z or lpng*.zip if you want DOS-style line endings.

Version 0.89 was the first official release of libpng.  Don't let the
fact that it's the first release fool you.  The libpng library has been
in extensive use and testing since mid-1995.  By late 1997 it had
finally gotten to the stage where there hadn't been significant
changes to the API in some time, and people have a bad feeling about
libraries with versions < 1.0.  Version 1.0.0 was released in
March 1998.

****
Note that some of the changes to the png_info structure render this
version of the library binary incompatible with libpng-0.89 or
earlier versions if you are using a shared library.  The type of the
"filler" parameter for png_set_filler() has changed from png_byte to
png_uint_32, which will affect shared-library applications that use
this function.

To avoid problems with changes to the internals of the png info_struct,
new APIs have been made available in 0.95 to avoid direct application
access to info_ptr.  These functions are the png_set_<chunk> and
png_get_<chunk> functions.  These functions should be used when
accessing/storing the info_struct data, rather than manipulating it
directly, to avoid such problems in the future.

It is important to note that the APIs did not make current programs
that access the info struct directly incompatible with the new
library, through libpng-1.2.x.  In libpng-1.4.x, which was meant to
be a transitional release, members of the png_struct and the
info_struct can still be accessed, but the compiler will issue a
warning about deprecated usage.  Since libpng-1.5.0, direct access
to these structs is not allowed, and the definitions of the structs
reside in private pngstruct.h and pnginfo.h header files that are not
accessible to applications.  It is strongly suggested that new
programs use the new APIs (as shown in example.c and pngtest.c), and
older programs be converted to the new format, to facilitate upgrades
in the future.
****

Additions since 0.90 include the ability to compile libpng as a
Windows DLL, and new APIs for accessing data in the info struct.
Experimental functions include the ability to set weighting and cost
factors for row filter selection, direct reads of integers from buffers
on big-endian processors that support misaligned data access, faster
methods of doing alpha composition, and more accurate 16->8 bit color
conversion.

The additions since 0.89 include the ability to read from a PNG stream
which has had some (or all) of the signature bytes read by the calling
application.  This also allows the reading of embedded PNG streams that
do not have the PNG file signature.  As well, it is now possible to set
the library action on the detection of chunk CRC errors.  It is possible
to set different actions based on whether the CRC error occurred in a
critical or an ancillary chunk.

For a detailed description on using libpng, read libpng-manual.txt.
For examples of libpng in a program, see example.c and pngtest.c.  For
usage information and restrictions (what little they are) on libpng,
see png.h.  For a description on using zlib (the compression library
used by libpng) and zlib's restrictions, see zlib.h

I have included a general makefile, as well as several machine and
compiler specific ones, but you may have to modify one for your own
needs.

You should use zlib 1.0.4 or later to run this, but it MAY work with
versions as old as zlib 0.95.  Even so, there are bugs in older zlib
versions which can cause the output of invalid compression streams for
some images.

You should also note that zlib is a compression library that is useful
for more things than just PNG files.  You can use zlib as a drop-in
replacement for fread() and fwrite(), if you are so inclined.

zlib should be available at the same place that libpng is, or at
https://zlib.net.

You may also want a copy of the PNG specification.  It is available
as an RFC, a W3C Recommendation, and an ISO/IEC Standard.  You can find
these at http://www.libpng.org/pub/png/pngdocs.html .

This code is currently being archived at libpng.sourceforge.io in the
[DOWNLOAD] area, and at http://libpng.download/src .

This release, based in a large way on Glenn's, Guy's and Andreas'
earlier work, was created and will be supported by myself and the PNG
development group.

Send comments/corrections/commendations to png-mng-implement at
lists.sourceforge.net (subscription required; visit
https://lists.sourceforge.net/lists/listinfo/png-mng-implement
to subscribe).

Send general questions about the PNG specification to png-mng-misc
at lists.sourceforge.net (subscription required; visit
https://lists.sourceforge.net/lists/listinfo/png-mng-misc to
subscribe).

Files in this distribution:

      ANNOUNCE      =>  Announcement of this version, with recent changes
      AUTHORS       =>  List of contributing authors
      CHANGES       =>  Description of changes between libpng versions
      KNOWNBUG      =>  List of known bugs and deficiencies
      LICENSE       =>  License to use and redistribute libpng
      README        =>  This file
      TODO          =>  Things not implemented in the current library
      TRADEMARK     =>  Trademark information
      example.c     =>  Example code for using libpng functions
      libpng.3      =>  manual page for libpng (includes libpng-manual.txt)
      libpng-manual.txt  =>  Description of libpng and its functions
      libpngpf.3    =>  manual page for libpng's private functions
      png.5         =>  manual page for the PNG format
      png.c         =>  Basic interface functions common to library
      png.h         =>  Library function and interface declarations (public)
      pngpriv.h     =>  Library function and interface declarations (private)
      pngconf.h     =>  System specific library configuration (public)
      pngstruct.h   =>  png_struct declaration (private)
      pnginfo.h     =>  png_info struct declaration (private)
      pngdebug.h    =>  debugging macros (private)
      pngerror.c    =>  Error/warning message I/O functions
      pngget.c      =>  Functions for retrieving info from struct
      pngmem.c      =>  Memory handling functions
      pngbar.png    =>  PNG logo, 88x31
      pngnow.png    =>  PNG logo, 98x31
      pngpread.c    =>  Progressive reading functions
      pngread.c     =>  Read data/helper high-level functions
      pngrio.c      =>  Lowest-level data read I/O functions
      pngrtran.c    =>  Read data transformation functions
      pngrutil.c    =>  Read data utility functions
      pngset.c      =>  Functions for storing data into the info_struct
      pngtest.c     =>  Library test program
      pngtest.png   =>  Library test sample image
      pngtrans.c    =>  Common data transformation functions
      pngwio.c      =>  Lowest-level write I/O functions
      pngwrite.c    =>  High-level write functions
      pngwtran.c    =>  Write data transformations
      pngwutil.c    =>  Write utility functions
      arm           =>  Contains optimized code for the ARM platform
      powerpc       =>  Contains optimized code for the PowerPC platform
      contrib       =>  Contributions
       arm-neon         =>  Optimized code for ARM-NEON platform
       powerpc-vsx      =>  Optimized code for POWERPC-VSX platform
       examples         =>  Example programs
       gregbook         =>  source code for PNG reading and writing, from
                            Greg Roelofs' "PNG: The Definitive Guide",
                            O'Reilly, 1999
       libtests         =>  Test programs
       mips-msa         =>  Optimized code for MIPS-MSA platform
       pngminim         =>  Minimal decoder, encoder, and progressive decoder
                            programs demonstrating use of pngusr.dfa
       pngminus         =>  Simple pnm2png and png2pnm programs
       pngsuite         =>  Test images
       testpngs
       tools            =>  Various tools
       visupng          =>  Contains a MSVC workspace for VisualPng
      intel             =>  Optimized code for INTEL-SSE2 platform
      mips              =>  Optimized code for MIPS platform
      projects      =>  Contains project files and workspaces for
                        building a DLL
       owatcom          =>  Contains a WATCOM project for building libpng
       visualc71        =>  Contains a Microsoft Visual C++ (MSVC)
                            workspace for building libpng and zlib
       vstudio          =>  Contains a Microsoft Visual C++ (MSVC)
                            workspace for building libpng and zlib
      scripts       =>  Directory containing scripts for building libpng:
                            (see scripts/README.txt for the list of scripts)

Good luck, and happy coding!

 * Cosmin Truta (current maintainer, since 2018)
 * Glenn Randers-Pehrson (former maintainer, 1998-2018)
 * Andreas Eric Dilger (former maintainer, 1996-1997)
 * Guy Eric Schalnat (original author and former maintainer, 1995-1996)
   (formerly of Group 42, Inc.)
