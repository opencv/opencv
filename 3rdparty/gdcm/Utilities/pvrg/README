
                 MPEG, CCITT H.261 (P*64), JPEG
  Image and Image sequence compression/decompression C software engines.


The Portable Video Research Group at Stanford have developed
image/image sequence compression and decompression engines (codecs)
for MPEG, CCITT H.261, and JPEG. The primary goal of these codecs is
to provide the functionality - these codecs are not optimized for
speed, rather completeness, and some of the code is kludgey.

Development of MPEG, P64, and JPEG engines has not been the primary
goal of the Portable Video Research Group.  Our research has been
focused on software and hardware for portable wireless digital video
communication.  The charter of this group ended in the summer of 1994.

COMMENTS/DISCLAIMERS:

This code has been compiled on the Sun Sparc and DECstation UNIX
machines; some code has been further checked on HP workstations.

For comments, bugs, and other mail relating to the source code, we
appreciate any comments. The code author can still be reached at Andy
C.  Hung at achung@cs.stanford.edu.  The standard public domain
disclaimer applies: Caveat Emptor - no guarantee on accuracy or
software support.

References related to these codecs should NOT use any author's name,
or refer to Stanford University.  Rather the Portable Video Research
Group or the acronym (PVRG) should be used, such as PVRG-MPEG,
PVRG-P64, PVRG-JPEG.

ANONYMOUS FTP:

The following files can be obtained through anonymous ftp from
havefun.stanford.edu, IP address [36.2.0.35].  The procedure is to use
ftp with the user name "anonymous" and an e-mail address for the
password.

CODEC DESCRIPTION:

I) PVRG-MPEG CODEC: (pub/mpeg/MPEGv1.2.1.tar.Z)

This public domain video encoder and decoder was generated according
to the Santa Clara August 1991 format.  It has been tested
successfully with decoders using the Paris December 1991 format. The
codec is capable of encoding all MPEG types of frames. The algorithms
for rate control, buffer-constrained encoding, and quantization
decisions are similar, but not identical, to that of the (simulation
model 1-3) MPEG document.  The rate control used is a simple
proportional Q-stepsize/Buffer loop that works though not very well -
better rate-control is the essence for good quality buffer-constrained
MPEG encoding.  Verification of the buffering is possible so as to
provide streams for real-time decoders.

The MPEG codec performs compression and decompression on raw raster
scanned YCbCr (also known as digital YUV) files. The companion display
program for the X window system is described in section IV) below.  A
manual of approximately 50 pages describes the program's use.

There are also MPEG compressed files from the table tennis sequence in
tennis.mpg and the flower garden sequence in flowg.mpg.

This codec was recently tested with the MPEG decoder of the Berkeley
Plateau Research group. If what you want is decoding and X display,
then you might want to look into their faster public domain MPEG
decoder/viewer. The Berkeley player is available via anonymous ftp
from toe.cs.berkeley.edu (128.32.149.117) in
/pub/multimedia/mpeg/mpeg-2.0.tar.Z.  There is also an encoder at that
site.  An ISO mpeg2 encoder and decoder is available by anonymous ftp
from ftp.netcom.com in the directory pub/c/cfogg/mpeg2 (alternate sites
may include ftp.uu.net).


II) PVRG-P64 CODEC: (pub/p64/P64v1.2.tar.Z)

This public domain video encoder and decoder is based on the CCITT
H.261 specification.  Some encoding algorithms are based on the RM 8
encoder.  We have tested it against a verified encoded sequence on the
CCITT 1992 specifications, but we would still appreciate anyone having
p64 video test sequences to let know.  Like the MPEG codec, it
supports all the encoding and decoding modes, and has provisions for
buffer-constrained encoding, so it can produce streams for real-time
decoders.

The H.261 codec takes the similar YCbCr raster scanned files as the MPEG
codec, and performs compression and decompresion on raster scanned
YCbCr files.  It can take standard CIF or NTSC-CIF files. The display
of these programs is described in section IV) below.  A manual of
approximately 50 pages describes its use.

There are also P64 compressed files from the table tennis sequence in
table.p64 and the flower garden sequence in flowg.p64.  The Inria
codec also performs H.261 video compression and is integrated into a
teleconferencing system; it can be obtained from avahi.inria.fr, in
/pub/h261.tar.Z.

III) PVRG-JPEG CODEC: (pub/jpeg/JPEGv1.2.tar.Z)

This public domain image encoder and decoder is based on the JPEG
Committee Draft.  It supports all of the baseline for encoding and
decoding.  The JPEG encoder is flexible in the variety of output
possible.  It also supports lossless coding, though not as speedy as
we would like.  The manual is approximately 50 pages long which
describes its use.  The display program for JFIF-style (YCbCr) files is
described in section IV) below.  The JFIF style is not a requirement
for this codec - it can compress and decompress CMYK, RGB, RGBalpha,
and other formats - this codec may be helpful if you wish to extract
information from non-JFIF encoded JPEG files.

This codec has been tested on publicly available JPEG data.  For
general purpose X display, you might want to try the program "xv"
(version 2.0 or greater).  The JPEG engine of the program "xv" is
based on the free, portable C code for JPEG compression available from
the Independent JPEG Group.  (anonymous login - ftp.uu.net (137.39.1.9
or 192.48.96.9) /graphics/jpeg/jpegsrc.v4.tar.Z).

IV) X VIEWER: (pub/cv/CVv1.2.1.tar.Z)

This viewer allows the user to look at image or image sequences
generated through the codecs described above. These image or image
sequences are in the YCbCr (also known as digital YUV) colorspace
(either JFIF specified or CCIR 601 specified) and may be 4:1:1 (CIF,
or MPEG 4:2:0 style) or 2:1:1 (CCIR-601 4:2:2 style) or 1:1:1
(non-decimated or CCIR-601 4:4:4 style). A short manual of
approximately 2 pages describes its use.

ACKNOWLEDGEMENTS:

Funded by the Defense Advanced Research Projects Agency.

I am especially grateful to Hewlett Packard and Storm Technology for
their financial support during the earlier stages of codec
development.  Any errors in the code and documentation are my own.
The following people are acknowledged for their advice and assistance.
Thanks, one and all.

	The Portable Video Research Group at Stanford:
	Teresa Meng, Peter Black, Navin Chaddha, Ben Gordon,
        Sheila Hemami, Wee-Chiew Tan, Eli Tsern.

	Adriaan Ligtenberg of Storm Technology.
	Jeanne Wiseman, Andrew Fitzhugh, Gregory Yovanof and
        Chuck Rosenberg of Hewlett Packard.
	Eric Hamilton and Jean-Georges Fritsch of C-Cube Microsystems.

	Lawrence Rowe of the Berkeley Plateau Research Group.
	Tom Lane of the Independent JPEG Group.
	Katsumi Tahara, Sony.
	Ciaran Mc Goldrick.
	Karl Lillevold.
	Mats Lofkvist.
	Hiroshi Saitoh, Panasonic.
	Frank Gadegast.
	Chad Fogg, Cascade.
	Thierry Turletti, Inria.
	Anders Klemets.
	Graham Logan.
	Jelle van Zeijl.
	George Warner, AT&T.
	Chris Adams, Hewlett Packard.
	Kent Murray,  Atlantic Centre For Remote Sensing Of The Oceans.
	I. C. Yang.
	Donald Lindsay.
	Harald A. Schiller, Tektronix.
	Ismail Dalgic.
	Tom Shield.
	Y. Fujii.
