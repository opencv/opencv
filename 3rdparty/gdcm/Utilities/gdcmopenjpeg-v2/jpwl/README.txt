===============================================================================
	JPEG2000 Part 11 (ISO/IEC 15444-11 JPWL) Software



		Version 20061213
===============================================================================





1. Scope
=============

This document describes the installation and use of the JPWL module in the framework of OpenJPEG library.

This implementation has been developed from OpenJPEG implementation of JPEG2000 standard, and for this reason it is written in C language.

If you find some bugs or if you have problems using the encoder/decoder, please send an e-mail to jpwl@diei.unipg.it


2. Installing the code
==========================

The JPWL code is integrated with the standard OpenJPEG library and codecs: it is activated by setting the macro USE_JPWL to defined in the preprocessor configuration options of your preferred C compiler.

2.1. Compiling the source code in Windows
-------------------------------------------

The "jpwl" directory is already populated with a couple of Visual C++ 6.0 workspaces

 * JPWL_image_to_j2k.dsw - Creates the encoder with JPWL functionalities
 * JPWL_j2k_to_image.dsw - Creates the decoder with JPWL functionalities

2.2. Compiling the source code in Unix-like systems
-----------------------------------------------------

Under linux, enter the jpwl directory and type "make clean" and "make".


3. Running the JPWL software
=========================

The options available at the command line are exactly the same of the base OpenJPEG codecs. In addition, there is a "-W" switch that activates JPWL functionalities.

3.1. JPWL Encoder
-------------------

-W           : adoption of JPWL (Part 11) capabilities (-W params)
               The parameters can be written and repeated in any order:
               [h<tile><=type>,s<tile><=method>,a=<addr>,z=<size>,g=<range>,...
                ...,p<tile:pack><=type>]

                 h selects the header error protection (EPB): 'type' can be
                   [0=none 1,absent=predefined 16=CRC-16 32=CRC-32 37-128=RS]
                   if 'tile' is absent, it applies to main and tile headers
                   if 'tile' is present, it applies from that tile
                     onwards, up to the next h<tile> spec, or to the last tile
                     in the codestream (max. 16 specs)

                 p selects the packet error protection (EEP/UEP with EPBs)
                  to be applied to raw data: 'type' can be
                   [0=none 1,absent=predefined 16=CRC-16 32=CRC-32 37-128=RS]
                   if 'tile:pack' is absent, it starts from tile 0, packet 0
                   if 'tile:pack' is present, it applies from that tile
                     and that packet onwards, up to the next packet spec
                     or to the last packet in the last tile in the codestream
                     (max. 16 specs)

                 s enables sensitivity data insertion (ESD): 'method' can be
                   [-1=NO ESD 0=RELATIVE ERROR 1=MSE 2=MSE REDUCTION 3=PSNR
                    4=PSNR INCREMENT 5=MAXERR 6=TSE 7=RESERVED]
                   if 'tile' is absent, it applies to main header only
                   if 'tile' is present, it applies from that tile
                     onwards, up to the next s<tile> spec, or to the last tile
                     in the codestream (max. 16 specs)

                 g determines the addressing mode: <range> can be
                   [0=PACKET 1=BYTE RANGE 2=PACKET RANGE]

                 a determines the size of data addressing: <addr> can be
                   2/4 bytes (small/large codestreams). If not set, auto-mode

                 z determines the size of sensitivity values: <size> can be
                   1/2 bytes, for the transformed pseudo-floating point value

                 ex.:
 h,h0=64,h3=16,h5=32,p0=78,p0:24=56,p1,p3:0=0,p3:20=32,s=0,s0=6,s3=-1,a=0,g=1,z=1
                 means
                   predefined EPB in MH, rs(64,32) from TPH 0 to TPH 2,
                   CRC-16 in TPH 3 and TPH 4, CRC-32 in remaining TPHs,
                   UEP rs(78,32) for packets 0 to 23 of tile 0,
                   UEP rs(56,32) for packets 24 to the last of tile 0,
                   UEP rs default for packets of tile 1,
                   no UEP for packets 0 to 19 of tile 3,
                   UEP CRC-32 for packets 20 of tile 3 to last tile,
                   relative sensitivity ESD for MH,
                   TSE ESD from TPH 0 to TPH 2, byte range with automatic
                   size of addresses and 1 byte for each sensitivity value

                 ex.:
                       h,s,p
                 means
                   default protection to headers (MH and TPHs) as well as
                   data packets, one ESD in MH

                 N.B.: use the following recommendations when specifying
                       the JPWL parameters list
                   - when you use UEP, always pair the 'p' option with 'h'

3.2. JPWL Decoder
-------------------

  -W <options>
    Activates the JPWL correction capability, if the codestream complies.
    Options can be a comma separated list of <param=val> tokens:
    c, c=numcomps
       numcomps is the number of expected components in the codestream
       (search of first EPB rely upon this, default is 3)


4. Known bugs and limitations
===============================

4.1. Bugs
-----------

* It is not possible to save a JPWL encoded codestream using the wrapped file format (i.e. JP2): only raw file format (i.e. J2K) is working

4.2. Limitations
------------------

* When specifying an UEP protection, you need to activate even TPH protection for those tiles where there is a protection of the packets
* RED insertion is not currently implemented at the decoder
* JPWL at entropy coding level is not implemented
