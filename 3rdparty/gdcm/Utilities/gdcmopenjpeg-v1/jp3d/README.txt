===============================================================================
	JPEG2000 Part 10 (ISO/IEC 15444-10 JP3D) Verification Model

				Version 1.1
===============================================================================


1. Scope
================

This document describes the installation and the use of the JP3D VM decoder and encoder under several operating systems (Linux, Unix, Windows, ...). Version 1.1 contains a complete JPEG 2000 Part 10 encoder, as well as a decoder.
The supported functionalities are compliant with the JPEG2000 part 10 algorithm as described in the WD 6.0.
The provided encoder and the decoder are compatible also with the International standard IS 15444-1 (core coding system).
This implementation has been developped from OpenJPEG implementation of JPEG2000 standard, and for this reason it is written in C language. 

If you find some bugs or if you have problems using the encoder/decoder, please send an e-mail to jp3d@lpi.tel.uva.es

2. Installing the code 
======================================================

- After decompressing the zip file provided, you should find 
at least the following files in the created 'jp3d_vm' directory:

 * libjp3dvm - This directory contains all library related code
 * codec - This directory contains all codec related code
 * tcltk - This directory contains the API scripts
 * README - The file you are reading
 * LICENCE - Copyright statement of the JP3D VM software

2.1. Compiling the source code in Windows
-------------------------------------------

This version has been compiled with Visual Studio 2003 using
the projects included in the distribution:

 * LibJp3dVM.vcproj - Creates the library with all the JP3D functionalities
 * jp3d_vm_enc.vcproj - Test encoder 
 * jp3d_vm_dec.vcproj - Test decoder 

2.2. Compiling the source code in Unix-like systems
-------------------------------------------

Library compilation
------------------------
This version of the library has been tested under the following OS:
- Fedora Core 

The installation process is as simple as this : 
1) Enter the 'jp3d_vm' directory
2) Build the distribution : 
make
make install
3) Clean all files produced during the build process
make clean

Codec compilation
------------------------
Once you've built the library, you should compile the JP3D codec.

1) Go to the 'codec' directory 
2) Build the encoder and decoder programs:

gcc convert.c volume_to_jp3d.c -o jp3d_vm_enc -I ../libjp3dvm/ -lm -ljp3dvm
gcc convert.c jp3d_to_volume.c -o jp3d_vm_dec -I ../libjp3dvm/ -lm -ljp3dvm

Note: You should add '-L ../libjp3dvm/' to those lines if you
did not use the 'install' target (and the 'clean' target neither...).

3. Running the JP3D VM
====================================================

3.1. JP3D ENCODER
====================================================

Required arguments
------------------------

 * Input file(s): -i Involume [*.bin, *.pgx]

Specifies the volume to compress. Accepted formats are *.BIN (raw binary data) or *.PGX files.
Both formats need some particular settings:

  a) BIN format. As it has no header, volume characteristics will be obtained from a .IMG file. Its location will be specified through the following argument:
	   -m Involumeinfo.IMG
     This file shall have the following structure, with the appropiate value in each case (bit per voxel, color map, dimensions in X,Y,Z):
	o Bpp	%d
	o Color Map	%d
	o Dimensions	%d %d %d

  b) PGX format. Program will consider it as a volume slice. In order to denote a volume through a sequence of slices, you can define the input filename with the common pattern of the set of PGX files followed by a dash (as a wildcard character for the sequence numbers).
	
 * Output file: -o Outfile [*.jp3d, *j2k]

Specifies the name of the file where the codestream will be saved. 
Part 1 compliant codestream will be created when an outfile has .j2k format.

Options 
--------

    * Rate values : -r 20,10,5
      This option offers the possibility to define the compression rate to apply. 
      Each value is a factor of compression (i.e. 20 : 1) and will generate a different quality layer. A lossless compression will be signified by the value 1.
      NOTE : The order used to define the different levels of compression is important and must be from left to right in descending order.

    * Quality values : -q 30,35,40
      This option offers the possibility to define the quality level to achieve. Each value is a psnr, to be given in dB, and represents a quality layer. 
      NOTE : The order used to define the different psnr-values is important and must be from left to right in ascending order.


    * Number of resolutions : -n 3,3,2
      This option offers the possibility to define the number of resolution levels computed for each dimension of the volume through the discret wavelet transform (DWT). Resolution in axial dimension can have a different value than in horizontal and vertical cases, but must be lower.
      DEFAULT VALUE : 3,3,1 


    * Switch modes : -M 3
      This option offers the possibility to use a mode switch during the encoding process:
          o BYPASS(LAZY)	[1]
          o RESET		[2]
          o RESTART(TERMALL)	[4]
          o VSC			[8]
          o ERTERM(SEGTERM)	[16]
          o SEGMARK(SEGSYM)	[32]
          o 3D_CONTEXT		[64]
      For several mode switch just sum corresponding values: i.e. -M 38  => RESTART(4) + RESET(2) + SEGMARK(32)
      DEFAULT VALUE: 0


    * Progression order : -p LRCP
      This option offers the possibility to specify the progression order. Possible progression orders are : LRCP, RLCP, RPCL, PCRL and CPRL.
      DEFAULT VALUE: LRCP.


    * Code-block size : -b 32,32,32
      This option offers the possibility to define the size of the code-block. The dimension must respect the constraint defined in the JPEG-2000 standard. The maximum value autorized is 64x64x64.
      DEFAULT VALUE: 64,64,64


    * Precinct size : -c [128,128,128],[128,128,128],...
      This option offers the possibility to define the size of the precincts at each resolution. Multiple records may be supplied, in which case the first record refers to the highest resolution level and subsequent records to lower resolution levels. The last specified record is right-shifted for each remaining lower resolution levels.
      NOTE : specified values must be power of 2.
      DEFAULT VALUE: 2^15 x 2^15 x 2^15


    * Tile size : -t 512,512,512
      This option offers the possibility to divide the volume in several tiles. The three values define the width, the heigth and the depth of the tile respectivily. 
      DEFAULT VALUE: Volume dimensions (one tile)


    * Subsampling factor : -s 2,2,2
      This option offers the possibility to apply a subsampling factor for X, Y and Z axis. Value higher than 2 can be a source of error ! 
      DEFAULT VALUE: 1,1,1


    * SOP marker before each packet : -SOP
      This option offers the possibility to add a specific marker before each packet. It is the marker SOP (Start of packet). If the option is not used no SOP marker will be added.


    * EPH marker after each packet header : -EPH
      This option offers the possibility to add a specific marker at the head of each packet header. It is the marker EPH (End of packet Header). If the option is not used no EPH marker will be added.


    * Offset of the volume origin : -d 150,300,10
      This option offers the possibility to move the origine of the volume in X, Y and/or Z axis. The division in tile could be modified as the anchor point for tiling will be different than the volume origin.
      NOTE : the offset of the volume can not be higher than the tile dimension if the tile option is used.
      DEFAULT VALUE: 0,0,0


    * Offset of the tile origin : -T 100,75,5
      This option offers the possibility to move the anchor point of the volume in X, Y and/or Z axis.
      NOTE : the tile anchor point can not be on the volume area.
      DEFAULT VALUE: 0,0,0


    * Display the help menu : -help
      This option displays on screen the content of this page

Additional options 
----------------------------------

    * Encoding information file: -x index_name.idx
      This option offers the possibility to create a text file with some structured information generated through the encoding. The name of the file must be specified, with .idx extension. The information structure is the following: 
          o Volume size:
                + VolW   + VolH   + VolD
          o Progression Order:
                + Prog
          o Tile size: 
                + TileW   + TileH   + TileD
          o Number of components:
                + NumComp
          o Number of layers:
                + NumLayer
          o Number of decompositions (=(number of resolutions - 1)):
                + NumDWTx   + NumDWTy   + NumDWTz
          o Precinct size:
                + [Precinct_width(NumDWT),Precinct_height(NumDWT),Precinct_depth(NumDWT)]
                + [Precinct_width(NumDWT-1),Precinct_height(NumDWT-1),Precinct_depth(NumDWT-1)]
                + ...
                + [Precinct_width(0),Precinct_height(0),Precinct_depth(0)] 
          o Main Header end position:
                + MH_EndPos
          o Codestream size:
                + CSSize
          o Tile 0 information:
                + TileNum (0) 
                + StartPos
                + TileHeader_EndPos
                + EndPos
                + TotalDisto (this is the sum of the distortion reductions brought by each packet belonging to this tile)
                + NumPix (this is the number of pixels in the tile)
                + MaxMSE (=TotalDisto/NumPix)
          o Tile1 information:
                + TileNum (1)
                + ...
          o ...
          o Tile N information:
                + TileNum (N)
                + ...
          o Packet 0 from Tile 0 information:
                + PackNum (0)
                + TileNum (0) 
                + LayerNum
                + ResNum
                + CompNum
                + PrecNum
                + StartPos
                + EndPos
                + Disto (distortion reduction brought by this packet)
          o Packet 1 from Tile 0 information:
                + PackNum (1)
                + ...
          o ...
          o Packet M from Tile 0 information
          o Packet 0 from Tile 1 information
          o ...
          o Packet M from Tile N information
          o Maximum distortion reduction on the whole volume:
                + MaxDisto
          o Total distortion on the whole volume (sum of the distortion reductions from all packets in the volume):
                + TotalDisto

3.2. JP3D DECODER
====================================================

Required arguments
------------------------

    * Infile : -i compressed file
      Currently accepts JP3D and J2K-files. The file type is identified based on its suffix (*.jp3d, *.j2k).


    * Outfile(s) : -o decompressed file(s)
      Currently accepts BIN-files and PGX-files. Binary data is written to the file (not ascii). 
      If a BIN-file is defined, decoder will create automatically the volume characteristic file appending a .IMG extension to the provided output filename.
      If a PGX-file is defined, decoder will understand this as a file pattern, appending corresponding indice from 0 to the number of decoded slices.
      NOTE : There will be as many output files as there are components: an indice starting from 0 will then be appended to the output filename, just before the extension.


Options available
------------------------

    * Reduce factor : -r 1,1,0
      Set the number of highest resolution levels to be discarded in each dimension. The decoded volume size is effectively divided by 2 to the power of the number of discarded levels. 
      NOTE : The reduce factor is limited by the smallest total number of decomposition levels among tiles.


    * Layer number : -l 2
      Set the maximum number of quality layers to decode. If there are less quality layers than the specified number, all the quality layers are decoded.


    * Performance comparisons : -O original-file
      This option offers the possibility to compute some quality results for the decompressed volume, like the PSNR value achieved or the global SSIM value. Needs the original file in order to compare with the new one.
      NOTE: Only valid when -r option is 0,0,0 (both original and decompressed volumes have same resolutions)
      NOTE: If original file is .BIN file, the volume characteristics file shall be defined with the -m option.
	    (i.e. -O original-BIN-file -m original-IMG-file)


    * Byte order (Big-endian / Little-endian) : -BE
      This option offers the possibility to save the decompressed volume with a predefined byte order. 
      DEFAULT VALUE: Little-endian


    * Display the help menu : -help
      This option displays on screen the content of this page


