/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/
/*
************************************************************
prototypes.h

This file contains the functional prototypes for typechecking.

************************************************************
*/

#ifndef PROTOTYPES_DONE
#define PROTOTYPES_DONE

/* jpeg.c */


extern void PrintImage();
extern void PrintFrame();
extern void PrintScan();
extern void MakeImage();
extern void MakeFrame();
extern void MakeScanFrequency();
extern void MakeScan();
extern void MakeConsistentFileNames();
extern void CheckValidity();
extern int CheckBaseline();
extern void ConfirmFileSize();
extern void JpegQuantizationFrame();
extern void JpegDefaultHuffmanScan();
extern void JpegFrequencyScan();
extern void JpegCustomScan();
extern void JpegEncodeScan();
extern void JpegLosslessFrequencyScan();
extern void JpegLosslessEncodeScan();

/* codec.c */

extern void FrequencyAC();
extern void EncodeAC();
extern void DecodeAC();
extern int DecodeDC();
extern void FrequencyDC();
extern void EncodeDC();
extern void ResetCodec();
extern void ClearFrameFrequency();
extern void AddFrequency();
extern void InstallFrequency();
extern void InstallPrediction();
extern void PrintACEhuff();
extern void PrintDCEhuff();
extern int SizeACEhuff();
extern int SizeDCEhuff();

extern int LosslessDecodeDC();
extern void LosslessFrequencyDC();
extern void LosslessEncodeDC();

/* huffman.c */

extern void MakeHuffman();
extern void SpecifiedHuffman();
extern void MakeDecoderHuffman();
extern void ReadHuffman();
extern void WriteHuffman();
extern int DecodeHuffman();
extern void EncodeHuffman();
extern void MakeXhuff();
extern void MakeEhuff();
extern void MakeDhuff();
extern void UseACHuffman();
extern void UseDCHuffman();
extern void SetACHuffman();
extern void SetDCHuffman();
extern void PrintHuffman();
extern void PrintTable();

/* io.c */


extern void ReadBlock();
extern void WriteBlock();
extern void ResizeIob();
extern void RewindIob();
extern void FlushIob();
extern void SeekEndIob();
extern void CloseIob();
extern void MakeIob();
extern void PrintIob();
extern int NumberBlocksIob();
extern int NumberBlockMDUIob();
extern void InstallIob();
extern void TerminateFile();

extern int NumberLineMDUIob();
extern void ReadLine();
extern void ReadPreambleLine();
extern void WriteLine();
extern int LineNumberMDUWideIob();
extern int LineNumberMDUHighIob();
extern void LineResetBuffers();

/* chendct.c */

extern void ChenDct();
extern void ChenIDct();

/* leedct.c */

extern void LeeIDct();
extern void LeeDct();

/* lexer.c */

extern void initparser();
extern void parser();

/* marker.c */

extern void WriteSoi();
extern void WriteEoi();
extern void WriteJfif();
extern void WriteSof();
extern void WriteDri();
extern void WriteDqt();
extern void WriteSos();
extern void WriteDnl();
extern void WriteDht();
extern void ReadSof();
extern void ReadDqt();
extern void ReadDht();
extern void ReadDri();
extern void ReadDnl();
extern void ReadSos();
extern void CheckScan();
extern void MakeConsistentFrameSize();

/* stream.c */

extern void initstream();
extern void pushstream();
extern void popstream();
extern void bpushc();
extern int bgetc();
extern void bputc();
extern void mropen();
extern void mrclose();
extern void mwopen();
extern void swbytealign();
extern void mwclose();
extern long mwtell();
extern long mrtell();
extern void mwseek();
extern void mrseek();
extern int megetb();
extern void meputv();
extern int megetv();
extern int DoMarker();
extern int ScreenMarker();
extern void Resync();
extern void WriteResync();
extern int ReadResync();
extern int ScreenAllMarker();
extern int DoAllMarker();

/* transform.c */

extern void ReferenceDct();
extern void ReferenceIDct();
extern void TransposeMatrix();
extern void Quantize();
extern void IQuantize();
extern void PreshiftDctMatrix();
extern void PostshiftIDctMatrix();
extern void BoundDctMatrix();
extern void BoundIDctMatrix();
extern void ZigzagMatrix();
extern void IZigzagMatrix();
extern int *ScaleMatrix();
extern void PrintMatrix();
extern void ClearMatrix();

#endif
