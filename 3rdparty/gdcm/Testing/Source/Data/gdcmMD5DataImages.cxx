/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// See tst2md5 in GDCM/Utilities/md5
//
// Another way to do it is:
//
// dcmdrle ACUSON-24-YBR_FULL-RLE.dcm bla.dcm
// gdcmraw -i bla.dcm -o bla.raw
// md5sum bla.raw
// So this md5 checksum should match the one in dcmtk...hopefully :)
//
static const char * const gdcmMD5DataImages[][2] = {
/* gdcm 512 512 4 8 1 */
{ "bfff320d1b058e91b4819aa4560c16f7" , "00191113.dcm" },
/* gdcm 256 256 1 16 1 */
{ "d594a5e2fde12f32b6633ca859b4d4a6" , "012345.002.050.dcm" },
/* gdcm 512 512 1 16 1 */
{ "c68cf1c4ae59903930e334498389ea88" , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
/* gdcm 448 336 1 16 1 */
{ "a9d455ca171294fe56d29bf714f4f845" , "05119865-mr-siemens-avanto-syngo.dcm" },
/* gdcm 192 192 1 16 1 */
{ "9acdd9969f5d0584ddd67e994f00b7c7" , "05148044-mr-siemens-avanto-syngo.dcm" },
/* gdcm 512 512 1 16 1 */
{ "c3541091a3794753065022f4defc4343" , "3E768EB7.dcm" },
/* gdcm 768 576 40 8 3 */
{ "6a796be399aefebc1479e924f6051d69" , "ACUSON-24-YBR_FULL_422-Jpeg_Baseline_1.dcm" },
/* gdcm 384 288 1 8 3 */
//{ "2a922d5c606354612bfdbece1421d863" , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
//{ "22b32f23beb118f7b64c13bf04bc2809" , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
{ "2d7a28cae6c3b3183284d1b4ae08307f" , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
/* gdcm 768 576 1 8 3 */
//{ "f9bb8a37acabdf8b0cfa4fd1b471e6aa" , "ACUSON-24-YBR_FULL-RLE.dcm" },
//{ "435c66f7e113d11d226d500294aae865" , "ACUSON-24-YBR_FULL-RLE.dcm" },
{ "429f31f0b70bd515b3feeda5dea5eac0" , "ACUSON-24-YBR_FULL-RLE.dcm" },
/* gdcm 768 576 25 8 3 */
{ "e36350b0711fd34eb86c386164554679" , "ACUSON-8-YBR_FULL-JPEG-TrailingInfo.dcm" },
/* gdcm 608 420 1 8 1 */
{ "7d8858e3419392b7f39a99fdc8028064" , "ALOKA_SSD-8-MONO2-RLE-SQ.dcm" },
/* gdcm 440 440 1 16 1 */
//{ "8acaa88edcc2c29d3be3ee373fbaed5e" , "CR-MONO1-10-chest.dcm" },
{ "1f772b4849727a9750931b60d920436f" , "CR-MONO1-10-chest.dcm" },
/* gdcm 512 512 1 16 1 */
{ "8c8b9d99ad12fb4d231182d4fc14c042" , "CT_16b_signed-UsedBits13.dcm" },
/* gdcm 512 512 1 16 1 */
{ "160c0b4432cfab8d36531b5a3693ff3e" , "CT-MONO2-12-lomb-an2.acr" },
/* gdcm 512 512 1 16 1 */
{ "90cca03ada67c6a1fcb48cfcc2b52eeb" , "CT-MONO2-16-ankle.dcm" },
/* gdcm 512 512 1 16 1 */
{ "a6cf43e05087b6c31644c1d360701ff2" , "CT-MONO2-16-brain.dcm" },
/* gdcm 512 400 1 16 1 */
{ "78bb9ea4b746ff2aa5559be567f95030" , "CT-MONO2-16-chest.dcm" },
/* gdcm 512 512 1 16 1 */
{ "dcb3aa1defd85d93d69d445e3e9b3074" , "CT-MONO2-16-ort.dcm" },
/* gdcm 512 512 1 8 1 */
{ "86d3e09a5858aa3844cb3be1b822a069" , "CT-MONO2-8-abdo.dcm" },
/* gdcm 512 614 1 16 1 */
{ "3695d167c298646b877efccaeff92682" , "CT_Phillips_JPEG2K_Decompr_Problem.dcm" },
/* gdcm 512 512 1 16 1 */
{ "3372195a35448b76daee682d23502090" , "CT-SIEMENS-Icone-With-PaletteColor.dcm" },
/* gdcm 512 512 1 16 1 */
{ "05bbdbcc81081791f6f9f8a1ffa648c8" , "D_CLUNIE_CT1_J2KI.dcm" },
/* gdcm 512 512 1 16 1 */
{ "f3a3d0e739e5f4fbeddd1452b81f4d89" , "D_CLUNIE_CT1_J2KR.dcm" },
/* gdcm 512 512 1 16 1 */
{ "f3a3d0e739e5f4fbeddd1452b81f4d89" , "D_CLUNIE_CT1_JPLL.dcm" },
/* gdcm 512 512 1 16 1 */
{ "f3a3d0e739e5f4fbeddd1452b81f4d89" , "D_CLUNIE_CT1_RLE.dcm" },
/* gdcm 512 512 1 16 1 */
{ "2e389ddbfc1b29d55c52c97e7f2c6f9c" , "D_CLUNIE_CT2_JPLL.dcm" },
/* gdcm 512 512 1 16 1 */
{ "2e389ddbfc1b29d55c52c97e7f2c6f9c" , "D_CLUNIE_CT2_RLE.dcm" },
/* gdcm 3064 4664 1 16 1 */
{ "d9b97ad9199d429960123dcc1e74bdbc" , "D_CLUNIE_MG1_JPLL.dcm" },
/* gdcm 3064 4664 1 16 1 */
{ "02742062fcad004500d73d7c61b9b9e6" , "D_CLUNIE_MG1_JPLY.dcm" },
/* gdcm 3064 4664 1 16 1 */
{ "d9b97ad9199d429960123dcc1e74bdbc" , "D_CLUNIE_MG1_RLE.dcm" },
/* gdcm 512 512 1 16 1 */
{ "7b7424e6115931c371f3c94c2f5d32d9" , "D_CLUNIE_MR1_JPLL.dcm" },
/* gdcm 512 512 1 16 1 */
{ "2824e914ecae250a755a8a0bb1a7d4b1" , "D_CLUNIE_MR1_JPLY.dcm" },
/* gdcm 512 512 1 16 1 */
{ "7b7424e6115931c371f3c94c2f5d32d9" , "D_CLUNIE_MR1_RLE.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "a70676f0e60a58f55a5ac517ff662e7e" , "D_CLUNIE_MR2_JPLL.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "981510df3a57e98141c7d192b45bd93f" , "D_CLUNIE_MR2_JPLY.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "a70676f0e60a58f55a5ac517ff662e7e" , "D_CLUNIE_MR2_RLE.dcm" },
/* gdcm 512 512 1 16 1 */
{ "fb03254fad02d2330d404225c3ea9b4e" , "D_CLUNIE_MR3_JPLL.dcm" },
/* gdcm 512 512 1 16 1 */
{ "d7009808a147f59a9bdf58d5c5924ef2" , "D_CLUNIE_MR3_JPLY.dcm" },
/* gdcm 512 512 1 16 1 */
{ "fb03254fad02d2330d404225c3ea9b4e" , "D_CLUNIE_MR3_RLE.dcm" },
/* gdcm 512 512 1 16 1 */
{ "14fa2ae9f63742af6944edd4a61145e8" , "D_CLUNIE_MR4_JPLL.dcm" },
/* gdcm 512 512 1 16 1 */
{ "a33ad864b49ae7daa59cfaabdf751976" , "D_CLUNIE_MR4_JPLY.dcm" },
/* gdcm 512 512 1 16 1 */
{ "14fa2ae9f63742af6944edd4a61145e8" , "D_CLUNIE_MR4_RLE.dcm" },
/* gdcm 256 1024 1 16 1 */
{ "6b5c1eff0ef65e36b0565f96507e96fd" , "D_CLUNIE_NM1_JPLL.dcm" },
/* gdcm 256 1024 1 16 1 */
{ "812050a7fc53b5735f7740b60969cb6b" , "D_CLUNIE_NM1_JPLY.dcm" },
/* gdcm 256 1024 1 16 1 */
{ "6b5c1eff0ef65e36b0565f96507e96fd" , "D_CLUNIE_NM1_RLE.dcm" },
/* gdcm 1841 1955 1 16 1 */
//{ "01518af70491372814fb056d536ffb7e" , "D_CLUNIE_RG1_JPLL.dcm" },
{ "ae141f6fb91f63769a6adc572a942fb9" , "D_CLUNIE_RG1_JPLL.dcm" },
/* gdcm 1841 1955 1 16 1 */
//{ "01518af70491372814fb056d536ffb7e" , "D_CLUNIE_RG1_RLE.dcm" },
{ "ae141f6fb91f63769a6adc572a942fb9" , "D_CLUNIE_RG1_RLE.dcm" },
/* gdcm 1760 2140 1 16 1 */
{ "06900ee4323a91b7f5ffab8655e3c845" , "D_CLUNIE_RG2_JPLL.dcm" },
/* gdcm 1760 2140 1 16 1 */
{ "27fa50d4cf6b31baa669e9746ce10f63" , "D_CLUNIE_RG2_JPLY.dcm" },
/* gdcm 1760 2140 1 16 1 */
{ "06900ee4323a91b7f5ffab8655e3c845" , "D_CLUNIE_RG2_RLE.dcm" },
/* gdcm 1760 1760 1 16 1 */
//{ "6588b7b8e6e53b2276d919a053316153" , "D_CLUNIE_RG3_JPLL.dcm" },
{ "e7c857ef7e6a2c81498297a072a0332e" , "D_CLUNIE_RG3_JPLL.dcm" },
/* gdcm 1760 1760 1 16 1 */
//{ "cb381c53172242404346b237bf741eb4" , "D_CLUNIE_RG3_JPLY.dcm" },
{ "cc2968949ffbb6548288ffde7e5202e4" , "D_CLUNIE_RG3_JPLY.dcm" },
/* gdcm 1760 1760 1 16 1 */
//{ "6588b7b8e6e53b2276d919a053316153" , "D_CLUNIE_RG3_RLE.dcm" },
{ "e7c857ef7e6a2c81498297a072a0332e" , "D_CLUNIE_RG3_RLE.dcm" },
/* gdcm 2048 2487 1 16 1 */
{ "bd0cccbfd8db465c0af306ba0f482d72" , "D_CLUNIE_SC1_JPLL.dcm" },
/* gdcm 2048 2487 1 16 1 */
{ "994a5abb70d3f5968672ce4970a9d4da" , "D_CLUNIE_SC1_JPLY.dcm" },
/* gdcm 2048 2487 1 16 1 */
{ "bd0cccbfd8db465c0af306ba0f482d72" , "D_CLUNIE_SC1_RLE.dcm" },
/* gdcm 640 480 1 8 3 */
{ "eb52dce9eed5ad677364baadf6144ac4" , "D_CLUNIE_US1_RLE.dcm" },
/* gdcm 756 486 1 8 3 */
{ "b07e34ec35ba1be62ee7d4a404cf0b90" , "D_CLUNIE_VL1_RLE.dcm" },
/* gdcm 756 486 1 8 3 */
{ "d215c88125359d34474a741d793c2215" , "D_CLUNIE_VL2_RLE.dcm" },
/* gdcm 756 486 1 8 3 */
{ "65cd359ea4c6c13ca89b906215a4b762" , "D_CLUNIE_VL3_RLE.dcm" },
/* gdcm 2226 1868 1 8 3 */
{ "e2fdf24d2c03dd0991b4f4e9d6e84ed6" , "D_CLUNIE_VL4_RLE.dcm" },
/* gdcm 2670 3340 1 8 3 */
{ "0ed86ef35d1fb443e1b63c28afe84bd0" , "D_CLUNIE_VL5_RLE.dcm" },
/* gdcm 756 486 1 8 3 */
{ "b825c0ed35c7c896fb707c14b534c233" , "D_CLUNIE_VL6_RLE.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "6111657e6b01ec7b243d63f5dec6ec48" , "D_CLUNIE_XA1_JPLL.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "51af0d83fe795f9c9544c20d0bbac11c" , "D_CLUNIE_XA1_JPLY.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "6111657e6b01ec7b243d63f5dec6ec48" , "D_CLUNIE_XA1_RLE.dcm" },
/* gdcm 117 181 1 8 3 */
{ "b4f442047a209a98af015c89b4a3c4ed" , "DermaColorLossLess.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "0b4dff77726ccf037fa83c42cc186a98" , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" },
/* gdcm 512 512 1 16 1 */
{ "6b92115ec7a394c4aaad74e88d7dbc98" , "FLAIR-wrong-group-length.dcm" },
/* gdcm 1670 2010 1 16 1 */
//{ "9ca80d44bfb1af2f96495fac4b57fa29" , "FUJI-10-MONO1-ACR_NEMA_2.dcm" },
{ "da2415a1e58b4ca2e588d0de18274f60" , "FUJI-10-MONO1-ACR_NEMA_2.dcm" },
/* gdcm 512 301 1 8 1 */
{ "59d9851ca0f214d57fdfd6a8c13bc91c" , "gdcm-ACR-LibIDO.acr" },
/* gdcm 750 750 1 8 1 */
{ "73f20916abaea83abebe9135c365d81a" , "gdcm-CR-DCMTK-16-NonSamplePerPix.dcm" },
/* gdcm 512 512 1 16 1 */
{ "f2c026beea9da0a78404f1299c4628bb" , "gdcm-JPEG-Extended-Allready_present.dcm" },
/* gdcm 512 512 1 16 1 */
{ "f2c026beea9da0a78404f1299c4628bb" , "gdcm-JPEG-Extended.dcm" },
/* gdcm 512 512 1 16 1 */
{ "ad67e448e8923c34f1e019c3395e616c" , "gdcm-JPEG-LossLess3a.dcm" },
/* gdcm 1876 2076 1 16 1 */
{ "c15c1e18a0c41970fbded48b20e834a1" , "gdcm-JPEG-LossLessThoravision.dcm" },
/* gdcm 128 128 1 16 1 */
{ "ad85be428c08ab4166347ef04bda9637" , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },
/* gdcm 160 64 1 16 1 */
{ "2f7f9ef80b49111c5a7cfdb60a97f523" , "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" },
/* gdcm 512 512 1 16 1 */
{ "864e2c5d6acf5a371fe9eaa7ee0dcf5f" , "gdcm-MR-SIEMENS-16-2.acr" },
/* gdcm 640 480 1 16 1 */
{ "f85ff02a143c426edc4b2f6b9a175305" , "gdcm-US-ALOKA-16.dcm" },
/* gdcm 512 512 1 16 1 */
{ "80527e9c17a4a3d12d408e9a354f37f9" , "GE_CT_With_Private_compressed-icon.dcm" },
/* gdcm 512 512 67 8 1 */
{ "b8bcbccd17b76a0f8e3d4c342f855f9f" , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },
/* gdcm 512 512 56 8 1 */
{ "71e4ea61df4f7ada2955799c91f93e74" , "GE_DLX-8-MONO2-Multiframe.dcm" },
/* gdcm 512 512 54 8 1 */
{ "51c998d3474c069b5703e98313258a1e" , "GE_DLX-8-MONO2-PrivateSyntax.dcm" },
/* gdcm 256 256 1 16 1 */
{ "8ac7f7891fb4506e2cd3ae2f0f7e9f46" , "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" },
/* gdcm 256 256 1 16 1 */
{ "1497fb9d7467b1eb36d5618e254aac76" , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" },
/* gdcm 640 480 1 8 3 */
{ "13fd8c7e533a3d7199bb78de45710f5c" , "GE_LOGIQBook-8-RGB-HugePreview.dcm" },
/* gdcm 512 512 1 16 1 */
{ "f3a3d0e739e5f4fbeddd1452b81f4d89" , "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" },
/* gdcm 1792 2392 1 16 1 */
//{ "821acfdb5d5ad9dc13275d3ad3827d43" , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
{ "c1ed06d39821a5fd65abc397982e2ac1" , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
/* gdcm 800 535 1 16 1 */
{ "70166425c4dca767e22d3f25f737922b" , "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" },
/* gdcm 800 535 1 16 1 */
{ "70166425c4dca767e22d3f25f737922b" , "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" },
/* gdcm 800 535 1 16 1 */
{ "70166425c4dca767e22d3f25f737922b" , "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" },
/* gdcm 800 535 1 8 3 */
{ "279e2b0363394a553ff8571cf3540c6c" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" },
/* gdcm 800 535 1 8 3 */
{ "38c2784aa485733fef45b6517479a4f5" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" },
/* gdcm 800 535 1 8 3 */
{ "279e2b0363394a553ff8571cf3540c6c" , "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" },
/* gdcm 800 535 1 8 1 */
{ "fa08fec923f34e009ec89f77232e52ad" , "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" },
/* gdcm 800 535 1 8 1 */
{ "3cd8bd92db17bff54e376885dfefdd8d" , "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" },
/* gdcm 800 535 1 8 1 */
{ "3cd8bd92db17bff54e376885dfefdd8d" , "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" },
/* gdcm 800 535 1 8 3 */
//{ "d613050ca0f9c924fb5282d140281fcc" , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
{ "16e999d6afc5574bcb075f296c3bcbbc" , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
/* gdcm 800 535 1 8 3 */
//{ "d613050ca0f9c924fb5282d140281fcc" , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
{ "16e999d6afc5574bcb075f296c3bcbbc" , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
/* duh ! This should be the exact same md5...if only the testing was a little
 * smarter */
{ "d613050ca0f9c924fb5282d140281fcc" , "LEADTOOLS_FLOWERS-8-PAL-RAW.dcm" },
/* gdcm 50 50 262 16 1 */
{ "ce1cc8ebb1efb86213d5912a1cfde843" , "LIBIDO-16-ACR_NEMA-Volume.dcm" },
/* gdcm 400 100 1 8 3 */
{ "81a40454eec2b18f4331cfd1ba4e501e" , "LIBIDO-24-ACR_NEMA-Rectangle.dcm" },
/* gdcm 128 128 1 8 1 */
{ "fc5db4e2e7fca8445342b83799ff16d8" , "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" },
/* gdcm 512 512 1 16 1 */
{ "b1476cacdd32216b3295d2a494af2945" , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },
/* gdcm 256 192 1 16 1 */
{ "d9f47017de79e8755e4bc5d3c9146ebd" , "MR-Brucker-CineTagging-NonSquarePixels.dcm" },
/* gdcm 512 512 1 16 1 */
{ "8fe67e8e1f849c1b61f59e70d2d53cf7" , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
/* gdcm 256 256 1 16 1 */
{ "f54c7ea520ab3ec32b6303581ecd262f" , "MR-MONO2-12-an2.acr" },
/* gdcm 256 256 1 16 1 */
//{ "48345bccbd67f57b4c13060b6a9a0d35" , "MR-MONO2-12-angio-an1.acr" },
//{ "19cd553b53c8c35b8f2c20f27ed31d2d" , "MR-MONO2-12-angio-an1.acr" },
{ "ae5c00b60a58849b19aaabfc6521eeed" , "MR-MONO2-12-angio-an1.acr" },
/* gdcm 1024 1024 1 16 1 */
{ "a70676f0e60a58f55a5ac517ff662e7e" , "MR-MONO2-12-shoulder.dcm" },
/* gdcm 256 256 1 16 1 */
{ "83be31fb5e5cee60dedaf485bf592ac3" , "MR-MONO2-16-head.dcm" },
/* gdcm 256 256 16 8 1 */
{ "01db0d71100c47013e588082d5f39bab" , "MR-MONO2-8-16x-heart.dcm" },
/* gdcm 256 256 1 16 1 */
{ "b606add66c681bbe674f972799c6d336" , "MR_Philips-Intera_BreaksNOSHADOW.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "0b4dff77726ccf037fa83c42cc186a98" , "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" },
/* gdcm 512 512 1 16 1 */
{ "f69bca6228b0ca07d97ee11c0ab3b989" , "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "0b4dff77726ccf037fa83c42cc186a98" , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" },
/* ?? */
{ "7775ffdd374994b7dd029f45f198844f" , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" },
/* gdcm 484 484 1 8 1 */
{ "8b636107a6d8e6a6b3d1d7eed966d7a0" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
/* gdcm 484 484 1 16 1 */
{ "3027eda10630e5c845f456264dc65210" , "MR-SIEMENS-DICOM-WithOverlays.dcm" },
/* gdcm 128 128 1 16 1 */
{ "6a925f871c58553f84ad24195e155c52" , "MR_SIEMENS_forceLoad29-1010_29-1020.dcm" },
/* gdcm 64 64 13 16 1 */
{ "c83ef2159abef677229d3afd26f9e6a0" , "NM-MONO2-16-13x-heart.dcm" },
/* gdcm 512 512 1 8 1 */
{ "a155c3004bb902ed3f2d78f482923b32" , "OT-MONO2-8-a7.dcm" },
/* gdcm 640 480 1 8 3 */
//{ "47715f0a5d5089268bbef6f83251a8ad" , "OT-PAL-8-face.dcm" },
{ "d7c30d57af821b02c67103250a744235" , "OT-PAL-8-face.dcm" },
/* gdcm 512 512 1 16 1 */
//{ "4b0021efe5a675f24c82e1ff28a1e2eb" , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
{ "d93d2f78d845c7a132489aab92eadd32" , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
/* gdcm 256 256 1 16 1 */
{ "b78366162d9d43b2852d2637c5365c89" , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
/* gdcm 256 256 1 16 1 */
{ "4842ccbaac5b563ce915d2e21eb4c06e" , "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" },
/* gdcm 512 512 76 8 1 */
{ "1f8951a6f8b599ad4ebc97efd7aab6be" , "PHILIPS_Integris_H-8-MONO2-Multiframe.dcm" },
/* gdcm 1024 1024 31 16 1 */
{ "3d005cd3270c6d2562c2a8d9069c9295" , "PHILIPS_Integris_V-10-MONO2-Multiframe.dcm" },
/* gdcm 1024 1024 1 16 1 */
{ "0b4dff77726ccf037fa83c42cc186a98" , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" },
/* gdcm 512 512 1 16 1 */
{ "6c9e477330d70d4a1360888121c7c3d3" , "PICKER-16-MONO2-Nested_icon.dcm" },
/* gdcm 512 512 1 16 1 */
{ "5ea911b29f472f371d21f2da2fd6b016" , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },
/* gdcm 512 512 1 16 1 */
{ "498f80fd27882351b9a09e6ceef470bc" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
/* gdcm 136 92 1 16 1 */
{ "dbbf39ac11a39372b1e961f40ac6f62a" , "RadBWLossLess.dcm" },
/* gdcm 600 430 1 16 3 */
//{ "964ea27345a7004325896d34b257f289" , "rle16sti.dcm" },
{ "f799773abbe36a1a8a3a881e27f8084d" , "rle16sti.dcm" },
/* gdcm 512 512 1 16 1 */
{ "80527e9c17a4a3d12d408e9a354f37f9" , "ser002img00026.dcm" },
/* gdcm 512 512 1 16 1 */
{ "7b55fd124331adde6276416678543048" , "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" },
/* gdcm 256 256 1 16 1 */
{ "ea24c09f475a4e9643e27f6d470edc67" , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr" },
/* gdcm 512 512 1 16 1 */
{ "864e2c5d6acf5a371fe9eaa7ee0dcf5f" , "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" },
/* gdcm 128 128 1 16 1 */
{ "a13466d96b2f068c4844240797069f13" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" },
/* gdcm 128 128 1 16 1 */
{ "73f5986082729c2661cdc8de81fd26d0" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" },
/* gdcm 128 128 1 16 1 */
{ "f932a194df62ec99aef676c563893496" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" },
/* gdcm 128 128 1 16 1 */
{ "b6e4780d8aa8c1d3642377a60a5302dd" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" },
/* gdcm 512 512 1 16 1 */
{ "4b426d4cd570bd4c998f3d19cfddfbb8" , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" },
/* gdcm 192 256 1 16 3 */
{ "faff9970b905458c0844400b5b869e25" , "SIEMENS-MR-RGB-16Bits.dcm" },
/* gdcm 512 512 1 16 1 */
{ "7ccf7c7c4b2a5fa9d337ea8b01f75c42" , "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" },
/* gdcm 192 192 1 16 1 */
{ "a3009bc70444148c5ea2441a099f9dc6" , "SIEMENS_Sonata-12-MONO2-SQ.dcm" },
/* gdcm 256 208 1 16 1 */
{ "017237320ccded3a367f07b44851788e" , "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" },
/* gdcm 512 512 1 8 1 */
{ "f845c8f283d39a0204c325654493ba53" , "test.acr" },
/* gdcm 256 256 1 16 1 */
{ "62c98e89a37c9a527d95d5ac3e2548b0" , "THERALYS-12-MONO2-Uncompressed-E_Film_Template.dcm" },
/* gdcm 256 256 1 16 1 */
{ "0121cd64c3b9957f76dd338d27454bc6" , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },
/* gdcm 512 512 1 16 1 */
{ "09661bd8516aeb5a6f09239f9ca1b092" , "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" },
/* gdcm 636 434 1 8 3 */
{ "23ec8ed09e1ecc353c2e6436a1de6cb2" , "US-GE-4AICL142.dcm" },
/* gdcm 640 480 1 8 1 */
{ "ba092234639594ee9091b46997532cce" , "US-IRAD-NoPreambleStartWith0003.dcm" },
/* gdcm 640 480 1 8 1 */
{ "1bde104ba256fb73528c5d9a02e363d7" , "US-IRAD-NoPreambleStartWith0005.dcm" },
/* gdcm 128 120 8 8 1 */
{ "bf63affde325b3fa81cd5a700f30bd5b" , "US-MONO2-8-8x-execho.dcm" },
/* gdcm 600 430 10 8 3 */
//{ "c70309b66045140b8e08c11aa319c0ab" , "US-PAL-8-10x-echo.dcm" },
{ "1785f4d8af4717c17bfb78ba74c18ea5" , "US-PAL-8-10x-echo.dcm" },
/* gdcm 640 480 1 8 3 */
//{ "fe2d477d699e327be2d3d65eb76203e9" , "US-RGB-8-epicard.dcm" },
{ "eb3433568c6b0cee90688ccf228ffc02" , "US-RGB-8-epicard.dcm" },
/* gdcm 256 120 1 8 3 */
{ "4b350b9353a93c747917c7c3bf9b8f44" , "US-RGB-8-esopecho.dcm" },
/* gdcm 512 512 12 8 1 */
{ "136eaf8f7d654bbb08741c201a945561" , "XA-MONO2-8-12x-catheter.dcm" },


// Those where added manually:
// If you are lucky and the image is raw (MONOCHROME2) you simply need to
// do (assuming you are under UNIX)
// $ ./bin/gdcmraw -i 42166745.dcm -o bla.raw
// $ md5sum bla.raw
// For JPEG and RLE file, you need to check result against dcmtk (dcmdjpeg,
// dcmdrle will help you).
// Image1 & Image2 are crap, some kind of decompression went wrong
// and PixelData is raw so not much you can do...
{ "1b0768a3518a6b6ed425c3c1d7a7ea3b" , "Image1.dcm" },
{ "a41c7f4e75cf637ae8912f5c3cd2c69d" , "Image2.dcm" },
{ "22c9be23446a7be61a90d3578f3c9739" , "deflate_image.dcm" },
{ "d5681b156af55899835293286c57d887" , "brain.dcm" },
{ "d4d365f0500f2ccff932317833d8804b" , "abdominal.dcm" },
{ "138d9bd642c6f1cdc427ef6f99132677" , "ankle.dcm" },
{ "ff8d450e47e8989478a1b6f19d0988cc" , "spine.dcm" },
{ "c78c1721a5ac585a12cf9a52abc25d4c" , "42166745.dcm" },
{ "d48ae6ccc815fd171169591a1048c5ed" , "JDDICOM_Sample4.dcm" },
{ "63c45c2e73403af781d07ae02247654f" , "I20051107115955.dcm" },
{ "938ac04374eadbe6ac4d7df80e5aa178" , "JDDICOM_Sample1.dcm" },
// This one can not be decompressed properly with dcmdjpeg. Until
// they fix dcmdjpeg I'll assume decompression went right
{ "33aa469ec024188d692262d03e7108a0" , "JDDICOM_Sample2.dcm" },
// Same problem
{ "308b1b6fbc01df4dc3fb168830777cb1" , "JDDICOM_Sample3.dcm" },
// Same thing
{ "b8b5030261f92574227fe91902738558" , "JDDICOM_Sample5.dcm" },
// This one was computed from dcmtk, gdcm 1.x fails to read it...
//{ "49ca8ad45fa7f24b0406a5a03ba8aff6" , "rle16loo.dcm" },
{ "04b42f011bdcf56e8a22607cb715447c" , "rle16loo.dcm" },
// Name is poorly choosen since it's actually a Dicom Objects bug
// Cannot only be read by gdcm 1.2.x
{ "f1436c1800ccbba8da82acb7f2dff29d" , "GE_JPEG-Broken.dcm" },
// For some reason the numbers gets funny
{ "73dd2a43ab7c2810714a8ea1079f3c38" , "10200901_8b_Palette_RLE_blinker.dcm" },
{ "609ede096636ae804dbb0fb4d78be2c2" , "10200900_8b_Palette_RLE_blinker.dcm" },
{ "ed6078d83f467f9eda5d8cd06f87080f" , "10200905_8b_Palette_RLE_blinker.dcm" },
// This is the same image as rle16loo ...
{ "49ca8ad45fa7f24b0406a5a03ba8aff6" , "rle16loop_16b_Palette_RLE.dcm" },
// dummy image, this is the same as US-IRAD-NoPreambleStartWith0005.dcm
{ "1bde104ba256fb73528c5d9a02e363d7" , "US.irad.27702.1.dcm"},
// ACR NEMA with a PhotometricInterpretation..
{ "1e3acacba2ae92b52011804ebddbf5df" , "acc-max.dcm" },
{ "d5037d66855c7815f55185b808e48750" , "Originale_015.dcm" },
{ "bbca0e3e26acdd1ab4f6c4871810ac50" , "PETAt001_PT001.dcm" },
{ "5a33e3a2cd414d734984c7a6a5a32a41" , "i0116.dcm" },
{ "5a0ebe04ffe50d4373624e444093b855" , "eclipse_dose.dcm" },
{ "c4e4589b884cfee49db81746cd18a41c" , "MRMN1786V1L.dcm" },
{ "bc8f9791b75916e85c91e458cd1364a3" , "MRMN1786V1T.dcm" },
{ "6cee5011d0dcce93d259d272eb94336f" , "MRMN1786V2L.dcm" },
{ "793d177bd10188e66483ddc04cbca9e7" , "gdcm-CT-GE-16-GantryTilt.dcm" },
{ "7db6d7da455d94ed1a3afda437c3f09e" , "MRMN1786V2T.dcm" },
// JPEG lossy:
{ "bea6e06fe30abd455288e35aaf47477d" , "angiograms.dcm" },
{ "350aea8820dbbdbb15eabc88f3dd3b16" , "image09-bis.dcm" },
// The same as US-IRAD-NoPreambleStartWith0003.dcm
{ "ba092234639594ee9091b46997532cce" , "US.irad.28317.1.dcm" },
{ "21a11f961726c81684be3241eb961819" , "PETAt001_PT204.dcm" },
{ "23b5d0a918fb882c08c458c1507c39f7" , "SiemensIcon.dcm" },
{ "0121cd64c3b9957f76dd338d27454bc6" , "RMI_Mattes_1_150_001_7_150_cEval0_038.dcm" },
{ "350aea8820dbbdbb15eabc88f3dd3b16" , "image09.dcm" },
{ "1bde104ba256fb73528c5d9a02e363d7" , "image12.dcm" },
{ "417ca9bf1db40dd8656cfb6de0aef181" , "JPEGLosslessLongSQ.dcm" },
{ "d5037d66855c7815f55185b808e48750" , "readWrite.dcm" },
// Visible Human written in DICOM with ITK-GDCM (1.2)
{ "e114252b9acf985cf53b06c39118a290" , "vhm.1001.dcm" },
{ "0b4dff77726ccf037fa83c42cc186a98" , "MR_Philips_Intera_Kosher.dcm" },
{ "f9d9898ad844a73c0656539388fec85c" , "0001.dcm" },
{ "c4031c607b7312952bb560e55cbbb072" , "exp000.dcm" },
{ "2bb620b2c516b66faeef7397d30b83f1" , "exp001.dcm" },
{ "0dfbe8aa4c20b52e1b8bf3cb6cbdf193" , "RMI_Mattes_unevenLengthTags.dcm" },
// The famous Siemens 0029-1010/1020
{ "6a925f871c58553f84ad24195e155c52" , "MR_forceLoad29-1010_29-1020.dcm" },
{ "255769e6c9d8a161fce520c958c94de7" , "MR_Siemens_ShadowGroupsImplicitVR.dcm" },
{ "5c25c435ecd8eeac124ab427668186de" , "01f7_7fdf.dcm" },
{ "0dfbe8aa4c20b52e1b8bf3cb6cbdf193" , "RMI_Mattes_1_1_001_7_1_cEval0_000.dcm" },
{ "620012cfec4ae2e6dd0137b2e16947d7" , "ElemZeroNotGroupLength.dcm" },
{ "57994245238730f4b4c5eaed21c4144b" , "File000138.dcm" },
{ "826b6bc1e36efda1b727b08248a71dd2" , "File000139.dcm" },
{ "6b26a5561678b26aa2f21a68ea334819" , "SYNGORTImage.dcm" },
{ "a198893279520631657781d47c4097b2" , "Implicit-0001.dcm" },
{ "99540b33d088646b8e424a57afdccbe6" , "CT_PET001_CT001.dcm" },
{ "343e27f0867b55af74b21a5ba55bd9cc" , "rattag.dcm" },
{ "0038402afc09d44d3c75ed207171976b" , "i0002.dcm" },

{ "0ad4eca25cc0f783bd60c07b2d73f8b0" , "fudd_post.dcm" },
{ "b85ca0e38d91ba0ee3a97f9c709415ac" , "brain_001.dcm" },
{ "f1436c1800ccbba8da82acb7f2dff29d" , "SignedShortLosslessBug.dcm" },
{ "eb87ca4a01e55dc7a7f1c92f0aa31017" , "ATTMA002_DS.dcm" },
{ "f17cf873a3c07f3d91dc88f34664bf0d" , "libido1.0-vol.acr" },
{ "d2fab61e0fff8869e448d69951f1084d" , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },
{ "e0221ddcc9febdc3c266bb8cd0fcf14f" , "undefined_length_un_vr.dcm" },
{ "a136e501bcd1b8ee0835981d2bc8dd87" , "fffc0000UN.dcm" },
{ "33aa469ec024188d692262d03e7108a0" , "JDDICOM_Sample2-dcmdjpeg.dcm" }, // mismatch DICOM / JPEG
{ "d2fab61e0fff8869e448d69951f1084d" , "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" },
{ "fc5db4e2e7fca8445342b83799ff16d8" , "simpleImageWithIcon.dcm" },

// Following need to be proof checked:
{ "0b4dff77726ccf037fa83c42cc186a98" , "imp.dcm" },
{ "eb5fc6c9e86a69c27b7208d72370a92d" , "Renal_Clearance.dcm" },
{ "1b130fa646f8d6b3c14b9443b35c80b4" , "IM0046A0.dcm" },
{ "8c00e7f797d8bb4a629ae1e4b08084b9" , "rvg_analysis.dcm" },
{ "0d7c2f8714ed3c7dec8194c6e4b359c3" , "BRTUM001.dcm" },
{ "e5dbfc8673ea25dd73c95f4df3353b10" , "3DDCM011.dcm" },
{ "12150f9067ea72bc56c75d83c42c8108" , "MIRA.dcm" },
{ "ae9a191c716dfda2651a147728910a89" , "NM.dcm" },
{ "d4444bcca0ee13919f867e486a8f3553" , "QCpatternC-v1.dcm" },
{ "ad85be428c08ab4166347ef04bda9637" , "gdcm-MR-PHILIPS-16-Multi-Seq-Correct.dcm" },
{ "9c390fc010105325427e3ff29a9b186c" , "SIEMENS_ORIGINAL.dcm" },
{ "0b4dff77726ccf037fa83c42cc186a98" , "venticles32004-11-09-08-55-57-6750000000.dcm" },
{ "bfff320d1b058e91b4819aa4560c16f7" , "test.dcm" },
{ "8c9666cef26a0cc76491a7a8a5239b85" , "Renal_Flow.dcm" },
{ "f3a3d0e739e5f4fbeddd1452b81f4d89" , "dclunie.dcm" },
{ "78dae0e2d9b7a124fe63300b6356ded4" , "Toshiba7005.dcm" },
{ "8240955d1e4b72989f8f49549f1617b2" , "PhilipsWith15Overlays.dcm" },
{ "3cd8bd92db17bff54e376885dfefdd8d" , "toto.dcm" },
{ "14ad8ae28cc5e5e5dc4c4d010c291a7f" , "test20.dcm" },
{ "e7408198028983cd6643a00fd7668c00" , "testacr-jpll.dcm" },
{ "6633fb8bce0a648763e7ed2b7de1db5c" , "1_172.dcm" },
{ "29cfe7038fae548abd4af4551b2a37cb" , "example.dcm" },
{ "6b18e445fbc917e7d6f55b616f17d3e1" , "WrongVRFor00271033.dcm" },
{ "0b4dff77726ccf037fa83c42cc186a98" , "ExplicitVRforPublicElementsImplicitVRforShadowElementsCorrected.dcm" },
{ "9570a06da59988ff3a7271cdde14abf5" , "3DT1_5138_0077.dcm" },
{ "20c5f63d65732703e3fffeb8d67c62ed" , "FluroWithDisplayShutter.dcm" },
{ "92b62f22fc44b8563c19f5536d9e499f" , "fudd_ant.dcm" },
{ "b92a22ef55d6a1fbd24a6a24a7a84a7f" , "yazhi.dcm" },
{ "2b69286289543afc3be7df0400abb8fc" , "IM_0001.dcm" },
//{ "67d86e26740882586b6bf3fe7e09c831" , "PMMA-Treppe.dcm" },
{ "287e002ebf84424fe66558b642c51bce" , "PMMA-Treppe.dcm" },
{ "82a8d3398c98d23e1b34f5364f967287" , "PERFORM_T1_Symphony_VA25.dcm" },
//{ "00b4b3282855c96744355892caf9d310" , "Al-Treppe.dcm" },
{ "2c0ccfeff0a6d4afd7929a672dec0cc0" , "Al-Treppe.dcm" },
{ "f891eae2cdfecd578b2204ebbb93f606" , "ImplicitDeclaredAsExplicit.dcm" },
{ "7ceb7886f9fbcb7ae979fdb6f1305414" , "CR-Preamble-NoMeta.dcm" },
{ "d2fab61e0fff8869e448d69951f1084d" , "db318_MR_STN_stimulator_T2_TSE_TRA__20071005161618048_4.dcm" },
{ "2b4d7fb6b039b43145f2766b2339c460" , "0019004_Baseline_IMG1.dcm" },
{ "f02e48f4b7df2ff4855733febc92e4e6" , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr.ovly.6002.dcm" },
{ "6a34b606135c4cd1c31511164e2cb235" , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr.ovly.6000.dcm" },
{ "65f961aca90fb61952d723feec6dec89" , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr.ovly.6006.dcm" },
{ "fc137690ae436bb44a695fb2f1e49d85" , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr.ovly.6004.dcm" },

// No tested:

{ "864e2c5d6acf5a371fe9eaa7ee0dcf5f" , "acr_image_with_non_printable_in_0051_1010.acr" },
{ "7efed1c332c62ea34d5edad901081c36" , "mr_r_opptak000_20000925_t2.nema" },
{ "d6995bd1d76eb36913c985c8397ff02d" , "mr_r_opptak000_20000925_t1.nema" },
{ "dc549f804f1f59bfee245ddf100ee26b" , "SIEMENS-IncompletePixelData.dcm" },
{ "51b7e83cdc569b8f355662d80bd7036d" , "20061119171007906.dcm" },
{ "01db0d71100c47013e588082d5f39bab" , "MR0001.dcm" },
{ "d9df75ff0394bb9a3753b625933d94fc" , "ModalityLUT.dcm" },
{ "acd3f756192a9f0484c971db2869d70b" , "PublicElementsImplicitInExplicitDataSet.dcm" },
{ "820575fcc05338f3e95296107d9ddf7a" , "MR3.dcm" },
{ "3c6984afa2af4257565d8f622f7f5e5b" , "SIEMENS-JPEG-CorruptFragClean.dcm" },
{ "cc2968949ffbb6548288ffde7e5202e4" , "D_CLUNIE_RG3_JPLY_dcmtk.dcm" },
{ "83656f4989a2d4be8f9ad56d83a907f5" , "5decf5b5-6695-4557-867e-a1f97321cd80.dcm" },
{ "92655f77a55eb9c316223b9e73d55800" , "PMS-IncompletePixelData.dcm" },
{ "6a23c9f150a360381c8fbceb2277efc5" , "TOSHIBA-CurveData2.dcm" },
{ "6ad75566935554fcc60ca35a003fba6d" , "IM_00003.dcm" },
{ "ab7cab6e7b5f9043f8fa7c0ae3f1b282" , "19872890.dcm" },
{ "48eb24e37d41faca8b629b3807c20d92" , "rm6_DICM.dcm" },
{ "201c9761ed911ad12508248252f4ff16" , "T2*_1484_0009.dcm" },
{ "90017033c3916259cad63615385a1d02" , "BogugsItemAndSequenceLength.dcm" },
{ "b09dc714f6d56d29005996b3204eec25" , "TOSHIBA-CurveData1.dcm" },
{ "5ed3d1de2a086ac7efd31a6f1463f129" , "SCIsMR.dcm" },
{ "9bbb1d908401299e73e636ad5b19a225" , "Siemens_CT_Sensation64_has_VR_RT.dcm" },
{ "13e41624a8b86c18ac257e3588f818ce" , "tst_ok.dcm" },
{ "17f514732d5f7e9484ed33c35bddc930" , "ImplicitVRInExplicitDataSet.dcm" },
{ "0826b8cc512a8e4d726c1c019af2eaf9" , "29627381.dcm" },
{ "727383f164107aaa9ee6c2182f94b91d" , "1.2.752.24.6.1820537353.20080220145707750.8.1.dcm" },
{ "5588652e673e745ad56441f5a26c2305" , "22484474.dcm" },
{ "cf6fde116c90e4e56512b22004503d96" , "PHILIPS_PrivateSeqWithPublicElems.dcm" },
{ "098bba76d4e75a3148c9b62aee35f950" , "GEMS-IncompletePixelData.dcm" },
{ "1fe71a52a211d1a56cc49cdcb71c1dbb" , "IM03.dcm" },
{ "b857b6cce2afd7033476526b89ed9678" , "TheralysGDCM1.dcm" },
{ "e998bbfca8b892fd0955c95b6f1584ea" , "Theralys2048Thingy.dcm" },
{ "0508d28682e209f87d28184ae10060bd" , "Compress_16bit_noPVRG.dcm" },
{ "90017033c3916259cad63615385a1d02" , "BogugsItemAndSequenceLengthCorrected.dcm" },
{ "51b7e83cdc569b8f355662d80bd7036d" , "debian_medcon_401529.dcm" },
{ "51b7e83cdc569b8f355662d80bd7036d" , "BogusItemStartItemEnd.dcm" },
{ "3c6984afa2af4257565d8f622f7f5e5b" , "SIEMENS-JPEG-CorruptFrag.dcm" },
{ "192f5ebd4d8f6164a4142421408172b2" , "PhilipsByteSwapping.dcm" },
{ "2438b8feef6c8f682b2f468aff4475e5" , "MismatchSOPClassUID.dcm" },
{ "738092e703655b6ae22a907d9f9f0c9c" , "YBRisGray.dcm" },
{ "635e947fc156c89a8d10ced8317ace82" , "2mAs.dcm" },
{ "54d648704507de7cd6e4be12061a5fb2" , "OSIRIX_MONOCHROME2_BUG.dcm" },
{ "1db40fc52b962c8b6823980fe9dad6d8" , "5mAs.dcm" },
{ "1b130fa646f8d6b3c14b9443b35c80b4" , "IM0046A0_2.dcm" },
{ "cdfb45de5ddad3e7e2b0053c848d0d2b" , "ImplicitVRInExplicitDataSet2.dcm" },
{ "ec5c421832df7cc43801d038705cd2cf" , "OverlayPrivateDataInBetween.dcm" },
{ "9a375c3b2a72a6ccf687250b17f646c9" , "ItemTerminatorVLnot0.dcm" },
{ "5269176094127c6f106cbcf9dbdf55b0" , "I0000017.dcm" },
{ "eb87ca4a01e55dc7a7f1c92f0aa31017" , "EmptyItemStarter.dcm" },
{ "cae8f644373163cb3970999f7ac00fd3" , "169585.dcm" },
{ "5ea911b29f472f371d21f2da2fd6b016" , "VRUNInMetaHeader.dcm" },
{ "ad7b664eed26e95f790c24cdb3060fb9" , "IM000450.dcm" },
{ "4f113f33863067b8ca8d560cb194da09" , "TOSHIBA-CurveData3.dcm" },
{ "14fa2ae9f63742af6944edd4a61145e8" , "mrbrainR.dcm" },
{ "ba17ae006314a097fef7a32fa1a624b0" , "c42g57hg64ca02.dcm" },
{ "10047ec574a6401ad90606e63304de6e" , "0020.DCM" },
{ "5184839806c9a9583fe192d43cede939" , "001005XA" },
{ "5291e36a7d54d4c8e00d5765cd0b3d17" , "001007XA" },
{ "a57df7bc23977977832f98bfb662cdb4" , "001003XA" },
{ "b36329c27f359daede3205d7c57cccd1" , "001009XA" },
{ "1d173a6f31e0e4dd1078adf3557774c7" , "001008XA" },
{ "9b42aa4e5d3b5f889164b5a734b954b3" , "001010XA" },
{ "87f72a961292c83f8a9230e8eefecc6b" , "001004XA" },
{ "6c7a303d25b418988ef9a37c4315a746" , "001001XA" },
{ "9dde002e4e99218891df5f98da56ec9d" , "001002XA" },
{ "6c7a303d25b418988ef9a37c4315a746" , "001001XA.1" },
{ "8891d44376bc53c7bd0d36d2b776cd9b" , "001006XA" },
{ "a6d10a3de9499d8853c6667b82628e86" , "TOSHIBA-EnhancedCT.dcm" },
{ "71b3b19571f5b9e7ec931071bf5157dd" , "CroppedArm.dcm" },
{ "bab06b7112dbce4f528bab5c1b89abc1" , "Bug_Philips_ItemTag_3F3F.dcm" },


{ "8d3a64a67b4d8d15561fb586fd0706ee" , "Nm.dcm" },
{ "05392d4f7a0c05223eeb957ee60203a9" , "MergeCOM3_330_IM50.dcm" },
{ "635c2dbedea549a89f88e959934ed93c" , "ELSCINT1_OW.dcm" },
{ "594995a6eb12a565247cd98f967a378d" , "KONICA_VROX.dcm" },
{ "1497b865a4c6ab73cd6797b8834baa9f" , "TheralysNoModalityNoObject.dcm" },
{ "d1f8cbdb279d038e2674ec0907afffe1" , "gastrolateral.dcm" },
{ "ea24c09f475a4e9643e27f6d470edc67" , "SIEMENS_GBS_III-16-ACR_NEMA_1-ULis2Bytes.dcm" },
{ "ecd26fb1fa914918ff9292c4f7340050" , "MR00010001.dcm" },
{ "c56567bbb8f1924d6c0c6fd8ca7f239d" , "006872.003.107.dcm" },
{ "228e9af71b8ce00cae3066e3fdd3641f" , "SIEMENS_MONOCHROME2_LUT_MOCO.dcm" },
{ "ef8c3435ee7a9b332976f2dc56833d3a" , "GENESIS_SIGNA-JPEG-CorruptFrag.dcm" }, // FIXME !
{ "42a4f33eb34456e3e82f30d707c84870" , "DCMOBJ_IMG57.dcm" },
{ "3a85617df95abbb63cd84a183515c697" , "image.acr" },
{ "8c5a627c461688cfa5dc708a170c5eda" , "IM-0001-0001.dcm" },
{ "ec87a838931d4d5d2e94a04644788a55" , "test_att.acr" },
{ "0621954acd5815e0b4f7b65fcc6506b1" , "SIEMENS_ImageLocationUN.dcm" },

{ "e1e34e81050d17b07a20c0e43c355187" , "GDCMFakeJPEG.dcm" },
{ "ef9f915086db838334ddc656a10486f2" , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" },
{ "498f80fd27882351b9a09e6ceef470bc" , "ELSCINT1_GDCM12Bug.dcm" },
{ "34b5c1ce40f09f0dbede87ebf3f6ed3c" , "korean_agfa_infinitt_2008-3.dcm" },
{ "2c8b8ee9950582af472cf652005f07d4" , "c_vf1001.dcm" },
{ "a1ea6633f17ef1e0d702cdd46434b393" , "MARTIN_ALBERT-20010502-2-1.nema" },
{ "d85237c7ff838f5246643a027d3727ae" , "GDCMPrinterDisplayUNasSQ.dcm" },
{ "2dbe2da7fbcf1bd73e2221e9cd4ad7ee" , "MissingPixelRepresentation.dcm" },
{ "a3009bc70444148c5ea2441a099f9dc6" , "E00001S03I0015.dcm" },
{ "afe5937be25ac657e48b1270b52e6d98" , "Martin.dcm" },



{ "4d790e17ee35572d64e37c55dbc36725" , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
{ "93ddc0c3d642af60f55630232d2801ae" , "CT_Image_Storage_multiframe.dcm" },
{ "6127740892542f23e37e6ab516b17caf" , "ELSCINT1_JP2vsJ2K.dcm" },
{ "b620a57170941e26dfd07ff334c73cb4" , "GE_MR_0025xx1bProtocolDataBlock.dcm" },
{ "4b5423b34ab4e104c222359a91448a5d" , "CT-SIEMENS-MissingPixelDataInIconSQ.dcm" },
{ "79b8705f2e6c7464bd3e2fc7e1d3483b" , "KODAK_CompressedIcon.dcm" },
{ "59071590099d21dd439896592338bf95" , "ima43.dcm" },
{ "46bf12c412590767bb8cd7f0d53eaa87" , "TG18-CH-2k-01.dcm" },
{ "9214cc4f62fbea873ffad88e1be877c5" , "LJPEG_BuginGDCM12.dcm" },
{ "52607a16af1eaddbbc71c14d32e489d8" , "DX_J2K_0Padding.dcm" },


{ "6af53848fe77feb56a12aba74dadea8e" , "TheralysGDCM120Bug.dcm" },
{ "79b8705f2e6c7464bd3e2fc7e1d3483b" , "TestVRSQUN2.dcm" },
{ "71e4ea61df4f7ada2955799c91f93e74" , "TestVRSQUN1.dcm" },

{ "6396332b75b15bf30b1dd1cd0f212691" , "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" },
{ "ba8dae4b43075c7e8562f5addf5f95c3" , "DX_GE_FALCON_SNOWY-VOI.dcm" },
{ "d6fb1fb06318dd305a461db9c84cf825" , "JPEG_LossyYBR.dcm" },
{ "43f33c06d56f239ce9ed5ca0d66a69d2" , "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" },
{ "dae9d2e2b412646fd0a0f31dc2d17aa4" , "BugGDCM2_UndefItemWrongVL.dcm" },


{ "62687f0a17e9c4153f18b55c8abfcef3" , "SIEMENS_CSA2.dcm" },
{ "a02fa05065b3a93a391f820ac38bd9ee" , "MAROTECH_CT_JP2Lossy.dcm" },
{ "8b5e38699887158c3defd47900a68fc5" , "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" },


{ "c57035e2dac52e339b27e8c965251b3d" , "DCMTK_JPEGExt_12Bits.dcm" }, // checked with dcmdjpeg v3.5.4+ 2009-05-07

{ "d7673c8575cb2765f8ae25aa3899c77e" , "PHILIPS_GDCM12xBug.dcm"},
{ "a597540a79306be4710f4f04497fc23a" , "PHILIPS_GDCM12xBug2.dcm"},

{ "1e8843c2d247f9e9e7a44c9c6de43f6d" , "multiframegrayscalewordscis.dcm" },
{ "928b41468193f0eecaea216866bbe735" , "signedtruecoloroldsc.dcm" },
{ "209bc9b02004a712f0436a1ca5e676b4" , "multiframesinglebitscis.dcm" },
{ "c8698fa1ec0b227113f244954b8e88f4" , "multiframegrayscalebytescis.dcm" },
{ "dce1513162a762bf43dcc3c9d5c5c3f7" , "multiframetruecolorscis.dcm" },

{ "6bf95a48f366bdf8af3a198c7b723c77" , "SinglePrecisionSC.dcm" },

{ "a870a7a7cab8c17646d118ae146be588" , "MR16BitsAllocated_8BitsStored.dcm" },

{ "da153c2f438d6dd4277e0c6ad2aeae74" , "OsirixFake16BitsStoredFakeSpacing.dcm" },
{ "1c485e1ac2b2bdbeeba14391b8c1e0c8" , "JPEGDefinedLengthSequenceOfFragments.dcm" },

{ "ae1290d59c63b0c334a4834c5995fe45" , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" },

{ "f3a3d0e739e5f4fbeddd1452b81f4d89" , "D_CLUNIE_CT1_JLSL.dcm" },
{ "5e9085c66976d2f5f9989d88bf7a90c4" , "D_CLUNIE_CT1_JLSN.dcm" },

{ "9eb513314b2fcf25d895e18ffb2ead0b" , "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
{ "12d1567ed81236cf3b01dc12766581a0" , "IM-0001-0066.CommandTag00.dcm" },
{ "6f26e552a1b71d386483118779d192ad" , "NM_Kakadu44_SOTmarkerincons.dcm" },
{ "56238d3665ebdb0251d1161fb7f4edc6" , "GDCMJ2K_TextGBR.dcm" },
{ "f8a1f4ce85b51527267e670a8aa0c308" , "PhilipsInteraSeqTermInvLen.dcm" },

// following 3 md5 computes with kdu_expand v6.3.1
{ "d6347ed051d7b887bdaad1a91433c6ba" , "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
{ "d6347ed051d7b887bdaad1a91433c6ba" , "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
{ "94414d8b4300aa3d8cbe4475d34e8e54" , "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },

{ "304f147752d46adfdcff71a30cd03d0a" , "NM-PAL-16-PixRep1.dcm" },
{ "20b1e4de7b864d44fcd0dda1fc42402c" , "MEDILABInvalidCP246_EVRLESQasUN.dcm" },

// gdcmDataExtra:
{ "cb26bfacea534b5cd6881bc36520ecfc" , "US_512x512x2496_JPEG_BaseLine_Process_1.dcm" },
{ "d40e2d27988f0c546b0daeb67fcbfba8" , "i32.XADC.7.215MegaBytes.dcm" },
{ "c383b244fd43cb0a9db764b71cb59741" , "1.3.46.670589.7.5.10.80002138018.20001204.181556.9.1.1.dcm" },

{ "f111ed4eea73f535261039d3f7b112e9" , "JPEGInvalidSecondFrag.dcm" },

// Computed with kakadu:
{ "a218fcee00e16d430f30ec7ebd4937dc" , "lena512_rot90.j2k.dcm" },

{ "dbf83984984741c98205d77a947b442c" , "SC16BitsAllocated_8BitsStoredJ2K.dcm" },
{ "c164a73ba18ab4e88977921ffc7c3a65" , "SC16BitsAllocated_8BitsStoredJPEG.dcm" },


/* Stopping condition */
{ 0 ,0 }
};
