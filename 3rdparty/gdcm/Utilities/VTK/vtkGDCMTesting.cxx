/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMTesting.h"

#include "vtkObjectFactory.h"
#include "vtkToolkits.h"
#include "gdcmTesting.h"
#include "gdcmFilename.h"

vtkCxxRevisionMacro(vtkGDCMTesting, "$Revision: 1.31 $")
vtkStandardNewMacro(vtkGDCMTesting)

// DICOM Filename, MHD MD5, RAW MD5
static const char * const vtkgdcmMD5MetaImages[][3] = {
{ "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm","3d77fa60702897dd0ad601ee728f93d3", "8b636107a6d8e6a6b3d1d7eed966d7a0" },
{ "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm","3b22e8e2aea4354ed947786152504d4d", "cd301effb5e2020f539a2f5d24a429d7" },
{ "SIEMENS_GBS_III-16-ACR_NEMA_1.acr","3864799d39588714e16c3a4482137c74", "ea24c09f475a4e9643e27f6d470edc67" },
{ "SIEMENS_GBS_III-16-ACR_NEMA_1-ULis2Bytes.dcm","f285ebea4a7675457b916c16ff770934", "ea24c09f475a4e9643e27f6d470edc67" },
{ "test.acr","06165c1b4e64898394fdc0598eb5971a", "f845c8f283d39a0204c325654493ba53" },
{ "MR-MONO2-12-angio-an1.acr","849ce36c711f742e64c5bd5a161bb849", "ae5c00b60a58849b19aaabfc6521eeed" },
{ "CT-MONO2-12-lomb-an2.acr","554cc0adb0dea2023d255d0eccbb21a6", "672d4cd82896c2b0dc09b6fe3eb748b0" },
{ "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr","ac6311f0db183e427e69650ba2e632b4", "a909e448b66303df6f5a512d169921d5" },
{ "gdcm-MR-SIEMENS-16-2.acr","670ba858ad6f6eefb4b59cb24e5a3b31", "864e2c5d6acf5a371fe9eaa7ee0dcf5f" },
{ "LIBIDO-8-ACR_NEMA-Lena_128_128.acr","e186747fca1e99d9f5b0c1607f19a9cb", "fc5db4e2e7fca8445342b83799ff16d8" },
{ "libido1.0-vol.acr","b10050e7aa8fa72228d8d33bf0504238", "f17cf873a3c07f3d91dc88f34664bf0d" },
{ "gdcm-ACR-LibIDO.acr","eb59c5fd7a6cc0bc1223e15e08a3637d", "59d9851ca0f214d57fdfd6a8c13bc91c" },
{ "MR-MONO2-12-an2.acr","356cb716cc543209113c0e2b535faba9", "f54c7ea520ab3ec32b6303581ecd262f" },
{ "012345.002.050.dcm","ea9ca1be179f269583c5c27e89c0a46d", "d594a5e2fde12f32b6633ca859b4d4a6" },
{ "GE_MR_0025xx1bProtocolDataBlock.dcm","c9d8d1f0e94a06a98469926ec83a08da", "b620a57170941e26dfd07ff334c73cb4" },
{ "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm","91fb4352059cf7f7ba023599ec2c39a9", "8fe67e8e1f849c1b61f59e70d2d53cf7" },
{ "gdcm-JPEG-LossLessThoravision.dcm","e5cb57f774e3be543169af242d27c93d", "c15c1e18a0c41970fbded48b20e834a1" },
{ "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm","0a6af461c022adb1cdbef8836a3426f7", "4b426d4cd570bd4c998f3d19cfddfbb8" },
{ "MR-MONO2-12-shoulder.dcm","ab968254870fdc9e41055d2adfe9ffef", "a70676f0e60a58f55a5ac517ff662e7e" },
{ "PICKER-16-MONO2-No_DicomV3_Preamble.dcm","ff5f52a1ffd97d0cc6dad2c59e371b95", "5ea911b29f472f371d21f2da2fd6b016" },
{ "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm","92629f1a782a3d10998d01a8af5df991", "16e999d6afc5574bcb075f296c3bcbbc" },
{ "D_CLUNIE_CT1_JPLL.dcm","74360148191955d7e5664952d85ad01c", "a109da372df2b85ff0a705fba62e9589" },
{ "DMCPACS_ExplicitImplicit_BogusIOP.dcm","3f434e2196fbe53d0024704c707254e9", "ef9f915086db838334ddc656a10486f2" },
{ "PHILIPS_GDCM12xBug2.dcm","f3bb3e8fbf262a37621a805357b0b1c6", "a597540a79306be4710f4f04497fc23a" },
{ "PICKER-16-MONO2-Nested_icon.dcm","2642d6b9be2d28d57f85abffc661bdb7", "954c99e48a083cc54de16ad07d553005" },
{ "RadBWLossLess.dcm","1cbcd5c44f8ffe7c448520971233afb4", "dbbf39ac11a39372b1e961f40ac6f62a" },
{ "gdcm-JPEG-Extended.dcm","4873a6ad617c2a8f86d130c853c5e127", "f05174b105604a462af0fa733fceebde" },
{ "DX_J2K_0Padding.dcm","6986a0e25c3c39e30ad30f5cdb17cbfe", "52607a16af1eaddbbc71c14d32e489d8" },
{ "CT-MONO2-16-ort.dcm","f758d2ad7de59ccf49473db31bb07825", "ca3c1630965318e6321daac7b39d8883" },
{ "D_CLUNIE_XA1_RLE.dcm","6992adfe56cdd43fdb5c4ae832be0bb3", "6111657e6b01ec7b243d63f5dec6ec48" },
{ "OT-MONO2-8-a7.dcm","7611df3cbfd9ac2e137a78d732d98603", "a155c3004bb902ed3f2d78f482923b32" },
{ "fffc0000UN.dcm","1ca6fabea6962185ade0592a7e0db06a", "a136e501bcd1b8ee0835981d2bc8dd87" },
{ "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm","9e1e8d58a04142531c21f3a39664edb0", "70166425c4dca767e22d3f25f737922b" },
{ "gdcm-US-ALOKA-16.dcm","6dfc2e8a30cd515b12a3c0891b1496b5", "f85ff02a143c426edc4b2f6b9a175305" },
{ "TG18-CH-2k-01.dcm","ff2121ad1e7471cd9f5c75d74664c8a1", "46bf12c412590767bb8cd7f0d53eaa87" },
{ "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm","5d007c4979f5c401c623b8b1d095a339", "c68cf1c4ae59903930e334498389ea88" },
{ "D_CLUNIE_RG1_RLE.dcm","e39a383f71c5f996cd2116e0312fc808", "ae141f6fb91f63769a6adc572a942fb9" },
{ "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm","675cb24b99d9c6cca3259e5e87bb6aea", "7f4158c2946981465bf409f2433ebaa7" },
{ "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm","778a5b6f00dd312b2ecad0deac6b9577", "7775ffdd374994b7dd029f45f198844f" },
{ "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm","bd7bdc10de0df526ee6858da2c13018c", "70166425c4dca767e22d3f25f737922b" },
{ "gdcm-JPEG-LossLess3a.dcm","62822559a854373b843ce8de2cc3ed7d", "a341e8cb1ec2aa07cb0e5b71bbe87087" },
{ "D_CLUNIE_MR1_JPLY.dcm","0a4ce0b9f33ccf29ced7537327052978", "2824e914ecae250a755a8a0bb1a7d4b1" },
{ "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm","a53035663a2de05df96871cf6a2773f1", "d2fab61e0fff8869e448d69951f1084d" },
{ "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm","adecccb2b55c791fb9a5f9adb9b5c7be", "b78366162d9d43b2852d2637c5365c89" },
{ "SignedShortLosslessBug.dcm","dbf67ea5171e87a11ffbadc1ac8af3e8", "bdec90fcb90f68b34f0a306c40443610" },
{ "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm","cc1e8af8bb7bf4d1e9ec20bad537cdfd", "f932a194df62ec99aef676c563893496" },
{ "BugGDCM2_UndefItemWrongVL.dcm","84fc423a48309bdb0256ace6c7cf1458", "dae9d2e2b412646fd0a0f31dc2d17aa4" },
{ "US-PAL-8-10x-echo.dcm","e83577771dbdb47f3046277bcdfe2758", "1785f4d8af4717c17bfb78ba74c18ea5" },
{ "MR_Philips-Intera_BreaksNOSHADOW.dcm","aa9341fd51dd1dd640fa6420b274f507", "b606add66c681bbe674f972799c6d336" },
{ "D_CLUNIE_RG3_RLE.dcm","df3d8699a986204038e3f9debde3a1d2", "e7c857ef7e6a2c81498297a072a0332e" },
{ "JPEGDefinedLengthSequenceOfFragments.dcm","cc21f41e99625aac7308d9fdd01481d6", "484754f6dd1cc59325e9a5bbf76c5f2c" },
{ "SIEMENS-MR-RGB-16Bits.dcm","56f175c3a227ca322245e90a625470a4", "faff9970b905458c0844400b5b869e25" },
{ "JDDICOM_Sample2.dcm","76ef935051cf0e816a0e8864f88d4814", "33aa469ec024188d692262d03e7108a0" },
{ "D_CLUNIE_MR2_JPLL.dcm","c3281801885ac42ec0e82af71636798d", "a70676f0e60a58f55a5ac517ff662e7e" },
{ "PHILIPS_Intera-16-MONO2-Uncompress.dcm","89fc7c6e5ccc5ed3320197d9bdfc4dc6", "0b4dff77726ccf037fa83c42cc186a98" },
{ "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm","2a8441911c2135e94cd16f231c7ad8e6", "0121cd64c3b9957f76dd338d27454bc6" },
{ "D_CLUNIE_CT2_RLE.dcm","74e31e7ac318cdcaffb172277f166e2e", "2e389ddbfc1b29d55c52c97e7f2c6f9c" },
{ "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm","6f97e45b637beb9282e85c7b81b1473b", "0b4dff77726ccf037fa83c42cc186a98" },
{ "D_CLUNIE_MR2_RLE.dcm","8bc2159b48a5a069ac7371f69694d761", "a70676f0e60a58f55a5ac517ff662e7e" },
{ "SIEMENS_ImageLocationUN.dcm","4fd3c5e0b413ccfbec2b4706ad23200b", "0621954acd5815e0b4f7b65fcc6506b1" },
{ "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm","a68236a0584e518d5ca9fce9136da38f", "09661bd8516aeb5a6f09239f9ca1b092" },
{ "ACUSON-24-YBR_FULL-RLE-b.dcm","479d056e0a591bc2850eef632fc2345c", "22b32f23beb118f7b64c13bf04bc2809" },
{ "D_CLUNIE_RG3_JPLL.dcm","351293952be98d7ce8e73ab574648daa", "e7c857ef7e6a2c81498297a072a0332e" },
{ "AMIInvalidPrivateDefinedLengthSQasUN.dcm","0f5c34ef47b32554270a52201ce26daf", "ae1290d59c63b0c334a4834c5995fe45" },
{ "CT-SIEMENS-Icone-With-PaletteColor.dcm","f0dc9d6c14c010948fdb64a4c6466565", "c6bdfac1d97c4dc60423b5b63a39a64a" },
{ "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm","d351dce81724bbd2b0bea467ad6942ff", "a13466d96b2f068c4844240797069f13" },
{ "LJPEG_BuginGDCM12.dcm","fb9e4419f90eaa9c2a6c4ae98b750348", "9214cc4f62fbea873ffad88e1be877c5" },
{ "XA-MONO2-8-12x-catheter.dcm","639e46ee89a1204df2ed708e33c3357b", "136eaf8f7d654bbb08741c201a945561" },
{ "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm","e086859c0b556c3d22d51bf2a05db26a", "70166425c4dca767e22d3f25f737922b" },
{ "TheralysGDCM120Bug.dcm","3bbfb2b54ec9d9e3f89fe046e5a954a4", "6af53848fe77feb56a12aba74dadea8e" },
{ "D_CLUNIE_VL1_RLE.dcm","668904d73866455812fee0aabdbcff1e", "b07e34ec35ba1be62ee7d4a404cf0b90" },
{ "D_CLUNIE_MR1_JPLL.dcm","087e123efb3e18f5e7693390926f08c6", "7b7424e6115931c371f3c94c2f5d32d9" },
{ "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm","61c2a3757ad053c52eeeaa94f5d6cadd", "b8bcbccd17b76a0f8e3d4c342f855f9f" },
{ "SIEMENS_Sonata-12-MONO2-SQ.dcm","91ee14eaab0a6f98b366ff3406b8d295", "a3009bc70444148c5ea2441a099f9dc6" },
{ "3E768EB7.dcm","a8f6eb80f1aebdcc2716435b7049be6c", "a0fcdd5b68358d7078421e260fa749b7" },
{ "DX_GE_FALCON_SNOWY-VOI.dcm","d3efd7ae0f764f26c22dfbfec46b13e9", "ba8dae4b43075c7e8562f5addf5f95c3" },
{ "US-RGB-8-epicard.dcm","d139728ec75594e11fd2d489049756f5", "fe2d477d699e327be2d3d65eb76203e9" },
{ "D_CLUNIE_NM1_JPLL.dcm","0210138626fc112253519cd202dac179", "6b5c1eff0ef65e36b0565f96507e96fd" },
{ "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm","3ec03618a9dcb12c636dae46cfa12bc5", "864e2c5d6acf5a371fe9eaa7ee0dcf5f" },
{ "MR_SIEMENS_forceLoad29-1010_29-1020.dcm","d887c10d6058b568dd3a07c573faf5e1", "6a925f871c58553f84ad24195e155c52" },
{ "D_CLUNIE_CT1_RLE.dcm","87ca724fcddef83277f918f7e7d4b235", "a109da372df2b85ff0a705fba62e9589" },
{ "D_CLUNIE_XA1_JPLY.dcm","df8bca314462d849da13af4c971eb616", "51af0d83fe795f9c9544c20d0bbac11c" },
{ "D_CLUNIE_MR3_JPLY.dcm","c2872e85ff9f155d80b6c703add04d8f", "d7009808a147f59a9bdf58d5c5924ef2" },
{ "D_CLUNIE_CT1_J2KI.dcm","2923abe11b2f7898af31cfada0005cea", "bc127eee2ebf5f2ee2a6a1daeac364ce" },
{ "CT_16b_signed-UsedBits13.dcm","2122e1b307aa75d602c665852a2daaff", "8c8b9d99ad12fb4d231182d4fc14c042" },
{ "D_CLUNIE_VL6_RLE.dcm","04550649448200f4f3b5e478044c021a", "b825c0ed35c7c896fb707c14b534c233" },
{ "DCMTK_JPEGExt_12Bits.dcm","1a9268135f0c200ffee17babd894352d", "c57035e2dac52e339b27e8c965251b3d" },
{ "D_CLUNIE_MR4_RLE.dcm","d682f844f5cc19df26dc9acf434da818", "14fa2ae9f63742af6944edd4a61145e8" },
{ "simpleImageWithIcon.dcm","eaf8ce3d45bf4f6e4991db6acbb4d673", "fc5db4e2e7fca8445342b83799ff16d8" },
{ "gdcm-MR-PHILIPS-16-Multi-Seq.dcm","0af3267489f800c8e272f607f1d9a938", "ad85be428c08ab4166347ef04bda9637" },
{ "D_CLUNIE_VL3_RLE.dcm","8ba4764396aa1d1c7fd84bf4132b273e", "65cd359ea4c6c13ca89b906215a4b762" },
{ "CT-MONO2-16-brain.dcm","cdcf9ee879a4159cc686004f76745148", "a6cf43e05087b6c31644c1d360701ff2" },
{ "CT-MONO2-16-ankle.dcm","803a459d3a6df6473c3f3e46129df147", "13e853d75ffe289b8210d1336e2394dd" },
{ "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm","4bb845dfa4b1f9b46fa53006e82bf4c8", "4d790e17ee35572d64e37c55dbc36725" },
{ "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm","e91814bf121ae0dad75be63fbf1b1e93", "38c2784aa485733fef45b6517479a4f5" },
{ "undefined_length_un_vr.dcm","fd3d95cd61c0fc391733f28f607fa506", "e0221ddcc9febdc3c266bb8cd0fcf14f" },
{ "JPEG_LossyYBR.dcm","351424d0392db74bc69cf6f6537ddb88", "d6fb1fb06318dd305a461db9c84cf825" },
{ "rle16loo.dcm","fd66e0089e5c343d3f24642710eb0129", "04b42f011bdcf56e8a22607cb715447c" },
{ "D_CLUNIE_MR2_JPLY.dcm","013b4754a4224b10cb503dead5a6a79e", "981510df3a57e98141c7d192b45bd93f" },
{ "DermaColorLossLess.dcm","2d02d5385526970fffbb5cebafd6b4df", "b4f442047a209a98af015c89b4a3c4ed" },
{ "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm","601f6ff54e324ccc4e9e5627c0593027", "c1ed06d39821a5fd65abc397982e2ac1" },
{ "D_CLUNIE_RG1_JPLL.dcm","08a9f8187cfad52150fffc80e76efa9e", "ae141f6fb91f63769a6adc572a942fb9" },
{ "D_CLUNIE_RG3_JPLY.dcm","3c39fb636e231a66e425d6c8645e086c", "cc2968949ffbb6548288ffde7e5202e4" },
{ "D_CLUNIE_MR3_RLE.dcm","847c8b74671107b463b90cbfa6535f7f", "fb03254fad02d2330d404225c3ea9b4e" },
{ "NM-MONO2-16-13x-heart.dcm","7bfdb71668190142a547899329c1d9c3", "c83ef2159abef677229d3afd26f9e6a0" },
{ "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm","bcfdeefa6f9808ae020586b164a29d99", "fa08fec923f34e009ec89f77232e52ad" },
{ "CT-MONO2-16-chest.dcm","f9110218ff6bd9310e6eb089fdb4ab0a", "78bb9ea4b746ff2aa5559be567f95030" },
{ "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm","aafebb62a52ad79ca0aea428fdfc7d1e", "017237320ccded3a367f07b44851788e" },
{ "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm","3e1c33b362a0fb5a3c3009e681a6e546", "e5a0f7083de19fcc63dfdec9d754470f" },
{ "US-GE-4AICL142.dcm","a9142c48ce2499d1f67e67db1b6f87d6", "23ec8ed09e1ecc353c2e6436a1de6cb2" },
{ "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm","68b7e492026a8a4da1e579802c2ef1ac", "2f7f9ef80b49111c5a7cfdb60a97f523" },
{ "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm","7433cf1900fbc331517e196daca6a4ef", "6396332b75b15bf30b1dd1cd0f212691" },
{ "KODAK_CompressedIcon.dcm","2ad7b4697b63a2fe0082374ff6ea58d8", "79b8705f2e6c7464bd3e2fc7e1d3483b" },
{ "D_CLUNIE_US1_RLE.dcm","14baf3e73f87d5b0893a9f305b11c87e", "eb52dce9eed5ad677364baadf6144ac4" },
{ "D_CLUNIE_CT1_J2KR.dcm","0eb834e9d393fd41cf0ceb5b9c0c8959", "a109da372df2b85ff0a705fba62e9589" },
{ "MAROTECH_CT_JP2Lossy.dcm","13a06166e548dc268ece0e22c860c387", "0e78a01c664550949071796339ac280e" },
{ "CR-MONO1-10-chest.dcm","0cccf203c6c2a097383e8a017322b77b", "1f772b4849727a9750931b60d920436f" },
{ "rle16sti.dcm","a225d3ea83666526eca91750e478b3ee", "f799773abbe36a1a8a3a881e27f8084d" },
{ "D_CLUNIE_MR4_JPLL.dcm","355a76b3d43982ad2256e010d0230835", "14fa2ae9f63742af6944edd4a61145e8" },
{ "US-IRAD-NoPreambleStartWith0003.dcm","27d939385c633d1db6943628a4e778cf", "ba092234639594ee9091b46997532cce" },
{ "00191113.dcm","7c17afc7815f18ebaecd99b9b07950f1", "bfff320d1b058e91b4819aa4560c16f7" },
{ "MR16BitsAllocated_8BitsStored.dcm","9d0266b04310ea377cf4d4c704ae44ba", "49b62d0b9004c2e1579317a36825cc5f" },
{ "D_CLUNIE_XA1_JPLL.dcm","462a9583d0241ead2f5a1bab318e177a", "6111657e6b01ec7b243d63f5dec6ec48" },
{ "MR-MONO2-8-16x-heart.dcm","379f882b2df9bdf115c38c4c68aea867", "01db0d71100c47013e588082d5f39bab" },
{ "ACUSON-24-YBR_FULL-RLE.dcm","1715cfceb7141080e09954ac55f1640b", "435c66f7e113d11d226d500294aae865" },
{ "D_CLUNIE_RG2_JPLL.dcm","ec91a9e10b3b9ba44305e62400db1ce9", "06900ee4323a91b7f5ffab8655e3c845" },
{ "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm","366582d77fed1513d96acce3ba37da11", "b6e4780d8aa8c1d3642377a60a5302dd" },
{ "GE_CT_With_Private_compressed-icon.dcm","48dd87fff8790e7fc2b1d31b03acd65a", "67206c434201b17a3a52ef42588b02e2" },
{ "D_CLUNIE_NM1_JPLY.dcm","b0227cfc9911bb0d18402685fd437a84", "812050a7fc53b5735f7740b60969cb6b" },
{ "US-RGB-8-esopecho.dcm","0916c89d6e3f3f28a52933829e5d0b3c", "4b350b9353a93c747917c7c3bf9b8f44" },
{ "FUJI-10-MONO1-ACR_NEMA_2.dcm","98190f9abe6cdf82e565aee55288aaba", "da2415a1e58b4ca2e588d0de18274f60" },
{ "JDDICOM_Sample2-dcmdjpeg.dcm","d883a2a8e69820f465865200cc726a4f", "33aa469ec024188d692262d03e7108a0" },
{ "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm","f89bb9670acce806d40f84e15a3143b0", "f69bca6228b0ca07d97ee11c0ab3b989" },
{ "D_CLUNIE_RG2_JPLY.dcm","e4e64e50877ce1134a58824e4b3b0cda", "27fa50d4cf6b31baa669e9746ce10f63" },
{ "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm","31a2d7476b505c01d77bf06333a3b797", "83601a7a4fa987c61b865b4b92f3caa0" },
{ "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm","12d33998ab1c371f3c2401fa362b2663", "a109da372df2b85ff0a705fba62e9589" },
{ "PHILIPS_GDCM12xBug.dcm","7a2531acbdaf8053365099a844f46df8", "d7673c8575cb2765f8ae25aa3899c77e" },
{ "OT-PAL-8-face.dcm","42c16b6897e32fbac5fd68a2d2523642", "d7c30d57af821b02c67103250a744235" },
{ "D_CLUNIE_SC1_JPLY.dcm","46dc0d3ad79883f0f5866ceeeb43c979", "994a5abb70d3f5968672ce4970a9d4da" },
{ "CT-MONO2-8-abdo.dcm","e94fa94ff3b7a34db63d40790995ce9c", "86d3e09a5858aa3844cb3be1b822a069" },
{ "GE_LOGIQBook-8-RGB-HugePreview.dcm","08a05020b521684a67d2ec40c2dc1600", "13fd8c7e533a3d7199bb78de45710f5c" },
{ "MR-Brucker-CineTagging-NonSquarePixels.dcm","e9c9b806ebf112c342426fc6b35f4471", "d9f47017de79e8755e4bc5d3c9146ebd" },
{ "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm","21f5f8c1d2b221833b665fae40327ce9", "698d6a3e88b270d4ad5ecfb00c11b634" },
{ "D_CLUNIE_CT2_JPLL.dcm","92f22a280db62906e4b894d0e7882550", "2e389ddbfc1b29d55c52c97e7f2c6f9c" },
{ "SIEMENS_CSA2.dcm","eb1db86d50a09dcf607af389e823d985", "62687f0a17e9c4153f18b55c8abfcef3" },
{ "MR-MONO2-16-head.dcm","49c0aea0ba79ac781a53dac56db530e1", "83be31fb5e5cee60dedaf485bf592ac3" },
{ "ELSCINT1_JP2vsJ2K.dcm","7d24121c34065f054fd398b5cf663a43", "87f5809b641b7235bfe5900a6281862b" },
{ "D_CLUNIE_MR4_JPLY.dcm","eb5b39beed663bec16b26bcfcaedddfe", "a33ad864b49ae7daa59cfaabdf751976" },
{ "MR-SIEMENS-DICOM-WithOverlays.dcm","9ad170d29a5e16ec04229b356f34ad45", "3027eda10630e5c845f456264dc65210" },
{ "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm","24a56cb80de943a06d82042933d22ea2", "8ac7f7891fb4506e2cd3ae2f0f7e9f46" },
{ "OsirixFake16BitsStoredFakeSpacing.dcm","a7f05bcbea7af6ab00c58950144fa86c", "68269299e8f4341120aaa13a933c6c11" },
{ "D_CLUNIE_VL2_RLE.dcm","017c67d4a3fedafb9d37264b4a4a68a1", "d215c88125359d34474a741d793c2215" },
{ "ALOKA_SSD-8-MONO2-RLE-SQ.dcm","6a8d9e3420077ef24de8cd0e490895ce", "7d8858e3419392b7f39a99fdc8028064" },
{ "D_CLUNIE_NM1_RLE.dcm","926811c358070765622fb131b9d4a39f", "6b5c1eff0ef65e36b0565f96507e96fd" },
{ "D_CLUNIE_SC1_RLE.dcm","2085edf780950a21167765481ad92811", "bd0cccbfd8db465c0af306ba0f482d72" },
{ "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm","ba1e9ecee0fee60f3037cd7fb2e7023e", "0b4dff77726ccf037fa83c42cc186a98" },
{ "GE_DLX-8-MONO2-PrivateSyntax.dcm","24e671036bdc9facc372b1ec4737e751", "51c998d3474c069b5703e98313258a1e" },
{ "D_CLUNIE_MR3_JPLL.dcm","baac36788bfa8d192455ca79f9b7af36", "fb03254fad02d2330d404225c3ea9b4e" },
{ "05148044-mr-siemens-avanto-syngo.dcm","60d33c5aecfe33a238e8ab8957194f4a", "9acdd9969f5d0584ddd67e994f00b7c7" },
{ "US-IRAD-NoPreambleStartWith0005.dcm","3c0377fbaca00a9dcd17fa0258caa2c0", "1bde104ba256fb73528c5d9a02e363d7" },
{ "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm","95d5f4e573800918664796910c238ed0", "3cd8bd92db17bff54e376885dfefdd8d" },
{ "US-MONO2-8-8x-execho.dcm","14a483a14e070ead0a31a96d86146311", "bf63affde325b3fa81cd5a700f30bd5b" },
{ "D_CLUNIE_RG2_RLE.dcm","bda1db6bbe98fcf680c4073df99e7b18", "06900ee4323a91b7f5ffab8655e3c845" },
{ "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm","d7292927bd1aef2298c25524055cc2ab", "279e2b0363394a553ff8571cf3540c6c" },
{ "CT-SIEMENS-MissingPixelDataInIconSQ.dcm","eed1e36b9333d9a50458e19bd0203455", "a86e48680d8ca941aa694d81872e8490" },
{ "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm","ec48c6dda2fdff2dd59366764d1fb0bd", "16e999d6afc5574bcb075f296c3bcbbc" },
{ "D_CLUNIE_VL4_RLE.dcm","c1bdaec5bf045ae322068790c2f304a8", "e2fdf24d2c03dd0991b4f4e9d6e84ed6" },
{ "GE_GENESIS-16-MONO2-WrongLengthItem.dcm","97e495e5e6b9fd9e080c7b9e40589b84", "1497fb9d7467b1eb36d5618e254aac76" },
{ "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm","a7da0400b907b86ef97af8e61658b569", "6bbe3a067ff90eae950bc901f0f92f9e" },
{ "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm","09dc2ab46d45d47a3ad41e9d53bf3c8d", "279e2b0363394a553ff8571cf3540c6c" },
{ "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm","c1bcf58ffa46cfe50a989847cdd3e3c9", "73f5986082729c2661cdc8de81fd26d0" },
{ "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm","3c10ecf511e1316d367399eeb4ee70dd", "d2fab61e0fff8869e448d69951f1084d" },
{ "D_CLUNIE_MR1_RLE.dcm","a9c04692b4126a69e2cffc81f567a516", "7b7424e6115931c371f3c94c2f5d32d9" },
{ "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm","eab7f8484a6c3d6e76a8e281213e56e4", "0b4dff77726ccf037fa83c42cc186a98" },
{ "GE_DLX-8-MONO2-Multiframe.dcm","67c44b08178efe2857221f3015b0510f", "71e4ea61df4f7ada2955799c91f93e74" },
{ "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm","4a967f8f35bfb93262dc90e0b8acf6b8", "3cd8bd92db17bff54e376885dfefdd8d" },
{ "00191113.dcm","7c17afc7815f18ebaecd99b9b07950f1", "bfff320d1b058e91b4819aa4560c16f7" },
{ "D_CLUNIE_CT1_JLSL.dcm","4efffce4cded3f8785d5ac4d3b11da41", "a109da372df2b85ff0a705fba62e9589" },
{ "D_CLUNIE_CT1_JLSN.dcm","bcf17a67d96046ce7a2b4de4495dbd4c", "7ca273fff6311586bd02ac983ccfbb6b" },
{ "IM-0001-0066.CommandTag00.dcm","51450a7a49ce7bc0e6a4ce18eaef2fff", "12d1567ed81236cf3b01dc12766581a0" },
{ "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm","2b1296ded36f37beb83e28fae108c280", "d93d2f78d845c7a132489aab92eadd32" },
{ "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm","f507ab11e56de33d6f0195dd185b5c62", "9eb513314b2fcf25d895e18ffb2ead0b" },
{ "GDCMJ2K_TextGBR.dcm","9935ec525ea42203f3a535d4c1187ed4", "56238d3665ebdb0251d1161fb7f4edc6" },
{ "NM_Kakadu44_SOTmarkerincons.dcm","99b60ec88d410b437adc06a809590127", "6f26e552a1b71d386483118779d192ad" },
{ "PhilipsInteraSeqTermInvLen.dcm","6ccabe05841e29633d9e4fe1aec81039", "f8a1f4ce85b51527267e670a8aa0c308" },
{ "LIBIDO-24-ACR_NEMA-Rectangle.dcm","28376a8bb2a64a9c8f6a50abfdbb336f", "81a40454eec2b18f4331cfd1ba4e501e" },
{ "TOSHIBA_J2K_SIZ1_PixRep0.dcm","b7551911535d7ee21a336b700e4f0e4f", "d6347ed051d7b887bdaad1a91433c6ba" },
{ "TOSHIBA_J2K_OpenJPEGv2Regression.dcm","f417f1a4552aeb03137ed91a04345fa2", "94414d8b4300aa3d8cbe4475d34e8e54" },
{ "TOSHIBA_J2K_SIZ0_PixRep1.dcm","fcaab3ded5c41993b82d831c4251526e", "d6347ed051d7b887bdaad1a91433c6ba" },
{ "NM-PAL-16-PixRep1.dcm","a04de15a87306cf77cd85ae5b720a0b2", "304f147752d46adfdcff71a30cd03d0a" },
{ "MEDILABInvalidCP246_EVRLESQasUN.dcm","da13f42e41d36ef065af2cb5d770f70b", "d99ace99196e148522c8599803bacc28" },
{ "JPEGInvalidSecondFrag.dcm","32ea68aea0c81d4cc75c5a234e1719ed", "8fb3a76c1f18b71b52d139ebd8406b50" },


/* Stopping condition */
{ 0 , 0 , 0 }
};

//static MD5MetaImagesType vtkGDCMTesting::GetMD5MetaImages()
//{
//}

unsigned int vtkGDCMTesting::GetNumberOfMD5MetaImages()
{
  // Do not count NULL value:
  static const unsigned int size = sizeof(vtkgdcmMD5MetaImages)/sizeof(*vtkgdcmMD5MetaImages) - 1;
  return size;
}

const char * const * vtkGDCMTesting::GetMD5MetaImage(unsigned int file)
{
  if( file < vtkGDCMTesting::GetNumberOfMD5MetaImages() ) return vtkgdcmMD5MetaImages[file];
  return NULL;
}

const char * vtkGDCMTesting::GetMHDMD5FromFile(const char *filepath)
{
  if(!filepath) return NULL;
  unsigned int i = 0;
//  MD5DataImagesType md5s = GetMD5DataImages();
  MD5MetaImagesType md5s = vtkgdcmMD5MetaImages;
  const char *p = md5s[i][0];
  gdcm::Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = md5s[i][0];
    }
  // \postcondition always valid (before sentinel)
//  assert( i <= GetNumberOfMD5DataImages() );
  return md5s[i][1];
}

const char * vtkGDCMTesting::GetRAWMD5FromFile(const char *filepath)
{
  if(!filepath) return NULL;
  unsigned int i = 0;
//  MD5DataImagesType md5s = GetMD5DataImages();
  MD5MetaImagesType md5s = vtkgdcmMD5MetaImages;
  const char *p = md5s[i][0];
  gdcm::Filename comp(filepath);
  const char *filename = comp.GetName();
  while( p != 0 )
    {
    if( strcmp( filename, p ) == 0 )
      {
      break;
      }
    ++i;
    p = md5s[i][0];
    }
  // \postcondition always valid (before sentinel)
//  assert( i <= GetNumberOfMD5DataImages() );
  return md5s[i][2];
}

//----------------------------------------------------------------------------
vtkGDCMTesting::vtkGDCMTesting()
{
}

vtkGDCMTesting::~vtkGDCMTesting()
{
}

//----------------------------------------------------------------------------
const char *vtkGDCMTesting::GetVTKDataRoot()
{
#ifdef VTK_DATA_ROOT
  return VTK_DATA_ROOT;
#else
  return NULL;
#endif
}

//----------------------------------------------------------------------------
const char *vtkGDCMTesting::GetGDCMDataRoot()
{
#ifdef GDCM_BUILD_TESTING
  return gdcm::Testing::GetDataRoot();
#else
  return NULL;
#endif
}

//----------------------------------------------------------------------------
void vtkGDCMTesting::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
