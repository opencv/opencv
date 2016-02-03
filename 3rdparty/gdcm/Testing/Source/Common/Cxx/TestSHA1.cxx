/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSHA1.h"
#include "gdcmFilename.h"
#include "gdcmTrace.h"
#include "gdcmTesting.h"

#include <iostream>
#include <cstring> // strcmp

static const char * const gdcmSHA1SumFiles[][2] = {
{ "265465a2e3b204ab9a094f2de56bbec96d55ab74" , "00191113.dcm" },
{ "ab316e51539a56053e216017c8445a246a978590" , "012345.002.050.dcm" },
{ "750ff7a9df7a12c13f7149c514664db7c68c3125" , "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm" },
{ "4efeffa6a7b535df85dda198a26e074866b0ed88" , "05148044-mr-siemens-avanto-syngo.dcm" },
{ "9d1e3fdf69f8d5b46105e4c259b3d84e7f281a45" , "3E768EB7.dcm" },
{ "f08ff36b45cd50d4349f4c6c17a75b3f653bb11e" , "ACUSON-24-YBR_FULL-RLE-b.dcm" },
{ "dac2ffa6ccf59f2d91e78f632e45b08d1537715c" , "ACUSON-24-YBR_FULL-RLE.dcm" },
{ "cf749c5323866c343255805cf781ae87e6bb598a" , "ALOKA_SSD-8-MONO2-RLE-SQ.dcm" },
{ "7202bfac6d4196aeb2655e879bc91e4dcf40a463" , "BugGDCM2_UndefItemWrongVL.dcm" },
{ "1d227106140c2a24dbe8c6becd4649761dd117a2" , "CR-MONO1-10-chest.dcm" },
{ "74e7de8b0f5903b8571294e2d8f960dc7329a0ca" , "CT_16b_signed-UsedBits13.dcm" },
{ "c0e08333984692bcef149c946e23bb46e01b9b7c" , "CT-MONO2-12-lomb-an2.acr" },
{ "594ce3215ca94f640fd7502d614651cc755a9d7e" , "CT-MONO2-16-ankle.dcm" },
{ "ce0bd0c9635db3f04b4996648c1d41e10daffa77" , "CT-MONO2-16-brain.dcm" },
{ "f9ad8311fa69835c55314e740ca55ab71e038c17" , "CT-MONO2-16-chest.dcm" },
{ "507fdef01d9357dfb99981ae6c18ab88931271b5" , "CT-MONO2-16-ort.dcm" },
{ "ad5c2b6d0816761d5f5dc1fcc83e44f3eea4666d" , "CT-MONO2-8-abdo.dcm" },
{ "76e6ae295b326a46d37f1ad5056ee18b6ee3d6f2" , "CT-SIEMENS-Icone-With-PaletteColor.dcm" },
{ "7bce2465ca918a5d2a9b8f9c6a9cce1b6b94b44b" , "CT-SIEMENS-MissingPixelDataInIconSQ.dcm" },
{ "1b75f84318351449b5d841d094297d7975e8ac5b" , "D_CLUNIE_CT1_J2KI.dcm" },
{ "1d40f08bf78a29e0729742ff5f5731477ab5a9ff" , "D_CLUNIE_CT1_J2KR.dcm" },
{ "6f6027b38ec184af8cdf31077cd669bd35270ccb" , "D_CLUNIE_CT1_JPLL.dcm" },
{ "3df887253b24f3296f078fc618e6a682cae4c1d6" , "D_CLUNIE_CT1_RLE.dcm" },
{ "5336a791f09fe080688b5c1fad04691b39588978" , "D_CLUNIE_CT2_JPLL.dcm" },
{ "37beaf412c74e435cfdf351cc95f02b62820e3ea" , "D_CLUNIE_CT2_RLE.dcm" },
{ "fa118c4661e14878a9113032b6549dc7666b773b" , "D_CLUNIE_MR1_JPLL.dcm" },
{ "f464c45fceb6f5dc234150064c1b7999cd0db2ff" , "D_CLUNIE_MR1_JPLY.dcm" },
{ "4f6a484fd77238f8bc245fbfadc449cb3ec40a75" , "D_CLUNIE_MR1_RLE.dcm" },
{ "111bed6926dbf4ad3f61bd9c95e36095475265a1" , "D_CLUNIE_MR2_JPLL.dcm" },
{ "eec3ed081ec196e036d2cac9e0043bef1038665e" , "D_CLUNIE_MR2_JPLY.dcm" },
{ "c2ab10273241fe18401f4584a42c76c5257219f0" , "D_CLUNIE_MR2_RLE.dcm" },
{ "fb04e8143117a8ecdd9cece68c08d4ee3b3ed040" , "D_CLUNIE_MR3_JPLL.dcm" },
{ "ef2d3fe28424e42963ca52f097b3d1309c95f43d" , "D_CLUNIE_MR3_JPLY.dcm" },
{ "08dd62d2f452e145dea7905b64e43836504c8a2e" , "D_CLUNIE_MR3_RLE.dcm" },
{ "e2fa66c4074e9f75b582eb6cf5261a6ac064c5cb" , "D_CLUNIE_MR4_JPLL.dcm" },
{ "7426ae6aa6cb2f13138878a68183d69ed8d42b1e" , "D_CLUNIE_MR4_JPLY.dcm" },
{ "2cf735f7ced248c20d4bd8af760523e8b1c88c44" , "D_CLUNIE_MR4_RLE.dcm" },
{ "f0632988a652cf186a2c1199bb19389534617952" , "D_CLUNIE_NM1_JPLL.dcm" },
{ "8e4a062c60bda83a7a118b505f9da29454eb77e1" , "D_CLUNIE_NM1_JPLY.dcm" },
{ "0933067b799d0b4407e190e0b3a58f0e97c92371" , "D_CLUNIE_NM1_RLE.dcm" },
{ "c72d14184a995b367682f42c3e3b62de7a5b4c7e" , "D_CLUNIE_RG1_JPLL.dcm" },
{ "b71f33df55502266ca2f4966cd5fc88b2b766385" , "D_CLUNIE_RG1_RLE.dcm" },
{ "8c337359cca8d2bf1002209589ff19699e530908" , "D_CLUNIE_RG2_JPLL.dcm" },
{ "c6188f2ae06a55d596d4d6b4d914b404f7bf5786" , "D_CLUNIE_RG2_JPLY.dcm" },
{ "5035a6298cf8590932605cb4bb10f128df971fcb" , "D_CLUNIE_RG2_RLE.dcm" },
{ "d2a56de2e035095c3dea9be80878f9607c63728d" , "D_CLUNIE_RG3_JPLL.dcm" },
{ "41abc80638b12a2601e7764805a9c4ddc385bdc5" , "D_CLUNIE_RG3_JPLY.dcm" },
{ "eb417a88c6b3fe6c5768ab705990fc0d54558816" , "D_CLUNIE_RG3_RLE.dcm" },
{ "8e27c853a156038e1f6daebb04f2c9059ebfbd82" , "D_CLUNIE_SC1_JPLY.dcm" },
{ "6a6ca811ae87ae1ca4549f49724b58562eea714b" , "D_CLUNIE_SC1_RLE.dcm" },
{ "94737b44e512c8b84e1fb880e04b0ae6f069bd96" , "D_CLUNIE_US1_RLE.dcm" },
{ "6cc2ad34df402a9aa5c6a2c5436ff0e775a59768" , "D_CLUNIE_VL1_RLE.dcm" },
{ "4432f4bba584c5bf93264050a0aef0237c8cc9a5" , "D_CLUNIE_VL2_RLE.dcm" },
{ "c11e1487e94b332f32c041d7e8889d1fe91c2db6" , "D_CLUNIE_VL3_RLE.dcm" },
{ "345bde73f67171cc6f90cd890d67108aa6039656" , "D_CLUNIE_VL4_RLE.dcm" },
{ "e52694c1e9be0f4d1c44eab3ac44aa4225b593b0" , "D_CLUNIE_VL6_RLE.dcm" },
{ "37bf802b1918f36fcf7c6fb89b40ee1da1124419" , "D_CLUNIE_XA1_JPLL.dcm" },
{ "ff7b9c99731732b55bcff577037542689efe9c72" , "D_CLUNIE_XA1_JPLY.dcm" },
{ "f384b5f1ae912d97e7784558d5519dc74b853f10" , "D_CLUNIE_XA1_RLE.dcm" },
{ "cc3641d1f25549a0ee06d2a5e87081cf6c407bf0" , "D_CLUNIE_CT1_JLSL.dcm" },
{ "ac9f4592e6cfad438de5c928b9e01591006683d8" , "D_CLUNIE_CT1_JLSN.dcm" },
{ "cc72ee9f3e671321e231103eb562063078b67bf8" , "DCMTK_JPEGExt_12Bits.dcm" },
{ "fb9da9ae84e5da97c992398ac0222fffafa028f6" , "DermaColorLossLess.dcm" },
{ "6f53ebd046c0c39ad04b7aa3767ced2647b02c83" , "DICOMDIR" },
{ "436f0de1319089bc203b699a6e64687a9b27fd8c" , "dicomdir_Acusson_WithPrivate_WithSR" },
{ "3c73737e44852b9f30cc4fe4e8cc600cdf63ee0a" , "DICOMDIR_MR_B_VA12A" },
{ "86687243468f6e8243081adbff3374c8fbeb1166" , "DICOMDIR-Philips-EasyVision-4200-Entries" },
{ "623523dedebccddb38ab6fbb76e01c1458bd0d98" , "dicomdir_Pms_With_heavy_embedded_sequence" },
{ "0c0c0b72d980a6675897fc50854c98176582e7d5" , "dicomdir_Pms_WithVisit_WithPrivate_WithStudyComponents" },
{ "9a90f8c949d99f9e9e7523f442b8995bb2b2ae3d" , "dicomdir_With_embedded_icons" },
{ "eb0b632f7feccde2b7d88d6a82f51857ef5de521" , "DMCPACS_ExplicitImplicit_BogusIOP.dcm" },
{ "abb8fb269447e046fe10306e4e5f54fe988a0069" , "DX_GE_FALCON_SNOWY-VOI.dcm" },
{ "861ff0760754f6ebb0056f52adcdcd60f0cd3bfa" , "DX_J2K_0Padding.dcm" },
{ "8275a7e217807273fb0027a64f46cba8656f699a" , "ELSCINT1_JP2vsJ2K.dcm" },
{ "399e80d60843e62b817322ba9e82278258f9a222" , "ELSCINT1_LOSSLESS_RICE.dcm" },
{ "3598e59bf8ff6cb5217473f9f23ad763638a27e0" , "ELSCINT1_PMSCT_RLE1.dcm" },
{ "3514c92127b0c78b63efce1fb6053e202da86d3f" , "ExplicitVRforPublicElementsImplicitVRforShadowElements.dcm" },
{ "cc80ee94edf9b3d7fd4c1402b2c6ae99eb832685" , "fffc0000UN.dcm" },
{ "374bf25ac4279a7458d848b595b23ed50580400b" , "FUJI-10-MONO1-ACR_NEMA_2.dcm" },
{ "ada05095c6d64767af77eafeb2e6fc305f2ba2c8" , "gdcm-ACR-LibIDO.acr" },
{ "f2d333278c7ebb985916011ecb808342205b9815" , "gdcm-CR-DCMTK-16-NonSamplePerPix.dcm" },
{ "09f0d023545047704beb2fa8168bd7fadca845e2" , "gdcm-JPEG-Extended.dcm" },
{ "70da750cb898c6510ba079a1e71b4869faf66dd7" , "gdcm-JPEG-LossLess3a.dcm" },
{ "736c14572e7ee964d24b2b5c5b2a607ca50ebc05" , "gdcm-JPEG-LossLessThoravision.dcm" },
{ "765541a6e32671891fc0787e69c4917c7166a77a" , "gdcm-MR-PHILIPS-16-Multi-Seq.dcm" },
{ "b7d171ebfcb288ec44647f3000515f3683dfef50" , "gdcm-MR-PHILIPS-16-NonSquarePixels.dcm" },
{ "6f1e68d6a8586862c31ac7d7a2d2c8d317865263" , "gdcm-MR-SIEMENS-16-2.acr" },
{ "a525344659d1767787c6929bb2b080286106e881" , "gdcm-US-ALOKA-16.dcm" },
{ "5b8c8d45b6c06f5d2fb90df6205606accfedfc6c" , "GE_CT_With_Private_compressed-icon.dcm" },
{ "58c81390cbb3a033c0f1fe15760bd14d87ca8313" , "GE_DLX-8-MONO2-Multiframe.dcm" },
{ "53b514276db021a5a8e26cddd3f09c43dad49747" , "GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm" },
{ "86ae2ce4b9676110db10ebc6aaba1d49bfa26537" , "GE_DLX-8-MONO2-PrivateSyntax.dcm" },
{ "34a42605c83313dbd6d0af4670c2dfabaf270319" , "GE_GENESIS-16-MONO2-Uncompressed-UnusualVR.dcm" },
{ "a3f130a256fea9804cc2b9e05dedb4b3af517421" , "GE_GENESIS-16-MONO2-WrongLengthItem.dcm" },
{ "da85472c37599dc02c83f2e4c24e3fd5a2617f92" , "GE_LOGIQBook-8-RGB-HugePreview.dcm" },
{ "a47f7d0d1d40c213666ccc67d3ce83464a5fa9b8" , "GE_MR_0025xx1bProtocolDataBlock.dcm" },
{ "36bcff2d268dbf8e92b296a9cd0894f5b4dcb5ed" , "GE_RHAPSODE-16-MONO2-JPEG-Fragments.dcm" },
{ "f84803d8879bc0a461f03c39ec896704796d8376" , "IM-0001-0066.dcm" },
{ "fb563954a0b3e8f521248cfac294233e85c517c9" , "ITK_GDCM124_MultiframeSecondaryCaptureInvalid.dcm" },
{ "5a21779c6db335b0f6a989c292b66e2d1939aaae" , "JDDICOM_Sample2.dcm" },
{ "9f2559b3e74c09e04c1ccca0424f6bbb3480fe9e" , "JDDICOM_Sample2-dcmdjpeg.dcm" },
{ "9af362dd8ba13c358b869ebaf9c70dd2c34a9b49" , "JPEG_LossyYBR.dcm" },
{ "e99bd082c18b4f0a150737d5f1a18e00c6bb73d8" , "KODAK-12-MONO1-Odd_Terminated_Sequence.dcm" },
{ "12fc320b12b35822da90155a2b92586f8ff4adf0" , "KODAK_CompressedIcon.dcm" },
{ "3e554827c2110cd94ab29f65ea4bb8155ae11be8" , "LEADTOOLS_FLOWERS-16-MONO2-JpegLossless.dcm" },
{ "1b8c958f48da2aaa1bbc085aec4410f5c2545875" , "LEADTOOLS_FLOWERS-16-MONO2-RLE.dcm" },
{ "7be91a03cca70febe383fd4df821f5475cc02c8d" , "LEADTOOLS_FLOWERS-16-MONO2-Uncompressed.dcm" },
{ "f6046930f93017ac0586f389d2f69cc47d139615" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossless.dcm" },
{ "7de6449beccbeda52d95b027d24448abad1194bc" , "LEADTOOLS_FLOWERS-24-RGB-JpegLossy.dcm" },
{ "6a8f88021d04488bf1d8f5160b49bc9245a5da2c" , "LEADTOOLS_FLOWERS-24-RGB-Uncompressed.dcm" },
{ "d5b043ad7c367c4ac5156c8137125d481201f8b6" , "LEADTOOLS_FLOWERS-8-MONO2-JpegLossy.dcm" },
{ "3f3d48adcbbe43cd88ec4dd38c0eabac1518cbb1" , "LEADTOOLS_FLOWERS-8-MONO2-RLE.dcm" },
{ "a6e1d158e76362545a32d90adfdbc06d0a795aa8" , "LEADTOOLS_FLOWERS-8-MONO2-Uncompressed.dcm" },
{ "53e1a8442e52b8e7fb0024f38a3ba47cd1557af4" , "LEADTOOLS_FLOWERS-8-PAL-RLE.dcm" },
{ "f1cf42d9fbd847f7bebaa2b2e05ad669401bceeb" , "LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm" },
{ "5c768b7fa2940546d9f3c49e76d8734ebae41b9b" , "libido1.0-vol.acr" },
{ "fe7170c9acf67f99977f12ab41429e082bca4440" , "LIBIDO-16-ACR_NEMA-Volume.dcm" },
{ "4d3d23112591bcbbf130e2d83768c59af69c6740" , "LIBIDO-24-ACR_NEMA-Rectangle.dcm" },
{ "971bf05ef2a29cdddfb28279b73444fd5dedb046" , "LIBIDO-8-ACR_NEMA-Lena_128_128.acr" },
{ "f6cf7fb8470a33f82beef9dcf55b7379464db068" , "LJPEG_BuginGDCM12.dcm" },
{ "483032c54da529824e6d1f587dba9a9c5c6f4888" , "MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm" },
{ "9033fd528b9008273cbb277f0adb21abe8a1d35a" , "MAROTECH_CT_JP2Lossy.dcm" },
{ "88534f6f5c247a198b860108c4d3a5b5f7fb7048" , "MR-Brucker-CineTagging-NonSquarePixels.dcm" },
{ "73817e417b440a961dfcc273eeae6007fc004f61" , "MR_ELSCINT1_00e1_1042_SQ_feff_00e0_Item.dcm" },
{ "2f210cc06d8e05b762b8fd996ac730f2442b258a" , "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm" },
{ "73eae004f6305c3ef10f710dd51ef750b7ced8ab" , "MR-MONO2-12-an2.acr" },
{ "80cb279b9cfe4cd5f8209dc1fd81a0de789fb561" , "MR-MONO2-12-angio-an1.acr" },
{ "30a35164282f20ffc38eeb4d0a94b74159f72a38" , "MR-MONO2-12-shoulder.dcm" },
{ "c6376ebdfda8ca05bb3d2f2796b2f4ce4cf5417a" , "MR-MONO2-16-head.dcm" },
{ "e4b6f0ea147335d048ef1cf271f1f8ea9b3a947a" , "MR-MONO2-8-16x-heart.dcm" },
{ "482282eab5270a9b49da1fe52955029c90665318" , "MR_Philips-Intera_BreaksNOSHADOW.dcm" },
{ "0edaefb39107ede69cbfef9fa5d646734859560a" , "MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm" },
{ "002496c378b8d74cde8034c8e64ac06278aaac64" , "MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm" },
{ "7b32e317ab15df08cbc22e5c71c19be0236c4a71" , "MR_Philips_Intera_PrivateSequenceImplicitVR.dcm" },
{ "d201a0399638b90dab2d147e8505b12dc3f22abf" , "MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm" },
{ "f86689b7eacfa0c477168f6d95e5f554ba9db2f6" , "MR-SIEMENS-DICOM-WithOverlays.dcm" },
{ "4ca412d03202693ea4dbe5189681afb25341e7d7" , "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm" },
{ "ddc7ed9fc533413dec8795c35c0dc45485a64422" , "MR_SIEMENS_forceLoad29-1010_29-1020.dcm" },
{ "af658128f2acfb72b95892435a34d983194adb85" , "MR_Spectroscopy_SIEMENS_OF.dcm" },
{ "a28aa38246dbce06692f5a977ca320cf26bb9260" , "NM-MONO2-16-13x-heart.dcm" },
{ "5ca8093b12ff8c78875af95091a96965a94acb19" , "OT-MONO2-8-a7.dcm" },
{ "9bf3ca1db0420baede3c257e9c102fe5d27a0554" , "OT-PAL-8-face.dcm" },
{ "38a8505e4a5f4119c76636e6eaf754edff1e8122" , "PET-cardio-Multiframe-Papyrus.dcm" },
{ "a011aaddce1295f42d9d8fbd14b1558ed04b0b6c" , "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm" },
{ "cd88e741f78d2267a7c8a83a0fb4b009fce706f8" , "PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm" },
{ "05fe55371a99bd6e233b4605eef8b47d5d21f243" , "PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm" },
{ "09a07901bb4b6ca120648c3fa478a4172811b4db" , "PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm" },
{ "d4a5ce137aac5d979a68b5a62593cf99096a485d" , "PHILIPS_Intera-16-MONO2-Uncompress.dcm" },
{ "f64181bf0abfacf2853ce3eeb60149f2c3cf7507" , "PICKER-16-MONO2-Nested_icon.dcm" },
{ "aef5bc2a2132e2c11a4f8fc5e1dd7655300d0c1f" , "PICKER-16-MONO2-No_DicomV3_Preamble.dcm" },
{ "0fc0e0168128517472e3a4a69808331aaffe4bb5" , "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm" },
{ "38e925d7c30ef5b0cf829ea520e271482ea89c7d" , "RadBWLossLess.dcm" },
{ "7a57d75599fd8521210835f6506dd77204f16883" , "rle16loo.dcm" },
{ "77d3e4e78f0452f4ebb0855fa367e19044014244" , "rle16sti.dcm" },
{ "3d9e8bf00e3509fc7210a8a67c3d15622c03b15e" , "SIEMENS-12-Jpeg_Process_2_4-Lossy-a.dcm" },
{ "292433a76c4ef0a8341c76b23d030950fbe7865a" , "SIEMENS_CSA2.dcm" },
{ "b67eecea6c2099fa0bb0d558ad2fb281e0c14811" , "SIEMENS_GBS_III-16-ACR_NEMA_1.acr" },
{ "89c6d755e1f183b18c3711ccce0bdb243d11f05c" , "SIEMENS_GBS_III-16-ACR_NEMA_1-ULis2Bytes.dcm" },
{ "521b21eb0b5ec9ebb77de6d7730971a3d5ce5542" , "SIEMENS_ImageLocationUN.dcm" },
{ "6f1e68d6a8586862c31ac7d7a2d2c8d317865263" , "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm" },
{ "32da5e531a2df100b4a6bc5334c0128a7603c62f" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm" },
{ "f7cf7f2d62113863f76580bac6a0465638ebd050" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm" },
{ "8708d74b623b3317631a982b9c316c8521c1b220" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm" },
{ "461959350160a422e7806a9e8201146b29a3a21f" , "SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm" },
{ "e5152fe2ada738f5b0c86af0858dc6d6b3ed9520" , "SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm" },
{ "bcba71eae5a0e3e43d836c410118e5992f6a6fb7" , "SIEMENS_MAGNETOM-12-MONO2-Uncompressed.dcm" },
{ "38284e6f4591418be8a572aaa190a86b9487ed6f" , "SIEMENS_MAGNETOM-12-MONO2-VRUN.dcm" },
{ "61b1b8768cde82a51fa8f5238713d65f16aeff68" , "SIEMENS_MOSAIC_12BitsStored-16BitsJPEG.dcm" },
{ "a3646e926f9b8dd6a536b95ab8a86ee35f74f612" , "SIEMENS-MR-RGB-16Bits.dcm" },
{ "542213ababb9f3410be5516d28adece7ca9aa4ad" , "SIEMENS_SOMATOM-12-ACR_NEMA-ZeroLengthUs.acr" },
{ "6d08c8b6f34de509166fa4d1e456e12861ab86e7" , "SIEMENS_Sonata-12-MONO2-SQ.dcm" },
{ "4204cf122c096365f7fa14009a4dd62c49e9e5cf" , "SIEMENS_Sonata-16-MONO2-Value_Multiplicity.dcm" },
{ "cd2c42d837702aab1f85de4dc000962dda1fe2f7" , "SignedShortLosslessBug.dcm" },
{ "d6e3e55033bd06e8b3e67ef83168d7425cf63c2f" , "simpleImageWithIcon.dcm" },
{ "55d4b449f51012c6bcad025a99eea672ba880a60" , "test.acr" },
{ "918939283378898524dfa449fd9c9bc12b6dd189" , "TG18-CH-2k-01.dcm" },
{ "d70be071b03866d8b561588588b9541ad4c370d7" , "THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm" },
{ "1f06affc50bfd1382efbf1960f5c601f79db841d" , "TheralysGDCM120Bug.dcm" },
{ "16d9e8000e9b19dd3bdb36b3a829780aa43c8b74" , "TOSHIBA_MRT150-16-MONO2-ACR_NEMA_2.dcm" },
{ "f188be58d7378e0949b0ee21ef72906ec1c7e236" , "undefined_length_un_vr.dcm" },
{ "acf081a99759cef05711ec51d5da9e910396024d" , "US-GE-4AICL142.dcm" },
{ "7330de615594148320e2f48a6aecb2e65607fa31" , "US-IRAD-NoPreambleStartWith0003.dcm" },
{ "3ffd40a0aba42431d2b2deeed07013e66afb0328" , "US-IRAD-NoPreambleStartWith0005.dcm" },
{ "035d7791c0fce801078854697fef86292a97eef1" , "US-MONO2-8-8x-execho.dcm" },
{ "8e9ef1a7516ab9bc0fbe894e25c28c7d15dd848d" , "US-PAL-8-10x-echo.dcm" },
{ "b401f77c8c78a82422baad6f6d42dcc275b7f0e0" , "US-RGB-8-epicard.dcm" },
{ "f4368f3e10ebbd70fbe00c3d4b29d30df3adee85" , "US-RGB-8-esopecho.dcm" },
{ "750f15609a6f442c0517173138e404894849b374" , "XA-MONO2-8-12x-catheter.dcm" },
{ "cdc423e6abfb6fd678a3e65675cd5ce42f304b0d"  , "PHILIPS_GDCM12xBug2.dcm" },
{ "916938d82884fcb77a54dade49481e15049e50b6"  , "PHILIPS_GDCM12xBug.dcm" },
{ "0f058542379c201e26ef233a4b116862283bc81e"  , "AMIInvalidPrivateDefinedLengthSQasUN.dcm" },
{ "ec4a0d2ff24e9c2f3281b75c479914aff3215fb7"  , "OsirixFake16BitsStoredFakeSpacing.dcm" },
{ "27277c296a9703f22b2ef033ec326dd22b4bacc9"  , "MR16BitsAllocated_8BitsStored.dcm" },
{ "6260235b715af050f47494e098fcaa2ada3a11a5"  , "JPEGDefinedLengthSequenceOfFragments.dcm" },
{ "b1977dfddff0d88742bc6f6368d56eb9d3325dd9"  , "IM-0001-0066.CommandTag00.dcm" },
{ "f9d31313e26c335d26e4fd9126d88f1467425117"  , "UnexpectedSequenceDelimiterInFixedLengthSequence.dcm" },
{ "ac491044a7c69d3a365f31f163db3a4bf1085afa"  , "GDCMJ2K_TextGBR.dcm" },
{ "ed38be73d76c7a326012f86034c7472aa02d83ee"  , "NM_Kakadu44_SOTmarkerincons.dcm" },
{ "c06aae794ed5690ae82237f4f153ca7795ddd5b7"  , "PhilipsInteraSeqTermInvLen.dcm" },
{ "c14634b31dc52e4ed2b0c626aa61af4de207f6e4"  , "TOSHIBA_J2K_SIZ1_PixRep0.dcm" },
{ "bf68f33e0922508d6075f0893298d339d817aa86"  , "TOSHIBA_J2K_OpenJPEGv2Regression.dcm" },
{ "6d21ac7b5b4ad32b7d91750a70b574a732b679a7"  , "TOSHIBA_J2K_SIZ0_PixRep1.dcm" },
{ "94cadb9e79e0d04f3c212cf6fa069b3cf3f09a76"  , "NM-PAL-16-PixRep1.dcm" },
{ "156463350047cada3ec091396695d3f3ef660c9a"  , "MEDILABInvalidCP246_EVRLESQasUN.dcm" },
{ "9775e8206011b9d5cedfcba16946060cb047f826"  , "JPEGInvalidSecondFrag.dcm" },
{ "d17a34f8ae066048442ab5b110d43c412472ea7e"  , "SC16BitsAllocated_8BitsStoredJ2K.dcm" },
{ "71517fce6c32625f1051b72085cfceeee58bd164"  , "SC16BitsAllocated_8BitsStoredJPEG.dcm" },

{ NULL, NULL}
};

int TestSHA1Func(const char* filename, const char *sha1ref, bool verbose = false)
{
  if( !filename || !sha1ref) return 1;

  if( verbose )
    std::cout << "TestRead: " << filename << std::endl;
  const char *dataroot = gdcm::Testing::GetDataRoot();
  std::string path = dataroot;
  path += "/";
  path += filename;
  path = filename;
  char sha1[2*20+1] = {};
  bool b = gdcm::SHA1::ComputeFile( path.c_str(), sha1 );
  if( !b )
    {
    std::cerr << "Fail ComputeFile: " << path << std::endl;
    return 1;
    }
  if( strcmp( sha1, sha1ref) != 0 )
    {
    std::cout << "Problem with: " << path << std::endl;
    std::cout << "Ref: " << sha1ref << " vs " << sha1 << std::endl;
    return 1;
    }
  return 0;
}

static const char *GetSHA1Sum(const char *filename)
{
  typedef const char * const (*sha1pair)[2];
  const char *sha1filename;
  sha1pair sha1filenames = gdcmSHA1SumFiles;
  int i = 0;
  while( ( sha1filename = sha1filenames[i][1] ) )
    {
    gdcm::Filename fn( filename );
    if( strcmp( sha1filename, fn.GetName() ) == 0 )
      {
      return sha1filenames[i][0];
      }
    ++i;
    }
  std::cerr << "Missing SHA1 for: " << filename << std::endl;
  return 0;
}


int TestSHA1(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    const char *sha1 = GetSHA1Sum( filename );
    return TestSHA1Func(filename, sha1, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    const char *sha1 = GetSHA1Sum( filename );
    r += TestSHA1Func( filename, sha1 );
    ++i;
    }

  return r;
}
