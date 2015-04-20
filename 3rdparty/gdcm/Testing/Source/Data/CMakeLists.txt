#-----------------------------------------------------------------------------

# We need to test the reading of all dicom images in the gdcmData directory
# First parse this directory and extract all images

set(GDCM_DATA_IMAGE_FILENAMES_GLOB_EXPR
  "${GDCM_DATA_ROOT}/*.acr"
  "${GDCM_DATA_ROOT}/*.nema"
  "${GDCM_DATA_ROOT}/*.dcm"
)

set(GDCM_DATA_DICOMDIR_FILENAMES_GLOB_EXPR
  # Those files do not have any extension, special regex:
  "${GDCM_DATA_ROOT}/dicomdir*"
)

# Case sensitive system:
if(UNIX)
  set(GDCM_DATA_IMAGE_FILENAMES_GLOB_EXPR
    ${GDCM_DATA_IMAGE_FILENAMES_GLOB_EXPR}
    "${GDCM_DATA_ROOT}/*.DCM"
  )
  set(GDCM_DATA_DICOMDIR_FILENAMES_GLOB_EXPR
    ${GDCM_DATA_DICOMDIR_FILENAMES_GLOB_EXPR}
    "${GDCM_DATA_ROOT}/DICOMDIR*"
  )
endif()

# GLOB expression !
file(GLOB GDCM_DATA_IMAGE_FILENAMES_GLOB
  ${GDCM_DATA_IMAGE_FILENAMES_GLOB_EXPR}
  )

# Same for DICOMDIR:
file(GLOB GDCM_DATA_DICOMDIR_FILENAMES_GLOB
  ${GDCM_DATA_DICOMDIR_FILENAMES_GLOB_EXPR}
  )

set(GDCM_DATA_FILENAMES_GLOB
  ${GDCM_DATA_IMAGE_FILENAMES_GLOB}
  ${GDCM_DATA_DICOMDIR_FILENAMES_GLOB}
)

# List of images that are technically difficult to read
# Hopefully we will soon be able to read them
set(BLACK_LIST_READER
# grrrrrr do not handle swapping correctly (implicit dataset are such a pain)
  LIBIDO-16-ACR_NEMA-Volume.dcm
# Papyrus:
  PET-cardio-Multiframe-Papyrus.dcm
# No way to read this image:
  ELSCINT1_LOSSLESS_RICE.dcm
# Toshiba US Private Data Storage 1.2.392.200036.9116.7.8.1.1.1
  TOSHIBA_MDW_HEADER.dcm
# TODO, nasty:
  PhilipsLosslessRice.dcm

# This file is incorrect
  IM-0001-0066.dcm
# Problem with RLE
  SIEMENS_GBS_III-16-ACR_NEMA_1.acr
  SIEMENS_GBS_III-16-ACR_NEMA_1-ULis2Bytes.dcm
# TestImageChangeTransferSyntax1
  MR-MONO2-12-angio-an1.acr
# Openjpeg / Part 2:
  lena512_rot90.j2k.dcm
)
if(NOT GDCM_USE_PVRG)
  set(BLACK_LIST_READER
    # The following should really fails according to JPEG spec, IJG is failing and even when forcing
    # decompression using IJG output is different from PVRG...
    # Apparently pvrg likes it this way...
    #  PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm
    ${BLACK_LIST_READER}
    PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm
  )
endif()
if(GDCM_WORDS_BIGENDIAN)
  set(BLACK_LIST_READER
    ${BLACK_LIST_READER}
    MR_Philips_Intera_PrivateSequenceExplicitVR_in_SQ_2001_e05f_item_wrong_lgt_use_NOSHADOWSEQ.dcm
    MR_Philips_Intera_SwitchIndianess_noLgtSQItem_in_trueLgtSeq.dcm
    PHILIPS_Intera-16-MONO2-Uncompress.dcm
    SIEMENS_MAGNETOM-12-MONO2-GDCM12-VRUN.dcm
    PICKER-16-MONO2-No_DicomV3_Preamble.dcm
  )
endif()

if(NOT GDCM_SUPPORT_BROKEN_IMPLEMENTATION)
  set(BLACK_LIST_READER
    ${BLACK_LIST_READER}
    GE_GENESIS-16-MONO2-WrongLengthItem.dcm # can't read odd length
    PHILIPS_GDCM12xBug2.dcm # odd length
    KODAK-12-MONO1-Odd_Terminated_Sequence.dcmq # odd length
    PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm # odd length
    PHILIPS_Gyroscan-8-MONO2-Odd_Sequence.dcm # odd length
    BugGDCM2_UndefItemWrongVL.dcm # odd length
    THERALYS-12-MONO2-Uncompressed-Even_Length_Tag.dcm
    CT-SIEMENS-MissingPixelDataInIconSQ.dcm # odd length
    00191113.dcm
    MR_Philips_Intera_PrivateSequenceImplicitVR.dcm
    PICKER-16-MONO2-Nested_icon.dcm
    PHILIPS_GDCM12xBug.dcm # odd length
    MR_Philips-Intera_BreaksNOSHADOW.dcm
    NM_Kakadu44_SOTmarkerincons.dcm
    MARCONI_MxTWin-12-MONO2-JpegLossless-ZeroLengthSQ.dcm
    )
endif()

set(GDCM_DATA_FILENAMES)
set(GDCM_BLACK_LIST_READER_DATA_FILENAMES)
foreach(filename ${GDCM_DATA_FILENAMES_GLOB})
  get_filename_component(filename_name ${filename} NAME)
  string(REGEX MATCH ${filename_name} bad_dicom ${BLACK_LIST_READER})
  if(NOT bad_dicom)
    set(GDCM_DATA_FILENAMES "${GDCM_DATA_FILENAMES}\n\"${filename}\",")
  else()
    set(GDCM_BLACK_LIST_READER_DATA_FILENAMES "${GDCM_BLACK_LIST_READER_DATA_FILENAMES}\n\"${filename}\",")
  endif()
endforeach()


# Populate GDCM_DATA_FILENAMES:
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/gdcmDataFileNames.cxx.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gdcmDataFileNames.cxx"
  )
