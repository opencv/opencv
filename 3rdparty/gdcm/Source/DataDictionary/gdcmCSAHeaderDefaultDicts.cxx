
// GENERATED FILE DO NOT EDIT
// $ xsltproc CSADefaultDicts.xsl CSAHeader.xml > gdcmCSAHeaderDefaultDicts.cxx

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMCSAHEADERDEFAULTDICT_CXX
#define GDCMCSAHEADERDEFAULTDICT_CXX

#include "gdcmCSAHeaderDict.h"
#include "gdcmCSAHeaderDictEntry.h"
#include "gdcmVR.h"
#include "gdcmVM.h"


namespace gdcm {
typedef struct
{
  const char *name;
  const char *type;
  VR::VRType vr;
  VM::VMType vm;
  const char *description;
} CSA_DICT_ENTRY;

static const CSA_DICT_ENTRY CSAHeaderDataDict [] = {
  {"AcquisitionDate","3",VR::DA,VM::VM1,"A number identifying the single continuous gathering of data over a period of time which resulted in this image"},
  {"ContentDate","2C",VR::DA,VM::VM1,"The date the image data creation started. Required if image is part of a series in which the images are temporally related"},
  {"AcquisitionTime","3",VR::TM,VM::VM1,"The time the acquisition of data that resulted in this image started"},
  {"ContentTime","2C",VR::TM,VM::VM1,"The time the image pixrl data creation started. Required if image is part of a series in which the images are temporally related"},
  {"Acquisition Datetime","3",VR::DT,VM::VM1,"The date and time that the acquisition of dada that result in this image started"},
  {"DerivationDescription","3",VR::ST,VM::VM1,"A text description of how this image was derived"},
  {"AcquisitionNumber","2",VR::IS,VM::VM1,"Image identification characteristics"},
  {"InstanceNumber","2",VR::IS,VM::VM1,"A number that identifies this image"},
  {"ImageComments","3",VR::LT,VM::VM1,"User-defined comments about the image"},
  {"ImageInAcquisition","3",VR::IS,VM::VM1,"Number of images that resulted from this acquisition of data"},
  {">Referenced SOP Class UID","1C",VR::UI,VM::VM1,"Uniquely identifies the referenced SOP Class"},
  {">Referenced SOP Instance UID","1C",VR::UI,VM::VM1,"Uniquely identifies the referenced SOP Instance"},
  {"> Referenced Frame Number","3",VR::IS,VM::VM1_n,"References one or more image frames of a Multi-frame Image SOP Instance, identifying which frames are siginificantly related to this image"},
  //{">Referenced SOP Class UID","1C",VR::UI,VM::VM1,"Uniquely identifies the referenced SOP Class"},
  //{">Referenced SOP Instance UID","1C",VR::UI,VM::VM1,"Uniquely identifies the referenced SOP Instance"},
  //{"> Referenced Frame Number","3",VR::IS,VM::VM1_n,"References one or more image frames of a Multi-frame Image SOP Instance, identifying which frames are siginificantly related to this image"},
  {"ImageType","3",VR::CS,VM::VM1_n,"Image identification characteristics"},
  {"Lossy Image Compression","3",VR::CS,VM::VM1,"Specifies whether an image has undergone lossy compression. Enumerated values: 00 = image has not been subjected to lossy compression. 01 = image has been subjected to lossy compression"},
  {"Quality Control Image","1C",VR::CS,VM::VM1,"Indicates whether or not this image is a quality control or phantom image. Enumerated values: YES, NO"},
  {"PatientOrientation","2",VR::CS,VM::VM1,"Patient direction of the rows and columns of the image. Not required for MR images"},
  //{"ImageType","1",VR::CS,VM::VM1_n,"Image identification characteristics"},
  {"Samples per Pixel","1",VR::US,VM::VM1,"Number of samples (planes) in this image"},
  {"PhotometricInterpretation","1",VR::CS,VM::VM1,"Specifies the intended interpretation of the pixel data"},
  {"BitsAllocated","1",VR::US,VM::VM1,"Number of bits allocated for each pixel sample. Each sample shall have the same number of bits allocated"},
  {"Scanning Sequence","1",VR::CS,VM::VM1_n,"Description of the type of data taken. Enumerated values: SE = Spin Echo; IR = Inversion Recovery; GR = Gradient Recalled; EP = Echo Planar; RM = Research Mode"},
  {"SequenceVariant","1",VR::CS,VM::VM1_n,"Variant of the Scanning Sequence"},
  {"ScanOptions","2",VR::CS,VM::VM1_n,"Parameters of scanning sequence"},
  {"MRAcquisitionType","2",VR::CS,VM::VM1,"Identification of date encoding scheme. Enumerated Values: 2D = frequency x phase 3D = frequency x phase x phase"},
  {"SequenceName","3",VR::SH,VM::VM1,"User defined name for the Scanning Sequence and Sequence Varaint combination"},
  {"AngioFlag","3",VR::CS,VM::VM1,"Angio image indicator. Primary image for angio processing. Enumerated values: Y = image is Angio N = image is not Angio"},
  {"RepetitionTime","2C",VR::DS,VM::VM1,"The period of time in msec between the beginning of a pulse sequence and the beginning of a succeeding (essentially identical) pulse sequence. Required except when Scanning Sequence is EP and Sequnece Variant is not segmented k-space"},
  {"EchoTime","2",VR::DS,VM::VM1,"Time in msec between the middle of the excitation pulse and the peak of the echo produced. In the case of segmented k-space, the TE(eff) is the time between the middle of the excitation pulse to the peak of the echo that is used to cover the center of k-space)"},
  {"InversionTime","2C",VR::DS,VM::VM1,"Time in msec after the middle of inverting RF pulse to middle of excitation pulse to detect the amount of longitudinal magnetization. Required if Scanning Sequence has value of IR"},
  {"NumberOfAverages","3",VR::DS,VM::VM1,"Number of times a given pulse sequence is repeated before any parameter has changed"},
  {"ImagingFrequency","3",VR::DS,VM::VM1,"Precession frquency in MHz of the nucleus being addressed"},
  {"ImagedNucleus","3",VR::SH,VM::VM1,"Nucleus that is resonant at the imaging frequency"},
  {"EchoNumbers","3",VR::IS,VM::VM1_n,"The echo number used in generating this image. In the case of segmented k-space, it is the effective Echo Number"},
  {"MagneticFieldStrength","3",VR::DS,VM::VM1,"Nominal Field strength of MR magnet, in Tesla"},
  {"SpacingBetweenSlices","3",VR::DS,VM::VM1,"Spacing between slices, in mm. The spacing is measured from the center-to-center of each slice"},
  {"NumberOfPhaseEncodingSteps","3",VR::IS,VM::VM1,"Total number of lines in k-space in the y-direction collecting during acquisition"},
  {"EchoTrainLength","2",VR::IS,VM::VM1,"Number of lines in k-space acquired per excitation per image"},
  {"PercentSampling","3",VR::DS,VM::VM1,"Fraction of acquisition matrix lines acquired, expressed as a percent"},
  {"PercentPhaseFieldOfView","3",VR::DS,VM::VM1,"Ration of field of view dimension in phase direction to field of view dimension in frequency direction, expressed as a percent"},
  {"PixelBandwidth","3",VR::DS,VM::VM1,"Reciprocal of the total sampling period, in hertz per pixel"},
  {"BeatRejectionFlag","3",VR::CS,VM::VM1,"Beat length sorting has been applied. Enumerated values: Y = yes N = no"},
  {"Low R-R Value","3",VR::IS,VM::VM1,"R-R interval low limit for beat rejection, in msec"},
  {"High R-R Value","3",VR::IS,VM::VM1,"R-R interval high limit for beat rejection, in msec"},
  {"TriggerTime","2C",VR::DS,VM::VM1,"Time, in msec, between peak of the R wave and the peak of the echo produced. In the case of segmented k-space, the TE(eff) is the time between the peak of the echo that is used to cover the center of k-space. Required for Scan Options which include heart gating"},
  {"NominalInterval","3",VR::IS,VM::VM1,"Average R-R interval used for the scans, in msec"},
  {"IntervalsAcquired","3",VR::IS,VM::VM1,"Number of R-R intervals acquired"},
  {"IntervalsRejected","3",VR::IS,VM::VM1,"Number of R-R intervals rejected"},
  {"PVC Rejection","3",VR::LO,VM::VM1,"Description of type of PVC rejection criteria used"},
  {"Skip Beats","3",VR::IS,VM::VM1,"Number of beats skipped after a detected arrhythmia"},
  {"HeartRate","3",VR::IS,VM::VM1,"Beats per minute"},
  {"CardiacNumberOfImages","3",VR::IS,VM::VM1,"Number of imagesb per cardiac cycle"},
  {"TriggerWindow","3",VR::IS,VM::VM1,"Percent of R-R interval, based on Heart Rate (0018,1088), prescriped as a window for a valid/usable trigger"},
  {"ReconstructionDiameter","3",VR::DS,VM::VM1,"Diameter in mm of the region from within which data were used in creating the reconstruction of the image. Data may exist outside this region and portions of the patient may exist outside the region"},
  {"Receive Coil","3",VR::SH,VM::VM1,"Received coil used"},
  {"TransmittingCoil","3",VR::SH,VM::VM1,"Transmitted coil used"},
  {"AcquisitionMatrix","3",VR::US,VM::VM4,"Dimension of acquired frequency/phase data before reconstruction. Multi-valued: frequency rows/frequency columns/phase rows/phase columns"},
  {"PhaseEncodingDirection","3",VR::CS,VM::VM1,"The axis of phase encoding with respect to the image. Enumerated Values: ROW = phase encoded in rows COL = phase encoded in columns"},
  {"FlipAngle","3",VR::DS,VM::VM1,"Steady state angle in degrees to which the magnetic vector is flipped from the magnetic vector to the primary field"},
  {"VariableFlipAngleFlag","3",VR::CS,VM::VM1,"Flip angle variation applied during image acquisition. Enumerated Values: Y = yes N = no"},
  {"SAR","3",VR::DS,VM::VM1,"Calculated whole body specific absortion rate in watts/kilogram"},
  {"dBdt","3",VR::DS,VM::VM1,"The rate of change of the gradient coil magnetic flux density with time (T/s)"},
  {"ContrastBolusAgent","2",VR::LO,VM::VM1,"Contrast or bolus agent"},
  {"ContrastBolusStartTime","3",VR::TM,VM::VM1,"Time of start of injection"},
  {"ContrastBolusStopTime","3",VR::TM,VM::VM1,"Time of end of contrast injection"},
  {"ContrastBolusTotalDose","3",VR::DS,VM::VM1,"Total amount in milliliters of the indiluted contrast agent"},
  {"Contrast Flow Rate(s)","3",VR::DS,VM::VM1_n,"Rate(s) of injection(s) in millilitres/sec"},
  {"ContrastBolusVolume","3",VR::DS,VM::VM1,"Volume injected in milliliters of diluted contrast agent"},
  {"ContrastFlowDurations","3",VR::DS,VM::VM1_n,"Duration(s) of injection(s) in seconds. Each Contrast Flow Duration value shall correspond to a value of Contrast Flow Rate (0018,1046)"},
  {"Contrast/Bolus Ingredient","3",VR::CS,VM::VM1,"Active ingredient of agent, Defined Terms: IODINE, GADOLIMIUM, CARBON DIOXIDE, BARIUM"},
  {"Contrast/Bolus Ingredient Concentration","3",VR::DS,VM::VM1,"Milligrams of active ingredient per milliliter of (diluted) agent"},
  {"SamplesPerPixel","1",VR::US,VM::VM1,"Number of samples (planes) in this image"},
  //{"PhotometricInterpretation","1",VR::CS,VM::VM1,"Specifies the intended interpretation of the pixel data"},
  {"Rows","1",VR::US,VM::VM1,"Number of rows in the image"},
  {"Columns","1",VR::US,VM::VM1,"Number of columns in the image"},
  {"PixelAspectRatio","1C",VR::IS,VM::VM2,"Ratio of the vertical size and horizontal size of the pixels in the image specified by a pair of integer values where the first value is the vertical pixel size, and the second value is the horizontal pixel size. Required if the aspect ratio is not 1/1 and the Image Plane Module is not applicable to this Image"},
  //{"BitsAllocated","1",VR::US,VM::VM1,"Number of bits allocated for each pixel sample. Each sample shall have the same number of bits allocated"},
  {"BitsStored","1",VR::US,VM::VM1,"Number of bits stored for each pixel sample. Each sample shall have the same number of bits stored"},
  {"HighBit","1",VR::US,VM::VM1,"Most significant bit for pixel sample data. Each sample shall have the same high bit"},
  {"PixelRepresentation","1",VR::US,VM::VM1,"Data representation of the pixel samples. Each sample shall have the same pixel representation. Enumerated values: 0000H = unsigned integer. 0001H = 2's complement."},
  {"PixelData","1",VR::OB_OW,VM::VM1,"A data stream of the pixel samples which comprises the image"},
  {"Smallest Image Pixel Value","3",VR::US_SS,VM::VM1,"The minimum actual pixel value encountered in this image"},
  {"Largest Image Pixel Value","3",VR::US_SS,VM::VM1,"The maximum actual pixel value encountered in this image"},
  {"SliceThickness","2",VR::DS,VM::VM1,"Nominal slice thickness, in mm"},
  {"ImageOrientationPatient","1",VR::DS,VM::VM6,"The direction cosines of the first row and the first column with respect to the patient"},
  {"ImagePositionPatient","1",VR::DS,VM::VM3,"The x,y and z coordinates of the upper left and hand corner (first pixel transmitted) of the image, in mm"},
  {"SliceLocation","3",VR::DS,VM::VM1,"Relative position of exposure expressed in mm"},
  {"PixelSpacing","1",VR::DS,VM::VM2,"Physical distance in the patient between the center of each pixel, specified by a numeric pair-adjacent row spacing (delimiter) adjacent column spacing in mm"},
  {"WindowCenter","3",VR::DS,VM::VM1_n,"Window center"},
  {"WindowWidth","1C",VR::DS,VM::VM1_n,"Window width"},
  {"WindowCenterWidthExplanation","3",VR::LO,VM::VM1_n,"Free form explanation of the meaning of the Window Center and Width. Multiple values correspond to multiple Window Center and Width values"},
  {"Manufacturer","2",VR::LO,VM::VM1,"Manufacturer of the equipment that produced the digital images"},
  {"InstitutionName","3",VR::LO,VM::VM1,"Institution where the equipment is located that produced the digital images"},
  {"InstitutionAddress","3",VR::ST,VM::VM1,"Mailing address of the institution where the equipment is located that produced the digital images"},
  {"ManufacturersModelName","3",VR::LO,VM::VM1,"Manufacturer's model number of the equipment that produced the digital images"},
  {"DeviceSerialNumber","3",VR::LO,VM::VM1,"Manufacturer's seriel number of the equipment that produced the digital images"},
  {"SoftwareVersions","3",VR::LO,VM::VM1_n,"Manufacturer's designation of software of the equipment that produced the digital images"},
  {"OverlayRows","1",VR::US,VM::VM1,"Number of Rows in Overlay"},
  {"OverlayColumns","1",VR::US,VM::VM1,"Number of Columns in Overlay"},
  {"OverlayType","1",VR::CS,VM::VM1,"Indicates whether this overlay represents a region of interest or other graphics"},
  {"Overlay Origin","1",VR::US,VM::VM1,"Location of first overlay point with respect to pixel in the image, given as row/column"},
  {"OverlayBitsAllocated","1",VR::US,VM::VM1,"Number of bits allocated in the overlay"},
  {"OverlayBitPosition","1",VR::US,VM::VM1,"Bit in which overlay is stored"},
  {"Overlay Data","1C",VR::OB_OW,VM::VM1,"Overlay pixel data"},
  {"HeaderType","1C",VR::CS,VM::VM1,"Medcom header characteristics. Defined Terms: MEDCOM 1"},
  {"HeaderVersion","2C",VR::LO,VM::VM1,"Version of Medcom header"},
  {"HeaderInfo","3",VR::OB,VM::VM1,"Manufacturer header info"},
  {"HistoryInfo","3",VR::OB,VM::VM1,"Patient registration history"},
  {"EchoLinePosition","1",VR::IS,VM::VM1,"Fourier line position with the maximal echo for the performed acquisition"},
  {"EchoColumnPosition","1",VR::IS,VM::VM1,"Echo column position for the performed acquisition"},
  {"EchoPartitionPosition","1",VR::IS,VM::VM1,"Echo partition position for the performed acquisition"},
  {"UsedChannelMask","1",VR::UL,VM::VM1,"8 bit mask of the used receiver channels for the performed acquisition. Example: channel 0: 00000001 channel 3: 00000111"},
  {"Actual3DImaPartNumber","1",VR::IS,VM::VM1,"Number of a 3D partitions beginning with 0"},
  {"ICE_Dims","1",VR::LO,VM::VM1,"The 9 used ICE object dimensions of the performed acquisition. Combined/unset dimensions will be marked with 'X'. E.g.: X_2_1_1_1_1_2_1_1"},
  {"B_value","1;",VR::IS,VM::VM1,"Diffusion effect in s/mm*mm of the ICE program for the performed acquisition"},
  {"Filter1","1",VR::IS,VM::VM1,"Context vision filter"},
  {"Filter2","1",VR::IS,VM::VM1,"not used"},
  {"ProtocolSliceNumber","1",VR::IS,VM::VM1,"Number of the slice beginning with 0"},
  {"RealDwellTime","1",VR::IS,VM::VM1,"The time in ns between the beginning of sampling one data point and the beginning of sampling of next data point in the acquired signal. This means the dwell time is the sampling rate during digital conversion of an acquired signal"},
  {"PixelFile","1",VR::UN,VM::VM1,"Used raw data file for the performed acquisition"},
  {"PixelFileName","1",VR::UN,VM::VM1,"Used raw data file name for the performed acquisition"},
  {"SliceMeasurementDuration","1",VR::DS,VM::VM1,"Time duration between two slices of the performed acquisition"},
  {"AcquisitionMatrixText","1",VR::SH,VM::VM1,"Used acquisition matrix description"},
  {"SequenceMask","1",VR::UL,VM::VM1,"Parameters used for acquisition, e.g. door open, interpolation, raw filter, Siemens seqence ...."},
  {"MeasuredFourierLines","1",VR::IS,VM::VM1,"Number of performed fourier lines"},
  {"FlowEncodingDirection","1",VR::IS,VM::VM1,"Flow encoding direction"},
  {"FlowVenc","1",VR::FD,VM::VM1,"Flow Quant attribute"},
  {"PhaseEncodingDirectionPositive","1",VR::IS,VM::VM1,"Phase encoding direction: 0 = negative; 1 = positive"},
  {"NumberOfImagesInMosaic","1",VR::US,VM::VM1,"Number of slices in a mosaic image"},
  {"DiffusionGradientDirection","1",VR::FD,VM::VM3,"Diffusion in gradient direction"},
  {"ImageGroup","1",VR::US,VM::VM1,"Group of images"},
  {"SliceNormalVector","1",VR::FD,VM::VM3,"X,y and z normal vector of the slices"},
  {"DiffusionDirection","1",VR::CS,VM::VM1,"Diffusion direction"},
  {"TimeAfterStart","1",VR::DS,VM::VM1,"Time delay after start of measurment"},
  //{"FlipAngle","1",VR::DS,VM::VM1,"Flip angle for SC images"},
  //{"SequenceName","1",VR::SH,VM::VM1,"Sequence name for SC images"},
  //{"RepetitionTime","1",VR::DS,VM::VM1,"Repetition time for SC images"},
  //{"EchoTime","1",VR::DS,VM::VM1,"Echo time for SC images"},
  //{"NumberOfAverages","1",VR::DS,VM::VM1_n,"Number of averages for SC images"},
  {"NonimageType","1",VR::CS,VM::VM1,"Data identification characteristics. Defined terms: RAW DATA NUM 4; SPEC NUM4"},
  {"NonimageVersion","3",VR::LO,VM::VM1,"Version of Non-image data"},
  {"NonimageInfo","3",VR::OB,VM::VM1,"Description of Non-image data"},
  {"NonimageData","2",VR::OB,VM::VM1,"Binary data stream"},
  //{"ImageType","3",VR::CS,VM::VM1_n,"Image identification characteristics"},
  //{"AcquisitionDate","3",VR::DA,VM::VM1,"The date the acquisition of data started"},
  //{"AcquisitionTime","3",VR::TM,VM::VM1,"The time the acquisition of data started"},
  //{"DerivationDescription","3",VR::ST,VM::VM1,"A text description of how this data set was derived"},
  //{"AcquisitionNumber","2",VR::IS,VM::VM1,"A number identifying the gathering of data over a period of time which resulted in this data set"},
  {"ImageNumber","1",VR::IS,VM::VM1,"A number that identifies this image"},
  //{"ImageComments","1",VR::LT,VM::VM1,"User-defined comments about the image"},
  {"ReferencedImageSequence","1",VR::UI,VM::VM1_n,"A sequence which provides reference to a set of Image SOP Class/Instance identifying other images significantly related to this image (localizer images)"},
  //{"PatientOrientation","1",VR::CS,VM::VM1,"Patient direction of the rows and columns of the image. Not required for MR images"},
  {"ScanningSequence","1",VR::CS,VM::VM1_n,"Description of the type of data taken. Enumerated values: SE = Spin Echo; IR = Inversion Recovery; GR = Gradient Recalled; EP = Echo Planar; RM = Research Mode"},
  //{"SequenceName","1",VR::SH,VM::VM1,"User defined name for the Scanning Sequence and Sequence Varaint combination"},
  //{"RepetitionTime","1",VR::DS,VM::VM1,"The period of time in msec between the beginning of a pulse sequence and the beginning of a succeeding (essentially identical) pulse sequence. Required except when Scanning Sequence is EP and Sequnece Variant is not segmented k-space"},
  //{"EchoTime","1",VR::DS,VM::VM1,"Time in msec between the middle of the excitation pulse and the peak of the echo produced. In the case of segmented k-space, the TE(eff) is the time between the middle of the excitation pulse to the peak of the echo that is used to cover the center of k-space)"},
  //{"InversionTime","1",VR::DS,VM::VM1,"Time in msec after the middle of inverting RF pulse to middle of excitation pulse to detect the amount of longitudinal magnetization. Required if Scanning Sequence has value of IR"},
  //{"NumberOfAverages","1",VR::DS,VM::VM1,"Number of averages"},
  //{"ImagingFrequency","1",VR::DS,VM::VM1,"Precession frquency in MHz of the nucleus being addressed"},
  //{"ImagedNucleus","1",VR::SH,VM::VM1,"Nucleus that is resonant at the imaging frequency"},
  //{"EchoNumbers","1",VR::IS,VM::VM1,"The echo number used in generating this image. In the case of segmented k-space, it is the effective Echo Number"},
  //{"MagneticFieldStrength","1",VR::DS,VM::VM1,"Nominal Field strength of MR magnet, in Tesla"},
  //{"NumberOfPhaseEncodingSteps","1",VR::IS,VM::VM1,"Total number of lines in k-space in the y-direction collecting during acquisition"},
  //{"EchoTrainLength","1",VR::IS,VM::VM1,"Number of lines in k-space acquired per excitation per image"},
  //{"PercentSampling","1",VR::DS,VM::VM1,"Fraction of acquisition matrix lines acquired, expressed as a percent"},
  //{"PercentPhaseFieldOfView","1",VR::DS,VM::VM1,"Ration of field of view dimension in phase direction to field of view dimension in frequency direction, expressed as a percent"},
  //{"TriggerTime","1",VR::DS,VM::VM1,"Time, in msec, between peak of the R wave and the peak of the echo produced. In the case of segmented k-space, the TE(eff) is the time between the peak of the echo that is used to cover the center of k-space. Required for Scan Options which include heart gating"},
  {"ReceivingCoil","1",VR::SH,VM::VM1,"Received coil used"},
  {"TransmittingColi","1",VR::SH,VM::VM1,"Transmitted coil used"},
  //{"AcquisitionMatrixText","1",VR::US,VM::VM4,"Dimension of acquired frequency/phase data before reconstruction. Multi-valued: frequency rows/frequency columns/phase rows/phase columns"},
  //{"PhaseEncodingDirection","1",VR::CS,VM::VM1,"The axis of phase encoding with respect to the image. Enumerated Values: ROW = phase encoded in rows COL = phase encoded in columns"},
  //{"FlipAngle","1",VR::DS,VM::VM1,"Steady state angle in degrees to which the magnetic vector is flipped from the magnetic vector to the primary field"},
  //{"VariableFlipAngleFlag","1",VR::CS,VM::VM1,"Flip angle variation applied during image acquisition. Enumerated Values: Y = yes N = no"},
  //{"SAR","1",VR::DS,VM::VM1,"Calculated whole body specific absortion rate in watts/kilogram"},
  //{"dBdt","1",VR::DS,VM::VM3,"The rate of change of the gradient coil magnetic flux density with time (T/s)"},
  //{"Rows","1",VR::US,VM::VM1,"Number of rows in the image"},
  //{"Columns","1",VR::US,VM::VM1,"Number of columns in the image"},
  //{"SliceThickness","1",VR::DS,VM::VM1,"Nominal slice thickness, in mm"},
  //{"ImagePositionPatient","1",VR::DS,VM::VM3,"The direction cosines of the first row and the first column with respect to the patient"},
  //{"ImageOrientationPatient","1",VR::DS,VM::VM6,"The x,y and z coordinates of the upper left and hand corner (first pixel transmitted) of the image, in mm."},
  //{"SliceLocation","1",VR::DS,VM::VM1,"Relative position of exposure expressed in mm"},
  //{"EchoLinePosition","1",VR::IS,VM::VM1,"Fourier line position with the maximal echo for the performed acquisition"},
  //{"EchoColumnPosition","1",VR::IS,VM::VM1,"Echo column position for the performed acquisition"},
  //{"EchoPartitionPosition","1",VR::IS,VM::VM1,"Echo partition position for the performed acquisition"},
  //{"Actual3DImaPartNumber","1",VR::IS,VM::VM1,"Number of a 3D partitions beginning with 0"},
  //{"RealDwellTime","1",VR::IS,VM::VM1,"The time in ns between the beginning of sampling one data point and the beginning of sampling of next data point in the acquired signal. This means the dwell time is the sampling rate during digital conversion of an acquired signal"},
  //{"ProtocolSliceNumber","1",VR::UN,VM::VM1,"Number of the slice beginning with 0"},
  //{"PixelFile","1",VR::UN,VM::VM1,"Used raw data file for the performed acquisition"},
  //{"PixelFileName","1",VR::LO,VM::VM1,"Used raw data file name for the performed acquisition"},
  //{"ICE_Dims","1",VR::DS,VM::VM1,"The 9 used ICE object dimensions of the performed acquisition. Combined/unset dimensions will be marked with 'X'. E.g.: X_2_1_1_1_1_2_1_1"},
  //{"PixelSpacing","1",VR::DS,VM::VM2,"Physical distance in the patient between the center of each pixel, specified by a numeric pair-adjacent row spacing (delimiter) adjacent column spacing in mm"},
  {"SourceImageSequence","1",VR::UI,VM::VM1_n,"A sequence which identifies the set of Image SOP Class/Instance pairs of the images which were use to derive this image"},
  //{"PixelBandwidth","1",VR::DS,VM::VM1,"Reciprocal of the total sampling period, in hertz per pixel"},
  //{"SliceMeasurementDuration","1",VR::DS,VM::VM1,"Time duration between two slices of the performed acquisition"},
  //{"SequenceMask","1",VR::UL,VM::VM1,"Parameters used for acquisition, e.g. door open, interpolation, raw filter, Siemens seqence .·"},
  //{"AcquisitionMatrixText","1",VR::SH,VM::VM1,"Used acquisition matrix description"},
  //{"MeasuredFourierLines","1",VR::IS,VM::VM1,"Number of performed fourier lines"},
  {"ulVersion","3",VR::UL,VM::VM1,"Protocol version"},
  {"tSequenceFileName","3",VR::ST,VM::VM1,"Sequence file name for actual measurement protocol"},
  {"tProtocolName","3",VR::ST,VM::VM1,"Name of actual measurement protocol"},
  {"tReferenceImage0","3",VR::LO,VM::VM1,"Referenced image"},
  {"tReferenceImage1","3",VR::LO,VM::VM1,"Referenced image"},
  {"tReferenceImage2","3",VR::LO,VM::VM1,"Referenced image"},
  {"lScanRegionPosSag","3",VR::FD,VM::VM1,"Desired table position in series block coordinate system"},
  {"lScanRegionPosCor","3",VR::FD,VM::VM1,"Desired table position in series block coordinate system"},
  {"lScanRegionPosTra","3",VR::FD,VM::VM1,"Desired table position in series block coordinate system"},
  {"ucScanRegionPosValid","3",VR::SH,VM::VM1,"Valid flag for desired table position in series block coordinate system"},
  {"aucSpare","3",VR::SH,VM::VM1,"Reserved"},
  {"lScanRegionDelta","3",VR::SL,VM::VM1,"Scan Region Position/Move [mm]"},
  {"sProtConsistencyInfo.tBaselineString","3",VR::LO,VM::VM1,"Baseline consistence info"},
  {"sProtConsistencyInfo.tGradCoilName","3",VR::SH,VM::VM1,"Coil name consistence info"},
  {"sProtConsistencyInfo.tGradAmplifierName","3",VR::LO,VM::VM1,"Gradient amplifier consistence info"},
  {"sProtConsistencyInfo.flNominalB0","3",VR::FD,VM::VM1,"Nominal Bo compensation consistence"},
  {"sGRADSPEC.sEddyCompensationX.aflAmplitude","3",VR::FD,VM::VM5,"Eddy compensation x amplitude gradient system specification"},
  {"sGRADSPEC.sEddyCompensationX.aflTimeConstant","3",VR::FD,VM::VM5,"Eddy compensation x time parameter gradient system specification"},
  {"sGRADSPEC.sEddyCompensationY.aflAmplitude","3",VR::FD,VM::VM5,"Eddy compensation y amplitude gradient system specification"},
  {"sGRADSPEC.sEddyCompensationY.aflTimeConstant","3",VR::FD,VM::VM5,"Eddy compensation y time parameter gradient system specification"},
  {"sGRADSPEC.sEddyCompensationZ.aflAmplitude","3",VR::FD,VM::VM5,"Eddy compensation z amplitude gradient system specification"},
  {"sGRADSPEC.sEddyCompensationZ.aflTimeConstant","3",VR::FD,VM::VM5,"Eddy compensation z time parameter gradient system specification"},
  {"sGRADSPEC.bB0CompensationValid","3",VR::SL,VM::VM1,"B0 compensation gradient system specification valid flag"},
  {"sGRADSPEC.sCrossTermCompensationXY.aflAmplitude","3",VR::FD,VM::VM5,"Crossterm compensation xy amplitude gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationXY.aflTimeConstant","3",VR::FD,VM::VM5,"Crossterm compensation xy time parameter gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationXZ.aflAmplitude","3",VR::FD,VM::VM5,"Crossterm compensation xz amplitude gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationXZ.aflTimeConstant","3",VR::FD,VM::VM5,"Crossterm compensation xz time parameter gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationYX.aflAmplitude","3",VR::FD,VM::VM5,"Crossterm compensation xz amplitude gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationYX.aflTimeConstant","3",VR::FD,VM::VM5,"Crossterm compensation yx time parameter gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationYZ.aflAmplitude","3",VR::FD,VM::VM5,"Crossterm compensation yz amplitude gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationYZ.aflTimeConstant","3",VR::FD,VM::VM5,"Crossterm compensation yz time parameter gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationZX.aflAmplitude","3",VR::FD,VM::VM5,"Crossterm compensation zx amplitude gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationZX.aflTimeConstant","3",VR::FD,VM::VM5,"Crossterm compensation zx time parameter gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationZY.aflAmplitude","3",VR::FD,VM::VM5,"Crossterm compensation zx amplitude gradient system specification"},
  {"sGRADSPEC.sCrossTermCompensationZY.aflTimeConstant","3",VR::FD,VM::VM5,"Crossterm compensation zy time parameter gradient system specification"},
  {"sGRADSPEC.bCrossTermCompensationValid","3",VR::SL,VM::VM1,"Crossterm compensation gradient system specification valid flag"},
  {"sGRADSPEC.lOffsetX","3",VR::SL,VM::VM1,"Gradient offset x direction [bit pattern]"},
  {"sGRADSPEC.lOffsetY","3",VR::SL,VM::VM1,"Gradient offset y direction [bit pattern]"},
  {"sGRADSPEC.lOffsetZ","3",VR::SL,VM::VM1,"Gradient offset z direction [bit pattern]"},
  {"sGRADSPEC.bOffsetValid","3",VR::SL,VM::VM1,"Gradient offsets valid flag"},
  {"sGRADSPEC.lDelayX","3",VR::SL,VM::VM1,"Gradient delay x direction"},
  {"sGRADSPEC.lDelayY","3",VR::SL,VM::VM1,"Gradient delay y direction"},
  {"sGRADSPEC.lDelayZ","3",VR::SL,VM::VM1,"Gradient delay z direction"},
  {"sGRADSPEC.bDelayValid","3",VR::SL,VM::VM1,"Gradient delay valid flag"},
  {"sGRADSPEC.flSensitivityX","3",VR::FD,VM::VM1,"Gradient sensitivity x direction [mT/m]"},
  {"sGRADSPEC.flSensitivityY","3",VR::FD,VM::VM1,"Gradient sensitivity y direction [mT/m]"},
  {"sGRADSPEC.flSensitivityZ","3",VR::FD,VM::VM1,"Gradient sensitivity z direction [mT/m]"},
  {"sGRADSPEC.bSensitivityValid","3",VR::SL,VM::VM1,"Gradient sensitivity valid flag"},
  {"sGRADSPEC.flGSWDMinRiseTime","3",VR::FD,VM::VM1,"Minimum gradient rise time for mode GRAD_GSWD_RISETIME"},
  {"sGRADSPEC.alShimCurrent","3",VR::SL,VM::VM16,"Shim current parameter [mA]"},
  {"sGRADSPEC.bShimCurrentValid","3",VR::SL,VM::VM1,"Shim current parameter valid flag"},
  {"sGRADSPEC.ucMode","3",VR::SH,VM::VM1,"Gradient mode: fast, normal, whisper"},
  {"sGRADSPEC.ucSpare0","3",VR::SH,VM::VM1,"Reserved"},
  {"sGRADSPEC.ucSpare1","3",VR::SH,VM::VM1,"Reserved"},
  {"sGRADSPEC.ucSpare2","3",VR::SH,VM::VM1,"Reserved"},
  {"sTXSPEC.asNucleusInfo[0].tNucleus","3",VR::SH,VM::VM1,"Transmitter system nucleus"},
  {"sTXSPEC.asNucleusInfo[0].lFrequency","3",VR::SL,VM::VM1,"Transmitter system frequency [Hz]"},
  {"sTXSPEC.asNucleusInfo[0].bFrequencyValid","3",VR::SL,VM::VM1,"Frequency valid flag"},
  {"sTXSPEC.asNucleusInfo[0].lDeltaFrequency","3",VR::SL,VM::VM1,"Offset from center frequency (lFrequency)"},
  {"sTXSPEC.asNucleusInfo[0].flReferenceAmplitude","3",VR::FD,VM::VM1,"Transmitter reference amplitude [V]"},
  {"sTXSPEC.asNucleusInfo[0].bReferenceAmplitudeValid","3",VR::SL,VM::VM1,"Reference amplitude valid flag"},
  {"sTXSPEC.asNucleusInfo[0].flAmplitudeCorrection","3",VR::FD,VM::VM1,"Transmitter amplitude correction factor, e.g. used for water suppression"},
  {"sTXSPEC.asNucleusInfo[0].bAmplitudeCorrectionValid","3",VR::SL,VM::VM1,"Amplitude correction valid flag"},

/* Manually added */
  {"sKSpace.ucSlicePartialFourier","1",VR::US,VM::VM1,"Partial Fourier Information 0x1,0x2,0x4,0x8,0x10 resp. 4/8,5/8,6/8,7/8,8/8"},
// http://www.nmr.mgh.harvard.edu/~greve/dicom-unpack
// http://web.archiveorange.com/archive/v/AW2a1uSBsvpLmfFd9xvy
/*
ASCCONV does contain most of the good info. In particular the slice
ordering is encoded thus:

    sSliceArray.ucMode -- should be in (1, 2, 4)
    enum SeriesMode
    {
      ASCENDING   = 0x01,
      DESCENDING  = 0x02,
      INTERLEAVED = 0x04
    };

And when interleaved, the slice ordering depends on whether the number
of slices is even or odd. I have it coded (counting from zero) as:

if ODD:
  slice_order = {0, 2, 4, ..., N-1, 1, 3, 5, ..., N-2}
else:
  slice_order = {1, 3, 5, ..., N-1, 0, 2, 4, ..., N-2}

which I can't immediately confirm, but I'm reasonably confident about it.

It's all a bit of a mess, but there is some code available on the CIRL
repo in bic/trunk/recon-tools/root/scanner/siemens.py

check out parse_siemens_hdr() in that module, where I also documented
various useful fields. And there are the helper functions (
strip_ascconv(), condense_array() ) in siemens_utils.py. No
docstrings, so ping me if you need clarification.
*/
  {"sSliceArray.ucMode","1",VR::US,VM::VM1,"slice ordering 0x1,0x2,0x4 resp. asc,desc,inter"},

  {0,0,VR::INVALID,VM::VM0,0 } // Gard
};

void CSAHeaderDict::LoadDefault()
{
   unsigned int i = 0;
   CSA_DICT_ENTRY n = CSAHeaderDataDict[i];
   while( n.name != 0 )
   {
     CSAHeaderDictEntry e( n.name, n.vr, n.vm, n.description );
     AddCSAHeaderDictEntry( e );
     n = CSAHeaderDataDict[++i];
   }
}

} // end namespace gdcm
#endif // GDCMCSAHEADERDEFAULTDICT_CXX
