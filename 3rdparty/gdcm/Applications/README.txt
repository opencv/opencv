I know I will regret that later, but for now Application are part of GDCM

Naming convention, pretty much the same used for tiff (tiffinfo and tiffdump),
 we have gdcmdump, gdcminfo (should I rename them dcmdump. dcminfo ?
 but it will collide with dcmtk...)

TODO:
I have this file: acc-max.dcm  it's pretty much a ACR-NEMA file,
but it contains Modality. It would be great if gdcmconv would handle this file
 and generate a file that would make dciodvfy happy
