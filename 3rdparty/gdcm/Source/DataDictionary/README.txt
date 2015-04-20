Part 3.6

The following files:

acuson.xml
agfa.xml
camtron.xml
dicom3.xml
elscint.xml
gems.xml
isg.xml
other.xml
papyrus.xml
philips.xml
picker.xml
siemens.xml
spi.xml
toshiba.xml

where all converted from the dicom3tools (see COPYRIGHT.dicom3tools) own format into an XML form





GroupName.dic is missing the Variable Pixel Data

Design of DICOMV3.xml
At one point in time the name attribute used to be in the character data of a description sub-element of entry. This was done to cope with case where the name would be a multi-line description. This was nice for some weird private attribute name in private dictionary...but this was adding more troubles than solving them. Indeed one would have to traverse each name to check for return line and handle them in a GUI. Therefore the attribute name in a public/private element should be single line (no \t, \n or \r)
TODO: should name be ASCII (issue with µ which is not ASCII) ?
