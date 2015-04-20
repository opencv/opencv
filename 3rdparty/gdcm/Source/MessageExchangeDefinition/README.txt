Part 3.7 / Part 3.8

The DICOM UL protocol consists of seven Protocol Data Units:
a) A-ASSOCIATE-RQ PDU
b) A-ASSOCIATE-AC PDU
c) A-ASSOCIATE-RJ PDU
d) P-DATA-TF PDU
e) A-RELEASE-RQ PDU
f) A-RELEASE-RP PDU
g) A-ABORT PDU

The encoding of the DICOM UL PDUs is defined as follows (Big Endian byte ordering):
Note: The Big Endian byte ordering has been chosen for consistency with the OSI and TCP/IP environment.
This pertains to the DICOM UL PDU headers only. The encoding of the PDV message fragments is
defined by the Transfer Syntax negotiated at association establishment.
a) Each PDU type shall consist of one or more bytes that when represented, are numbered
sequentially, with byte 1 being the lowest byte number.
b) Each byte within the PDU shall consist of eight bits that, when represented, are numbered 7 to
0, where bit 0 is the low order bit.
c) When consecutive bytes are used to represent a string of characters, the lowest byte numbers
represent the first character.
d) When consecutive bytes are used to represent a binary number, the lower byte number has
the most significant value.
e) The lowest byte number is placed first in the transport service data flow.
f) An overview of the PDUs is shown in Figures 9-1 and 9-2. The detailed structure of each PDU
is specified in the following sections.
