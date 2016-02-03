Part 3.8
PS 3.10-2006
Page 10
3.3 PRESENTATION SERVICE DEFINITIONS
This Part of the Standard makes use of the following terms defined in ISO 8822:
a. Abstract Syntax;
b. Abstract Syntax Name.
3.4 DICOM INTRODUCTION AND OVERVIEW DEFINITIONS
This Part of the Standard makes use of the following terms defined in PS 3.1 of the DICOM Standard:
- Attribute.
3.5 DICOM INFORMATION OBJECT DEFINITIONS
This Part of the Standard makes use of the following terms defined in PS 3.3 of the DICOM Standard:
a. Information Object Definition.
3.6 DICOM DATA STRUCTURE AND ENCODING DEFINITIONS
This Part of the Standard makes use of the following terms defined in PS 3.5 of the DICOM Standard:
a. Data Element;
b. Data Set;
c. Data Element Type;
d. Value;
e. Value Multiplicity;
f. Value Representation;
3.7 DICOM MESSAGE EXCHANGE DEFINITIONS
This Part of the Standard makes use of the following terms defined in PS 3.7 of the DICOM Standard:
a. Service Object Pair (SOP) Class;
b. Service Object Pair (SOP) Instance;
c. Implementation Class UID.
3.8 DICOM MEDIA STORAGE AND FILE FORMAT DEFINITIONS
The following definitions are commonly used in this Part of the Standard:
Application Profile: A Media Storage Application Profile defines a selection of choices at the various
layers of the DICOM Media Storage Model which are applicable to a specific need or context in which the
media interchange is intended to be performed.
DICOM File Service: The DICOM File Service specifies a minimum abstract view of files to be provided
by the Media Format Layer. Constraining access to the content of files by the Application Entities
through such a DICOM File Service boundary ensures Media Format and Physical Media independence.
DICOM File: A DICOM File is a File with a content formatted according to the requirements of this Part of
the DICOM Standard. In particular such files shall contain, the File Meta Information and a properly
formatted Data Set.
PS 3.10-2006
Page 11
DICOMDIR File: A unique and mandatory DICOM File within a File-set which contains the Media Storage
Directory SOP Class. This File is given a single component File ID, DICOMDIR.
File: A File is an ordered string of zero or more bytes, where the first byte is at the beginning of the file
and the last byte at the end of the File. Files are identified by a unique File ID and may by written, read
and/or deleted.
File ID: Files are identified by a File ID which is unique within the context of the File-set they belong to. A
set of ordered File ID Components (up to a maximum of eight) forms a File ID.
File ID Component: A string of one to eight characters of a defined character set.
File Meta Information: The File Meta Information includes identifying information on the encapsulated
Data Set. It is a mandatory header at the beginning of every DICOM File.
File-set: A File-set is a collection of DICOM Files (and possibly non-DICOM Files) that share a common
naming space within which File IDs are unique.
File-set Creator: An Application Entity that creates the DICOMDIR File (see section 8.6) and zero or
more DICOM Files.
File-set Reader: An Application Entity that accesses one or more files in a File-set.
File-set Updater: An Application Entity that accesses Files, creates additional Files, or deletes existing
Files in a File-set. A File-set Updater makes the appropriate alterations to the DICOMDIR file reflecting
the additions or deletions.
DICOM File Format: The DICOM File Format provides a means to encapsulate in a File the Data Set
representing a SOP Instance related to a DICOM Information Object.
Media Format: Data structures and associated policies which organizes the bit streams defined by the
Physical Media format into data file structures and associated file directories.
Media Storage Model: The DICOM Media Storage Model pertains to the data structures used at different
layers to achieve interoperability through media interchange.
Media Storage Services: DICOM Media Storage Services define a set of operations with media that
facilitate storage to and retrieval from the media of DICOM SOP Instances.
Physical Media: A piece of material with recording capabilities for streams of bits. Characteristics of a
Physical Media include form factor, mechanical characteristics, recording properties and rules for
recording and organizing bit streams in accessible structures
Secure DICOM File: A DICOM File that is encapsulated with the Cryptographic Message Syntax
specified in RFC 2630.
Secure File-set: A File-set in which all DICOM Files are Secure DICOM Files.
Secure Media Storage Application Profile: A DICOM Media Storage Application Profile that requires a
Secure File-set.
PS 3.10-2006
Page 12
4 Symbols and Abbreviations
The following symbols and abbreviations are used in this Part of the Standard.
ACC American College of Cardiology
ACR American College of Radiology
ASCII American Standard Code for Information Interchange
AE Application Entity
ANSI American National Standards Institute
CEN/TC/251 Comite Europeen de Normalisation - Technical Committee 251 - Medical
Informatics
DICOM Digital Imaging and Communications in Medicine
FSC File-set Creator
FSR File-set Reader
FSU File-set Updater
HL7 Health Level 7
HTML Hypertext Transfer Markup Language
IEEE Institute of Electrical and Electronics Engineers
ISO International Standards Organization
ID Identifier
IOD Information Object Definition
JIRA Japan Industries Association of Radiation Apparatus
MIME Multipurpose Internet Mail Extensions
NEMA National Electrical Manufacturers Association
OSI Open Systems Interconnection
SOP Service-Object Pair
TCP/IP Transmission Control Protocol/Internet Protocol
UID Unique Identifier
VR Value Representation
XML Extensible Markup Language
5 Conventions
Words are capitalized in this document to help the reader understand that these words have been
previously defined in Section 3 of this document and are to be interpreted with that meaning.
A Tag is represented as (gggg,eeee), where gggg equates to the Group Number and eeee equates to the
Element Number within that Group. Tags are represented in hexadecimal notation as specified in PS 3.5
of the DICOM Standard..
Attributes of File Meta Information are assigned a Type which indicates if a specific Attribute is required
depending on the Media Storage Services. The following Type designations are derived from the PS 3.5
designations but take into account the Media Storage environment:

