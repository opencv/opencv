################################################################################ 
#
#  Program: GDCM (Grassroots DICOM). A DICOM library
#
#  Copyright (c) 2006-2011 Mathieu Malaterre
#  All rights reserved.
#  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.
#
#     This software is distributed WITHOUT ANY WARRANTY; without even
#     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#     PURPOSE.  See the above copyright notice for more information.
#
#  File: ReadAndDumpDICOMDIR.py
#   
#  Author: Lukas Batteau (lbatteau gmail)
#
#  This example shows how to read and dump a DICOMDIR File.
#  Based on Tom Marynowski's (lordglub gmail) example.
#
#  Usage:
#  python ReadAndDumpDICOMDIR.py [DICOMDIR file]
############################################################################



import sys
import gdcm

if __name__ == "__main__":
    # Check arguments
    if (len(sys.argv) < 2):
        # No filename passed
        print "No input filename found"
        quit()
        
    filename = sys.argv[1]

        
    # Read file
    reader = gdcm.Reader()
    reader.SetFileName(filename)
    if (not reader.Read()):
        print "Unable to read %s" % (filename)
        quit()

    file = reader.GetFile()

    # Retrieve header information
    fileMetaInformation = file.GetHeader()
    print fileMetaInformation

    # Retrieve data set
    dataSet = file.GetDataSet()
    #print dataSet

    # Check media storage
    mediaStorage = gdcm.MediaStorage()
    mediaStorage.SetFromFile(file)
    if (gdcm.MediaStorage.GetMSType(str(mediaStorage)) != gdcm.MediaStorage.MediaStorageDirectoryStorage):
        # File is not a DICOMDIR
        print "This file is not a DICOMDIR (Media storage type: %s)" % (str(mediaStorage)) 
        quit()

    # Check Media Storage SOP Class
    if (fileMetaInformation.FindDataElement(gdcm.Tag(0x0002, 0x0002))):
        sopClassUid = str(fileMetaInformation.GetDataElement(gdcm.Tag(0x0002, 0x0002)).GetValue())
        # Check SOP UID
        if (sopClassUid != "1.2.840.10008.1.3.10"):
            # File is not a DICOMDIR
            print "This file is not a DICOMDIR"
    else:
        # Not present
        print "Media Storage SOP Class not present"
        quit()

    # Iterate through the DICOMDIR data set
    iterator = dataSet.GetDES().begin()
    while (not iterator.equal(dataSet.GetDES().end())):
        dataElement = iterator.next()

        # Check the element tag
        if (dataElement.GetTag() == gdcm.Tag(0x004, 0x1220)):
            # The 'Directory Record Sequence' element
            sequence = dataElement.GetValueAsSQ()

            # Loop through the sequence items
            itemNr = 1
            while (itemNr < sequence.GetNumberOfItems()):
                item = sequence.GetItem(itemNr)

                # Check the element tag
                if (item.FindDataElement(gdcm.Tag(0x0004, 0x1430))):
                    # The 'Directory Record Type' element
                    value = str(item.GetDataElement(gdcm.Tag(0x0004, 0x1430)).GetValue())

                    # PATIENT
                    while (value.strip() == "PATIENT"):
                        print value.strip()
                        # Print patient name
                        if (item.FindDataElement(gdcm.Tag(0x0010, 0x0010))):
                            value = str(item.GetDataElement(gdcm.Tag(0x0010, 0x0010)).GetValue())
                            print value

                        # Print patient ID
                        if (item.FindDataElement(gdcm.Tag(0x0010, 0x0020))):
                            value = str(item.GetDataElement(gdcm.Tag(0x0010, 0x0020)).GetValue())
                            print value

                        # Next
                        itemNr = itemNr + 1
                        item = sequence.GetItem(itemNr)
                        if (item.FindDataElement(gdcm.Tag(0x0004, 0x1430))):
                            value = str(item.GetDataElement(gdcm.Tag(0x0004, 0x1430)).GetValue())

                        # STUDY
                        while (value.strip() == "STUDY"):
                            print value.strip()

                            # Print study UID
                            if (item.FindDataElement(gdcm.Tag(0x0020, 0x000d))):
                                value = str(item.GetDataElement(gdcm.Tag(0x0020, 0x000d)).GetValue())
                                print value
                            
                            # Print study date
                            if (item.FindDataElement(gdcm.Tag(0x0008, 0x0020))):
                                value = str(item.GetDataElement(gdcm.Tag(0x0008, 0x0020)).GetValue())
                                print value
                            
                            # Print study description
                            if (item.FindDataElement(gdcm.Tag(0x0008, 0x1030))):
                                value = str(item.GetDataElement(gdcm.Tag(0x0008, 0x1030)).GetValue())
                                print value

                            # Next
                            itemNr = itemNr + 1
                            item = sequence.GetItem(itemNr)
                            if (item.FindDataElement(gdcm.Tag(0x0004, 0x1430))):
                                    value = str(item.GetDataElement(gdcm.Tag(0x0004, 0x1430)).GetValue())
                            
                            # SERIES
                            while (value.strip() == "SERIES"):
                                print value.strip()

                                # Print series UID
                                if (item.FindDataElement(gdcm.Tag(0x0020, 0x000e))):
                                    value = str(item.GetDataElement(gdcm.Tag(0x0020, 0x000e)).GetValue())
                                    print value
                                
                                # Print series modality
                                if (item.FindDataElement(gdcm.Tag(0x0008, 0x0060))):
                                    value = str(item.GetDataElement(gdcm.Tag(0x0008, 0x0060)).GetValue())
                                    print "Modality"
                                    print value

                                # Print series description
                                if (item.FindDataElement(gdcm.Tag(0x0008, 0x103e))):
                                    value = str(item.GetDataElement(gdcm.Tag(0x0008, 0x103e)).GetValue())
                                    print "Description"
                                    print value
                                
                                # Next
                                itemNr = itemNr + 1
                                item = sequence.GetItem(itemNr)
                                if (item.FindDataElement(gdcm.Tag(0x0004, 0x1430))):
                                    value = str(item.GetDataElement(gdcm.Tag(0x0004, 0x1430)).GetValue())
                            
                                # IMAGE
                                while (value.strip() == "IMAGE"):
                                    print value.strip()

                                    # Print image UID
                                    if (item.FindDataElement(gdcm.Tag(0x0004, 0x1511))):
                                        value = str(item.GetDataElement(gdcm.Tag(0x0004, 0x1511)).GetValue())
                                        print value
                                
                                    # Next
                                    if (itemNr < sequence.GetNumberOfItems()):
                                        itemNr = itemNr + 1
                                    else:
                                        break

                                    item = sequence.GetItem(itemNr)
                                    if (item.FindDataElement(gdcm.Tag(0x0004, 0x1430))):
                                            value = str(item.GetDataElement(gdcm.Tag(0x0004, 0x1430)).GetValue())

                # Next
                itemNr = itemNr + 1
