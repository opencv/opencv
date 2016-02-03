<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- XSL to convert XML Part4.xml UIDs into C++ code -->
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
  <xsl:template match="/sop-classes">
    <xsl:text>
// GENERATED FILE DO NOT EDIT
// $ xsltproc Part4ToC++.xsl Part4.xml &gt; gdcmSOPClassUIDToIOD.cxx

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
</xsl:text>
<xsl:text>
#include "gdcmSOPClassUIDToIOD.h"

namespace gdcm
{
  static const char * const SOPClassToIOD[][2] = {
</xsl:text>
    <xsl:for-each select="media-storage-standard-sop-classes/mapping">
        <xsl:text>{"</xsl:text>
        <xsl:value-of select="@sop-class-uid"/>
        <xsl:text>" , "</xsl:text>
        <xsl:value-of select="@iod"/>
        <xsl:text>"},</xsl:text>
<xsl:text>
</xsl:text>
    </xsl:for-each>
<xsl:text>
{ 0, 0 }
};

} // end namespace gdcm
</xsl:text>
  </xsl:template>
</xsl:stylesheet>
