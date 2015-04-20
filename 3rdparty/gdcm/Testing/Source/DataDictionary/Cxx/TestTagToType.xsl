<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- XSL to convert XML GDCM2 data dictionay into
     C++ template instantiation code
-->
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
<!-- The main template that loop over all dict/entry -->
  <xsl:template match="/">
    <xsl:text>
// GENERATED FILE DO NOT EDIT
// $ xsltproc TestTagToType.xsl DICOMV3.xml &gt; TestTagToType.cxx

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "gdcmTagToType.h"

int TestTagToType(int, char *[])
{
</xsl:text>
    <xsl:for-each select="dict/entry">
      <xsl:variable name="group" select="translate(@group,'x','0')"/>
      <xsl:variable name="element" select="translate(@element,'x','0')"/>
      <xsl:if test="contains(@element,'x') = true and contains(@element,'xx') = false and @vr != '' and @vr != 'US_SS' and @vr != 'US_SS_OW' and @vr != 'OB_OW'">
        <xsl:text>  gdcm::TagToType&lt;0x</xsl:text>
        <xsl:value-of select="$group"/>
        <xsl:text>,0x</xsl:text>
        <xsl:value-of select="$element"/>
        <xsl:text>&gt; </xsl:text>
        <xsl:value-of select="concat('x',concat($group,$element))"/>
        <xsl:text>; (void)</xsl:text>
        <xsl:value-of select="concat('x',concat($group,$element))"/>
        <xsl:text>;
</xsl:text>
      </xsl:if>
    </xsl:for-each>
    <xsl:text>
  return 0;
}
</xsl:text>
  </xsl:template>
</xsl:stylesheet>
