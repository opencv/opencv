<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- XSL to convert XML GDCM2 data dictionay into
     C++ template code
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
  <xsl:include href="VM.xsl"/>
<!-- The main template that loop over all dict/entry -->
  <xsl:template match="/">
    <xsl:text>
// GENERATED FILE DO NOT EDIT
// $ xsltproc TagToVR.xsl Part6.xml &gt; gdcmTagToVR.cxx

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmVR.h"
#include "gdcmVM.h"
#include "gdcmStaticAssert.h"

namespace gdcm {
VR::VRType GetVRFromTag( Tag const &amp; t ) {
if( t.IsGroupLength() ) return VR::UL;
uint32_t tag = t.GetElementTag();
switch( tag ) {
</xsl:text>
    <xsl:for-each select="dicts/dict/entry">
      <xsl:sort select="@group" data-type="text" order="ascending"/>
      <xsl:sort select="@element" data-type="text" order="ascending"/>
      <xsl:variable name="group" select="translate(@group,'x','0')"/>
      <xsl:variable name="element" select="translate(@element,'x','0')"/>
      <xsl:if test="contains(@element,'x') = true and contains(@element,'xx') = false and @vr != '' and @vr != 'US_SS_OW' and @vr != 'OB_OW'">
<xsl:variable name="classname">
        <xsl:text>case 0x</xsl:text>
        <xsl:value-of select="$group"/>
        <xsl:value-of select="$element"/>
        <xsl:text>:</xsl:text>
</xsl:variable>
        <xsl:value-of select="$classname"/>
        <xsl:text>return VR::</xsl:text>
        <xsl:value-of select="@vr"/>
        <xsl:text>;</xsl:text>
        <xsl:text>
</xsl:text>
      </xsl:if>
    </xsl:for-each>
default:
return VR::INVALID;
}
}
    <xsl:text>
} // end namespace gdcm
</xsl:text>
  </xsl:template>
</xsl:stylesheet>
