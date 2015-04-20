<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- XSL to convert XML GDCM2 data dictionay into the old
     GDCM1 data dictionary -->
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
    <xsl:for-each select="dict/entry">
      <xsl:sort select="@group"/>
      <xsl:sort select="@element"/>
      <xsl:if test="@element = '0000'">
      <xsl:text>
</xsl:text>
</xsl:if>
      <xsl:value-of select="translate(@group,'x','0')"/>
      <xsl:text> </xsl:text>
      <xsl:value-of select="translate(@element,'x','0')"/>
      <xsl:text> </xsl:text>
      <xsl:value-of select="@vr"/>
      <xsl:text> </xsl:text>
      <xsl:value-of select="@vm"/>
      <xsl:text> </xsl:text>
      <xsl:value-of select="@name"/>
      <!--<xsl:if test="@retired = &quot;true&quot;">-->
      <xsl:if test="@retired != &quot;false&quot;">
        <xsl:text> (RET)</xsl:text>
      </xsl:if>
      <xsl:text>
</xsl:text>
    </xsl:for-each>
  </xsl:template>
</xsl:stylesheet>
