<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- share common code -->
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
<!--life saver xsl script found at:
http://www.thescripts.com/forum/thread86881.html
-->
  <xsl:template name="upperCase">
    <xsl:param name="textToTransform"/>
    <xsl:variable name="head">
      <xsl:choose>
        <xsl:when test="contains($textToTransform, ' ')">
          <xsl:value-of select="substring-before($textToTransform, ' ')"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$textToTransform"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <xsl:variable name="tail" select="substring-after($textToTransform, ' ')"/>
    <xsl:variable name="firstTransform" select="concat(translate(substring($head, 1, 1), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), substring($head, 2))"/>
    <xsl:choose>
      <xsl:when test="$tail">
        <xsl:value-of select="$firstTransform"/>
        <xsl:call-template name="upperCase">
          <xsl:with-param name="textToTransform" select="$tail"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$firstTransform"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

</xsl:stylesheet>
