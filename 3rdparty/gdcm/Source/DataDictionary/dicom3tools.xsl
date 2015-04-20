<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- XSL to convert XML GDCM2 data dictionay into
     David Clunie's dicom3tools data dictionary
Checked against:
     dicom3tools_1.00.snapshot.20120808/libsrc/standard/elmdict/dicom3.tpl
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
<xsl:include href="uppercase.xsl"/>
<!-- The main template that loop over all dict/entry -->
  <xsl:template match="/">
    <xsl:for-each select="dicts/dict/entry">
    <xsl:sort select="@group"/>
    <xsl:sort select="@element"/>
      <xsl:text>(</xsl:text>
      <xsl:value-of select="translate(@group,'abcdef','ABCDEF')"/>
      <xsl:text>,</xsl:text>
      <xsl:choose>
        <xsl:when test="starts-with(@element,'xx') and @element != 'xxxx'">
      <xsl:value-of select="'00'"/>
      <xsl:value-of select="translate(substring(@element,3,2),'abcdef','ABCDEF')"/>
        </xsl:when>
      <xsl:otherwise>
      <xsl:value-of select="translate(@element,'abcdef','ABCDEF')"/>
      </xsl:otherwise>
      </xsl:choose>
      <xsl:text>) VERS="</xsl:text>
      <xsl:choose>
        <xsl:when test="@retired = 'true'">
        </xsl:when>
        <xsl:otherwise>
          <xsl:text>3</xsl:text>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:if test="@retired != 'false'">
        <xsl:text>RET</xsl:text>
      </xsl:if>
      <xsl:text>" VR="</xsl:text>
      <xsl:choose>
        <xsl:when test="not(@vr)">
        <xsl:value-of select="'NONE'"/>
        </xsl:when>
        <xsl:when test="@vr = 'OB_OW'">
        <xsl:value-of select="'OW/OB'"/>
      </xsl:when>
        <xsl:when test="@vr = 'US_SS_OW'">
        <xsl:value-of select="'US or OW'"/>
      </xsl:when>
        <xsl:when test="@vr = 'US_SS'">
        <xsl:value-of select="'US or SS'"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="@vr"/>
      </xsl:otherwise>
      </xsl:choose>
      <xsl:text>" VM="</xsl:text>
        <xsl:value-of select="@vm"/>
      <xsl:text>" Keyword="</xsl:text>
      <xsl:variable name="apos">'</xsl:variable>
      <!--translating an apostrophe is a pain ... better solution ? -->
      <xsl:variable name="description_apos">
        <xsl:value-of select="translate(@name, $apos, '')"/>
      </xsl:variable>
      <xsl:variable name="description_dash">
        <!-- the dicom3tools is not always consistent with capitalization.
             Assume that every time there is a - we want capitalization -->
        <xsl:value-of select="translate($description_apos,'-',' ')"/>
      </xsl:variable>
      <xsl:variable name="description_cap">
        <xsl:call-template name="upperCase">
          <xsl:with-param name="textToTransform" select="normalize-space($description_dash)"/>
        </xsl:call-template>
      </xsl:variable>
      <!-- remove remaining extra character -->
      <xsl:value-of select="translate(@keyword,'/(),','')"/>
      <xsl:text>" Name="</xsl:text>
      <xsl:value-of select="@name"/>
      <xsl:text>"</xsl:text>
      <xsl:text>
</xsl:text>
    </xsl:for-each>
  </xsl:template>
</xsl:stylesheet>
