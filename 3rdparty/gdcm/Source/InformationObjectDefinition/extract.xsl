<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="xml" indent="yes"/>
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
  <xsl:template match="entry">
    <xsl:variable name="name2" select="translate(@name,'&gt;','')"/>
    <entry group="{@group}" element="{@element}" name="{$name2}" /> <!-- type="{@type}"/>-->
  </xsl:template>

  <xsl:template match="iod">
  </xsl:template>

  <xsl:template match="module">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="macro">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="/">
    <dict>
    <xsl:apply-templates/>
    </dict>
  </xsl:template>
</xsl:stylesheet>
