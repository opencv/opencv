<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >

<xsl:output method="text" indent="no" encoding="ISO-8859-1"/>
<!--xsl:include href="uppercase.xsl"/-->
<xsl:strip-space elements="*"/>

<!-- MAIN -->

<xsl:template match="/dicts">
<xsl:text>#
#  Copyright (C) 1994-2012, OFFIS e.V.
#  All rights reserved.  See COPYRIGHT file for details.
#
#  This software and supporting documentation were developed by
#
#    OFFIS e.V.
#    R&amp;D Division Health
#    Escherweg 2
#    D-26121 Oldenburg, Germany
#
#
#  Module:  dcmdata
#
#  Author:  Andrew Hewett, Marco Eichelberg, Joerg Riesmeier
#
#  Purpose: This is the global standard DICOM data dictionary for the DCMTK.
#
# This file contains the complete data dictionary from the 2011 edition of the
# DICOM standard.  This also includes the non-private definitions from the
# DICONDE (Digital Imaging and Communication in Nondestructive Evaluation) and
# DICOS (Digital Imaging and Communications in Security) standard.
#
# In addition, the data dictionary entries from the following final text
# supplements and correction items have been incorporated:
# - Supplement 152.
# - CP 1064, 1123, 1137, 1138, 1147, 1188, 1204.
#
# Each line represents an entry in the data dictionary.  Each line has 5 fields
# (Tag, VR, Name, VM, Version).  Entries need not be in ascending tag order.
#
# Entries may override existing entries.
#
# Each field must be separated by a single tab.  The tag values (gggg,eeee)
# must be in hexedecimal and must be surrounded by parentheses.  Repeating
# groups are represented by indicating the range (gggg-gggg,eeee).  By default
# the repeating notation only represents even numbers.  A range where only
# odd numbers are valid is represented using the notation (gggg-o-gggg,eeee).
# A range can represent both even and odd numbers using the notation
# (gggg-u-gggg,eeee).  The element part of the tag can also be a range.
#
# Comments have a '#' at the beginning of the line.
#
# Tag		VR	Name			VM	Version
#
</xsl:text>

  <xsl:for-each select="dict/entry">
    <xsl:sort select="@group"/>
    <xsl:sort select="@element"/>
    <xsl:variable name="group_upper">
      <xsl:value-of select="translate(@group,'abcdef','ABCDEF')"/>
    </xsl:variable>
    <xsl:variable name="element_upper">
      <xsl:value-of select="translate(@element,'abcdef','ABCDEF')"/>
    </xsl:variable>

      <xsl:if test="not(@retired)">
    <xsl:variable name="attribute-name" select="@keyword"/>

    <!-- output tag -->
    <xsl:choose>
      <!-- repeating group -->
      <xsl:when test="contains(@group,'xx')">
        <xsl:value-of select="concat('(',substring(@group,1,2),'00-',substring(@group,1,2),'FF,',$element_upper,')&#9;')"/>
      </xsl:when>
      <xsl:when test="contains(@element,'xx')">
        <xsl:value-of select="concat('(',$group_upper,',',substring($element_upper,1,2),'00-',substring($element_upper,1,2),'FF)&#9;')"/>
      </xsl:when>
      <!-- standard case -->
      <xsl:otherwise>
        <xsl:value-of select="concat('(',$group_upper,',',$element_upper,')&#9;')"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output VR -->
    <xsl:choose>
      <!-- offset attribute for DICOMDIR -->
      <xsl:when test="@group='0004' and contains($attribute-name,'Offset')">
        <xsl:value-of select="'up&#9;'"/>
      </xsl:when>
      <!-- multiple value representations -->
      <xsl:when test="@vr='OB_OW'">
        <xsl:value-of select="'ox&#9;'"/>
      </xsl:when>
      <xsl:when test="@vr='US_SS'">
        <xsl:value-of select="'xs&#9;'"/>
      </xsl:when>
      <xsl:when test="@vr='US_SS_OW'">
        <xsl:value-of select="'lt&#9;'"/>
      </xsl:when>
      <!-- standard case -->
      <xsl:when test="string(@vr)">
        <xsl:value-of select="concat(@vr,'&#9;')"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'na&#9;'"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output name -->
    <xsl:choose>
      <xsl:when test="@version='2'">
        <xsl:value-of select="concat('ACR_NEMA_',$attribute-name,'&#9;')"/>
      </xsl:when>
      <xsl:when test="@retired='true'">
        <xsl:value-of select="concat('RETIRED_',$attribute-name,'&#9;')"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="concat($attribute-name,'&#9;')"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output VM -->
    <xsl:choose>
      <xsl:when test="string(@vm)">
        <xsl:value-of select="concat(@vm,'&#9;')"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'1&#9;'"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output version -->
    <xsl:choose>
      <xsl:when test="@version='2'">
        <xsl:value-of select="'ACR/NEMA2'"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'DICOM_2011'"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output newline -->
    <xsl:value-of select="'&#10;'"/>
      </xsl:if>
  </xsl:for-each>

  <!-- iterate over all entries -->
<xsl:text>
#
#---------------------------------------------------------------------------
#
# Private Creator Data Elements
#
(0009-o-ffff,0000)	UL	PrivateGroupLength	1	PRIVATE
(0009-o-ffff,0010-u-00ff)	LO	PrivateCreator	1	PRIVATE
(0001-o-0007,0000)	UL	IllegalGroupLength	1	ILLEGAL
(0001-o-0007,0010-u-00ff)	LO	IllegalPrivateCreator	1	ILLEGAL
#
#---------------------------------------------------------------------------
#
# A "catch all" for group length elements
#
(0000-u-ffff,0000)	UL	GenericGroupLength	1	GENERIC
#
#---------------------------------------------------------------------------
#
# Retired data elements from ACR/NEMA 2 (1988)
#
</xsl:text>

  <xsl:for-each select="dict/entry">
    <xsl:sort select="@group"/>
    <xsl:sort select="@element"/>

    <xsl:variable name="group_upper">
      <xsl:value-of select="translate(@group,'abcdef','ABCDEF')"/>
    </xsl:variable>
    <xsl:variable name="element_upper">
      <xsl:value-of select="translate(@element,'abcdef','ABCDEF')"/>
    </xsl:variable>

      <xsl:if test="@retired">
    <xsl:variable name="attribute-name" select="@keyword"/>
    <!-- output tag -->
    <xsl:choose>
      <!-- repeating group -->
      <xsl:when test="contains(@group,'xx')">
        <xsl:value-of select="concat('(',substring($group_upper,1,2),'00-',substring($group_upper,1,2),'FF,',$element_upper,')&#9;')"/>
      </xsl:when>
      <xsl:when test="contains(@element,'xx')">
        <xsl:value-of select="concat('(',$group_upper,',',substring(@element,1,2),'00-',substring($element_upper,1,2),'FF)&#9;')"/>
      </xsl:when>
      <!-- standard case -->
      <xsl:otherwise>
        <xsl:value-of select="concat('(',$group_upper,',',$element_upper,')&#9;')"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output VR -->
    <xsl:choose>
      <!-- offset attribute for DICOMDIR -->
      <xsl:when test="@group='0004' and contains($attribute-name,'Offset')">
        <xsl:value-of select="'up&#9;'"/>
      </xsl:when>
      <!-- multiple value representations -->
      <xsl:when test="@vr='OB_OW'">
        <xsl:value-of select="'ox&#9;'"/>
      </xsl:when>
      <xsl:when test="@vr='US_SS'">
        <xsl:value-of select="'xs&#9;'"/>
      </xsl:when>
      <xsl:when test="@vr='US_SS_OW'">
        <xsl:value-of select="'lt&#9;'"/>
      </xsl:when>
      <!-- standard case -->
      <xsl:when test="string(@vr)">
        <xsl:value-of select="concat(@vr,'&#9;')"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'na&#9;'"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output name -->
    <xsl:choose>
      <xsl:when test="@version='2'">
        <xsl:value-of select="concat('ACR_NEMA_',$attribute-name,'&#9;')"/>
      </xsl:when>
      <xsl:when test="@retired='true'">
        <xsl:value-of select="concat('ACR_NEMA_',$attribute-name,'&#9;')"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="concat($attribute-name,'&#9;')"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output VM -->
    <xsl:choose>
      <xsl:when test="string(@vm)">
        <xsl:value-of select="concat(@vm,'&#9;')"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'1&#9;'"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output version -->
    <xsl:choose>
      <xsl:when test="@retired='true'">
        <xsl:value-of select="'ACR/NEMA2'"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="'DICOM_2011'"/>
      </xsl:otherwise>
    </xsl:choose>
    <!-- output newline -->
    <xsl:value-of select="'&#10;'"/>
  </xsl:if>
  </xsl:for-each>
</xsl:template>

</xsl:stylesheet>
