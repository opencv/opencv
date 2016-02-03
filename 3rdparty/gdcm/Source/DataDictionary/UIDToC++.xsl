<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- XSL to convert XML Part6.xml UIDs into C++ code -->
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
  <xsl:template match="/dicts">
    <xsl:text>
// GENERATED FILE DO NOT EDIT
// $ xsltproc UIDToC++.xsl Part6.xml &gt; gdcmUIDs.cxx

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
#ifndef GDCMUIDS_H
#define GDCMUIDS_H

  typedef enum {
</xsl:text>
    <xsl:for-each select="table/uid">
        <xsl:choose>
          <xsl:when test="../@name = 'UID VALUES'">
            <xsl:text>uid_</xsl:text>
          </xsl:when>
          <xsl:when test="../@name = 'Well-known Frames of Reference'">
            <xsl:text>frameref_</xsl:text>
          </xsl:when>
          <xsl:otherwise>
            <xsl:text>unhandled_</xsl:text>
          </xsl:otherwise>
        </xsl:choose>
        <xsl:value-of select="translate(@value,'.','_')"/>
        <xsl:text> = </xsl:text>
        <xsl:number value="position()" format="1" />
        <xsl:text>, // </xsl:text>
        <xsl:value-of select="@name"/>
<xsl:text>
</xsl:text>
    </xsl:for-each>
<xsl:text>
} TSType;
  typedef enum {
</xsl:text>
    <xsl:for-each select="table/uid">
        <!--xsl:choose>
          <xsl:when test="../@name = 'UID VALUES'">
            <xsl:text>uid_</xsl:text>
          </xsl:when>
          <xsl:when test="../@name = 'Well-known Frames of Reference'">
            <xsl:text>frameref_</xsl:text>
          </xsl:when>
          <xsl:otherwise>
            <xsl:text>unhandled_</xsl:text>
          </xsl:otherwise>
        </xsl:choose-->
        <xsl:if test="starts-with(@name,'1')">
          <xsl:text>//</xsl:text>
        </xsl:if>
        <xsl:value-of select="translate(@name,'&amp;@[]/(),-: ','')"/>
        <xsl:if test="@retired = 'true'">
          <xsl:text>Retired</xsl:text>
        </xsl:if>
        <xsl:text> = </xsl:text>
        <xsl:number value="position()" format="1" />
        <xsl:text>, // </xsl:text>
        <xsl:value-of select="@name"/>
<xsl:text>
</xsl:text>
    </xsl:for-each>
<xsl:text>} TSName;
#endif // GDCMUIDS_H
</xsl:text>
#ifdef GDCMUIDS_CXX
        <xsl:text>static const char * const TransferSyntaxStrings[][2] = {
</xsl:text>
    <xsl:for-each select="table/uid">
        <xsl:text>{"</xsl:text>
        <xsl:value-of select="@value"/>
        <xsl:text>","</xsl:text>
        <xsl:value-of select="@name"/>
        <xsl:text>"},
</xsl:text>
    </xsl:for-each>
        <xsl:text>{ 0, 0 }
};
#endif // GDCMUIDS_CXX
</xsl:text>
  </xsl:template>
</xsl:stylesheet>
