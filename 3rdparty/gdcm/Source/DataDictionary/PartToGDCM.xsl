<?xml version="1.0" encoding="UTF-8"?>
<!--
  Program: GDCM (Grass Root DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="xml" indent="yes" encoding="UTF-8"/>
<!-- dict / dicts -->
  <xsl:template match="*|@*">
    <xsl:copy>
      <xsl:apply-templates select="@*|node()"/>
    </xsl:copy>
  </xsl:template>
<!--xsl:template match="text()|comment()|processing-instruction()"><xsl:copy/></xsl:template-->
<!--xsl:template match="document()"><xsl:copy/></xsl:template-->
  <xsl:template match="table">
  </xsl:template>
  <xsl:template match="dict">
    <xsl:apply-templates/>
  </xsl:template>
  <xsl:template match="/">
    <xsl:comment>
This file was generated using the following commands:

  $ xsltproc PartToGDCM.xsl Part6.xml &gt; tmp.xml
  $ xsltproc order.xsl tmp.xml &gt; DICOMV3.xml
    </xsl:comment>
    <xsl:comment>
  Program: GDCM (Grass Root DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
</xsl:comment>
    <xsl:element name="dict">
      <xsl:attribute name="edition">2009</xsl:attribute>
      <xsl:apply-templates/>
    </xsl:element>
  </xsl:template>
  <xsl:template match="dicts">
    <xsl:apply-templates/>
  </xsl:template>
  <xsl:template match="entry">
<!--xsl:sort select="@group"/-->
    <xsl:element name="entry">
      <xsl:copy-of select="@*"/>
    </xsl:element>
  </xsl:template>
</xsl:stylesheet>
