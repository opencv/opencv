<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="xml" indent="yes" encoding="UTF-8"/>
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
<!-- how to run:
$ xsltproc ../trunk/Source/InformationObjectDefinition/Part4.xsl ./standard/2008/08_04pu.xml
-->
<xsl:template match="text()" />
  <xsl:template match="informaltable">
	  <!--xsl:if test="tgroup/tbody/row/entry/para = 'SOP Class Name'"-->
	  <!--xsl:if test="tgroup/tbody/row/entry/para = 'SOP Class UID'"-->
	  <xsl:if test="tgroup/tbody/row/entry/para = 'Related General SOP Class Name'">
    <standard-and-related-general-sop-classes>
      <xsl:apply-templates/>
    </standard-and-related-general-sop-classes>
	  </xsl:if>
	  <xsl:if test="tgroup/tbody/row/entry/para = 'IOD Specification'">
    <media-storage-standard-sop-classes>
      <xsl:apply-templates/>
    </media-storage-standard-sop-classes>
	  </xsl:if>
	  <xsl:if test="tgroup/tbody/row/entry/para = 'IOD Specification(defined in PS 3.3)'">
    <standard-sop-classes>
      <xsl:apply-templates/>
    </standard-sop-classes>
	  </xsl:if>
  </xsl:template>

  <xsl:template match="para">
    <!--xsl:apply-templates/-->
  </xsl:template>

  <xsl:template match="tgroup">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="tbody">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="article">
    <xsl:variable name="section-number" select="'Table B.5-1STANDARD SOP CLASSES'"/>
    <xsl:variable name="section-anchor" select="para[starts-with(normalize-space(.),$section-number)]"/>
    <xsl:message><xsl:value-of select="$section-anchor"/></xsl:message>
	    <!--xsl:apply-templates select="article/sect1/sect2/informaltable"/-->
	    <!--xsl:apply-templates select="informaltable"/-->
	    <!--xsl:apply-templates select="sect1/sect2/informaltable"/-->
	    <!--xsl:apply-templates select="article/informaltable"/-->
    <sop-classes>
	    <xsl:apply-templates select="informaltable"/>
    </sop-classes>
  </xsl:template>

  <xsl:template match="row">
    <!--xsl:apply-templates/-->
    <xsl:variable name="classname" select="entry[1]/para"/>
    <xsl:variable name="classuid" select="entry[2]/para"/>
    <xsl:variable name="iod" select="entry[3]/para"/>
    <xsl:if test="$classname != 'SOP Class Name'">
    <mapping sop-class-name="{$classname}" sop-class-uid="{normalize-space($classuid)}" iod="{$iod}" />
    </xsl:if>
  </xsl:template>

  <!--
  <xsl:template match="entry">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="para">
    <xsl:value-of select="."/>
  </xsl:template>
  -->

  <xsl:template match="/">
    <xsl:comment>
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
</xsl:comment>
     <xsl:apply-templates select="article"/>
 </xsl:template>
</xsl:stylesheet>
