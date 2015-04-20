<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="xml" indent="yes"/>
<!-- XSL to convert Part7 into GDCM2 xml dict -->
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
   <entry group="{@group}"
  element="{@element}"
  vr="{@vr}"
  vm="{@vm}"
  retired="{@retired}"
  version="{@version}"
  >
  <description><xsl:value-of select="description"/></description>
</entry>
</xsl:template>

<xsl:template match="/">
    <xsl:processing-instruction name="xml-stylesheet">
type="text/xsl" href="gdcm2html.xsl"
</xsl:processing-instruction>
    <xsl:comment> to produce output use:
$ xsltproc gdcm2html.xsl GDCM2.xml
    </xsl:comment>
    <xsl:comment>
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
</xsl:comment>

<dict edition="2007">
    <xsl:for-each select="dicts/dict/entry">
      <xsl:sort select="@group"/>
      <xsl:sort select="@element"/>
  <xsl:apply-templates select="."/>
    </xsl:for-each>
</dict>
</xsl:template>

<xsl:template name="get-vr">
  <xsl:param name="representations"/>
    <xsl:choose>
      <xsl:when test="representations/representation[1]/@vr='US' and representations/representation[2]/@vr='SS' and representations/representation[3]/@vr='OW'">
        <xsl:value-of select="'US_SS_OW'"/>
      </xsl:when>
      <xsl:when test="representations/representation[1]/@vr='US' and representations/representation[2]/@vr='SS'">
        <xsl:value-of select="'US_SS'"/>
      </xsl:when>
      <xsl:when test="representations/representation[1]/@vr='OB' and representations/representation[2]/@vr='OW'">
        <xsl:value-of select="'OB_OW'"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="representations/representation[1]/@vr"/>
      </xsl:otherwise>
    </xsl:choose>
</xsl:template>

<xsl:template name="get-vm">
  <xsl:param name="representations"/>
        <xsl:value-of select="representations/representation[1]/@vm"/>
</xsl:template>


</xsl:stylesheet>
