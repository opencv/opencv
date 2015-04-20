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
<!-- The main template that loop over all dict/entry -->
  <xsl:template match="row">
  <entry group="{substring-before(substring-after(entry[2],'('),',')}"
  element="{substring-after(substring-before(entry[2],')'),',')}"
  vr="{entry[3]}"
  vm="{entry[4]}"
  retired="true"
  version="3"
  >
  <description><xsl:value-of select="entry[1]"/></description>
  </entry>
  </xsl:template>

  <xsl:template match="/tables/informaltable/tgroup/tbody">
    <xsl:apply-templates/>
  </xsl:template>
</xsl:stylesheet>
