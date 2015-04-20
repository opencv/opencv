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
  <!--
cleanup excel HTML:
$ tidy -asxml dicomhdr.html > toto.html
Important specify it's html:

$ xsltproc - -html dicomhdr.xsl toto.html


Looking at the HTML page, I think the 'unit' tr was done by hand thus should be discard (full of typos)
-->
  <xsl:template match="/">
    <dict>
      <xsl:for-each select="html/body/table/tr">
	  <xsl:variable name="name" select="normalize-space(td[1])"/>
	  <xsl:variable name="tag" select="normalize-space(td[2])"/>
	  <xsl:variable name="type" select="normalize-space(td[3])"/>
	  <xsl:variable name="vr" select="normalize-space(td[4])"/>
	  <xsl:variable name="vm" select="normalize-space(td[5])"/>
	  <xsl:variable name="desc" select="normalize-space(td[6])"/>
	  <xsl:variable name="creator" select="normalize-space(td[7])"/>
	  <xsl:variable name="unit" select="normalize-space(td[8])"/>
		  <!-- tag="{ $tag }" -->
	  <entry name="{ $name }"
		  type="{ $type }" vr="{ $vr }" vm="{ $vm }" description="{$desc}">
          <!--xsl:for-each select=".">
	  <xsl:variable name="vm" select="normalize-space(td[1])"/>
            <value>
              <xsl:value-of select="$vm"/>
            </value>
          </xsl:for-each-->
        </entry>
      </xsl:for-each>
    </dict>
  </xsl:template>
</xsl:stylesheet>
