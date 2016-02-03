<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:fo="http://www.w3.org/1999/XSL/Format" xmlns:java="http://xml.apache.org/xslt/java" version="1.1" exclude-result-prefixes="java">
  <xsl:output method="pdf"/>
<!-- fop -xml Part6.xml -xsl Part6PDF.xsl Part6.pdf -->
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
  <xsl:template match="/">
    <fo:root xmlns:fo="http://www.w3.org/1999/XSL/Format" xmlns:fox="http://xml.apache.org/fop/extensions">
      <fo:layout-master-set>
        <fo:simple-page-master master-name="A4-L" page-height="297mm" page-width="210mm" margin-top="10mm" margin-bottom="10mm" margin-left="10mm" margin-right="10mm">
<!--
    * <fo:region-body> defines the body region
    * <fo:region-before> defines the top region (header)
    * <fo:region-after> defines the bottom region (footer)
    * <fo:region-start> defines the left region (left sidebar)
    * <fo:region-end> defines the right region (right sidebar)
-->
          <fo:region-body margin="10mm"/>
          <fo:region-before extent="10mm"/>
          <fo:region-after extent="10mm"/>
          <fo:region-start extent="10mm"/>
          <fo:region-end extent="10mm"/>
        </fo:simple-page-master>
      </fo:layout-master-set>
      <fo:page-sequence master-reference="A4-L">
        <fo:static-content flow-name="xsl-region-before" font-size="10pt">
          <fo:block text-align="justify">
            PS 3.6-2007<fo:block><xsl:text>
</xsl:text></fo:block>Page <fo:page-number/>
          </fo:block>
        </fo:static-content>
        <fo:static-content flow-name="xsl-region-after" font-size="10pt">
          <fo:block text-align="center">
          - Standard -
          </fo:block>
        </fo:static-content>
        <fo:flow flow-name="xsl-region-body">
          <fo:block>
                <xsl:for-each select="dicts/dict">
            <fo:table table-layout="fixed" border-color="rgb(0,0,0)" border-width="1pt">
              <fo:table-column column-width="30mm"/><!--Tag-->
              <fo:table-column column-width="90mm"/><!--Name-->
              <fo:table-column column-width="7.5mm"/><!--VR-->
              <fo:table-column column-width="10mm"/><!--VM-->
              <fo:table-column column-width="15mm"/><!--Retired?-->
              <fo:table-header background-color="rgb(214,214,214)" font-size="10pt" font-weight="bold" text-align="justify">
                <fo:table-row text-align="justify" font-size="10pt">
                  <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                    <fo:block>Tag</fo:block>
                  </fo:table-cell>
                  <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                    <fo:block>Name</fo:block>
                  </fo:table-cell>
                  <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                    <fo:block>VR</fo:block>
                  </fo:table-cell>
                  <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                    <fo:block>VM</fo:block>
                  </fo:table-cell>
                  <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                    <fo:block>Retired</fo:block>
                  </fo:table-cell>
                </fo:table-row>
              </fo:table-header>
              <fo:table-body>
<!-- http://www.topxml.com/xsl/articles/caseconvert/ -->
                <xsl:variable name="lcletters">abcdefghijklmnopqrstuvwxyz</xsl:variable>
                <xsl:variable name="ucletters">ABCDEFGHIJKLMNOPQRSTUVWXYZ</xsl:variable>
                <xsl:for-each select="entry">
                  <xsl:variable name="my_font_style" select="italic"/>
<!--xsl:if test="@retired != 'false'">italic</xsl:if>
                    </xsl:variable-->
<!--fo:table-row text-align="center" font-size="10pt" font-style="{$my_font_style}"-->
                  <fo:table-row text-align="justify" font-size="10pt">
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block vertical-align="middle">
                        <xsl:text>(</xsl:text>
                        <xsl:value-of select="translate(@group,$lcletters,$ucletters)"/>
                        <xsl:text>,</xsl:text>
                        <xsl:value-of select="translate(@element,$lcletters,$ucletters)"/>
                        <xsl:text>)</xsl:text>
                      </fo:block>
                    </fo:table-cell>
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block vertical-align="middle">
                        <xsl:value-of select="description"/>
                      </fo:block>
                    </fo:table-cell>
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block vertical-align="middle">
                          <xsl:value-of select="@vr"/>
<!-- add a new line: -->
                      </fo:block>
                    </fo:table-cell>
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block vertical-align="middle">
                          <xsl:value-of select="@vm"/>
                      </fo:block>
                    </fo:table-cell>
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block vertical-align="middle">
                        <xsl:if test="@retired != 'false'">
                          <xsl:text>RET</xsl:text>
                        </xsl:if>
                      </fo:block>
                    </fo:table-cell>
                  </fo:table-row>
                </xsl:for-each>
              </fo:table-body>
            </fo:table>
                </xsl:for-each>
          </fo:block>
        </fo:flow>
      </fo:page-sequence>
    </fo:root>
  </xsl:template>
</xsl:stylesheet>
