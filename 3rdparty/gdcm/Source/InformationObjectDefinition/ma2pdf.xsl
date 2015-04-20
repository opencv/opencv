<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="html" indent="yes"/>
<!-- XSL to convert XML GDCM2 data dictionay into HTML form -->
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
            PS 3.3-2007<fo:block><xsl:text>
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
            <xsl:for-each select="tables/table">
<!-- fop does not support table and caption -->
<!--fo:table-and-caption>
            <fo:table-caption>
              <fo:block>Caption for this table</fo:block>
            </fo:table-caption-->
              <fo:block>
                <fo:marker marker-class-name="cont">
                  <fo:block/>
                </fo:marker>
                <xsl:value-of select="@ref"/>
                <xsl:text> </xsl:text>
                <xsl:value-of select="@name"/>
              </fo:block>
              <fo:table table-layout="fixed" border-color="rgb(0,0,0)" border-width="1pt">
                <fo:table-column column-width="50mm"/> <!--Name-->
                <fo:table-column column-width="20mm"/> <!--Tag-->
                <fo:table-column column-width="10mm"/> <!--Type-->
                <fo:table-column column-width="90mm"/> <!--Description-->
                <fo:table-header background-color="rgb(214,214,214)" font-size="10pt" font-weight="bold" text-align="justify">
                  <fo:table-row text-align="justify" font-size="10pt">
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block>Name</fo:block>
                    </fo:table-cell>
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block>Tag</fo:block>
                    </fo:table-cell>
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block>Type</fo:block>
                    </fo:table-cell>
                    <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                      <fo:block>Description</fo:block>
                    </fo:table-cell>
                  </fo:table-row>
                </fo:table-header>
                <fo:table-body>
                  <xsl:for-each select="entry">
                    <fo:table-row text-align="left" font-size="10pt">
                      <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                        <fo:block vertical-align="middle">
                          <xsl:value-of select="@name"/>
                        </fo:block>
                      </fo:table-cell>
                      <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                        <fo:block vertical-align="middle">
                          <xsl:text>(</xsl:text>
                          <xsl:value-of select="@group"/>
                          <xsl:text>,</xsl:text>
                          <xsl:value-of select="@element"/>
                          <xsl:text>)</xsl:text>
                        </fo:block>
                      </fo:table-cell>
                      <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                        <fo:block vertical-align="middle">
                          <xsl:value-of select="@type"/>
                        </fo:block>
                      </fo:table-cell>
                      <fo:table-cell border-color="rgb(0,0,0)" border-width="1pt">
                        <fo:block vertical-align="middle">
                          <xsl:value-of select="description"/>
                        </fo:block>
                      </fo:table-cell>
                    </fo:table-row>
                  </xsl:for-each>
                </fo:table-body>
              </fo:table>
<!--/fo:table-and-caption-->
<!-- Create a new page -->
              <fo:block break-after="page">
                <xsl:text>
</xsl:text>
              </fo:block>
            </xsl:for-each>
          </fo:block>
        </fo:flow>
      </fo:page-sequence>
    </fo:root>
  </xsl:template>
</xsl:stylesheet>
