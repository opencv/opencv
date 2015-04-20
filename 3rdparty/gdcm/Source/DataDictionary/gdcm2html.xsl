<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="html"
  media-type="text/html" encoding="UTF-8"
indent="no" omit-xml-declaration="yes" doctype-public="-//W3C//DTD HTML 4.01//EN" doctype-system="http://www.w3.org/TR/html4/strict.dtd"/>
<!--xsl:stylesheet xmlns:xsl=
"http://www.w3.org/1999/XSL/Transform"
xmlns=
"http://www.w3.org/TR/xhtml1/strict"
version="1.0">
<xsl:output method="xml" indent="yes"
encoding="iso-8859-1"/-->

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
    <xsl:variable name="has_owner" select="dict/entry/@owner"/>
    <xsl:variable name="has_retired" select="dict/entry/@retired"/>
    <html>
    <head>
    <title>
    DICOM DICTIONARY
    </title>
      <style type="text/css">
tr.normal
   {
   font-style:normal;
   }
tr.italic
   {
   font-style:italic;
   }
</style>

    </head>
      <body>
        <table border="1">
          <tr bgcolor="#d6d6d6">
<!--rgb(214,214,214) -->
            <th>Tag</th>
            <th>VR</th>
            <th>VM</th>
            <th>Description</th>
            <th>Version</th>
            <xsl:choose>
              <xsl:when test="$has_owner">
                <th>Owner</th>
              </xsl:when>
              <xsl:when test="$has_retired">
                <th>Retired</th>
              </xsl:when>
              <xsl:otherwise>
                <th>bla</th>
              </xsl:otherwise>
            </xsl:choose>
          </tr>
<!-- The main template that loop over all dict/entry -->
          <xsl:for-each select="dict/entry">
            <xsl:variable name="my_italic" value="@retired != 'false'"/>
            <xsl:variable name="my_class">
              <xsl:choose>
<!--xsl:when test="$has_owner">
              <xsl:text>normal</xsl:text>
              </xsl:when-->
<!--xsl:when test="$has_retired"-->
                <xsl:when test="@retired !='false'">
              italic
              </xsl:when>
                <xsl:otherwise>
              normal
              </xsl:otherwise>
              </xsl:choose>
            </xsl:variable>
            <tr class="{$my_class}">
              <td>
<!--xsl:if test="@retired != 'false'"><i></xsl:if-->
                <xsl:text>(</xsl:text>
                <xsl:value-of select="@group"/>
                <xsl:text>,</xsl:text>
                <xsl:value-of select="@element"/>
                <xsl:text>)</xsl:text>
<!--xsl:if test="$my_italic"></i></xsl:if-->
              </td>
              <td>
                <xsl:value-of select="@vr"/>
              </td>
              <td>
                <xsl:value-of select="@vm"/>
              </td>
              <td>
                <xsl:value-of select="description"/>
              </td>
              <td>
                <xsl:value-of select="@version"/>
              </td>
              <td>
                <xsl:choose>
                  <xsl:when test="$has_owner">
                    <xsl:value-of select="@owner"/>
                  </xsl:when>
                  <xsl:when test="$has_retired">
                    <xsl:if test="@retired != 'false'">
                      <xsl:text> (RET)</xsl:text>
                    </xsl:if>
                  </xsl:when>
                </xsl:choose>
              </td>
            </tr>
          </xsl:for-each>
        </table>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
