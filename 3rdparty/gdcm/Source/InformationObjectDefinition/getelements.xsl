<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
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
  <!-- TODO: Need to distinguish IODs, modules and macros -->

  <!-- an Entry line -->
  <xsl:template match="entry" mode="iod">
    <todo/>
  </xsl:template>

  <xsl:template match="entry" mode="module">
    <tr>
      <td>(<xsl:value-of select="@group"/>,<xsl:value-of select="@element"/>)</td>
      <td>
        <xsl:value-of select="@name"/>
      </td>
      <td>
        <xsl:value-of select="@type"/>
      </td>
      <td>
        <xsl:value-of select="description"/>
      </td>
    </tr>
  </xsl:template>


  <!-- an Include line -->
  <xsl:template match="include" mode="module">
    <tr>
      <td colspan="4"> <!-- FIXME hardcoded value -->
        <xsl:value-of select="@ref"/>
      </td>
    </tr>
  </xsl:template>


  <xsl:template match="/">
    <html>
      <body>
<!-- The main template that loop over all dict/entry -->
        <xsl:for-each select="tables/macro">
          <table border="1">
            <caption>
              <em>
                <xsl:value-of select="@name"/>
                <br/>
                <xsl:value-of select="@ref"/>
              </em>
            </caption>
            <tr bgcolor="#d6d6d6">
<!--rgb(214,214,214) -->
              <th>Tag</th>
              <th>Name</th>
              <th>Type</th>
              <th>Description</th>
            </tr>
            <xsl:apply-templates mode="module"/>
          </table>
        </xsl:for-each>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
