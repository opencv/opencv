<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="xml" encoding="UTF-8" indent="yes"/>
<!-- XSL to convert XML GDCM2 data dictionay into
     dcm4che data dictionary
Checked against:
    http://dcm4che.svn.sourceforge.net/viewvc/dcm4che/dcm4che2/trunk/dcm4che-core/src/xml/dictionary.xml
-->
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->

<!--life saver xsl script found at:
http://www.thescripts.com/forum/thread86881.html
-->
  <xsl:template name="upperCase">
    <xsl:param name="textToTransform"/>
    <xsl:variable name="head">
      <xsl:choose>
        <xsl:when test="contains($textToTransform, ' ')">
          <xsl:value-of select="substring-before($textToTransform, ' ')"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$textToTransform"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <xsl:variable name="tail" select="substring-after($textToTransform, ' ')"/>
    <xsl:variable name="firstTransform" select="concat(translate(substring($head, 1, 1), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), substring($head, 2))"/>
    <xsl:choose>
      <xsl:when test="$tail">
        <xsl:value-of select="$firstTransform"/>
        <xsl:call-template name="upperCase">
          <xsl:with-param name="textToTransform" select="$tail"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$firstTransform"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

<!--
http://www.dpawson.co.uk/xsl/sect2/replace.html
-->
<xsl:template name="replace-string">
    <xsl:param name="text"/>
    <xsl:param name="replace"/>
    <xsl:param name="with"/>
    <xsl:choose>
      <xsl:when test="contains($text,$replace)">
        <xsl:value-of select="substring-before($text,$replace)"/>
        <xsl:value-of select="$with"/>
        <xsl:call-template name="replace-string">
          <xsl:with-param name="text"
select="substring-after($text,$replace)"/>
          <xsl:with-param name="replace" select="$replace"/>
          <xsl:with-param name="with" select="$with"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$text"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template name="makeKeyword">
    <xsl:param name="textToTransform"/>
      <xsl:variable name="apos">'</xsl:variable>

      <!--translating an apostrophe is a pain ... better solution ? -->
      <xsl:variable name="description_apos3">
        <!--xsl:value-of select="translate($textToTransform, $apos, '')"/-->
<xsl:call-template name="replace-string">
<xsl:with-param name="text" select="$textToTransform"/>
<xsl:with-param name="replace">De-identification</xsl:with-param>
<xsl:with-param name="with" select="'Deidentification'"/>
</xsl:call-template>
      </xsl:variable>
      <xsl:variable name="description_apos2">
        <!--xsl:value-of select="translate($textToTransform, $apos, '')"/-->
<xsl:call-template name="replace-string">
<xsl:with-param name="text" select="$description_apos3"/>
<xsl:with-param name="replace">'s</xsl:with-param>
<xsl:with-param name="with" select="' '"/>
</xsl:call-template>
      </xsl:variable>
      <xsl:variable name="description_apos">
        <!--xsl:value-of select="translate($textToTransform, $apos, '')"/-->
<xsl:call-template name="replace-string">
<xsl:with-param name="text" select="$description_apos2"/>
<xsl:with-param name="replace">Sub-operations</xsl:with-param>
<xsl:with-param name="with" select="'Suboperations'"/>
</xsl:call-template>
      </xsl:variable>
      <xsl:variable name="description_dash2">
        <xsl:value-of select="translate($description_apos,'-/','  ')"/>
      </xsl:variable>
      <xsl:variable name="description_dash">
        <xsl:value-of select="translate($description_dash2,$apos,'')"/>
      </xsl:variable>
      <xsl:variable name="description_cap">
        <xsl:call-template name="upperCase">
          <xsl:with-param name="textToTransform" select="normalize-space($description_dash)"/>
        </xsl:call-template>
      </xsl:variable>
      <!-- remove remaining extra character -->
      <xsl:value-of select="translate($description_cap,'(),','')"/>
  </xsl:template>

<!-- The main template that loop over all dict/entry -->
  <xsl:template match="/">
    <dictionary tagclass="org.dcm4che2.data.Tag">
      <xsl:text>
</xsl:text>
    <xsl:for-each select="dicts/dict/entry">
    <xsl:sort select="@group"/>
    <xsl:sort select="@element"/>
      <xsl:variable name="description_cap">
      </xsl:variable>
      <xsl:variable name="keyword">
        <xsl:call-template name="makeKeyword">
          <xsl:with-param name="textToTransform" select="@name"/>
        </xsl:call-template>
      </xsl:variable>
      <xsl:variable name="group">
        <xsl:value-of select="translate(@group,'abcdef','ABCDEF')"/>
      </xsl:variable>
      <xsl:variable name="element">
        <xsl:value-of select="translate(@element,'abcdef','ABCDEF')"/>
      </xsl:variable>
      <xsl:variable name="vr">
        <xsl:value-of select="translate(@vr,'_','|')"/>
      </xsl:variable>
      <xsl:variable name="retired">
      <xsl:choose>
        <xsl:when test="@retired = 'true'">
          <xsl:text>RET</xsl:text>
        </xsl:when>
        <xsl:otherwise>
          <xsl:text></xsl:text>
        </xsl:otherwise>
      </xsl:choose>
      </xsl:variable>
      <element tag="{$group}{$element}" keyword="{$keyword}" vr="{$vr}" vm="{@vm}" ret="{$retired}"><xsl:value-of select="@name"/></element>
      <xsl:text>
</xsl:text>
    </xsl:for-each>
    </dictionary>
  </xsl:template>
</xsl:stylesheet>
