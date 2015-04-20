<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
<!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
  <xsl:output method="xml" indent="yes" encoding="UTF-8"/>
<!--
  MAIN template
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
    <dicts edition="2011">
      <xsl:apply-templates select="article/informaltable"/>
      <xsl:apply-templates select="article/sect1/informaltable"/>
      <xsl:apply-templates select="article/sect1/sect2/informaltable"/>
    </dicts>
  </xsl:template>
  <xsl:template match="article/sect1/sect2/informaltable">
    <xsl:if test="tgroup/tbody/row/entry[1]/para = 'Message Field'">
      <xsl:apply-templates select="." mode="data-elements">
        <xsl:with-param name="title" select="preceding::title[1]"/>
<!-- Get the table name -->
      </xsl:apply-templates>
    </xsl:if>
  </xsl:template>
  <xsl:template match="article/informaltable">
<!--xsl:for-each select="article/sect1/informaltable"-->
    <xsl:if test="tgroup/tbody/row/entry[1]/para = 'Tag'">
<!-- Does the table header contains ... -->
      <xsl:apply-templates select="." mode="data-elements">
        <xsl:with-param name="title" select="preceding::title[1]"/>
<!-- Get the table name -->
      </xsl:apply-templates>
    </xsl:if>
    <xsl:if test="tgroup/tbody/row/entry[1]/para = 'UID Value'">
<!-- Does the table header contains ... -->
      <xsl:apply-templates select="." mode="uid">
        <xsl:with-param name="title" select="preceding::para[1]"/>
<!-- Get the table name -->
      </xsl:apply-templates>
    </xsl:if>
<!--/xsl:for-each-->
  </xsl:template>
  <xsl:template match="article/sect1/informaltable">
<!--xsl:for-each select="article/sect1/informaltable"-->
    <xsl:if test="tgroup/tbody/row/entry[1]/para = 'Tag'">
<!-- Does the table header contains ... -->
      <xsl:apply-templates select="." mode="data-elements">
        <xsl:with-param name="title" select="preceding::title[1]"/>
<!-- Get the table name -->
      </xsl:apply-templates>
    </xsl:if>
    <xsl:if test="tgroup/tbody/row/entry[1]/para = 'UID Value'">
<!-- Does the table header contains ... -->
      <xsl:apply-templates select="." mode="uid">
        <xsl:with-param name="title" select="preceding::para[1]"/>
<!-- Get the table name -->
      </xsl:apply-templates>
    </xsl:if>
<!--/xsl:for-each-->
  </xsl:template>
<!--

template for a row in data-elements mode. Should be:

  Tag | Name | VR | VM | (RET)?

-->
  <xsl:template match="row" mode="data-elements-part7">
    <xsl:param name="retired" select="0"/>
    <xsl:if test="entry[1]/para != 'Message Field'">
      <xsl:variable name="keyword_value" select="normalize-space(entry[2]/para)"/>
      <xsl:variable name="tag_value" select="translate(entry[3]/para,'ABCDEF','abcdef')"/>
      <xsl:variable name="group_value" select="substring-after(substring-before($tag_value,','), '(')"/>
      <xsl:variable name="element_value" select="substring-after(substring-before($tag_value,')'), ',')"/>
      <xsl:variable name="vr">
        <xsl:call-template name="process-vr">
          <xsl:with-param name="text" select="normalize-space(entry[4]/para)"/>
        </xsl:call-template>
      </xsl:variable>
      <xsl:variable name="vm" select="normalize-space(entry[5]/para)"/>
      <xsl:variable name="name" select="normalize-space(entry[1]/para)"/>
      <xsl:variable name="description" select="entry[6]"/>
      <entry group="{$group_value}" element="{$element_value}" vr="{$vr}" vm="{$vm}" keyword="{$keyword_value}" name="{$name}">
        <xsl:if test="$retired = 1">
          <xsl:attribute name="retired">true</xsl:attribute>
        </xsl:if>
        <xsl:if test="$description">
          <xsl:element name="description">
            <xsl:value-of select="$description"/>
          </xsl:element>
        </xsl:if>
      </entry>
    </xsl:if>
  </xsl:template>
  <xsl:template match="row" mode="data-elements">
    <xsl:if test="entry[1]/para != 'Tag'">
<!-- skip the table header -->
      <xsl:variable name="tag_value" select="translate(entry[1]/para,'ABCDEF','abcdef')"/>
      <xsl:variable name="keyword_value" select="normalize-space(entry[3]/para)"/>
      <xsl:variable name="group_value" select="substring-after(substring-before($tag_value,','), '(')"/>
      <xsl:variable name="element_value">
        <xsl:variable name="tmp" select="substring-after(substring-before($tag_value,')'), ',')"/>
        <xsl:if test="$tmp = '3100 to 31ff'">
          <xsl:value-of select="'31xx'"/>
        </xsl:if>
        <xsl:if test="$tmp != '3100 to 31ff'">
          <xsl:value-of select="$tmp"/>
        </xsl:if>
      </xsl:variable>
<!--xsl:sort select="concat(@group_value,',',@element_value)"/-->
      <xsl:variable name="vr">
        <xsl:call-template name="process-vr">
          <xsl:with-param name="text" select="normalize-space(entry[4]/para)"/>
        </xsl:call-template>
      </xsl:variable>
      <xsl:if test="$group_value != '' and $element_value != ''">
        <xsl:variable name="name">
          <xsl:variable name="desc_value" select="normalize-space(translate(entry[2]/para,'&#160;',' '))"/>
          <xsl:if test="$desc_value != ''">
            <description>
<!-- some funny quote is in the way, replace it: -->
              <xsl:variable name="single_quote1">’–μ</xsl:variable>
              <xsl:variable name="single_quote2">'-µµ</xsl:variable>
              <xsl:value-of select="translate($desc_value,$single_quote1,$single_quote2)"/>
            </description>
          </xsl:if>
        </xsl:variable>
        <xsl:variable name="vm">
          <xsl:variable name="tmp" select="normalize-space(entry[5]/para[1])"/>
          <xsl:choose>
          <xsl:when test="$tmp = '1-n1'">
<!-- Special handling of LUT Data -->
            <xsl:value-of select="'1-n'"/>
          </xsl:when>
          <xsl:when test="$tmp = '1-n 1'">
<!-- Special handling of LUT Data -->
            <xsl:value-of select="'1-n'"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:value-of select="$tmp"/>
          </xsl:otherwise>
          </xsl:choose>
        </xsl:variable>
        <entry group="{ $group_value }" element="{ $element_value }" keyword="{$keyword_value}">
          <xsl:if test="$vr != ''">
            <xsl:attribute name="vr">
              <xsl:value-of select="$vr"/>
            </xsl:attribute>
          </xsl:if>
          <xsl:if test="$vm != ''">
            <xsl:attribute name="vm">
              <xsl:value-of select="$vm"/>
            </xsl:attribute>
          </xsl:if>
          <xsl:if test="normalize-space(entry[6]/para) = 'RET'">
            <xsl:attribute name="retired">
              <xsl:value-of select="'true'"/>
            </xsl:attribute>
          </xsl:if>
<!--xsl:attribute name="version">
                <xsl:value-of select="'3'"/>
              </xsl:attribute-->
          <xsl:if test="$name != ''">
            <xsl:attribute name="name">
              <xsl:value-of select="$name"/>
            </xsl:attribute>
          </xsl:if>
          <xsl:if test="$name = 'KVP'">
            <xsl:element name="correction">
<!-- vendor misuse of tags -->
              <xsl:value-of select="'kVp'"/>
            </xsl:element>
          </xsl:if>
          <xsl:if test="$name = ''">
            <xsl:element name="description">
<!-- vendor misuse of tags -->
              <xsl:value-of select="'SHALL NOT BE USED'"/>
            </xsl:element>
          </xsl:if>
<!--
              <xsl:if test="entry[3]/para != '' and entry[4]/para != ''">
                <representations>
                  <representation vr="{ entry[3]/para }" vm="{ entry[4]/para }"/>
                </representations>
              </xsl:if>
-->
        </entry>
      </xsl:if>
    </xsl:if>
  </xsl:template>
<!--
template for a row in UID mode. Should be:

  UID Value |  UID NAME |  UID TYPE | Part

-->
  <xsl:template match="row" mode="uid">
    <xsl:if test="entry[1]/para != 'UID Value'">
<!-- skip the table header -->
      <xsl:variable name="value" select="normalize-space(translate(entry[1]/para,'&#10;&#9;&#173;&#160;',''))"/>
      <!--xsl:variable name="garbage" select="&#160;"/--> <!-- pair C2 A0, non-breaking space -->
      <xsl:variable name="garbage1">&#160;</xsl:variable>
      <xsl:variable name="garbage2">&#173;</xsl:variable> <!-- C2 AD -->
      <xsl:variable name="name1" select="translate(entry[2]/para,'–&#9;','-')"/>
      <xsl:variable name="name2" select="translate($name1,$garbage1,' ')"/>
      <xsl:variable name="name3" select="translate($name2,$garbage2,' ')"/>
      <xsl:variable name="name" select="normalize-space($name3)"/>
      <xsl:variable name="type" select="normalize-space(translate(entry[3]/para,'&#9;&#160;',''))"/>

      <xsl:choose>
        <xsl:when test="contains(entry[2]/para,'(Retired)')">
          <xsl:variable name="name-retired">
            <xsl:value-of select="normalize-space(substring-before($name,'(Retired)'))"/>
          </xsl:variable>
          <uid value="{$value}" name="{$name-retired}" type="{$type}" part="{entry[4]/para}" retired="true"/>
        </xsl:when>
        <xsl:otherwise>
          <uid value="{$value}" name="{$name}" type="{$type}" part="{entry[4]/para}" retired="false"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:if>
  </xsl:template>
<!--
template for a row in Frame of Reference mode. Should be:

  UID Value |  UID NAME |  Normative Reference

-->
  <xsl:template match="row" mode="frameref">
    <xsl:if test="entry[1]/para != 'UID Value'">
<!-- skip the table header -->
      <uid value="{entry[1]/para}" name="{entry[2]/para}" normative-reference="{entry[3]/para}"/>
    </xsl:if>
  </xsl:template>
<!--
template to split table into two cases: UIDs or Normative Reference:
-->
  <xsl:template match="informaltable" mode="data-elements">
    <xsl:param name="title"/>
    <xsl:variable name="ref" select="substring-before($title,'&#9;')"/>
    <xsl:variable name="name" select="substring-after($title,'&#9;')"/>
    <dict ref="{$ref}" name="{$name}">
      <xsl:choose>
        <xsl:when test="tgroup/tbody/row/entry[4]/para = 'VR'">
          <xsl:choose>
            <!-- PS 3.7 -->
            <xsl:when test="tgroup/tbody/row/entry[1]/para = 'Message Field'">
              <xsl:variable name="retval" select="contains($title,'Retired')"/>
              <xsl:apply-templates select="tgroup/tbody/row" mode="data-elements-part7">
                <xsl:with-param name="retired" select="$retval"/>
              </xsl:apply-templates>
            </xsl:when>
            <!-- PS 3.6 -->
            <xsl:otherwise>
              <xsl:apply-templates select="tgroup/tbody/row" mode="data-elements"/>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:when>
      </xsl:choose>
    </dict>
  </xsl:template>
  <xsl:template match="informaltable" mode="uid">
    <xsl:param name="title"/>
    <table name="{$title}">
      <xsl:choose>
        <xsl:when test="tgroup/tbody/row/entry[3]/para = 'Normative Reference'">
          <xsl:apply-templates select="tgroup/tbody/row" mode="frameref"/>
        </xsl:when>
        <xsl:when test="tgroup/tbody/row/entry[3]/para = 'UID TYPE'">
          <xsl:apply-templates select="tgroup/tbody/row" mode="uid"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:message>Unhandled <xsl:value-of select="$title"/></xsl:message>
        </xsl:otherwise>
      </xsl:choose>
    </table>
  </xsl:template>
<!--
  template to process VR from PDF representation into GDCM representation
-->
  <xsl:template name="process-vr">
    <xsl:param name="text"/>
    <xsl:choose>
      <xsl:when test="$text='see note'">
        <xsl:value-of select="''"/>
      </xsl:when>
      <xsl:when test="$text='US or SS or OW'">
        <xsl:value-of select="'US_SS_OW'"/>
      </xsl:when>
      <xsl:when test="$text='US or SSor OW'">
        <xsl:value-of select="'US_SS_OW'"/>
      </xsl:when>
      <xsl:when test="$text='US or SS'">
        <xsl:value-of select="'US_SS'"/>
      </xsl:when>
      <xsl:when test="$text='OW or OB'">
        <xsl:value-of select="'OB_OW'"/>
      </xsl:when>
      <xsl:when test="$text='OB or OW'">
        <xsl:value-of select="'OB_OW'"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$text"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
</xsl:stylesheet>
