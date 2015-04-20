<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- share common code to transform a VM Part 6 string into a gdcm::VM type
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
  <xsl:include href="VM.xsl"/>
<!-- The main template that loop over all dict/entry -->
  <xsl:key name="entries" match="entry" use="@group"/>
  <xsl:template match="/">
    <xsl:text>
// GENERATED FILE DO NOT EDIT
// $ xsltproc DefaultDicts.xsl Part6.xml &gt; gdcmDefaultDicts.cxx

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMDEFAULTDICTS_CXX
#define GDCMDEFAULTDICTS_CXX

#include "gdcmDicts.h"
#include "gdcmVR.h"
#include "gdcmDict.h"
#include "gdcmDictEntry.h"

namespace gdcm {
typedef struct
{
  uint16_t group;
  uint16_t element;
  VR::VRType vr;
  VM::VMType vm;
  const char *name;
  const char *keyword;
  bool ret;
} DICT_ENTRY;

static const DICT_ENTRY DICOMV3DataDict [] = {
</xsl:text>
    <xsl:for-each select="dicts/dict/entry">
      <!-- need to sort based on text, since hex are not 'number' -->
      <xsl:sort select="@group" data-type="text" order="ascending"/>
      <xsl:sort select="@element" data-type="text" order="ascending"/>
      <xsl:variable name="group" select="translate(@group,'x','0')"/>
      <xsl:variable name="element" select="translate(@element,'x','0')"/>
      <xsl:choose>
        <xsl:when test="substring(@group,3) != 'xx' and substring(@element,3) = 'xx' and substring(@element,1,2) != '00' and substring(@element,1,2) != '10'">
          <xsl:call-template name="do-one-entry">
            <xsl:with-param name="count" select="0"/>
            <xsl:with-param name="do-element" select="1"/>
            <xsl:with-param name="group" select="@group"/>
            <xsl:with-param name="element" select="$element"/>
            <!--xsl:with-param name="owner" select="@owner"/-->
            <xsl:with-param name="vr" select="@vr"/>
            <xsl:with-param name="vm" select="@vm"/>
            <xsl:with-param name="retired" select="@retired"/>
            <xsl:with-param name="name" select="@name"/>
            <xsl:with-param name="keyword" select="@keyword"/>
          </xsl:call-template>
        </xsl:when>
        <xsl:when test="substring(@group,3) = 'xx' and contains(@element,'x') = false ">
          <xsl:call-template name="do-one-entry">
            <xsl:with-param name="count" select="0"/>
            <xsl:with-param name="do-group" select="1"/>
            <xsl:with-param name="group" select="$group"/>
            <xsl:with-param name="element" select="@element"/>
            <!--xsl:with-param name="owner" select="@owner"/-->
            <xsl:with-param name="vr" select="@vr"/>
            <xsl:with-param name="vm" select="@vm"/>
            <xsl:with-param name="retired" select="@retired"/>
            <xsl:with-param name="name" select="@name"/>
            <xsl:with-param name="keyword" select="@keyword"/>
          </xsl:call-template>
        </xsl:when>
        <xsl:when test="contains(@group,'x') = false and contains(@element,'x') = false">
          <xsl:call-template name="do-one-entry">
            <xsl:with-param name="count" select="255"/>
            <xsl:with-param name="group" select="@group"/>
            <xsl:with-param name="element" select="@element"/>
            <!--xsl:with-param name="owner" select="@owner"/-->
            <xsl:with-param name="vr" select="@vr"/>
            <xsl:with-param name="vm" select="@vm"/>
            <xsl:with-param name="retired" select="@retired"/>
            <xsl:with-param name="name" select="@name"/>
            <xsl:with-param name="keyword" select="@keyword"/>
          </xsl:call-template>
        </xsl:when>
        <!-- Private element e.g (0019,xx26) -->
        <xsl:when test="contains(@group,'x') = false and substring(@element,1,2) = 'xx' and not(contains(substring(@element,3,4),'x'))">
          <xsl:call-template name="do-one-entry">
            <xsl:with-param name="count" select="255"/>
            <xsl:with-param name="group" select="@group"/>
            <xsl:with-param name="element" select="$element"/> <!-- replaced xx with 00 which is what we want -->
            <!--xsl:with-param name="owner" select="@owner"/-->
            <xsl:with-param name="vr" select="@vr"/>
            <xsl:with-param name="vm" select="@vm"/>
            <xsl:with-param name="retired" select="@retired"/>
            <xsl:with-param name="name" select="@name"/>
            <xsl:with-param name="keyword" select="@keyword"/>
          </xsl:call-template>
        </xsl:when>
        <xsl:when test="contains(@group,'x') = false and @element = '00xx'">
          <xsl:call-template name="do-one-entry">
            <xsl:with-param name="count" select="255"/>
            <xsl:with-param name="group" select="@group"/>
            <xsl:with-param name="element" select="$element"/>
            <!--xsl:with-param name="owner" select="@owner"/-->
            <xsl:with-param name="vr" select="@vr"/>
            <xsl:with-param name="vm" select="@vm"/>
            <xsl:with-param name="retired" select="@retired"/>
            <xsl:with-param name="name" select="@name"/>
            <xsl:with-param name="keyword" select="@keyword"/>
          </xsl:call-template>
        </xsl:when>
        <xsl:otherwise>
          <xsl:message>Problem with element:(<xsl:value-of select="@group"/>,<xsl:value-of select="@element"/>)
</xsl:message>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
    <xsl:for-each select="//entry[generate-id() = generate-id(key('entries',@group)[1])]">
<!--
Note: We need to produce generic group length for all known groups but 0000 and 0002 since they have there own
already

Implementation note:
generating group length for arbitrary even group number seems to get my xsltproc on its knees
-->
      <xsl:if test="contains(@group,'x') = false and @group!='0000' and @group!='0002'">
        <xsl:call-template name="do-one-entry">
          <xsl:with-param name="count" select="0"/>
          <xsl:with-param name="group" select="@group"/>
          <xsl:with-param name="element" select="'0000'"/>
          <xsl:with-param name="vr" select="'UL'"/>
          <xsl:with-param name="vm" select="'1'"/>
          <!--xsl:with-param name="owner" select="@owner"/-->
          <xsl:with-param name="retired" select="'true'"/>
          <xsl:with-param name="name" select="concat('Group Length ',@group)"/>
        </xsl:call-template>
      </xsl:if>
    </xsl:for-each>
    <xsl:text>
 // FIXME: need a dummy element
  {0xffff,0xffff,VR::INVALID,VM::VM0,"","",true }, // dummy
  {0xffff,0xffff,VR::INVALID,VM::VM0,0,0,true } // Gard
};

void Dict::LoadDefault()
{
   unsigned int i = 0;
   DICT_ENTRY n = DICOMV3DataDict[i];
   while( n.name != 0 )
   {
      Tag t(n.group, n.element);
      DictEntry e( n.name, n.keyword, n.vr, n.vm, n.ret );
      AddDictEntry( t, e );
      n = DICOMV3DataDict[++i];
   }
}

/*
void PrivateDict::LoadDefault()
{
  // TODO
}
*/

} // end namespace gdcm
#endif // GDCMDEFAULTDICTS_CXX
</xsl:text>
  </xsl:template>
  <xsl:template name="do-group-length">
    <xsl:param name="count" select="0"/>
    <xsl:if test="$count &lt; 65535">
<!-- 0xffff -->
      <xsl:variable name="group-length">
        <xsl:call-template name="printHex">
          <xsl:with-param name="number" select="$count"/>
        </xsl:call-template>
      </xsl:variable>
<!--xsl:call-template name="do-one-entry">
        <xsl:with-param name="count" select="0"/>
        <xsl:with-param name="group" select="$group-length"/>
        <xsl:with-param name="element" select="'0000'"/>
        <xsl:with-param name="vr" select="'UL'"/>
        <xsl:with-param name="vm" select="'1'"/>
      </xsl:call-template-->
      <xsl:call-template name="do-group-length">
        <xsl:with-param name="count" select="$count + 2"/>
      </xsl:call-template>
    </xsl:if>
  </xsl:template>
  <xsl:template name="do-one-entry">
    <xsl:param name="count" select="0"/>
    <xsl:param name="do-group" select="0"/>
    <xsl:param name="do-element" select="0"/>
    <xsl:param name="group"/>
    <xsl:param name="element"/>
    <xsl:param name="owner" select="''"/>
    <xsl:param name="vr"/>
    <xsl:param name="vm"/>
    <xsl:param name="retired"/>
    <xsl:param name="name"/>
    <xsl:param name="keyword"/>
    <xsl:if test="$count &lt; 256">
      <xsl:text>  {0x</xsl:text>
      <xsl:value-of select="$group"/>
      <xsl:text>,0x</xsl:text>
      <xsl:value-of select="$element"/>
      <!--xsl:text>,"</xsl:text>
      <xsl:value-of select="$owner"/>
      <xsl:text>"</xsl:text-->
<!--xsl:value-of select="$temp"/-->
      <xsl:text>,VR::</xsl:text>
      <xsl:if test="not ($vr != '')">
<!-- FIXME -->
        <xsl:text>INVALID</xsl:text>
      </xsl:if>
      <xsl:if test="$vr != ''">
        <xsl:value-of select="$vr"/>
      </xsl:if>
      <xsl:text>,VM::</xsl:text>
      <xsl:call-template name="VMStringToVMType">
        <xsl:with-param name="vmstring" select="$vm"/>
      </xsl:call-template>
      <xsl:text>,"</xsl:text>
      <xsl:value-of select="$name"/>
      <xsl:text>","</xsl:text>
      <xsl:value-of select="$keyword"/>
      <xsl:text>",</xsl:text>
      <xsl:value-of select="$retired = 'true'"/>
      <xsl:text> },</xsl:text>
      <xsl:text>
</xsl:text>
    </xsl:if>
    <xsl:if test="$count &lt; 255">
<!--xsl:message><xsl:value-of select="$do-group"/></xsl:message-->
      <xsl:if test="$do-group != '0'">
        <xsl:variable name="temp">
          <xsl:call-template name="printHex">
            <xsl:with-param name="number" select="$count + 2"/>
          </xsl:call-template>
        </xsl:variable>
        <xsl:variable name="tail">
          <xsl:if test="string-length($temp) != 1">
            <xsl:value-of select="$temp"/>
          </xsl:if>
          <xsl:if test="string-length($temp) = 1">
            <xsl:value-of select="concat('0',$temp)"/>
          </xsl:if>
        </xsl:variable>
        <xsl:variable name="group_xx" select="concat(substring($group,1,2),$tail)"/>
        <xsl:call-template name="do-one-entry">
          <xsl:with-param name="count" select="$count + 2"/>
          <xsl:with-param name="do-group" select="$do-group"/>
          <xsl:with-param name="do-element" select="$do-element"/>
          <xsl:with-param name="group" select="$group_xx"/>
          <xsl:with-param name="element" select="$element"/>
          <xsl:with-param name="vr" select="$vr"/>
          <xsl:with-param name="vm" select="$vm"/>
          <!--xsl:with-param name="owner" select="$owner"/-->
          <xsl:with-param name="retired" select="$retired"/>
          <xsl:with-param name="name" select="$name"/>
          <xsl:with-param name="keyword" select="$keyword"/>
        </xsl:call-template>
      </xsl:if>
      <xsl:if test="$do-element != '0'">
        <xsl:variable name="temp">
          <xsl:call-template name="printHex">
            <xsl:with-param name="number" select="$count + 1"/>
          </xsl:call-template>
        </xsl:variable>
        <xsl:variable name="tail">
          <xsl:if test="string-length($temp) != 1">
            <xsl:value-of select="$temp"/>
          </xsl:if>
          <xsl:if test="string-length($temp) = 1">
            <xsl:value-of select="concat('0',$temp)"/>
          </xsl:if>
        </xsl:variable>
        <xsl:variable name="element_xx" select="concat(substring($element,1,2),$tail)"/>
        <xsl:call-template name="do-one-entry">
          <xsl:with-param name="count" select="$count + 1"/>
          <xsl:with-param name="do-group" select="$do-group"/>
          <xsl:with-param name="do-element" select="$do-element"/>
          <xsl:with-param name="group" select="$group"/>
          <xsl:with-param name="element" select="$element_xx"/>
          <xsl:with-param name="vr" select="$vr"/>
          <xsl:with-param name="vm" select="$vm"/>
          <!--xsl:with-param name="owner" select="$owner"/-->
          <xsl:with-param name="retired" select="$retired"/>
          <xsl:with-param name="name" select="$name"/>
          <xsl:with-param name="keyword" select="$keyword"/>
        </xsl:call-template>
      </xsl:if>
    </xsl:if>
  </xsl:template>
<!-- A function to convert a decimal into an hex -->
  <xsl:template name="printHex">
    <xsl:param name="number">0</xsl:param>
    <xsl:variable name="low">
      <xsl:value-of select="$number mod 16"/>
    </xsl:variable>
    <xsl:variable name="high">
      <xsl:value-of select="floor($number div 16)"/>
    </xsl:variable>
    <xsl:choose>
      <xsl:when test="$high &gt; 0">
        <xsl:call-template name="printHex">
          <xsl:with-param name="number">
            <xsl:value-of select="$high"/>
          </xsl:with-param>
        </xsl:call-template>
      </xsl:when>
<!--<xsl:otherwise>
      <xsl:text>0x</xsl:text>
    </xsl:otherwise>-->
    </xsl:choose>
    <xsl:choose>
      <xsl:when test="$low &lt; 10">
        <xsl:value-of select="format-number($low,&quot;0&quot;)"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:variable name="temp">
          <xsl:value-of select="$low - 10"/>
        </xsl:variable>
        <xsl:value-of select="translate($temp, '012345', 'abcdef')"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
</xsl:stylesheet>
