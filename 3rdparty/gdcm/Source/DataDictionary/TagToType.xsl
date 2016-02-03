<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
<!-- XSL to convert XML GDCM2 data dictionay into
     C++ template code
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
  <xsl:template match="/">
    <xsl:text>
// GENERATED FILE DO NOT EDIT
// $ xsltproc TagToType.xsl Part6.xml &gt; gdcmTagToType.h

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMTAGTOTYPE_H
#define GDCMTAGTOTYPE_H

#include "gdcmVR.h"
#include "gdcmVM.h"
#include "gdcmStaticAssert.h"

namespace gdcm {
// default template: the compiler should only pick it up when the element is private:
template &lt;uint16_t group,uint16_t element&gt; struct TagToType {
//GDCM_STATIC_ASSERT( group % 2 );
enum { VRType = VR::VRALL };
enum { VMType = VM::VM1_n };
};
// template for group length:
template &lt;uint16_t group&gt; struct TagToType&lt;group,0x0000&gt; {
static const char* GetVRString() { return "UL"; }
typedef VRToType&lt;VR::UL&gt;::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
</xsl:text>
    <xsl:for-each select="dicts/dict/entry">
      <xsl:sort select="@group" data-type="text" order="ascending"/>
      <xsl:sort select="@element" data-type="text" order="ascending"/>
      <xsl:variable name="group" select="translate(@group,'x','0')"/>
      <xsl:variable name="element" select="translate(@element,'x','0')"/>
      <xsl:if test="contains(@element,'x') = true and contains(@element,'xx') = false and @vr != '' and @vr != 'US_SS' and @vr != 'US_SS_OW' and @vr != 'OB_OW'">
<xsl:variable name="classname">
        <xsl:text>TagToType&lt;0x</xsl:text>
        <xsl:value-of select="$group"/>
        <xsl:text>,0x</xsl:text>
        <xsl:value-of select="$element"/>
        <xsl:text>&gt;</xsl:text>
</xsl:variable>
        <xsl:text>template &lt;&gt; struct </xsl:text>
        <xsl:value-of select="$classname"/>
        <xsl:text> {
</xsl:text>
        <xsl:text>static const char* GetVRString() { return "</xsl:text>
        <xsl:value-of select="@vr"/>
        <xsl:text>"; }
</xsl:text>
        <xsl:text>typedef VRToType&lt;VR::</xsl:text>
        <xsl:value-of select="@vr"/>
        <xsl:text>&gt;::Type Type;</xsl:text>
        <xsl:text>
</xsl:text>
        <xsl:text>enum { VRType = VR::</xsl:text>
        <xsl:value-of select="@vr"/>
        <xsl:text> };</xsl:text>
        <xsl:text>
</xsl:text>
        <xsl:text>enum { VMType = VM::</xsl:text>
        <xsl:call-template name="VMStringToVMType">
          <xsl:with-param name="vmstring" select="@vm"/>
        </xsl:call-template>
        <xsl:text> };</xsl:text>
        <xsl:text>
</xsl:text>
        <xsl:text>static const char* GetVMString() { return "</xsl:text>
        <xsl:value-of select="@vm"/>
        <xsl:text>"; }
</xsl:text>
        <xsl:text>};</xsl:text>
        <xsl:text>
</xsl:text>
        <!--xsl:text>const char </xsl:text><xsl:value-of select="$classname"/><xsl:text>::VRString[] = "</xsl:text>
        <xsl:text>";
</xsl:text>
        <xsl:text>const char </xsl:text><xsl:value-of select="$classname"/><xsl:text>::VMString[] = "</xsl:text>
        <xsl:value-of select="@vm"/>
        <xsl:text>";
</xsl:text-->
      </xsl:if>
    </xsl:for-each>
    <xsl:text>
} // end namespace gdcm
#endif // GDCMTAGTOTYPE_H
</xsl:text>
  </xsl:template>
</xsl:stylesheet>
