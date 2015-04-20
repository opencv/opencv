<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="text" indent="yes"/>
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
    <xsl:text>// GENERATED FILE DO NOT EDIT
// $ xsltproc TagKeywords.xsl Part6.xml &gt; gdcmTagKeywords.h

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2012 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMTAGKEYWORDS_H
#define GDCMTAGKEYWORDS_H

#include "gdcmAttribute.h"

namespace gdcm {
namespace Keywords {

</xsl:text>
<xsl:apply-templates select="dicts/dict/entry" mode="attribute" />
<xsl:text>
}
}

#endif
</xsl:text>

  </xsl:template>


  <xsl:template match="entry[description = 'SHALL NOT BE USED']" mode="attribute">
    <xsl:text>    // 0x</xsl:text>
    <xsl:value-of select="@group" />
    <xsl:text>, 0x</xsl:text>
    <xsl:value-of select="@element" />
    <xsl:text> SHALL NOT BE USED
</xsl:text>
  </xsl:template>

  <xsl:template match="entry" mode="attribute">
    <xsl:text>    typedef gdcm::Attribute&lt;0x</xsl:text>
    <xsl:value-of select="translate(@group, 'x', '0')" />
    <xsl:text>, 0x</xsl:text>
    <xsl:value-of select="translate(@element, 'x', '0')" />
    <xsl:text>&gt; </xsl:text>
    <xsl:value-of select="@keyword" />
    <xsl:text>;
</xsl:text>
  </xsl:template>

</xsl:stylesheet>
