<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:my="urn:my" version="2.0" exclude-result-prefixes="#all">
  <!--
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
-->
  <!-- =====================================================================
 |
 |  Copyright (C) 2007, ICSMED AG
 |
 |  This software and supporting documentation were developed by
 |
 |    ICSMED AG
 |    Escherweg 2
 |    26121 Oldenburg
 |    Germany
 |
 |  Purpose: Extract particular section from part 3 of the DICOM standard.
 |           Subsections and tables within the section are not extracted.
 |
 ======================================================================= -->
  <!--
Special Thanks to Joerg Riesmeier for the extract_section.xsl script !
-->
  <!--
TODO:
* Make sure a <include/> is indeed a Include `' ...
  eg. Fix RAW DATA KEYS, Key are recognized as `Include`

Usage: (you need a XSLT 2.0 processor)

$ java -jar ~/Software/saxon/saxon8.jar  08_03pu.xml Part3.xsl > ModuleAttributes.xml

or on debian

saxonb-xslt -o Part3.xml -s 09_03pu3.xml -xsl Part3.xsl
-->
  <xsl:output method="xml" indent="yes" encoding="UTF-8"/>
  <xsl:strip-space elements="*"/>
  <xsl:variable name="apos">'</xsl:variable>
  <xsl:variable name="doublequote">"</xsl:variable>
  <xsl:variable name="linebreak">
    <xsl:text>
</xsl:text>
  </xsl:variable>
  <!--

Special normalize-space

-->
  <xsl:function name="my:normalize-space" as="xs:string*">
    <xsl:param name="string" as="xs:string*"/>
    <xsl:value-of select="normalize-space(translate($string,'&#160;',' '))"/>
  </xsl:function>
<!--
-->
  <xsl:function name="my:normalize-paragraph" as="xs:string*">
    <xsl:param name="string" as="xs:string*"/>
    <xsl:sequence select="for $s in $string return string-join( for $word in tokenize($s, $linebreak) return normalize-space($word), $linebreak)"/>
  </xsl:function>
  <!--
  -->
  <xsl:function name="my:remove-trailing-dot" as="xs:string*">
    <xsl:param name="string" as="xs:string"/>
    <xsl:variable name="result" as="xs:string" select="if (ends-with($string,'.'))      then substring($string,1,string-length($string)-1)      else $string"/>
    <xsl:value-of select="$result"/>
  </xsl:function>
  <!--

Weird camel case function to get closer to docbook version

-->
  <xsl:function name="my:camel-case" as="xs:string*">
    <xsl:param name="stringraw" as="xs:string*"/>
    <xsl:variable name="string" select="normalize-space(translate($stringraw,'&#160;&#173;','  '))"/>
    <xsl:variable name="tmp0">
      <xsl:sequence select="for $s in $string return string-join( for $word in tokenize($s, '-| ') return concat( upper-case(substring($word, 1, 1)), lower-case(substring($word, 2))) , ' ')"/>
    </xsl:variable>
    <xsl:variable name="tmp1">
      <xsl:sequence select="for $s in $tmp0 return string-join( for $word in tokenize($s, '-| ') return if (string-length($word) = 2) then upper-case($word) else $word , ' ')"/>
    </xsl:variable>
    <xsl:variable name="tmp2" select="replace($tmp1,'Iod ','IOD ')"/>
    <xsl:variable name="tmp3" select="replace($tmp2,' Pdf ',' PDF ')"/>
    <xsl:variable name="tmp4" select="replace($tmp3,' Cda ',' CDA ')"/>
    <xsl:variable name="tmp5" select="replace($tmp4,'Sop ','SOP ')"/>
    <xsl:variable name="tmp6" select="replace($tmp5,'Xrf ','XRF ')"/>
    <xsl:variable name="tmp7" select="replace($tmp6,' Ecg ',' ECG ')"/>
    <xsl:variable name="tmp8" select="replace($tmp7,' Cad ',' CAD ')"/>
    <xsl:variable name="tmp9" select="replace($tmp8,'Xa/xrf ','XA/XRF ')"/>
    <xsl:variable name="tmp10" select="replace($tmp9,'Hl7 ','HL7 ')"/>
    <xsl:variable name="tmp11" select="replace($tmp10,'Icc ','ICC ')"/>
    <xsl:variable name="tmp12" select="replace($tmp11,' And ',' and ')"/>
    <xsl:variable name="tmp13" select="replace($tmp12,' OF ',' of ')"/>
    <xsl:variable name="tmp14" select="replace($tmp13,'Nm/pet ','NM/PET ')"/>
    <xsl:variable name="tmp15" select="replace($tmp14,'Pet ','PET ')"/>
    <xsl:variable name="tmp16" select="replace($tmp15,'Roi ','ROI ')"/>
    <xsl:variable name="tmp17" select="replace($tmp16,'Voi ','VOI ')"/>
    <xsl:variable name="tmp18" select="replace($tmp17,'Lut ','LUT ')"/>
    <xsl:variable name="tmp19" select="replace($tmp18,' AN ',' an ')"/>
    <xsl:variable name="tmp20" select="replace($tmp19,' IN ',' in ')"/>
    <xsl:variable name="tmp21" select="replace($tmp20,' Fov ',' FOV ')"/>
    <xsl:variable name="tmp22" select="replace($tmp21,' OR ',' or ')"/>
    <xsl:variable name="tmp23" select="replace($tmp22,' Ct/mr ',' CT/MR ')"/>
    <xsl:value-of select="$tmp23"/>
  </xsl:function>
  <!--
Function to parse a row from an informaltable specifically for a Macro/Module table:
-->
  <xsl:template match="row" mode="macro">
    <xsl:variable name="name">
      <xsl:for-each select="entry[1]/para">
        <xsl:value-of select="normalize-space(.)"/>
        <xsl:if test="position() != last()">
          <xsl:text> </xsl:text>
        </xsl:if>
      </xsl:for-each>
    </xsl:variable>
    <xsl:variable name="tag" select="normalize-space(string-join(entry[2]/para,' '))"/>
    <xsl:choose>
      <xsl:when test="substring($tag,1,1) = '(' and substring($tag,11,1) = ')'">
        <xsl:variable name="group" select="normalize-space(substring-after(substring-before($tag,','), '('))"/>
        <xsl:variable name="element" select="normalize-space(substring-after(substring-before($tag,')'), ','))"/>
        <!--used internally to find out if type is indeed type of if column type was missing ... not full proof -->
        <xsl:variable name="internal_type" select="normalize-space(string-join(entry[3]/para,' '))"/>
        <xsl:variable name="type">
          <xsl:value-of select="entry[3]/para" separator="{$linebreak}"/>
        </xsl:variable>
        <!-- some funny quote is in the way, replace it: -->
        <xsl:variable name="single_quote1">’“”– ­</xsl:variable>
        <xsl:variable name="single_quote2" select="concat(concat(concat($apos, $doublequote),$doublequote),'- µ')"/>
        <xsl:variable name="description_tmp">
          <xsl:value-of select="entry[4]/para" separator="{$linebreak}"/>
        </xsl:variable>
        <xsl:variable name="name_translate" select="normalize-space(translate($name,$single_quote1,$single_quote2))"/>
        <xsl:variable name="description" select="translate($description_tmp,$single_quote1,$single_quote2)"/>
        <!-- Attribute Name  Tag  Type  Attribute Description -->
        <xsl:choose>
          <!-- Try to figure if this table is busted (missing Type column -->
          <xsl:when test="string-length($internal_type) &gt; 0 and string-length($internal_type) &lt;= 2">
            <xsl:choose>
              <xsl:when test="$group != '' and $element != ''">
                <entry group="{$group}" element="{$element}" name="{$name_translate}" type="{normalize-space($type)}">
                  <xsl:variable name="n_description" select="my:normalize-paragraph($description)"/>
                  <!--xsl:call-template name="description-extractor">
                    <xsl:with-param name="desc" select="$n_description"/>
                  </xsl:call-template-->
                  <description>
                    <xsl:value-of select="$n_description"/>
                  </description>
                  <xsl:variable name="dummy">
                    <xsl:call-template name="get-description-reference">
                      <xsl:with-param name="description" select="$n_description"/>
                    </xsl:call-template>
                  </xsl:variable>
                  <!--xsl:if test="$dummy !='' and $dummy != 'C.10.4' and $dummy != 'C.7.6.4'"-->
                  <xsl:if test="$dummy !='' and $dummy != 'C.10.4'">
                    <!-- infinite recursion in C.10.4 -->
                    <!--xsl:message>reference found: <xsl:value-of select="$dummy"/></xsl:message-->
                    <!--
Here is how you would get to the article and extract the section specified:
-->
                    <xsl:call-template name="extract-section-paragraphs">
                      <xsl:with-param name="article" select="../../../.."/>
                      <xsl:with-param name="extractsection" select="$dummy"/>
                    </xsl:call-template>
                  </xsl:if>
                  <!--xsl:variable name="section" />
                    <xsl:apply-templates select="entry" mode="iod2"/>
                  </xsl:variable-->
                </entry>
              </xsl:when>
              <xsl:otherwise>
                <xsl:message>SHOULD NOT HAPPEN</xsl:message>
                <!--include ref="{translate($name_translate,'','µ')}" type="{normalize-space($type)}"/-->
              </xsl:otherwise>
            </xsl:choose>
          </xsl:when>
          <xsl:otherwise>
            <entry group="{$group}" element="{$element}" name="{$name_translate}">
              <!-- type ?? -->
              <!-- very specific -->
              <xsl:variable name="desc" select="translate($type,$single_quote1,$single_quote2)"/>
              <xsl:variable name="n_description" select="my:normalize-paragraph($desc)"/>
              <description>
                <xsl:value-of select="$n_description"/>
              </description>
              <xsl:variable name="dummy">
                <xsl:call-template name="get-description-reference">
                  <xsl:with-param name="description" select="$n_description"/>
                </xsl:call-template>
              </xsl:variable>
              <xsl:if test="$dummy !=''">
                <xsl:call-template name="extract-section-paragraphs">
                  <xsl:with-param name="article" select="../../../.."/>
                  <xsl:with-param name="extractsection" select="$dummy"/>
                </xsl:call-template>
              </xsl:if>
            </entry>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:when>
      <xsl:when test="$name = 'Attribute Name' or $name = 'Attribute name' or $name = 'Key'">
        <!-- This is supposed to be the very first line of each table -->
        <!-- someday I might add more stuff here... -->
      </xsl:when>
      <xsl:otherwise>
        <!-- I should check if this is indeed a Include line or not... -->
        <xsl:choose>
          <xsl:when test="entry[1]/@namest = 'c1' and entry[1]/@nameend = 'c2'">
            <xsl:choose>
              <xsl:when test="entry[2]/@namest = 'c3' and entry[2]/@nameend = 'c4'">
                <xsl:variable name="include" select="normalize-space(translate(translate(entry[1],'‘',$apos),'’',$apos))"/>
                <xsl:variable name="description" select="normalize-space(entry[2])"/>
                <include ref="{$include}">
                  <xsl:if test="$description != ''">
                    <xsl:attribute name="description" select="$description"/>
                  </xsl:if>
                </include>
              </xsl:when>
              <xsl:otherwise>
                <xsl:choose>
                  <xsl:when test="count(entry) = 3">
                    <xsl:variable name="include" select="normalize-space(translate(translate(entry[1],'‘',$apos),'’',$apos))"/>
                    <xsl:variable name="type" select="normalize-space(entry[2])"/>
                    <xsl:variable name="description" select="normalize-space(entry[3])"/>
                    <include ref="{$include}">
                      <xsl:if test="$type!= ''">
                        <xsl:attribute name="type" select="$type"/>
                      </xsl:if>
                      <xsl:if test="$description != ''">
                        <xsl:attribute name="description" select="$description"/>
                      </xsl:if>
                    </include>
                  </xsl:when>
                  <xsl:when test="count(entry) = 2">
                    <xsl:variable name="include" select="normalize-space(translate(translate(entry[1],'‘',$apos),'’',$apos))"/>
                    <xsl:variable name="description" select="normalize-space(entry[2])"/>
                    <include ref="{$include}">
                      <xsl:if test="$description != ''">
                        <xsl:attribute name="description" select="$description"/>
                      </xsl:if>
                    </include>
                  </xsl:when>
                  <xsl:otherwise>
                    <include ref="UNHANDLED"/>
                  </xsl:otherwise>
                </xsl:choose>
              </xsl:otherwise>
            </xsl:choose>
          </xsl:when>
          <xsl:when test="entry[1]/@namest = 'c1' and entry[1]/@nameend = 'c3'">
            <xsl:variable name="include" select="normalize-space(translate(translate(entry[1],'‘',$apos),'’',$apos))"/>
            <xsl:variable name="description" select="normalize-space(entry[2])"/>
            <include ref="{$include}">
              <xsl:if test="$description != ''">
                <xsl:attribute name="description" select="$description"/>
              </xsl:if>
            </include>
          </xsl:when>
          <xsl:when test="entry[1]/@namest = 'c1' and entry[1]/@nameend = 'c4'">
            <xsl:variable name="include" select="normalize-space(translate(translate(entry[1],'‘',$apos),'’',$apos))"/>
            <include ref="{$include}"/>
          </xsl:when>
          <xsl:when test="entry[1]/@namest = 'c1' and entry[1]/@nameend = 'c5'">
            <xsl:variable name="include" select="normalize-space(translate(translate(entry[1],'‘',$apos),'’',$apos))"/>
            <include ref="{$include}"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:variable name="include" select="normalize-space(string-join(entry,' '))"/>
            <!-- Table Table C.10-9 Waveform Module Attributes has two empty lines ... -->
            <xsl:if test="$include != ''">
              <!--include ref="{$include}"/-->
              <xsl:variable name="include" select="normalize-space(translate(translate(entry[1],'‘',$apos),'’',$apos))"/>
              <!-- nothing in entry[2] -->
              <!-- FIXME I should check that -->
              <xsl:variable name="type" select="normalize-space(entry[3])"/>
              <xsl:variable name="description" select="normalize-space(entry[4])"/>
              <include ref="{$include}">
                <xsl:if test="$type!= ''">
                  <xsl:attribute name="type" select="$type"/>
                </xsl:if>
                <xsl:if test="$description != ''">
                  <xsl:attribute name="description" select="$description"/>
                </xsl:if>
              </include>
            </xsl:if>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
  <!--

Function to parse an entry from a row in an IOD table
Take the ie name as input

-->
  <xsl:template match="entry" mode="iod">
    <xsl:param name="ie_name"/>
    <xsl:for-each select="entry">
      <xsl:if test="(position() mod 3 = 1)">
        <xsl:variable name="usage" select="translate(normalize-space(following-sibling::entry[2]/para),'– ','- ')"/>
        <xsl:variable name="usage_required" select="my:normalize-space(replace($usage,'required','Required'))"/>
        <entry ie="{$ie_name}" name="{normalize-space(para)}" ref="{normalize-space(following-sibling::entry[1]/para)}" usage="{$usage_required}"/>
      </xsl:if>
    </xsl:for-each>
  </xsl:template>
  <!--


-->
  <xsl:template match="entry" mode="iod2">
    <xsl:for-each select="entry">
      <xsl:variable name="usage" select="translate(entry[3]/para,'– ','- ')"/>
      <xsl:variable name="usage_required" select="my:normalize-space(replace($usage,'required','Required'))"/>
      <entry ie="{normalize-space(para)}" name="{normalize-space(following-sibling::entry[1]/para)}" ref="{normalize-space(following-sibling::entry[2]/para)}" usage="{$usage_required}"/>
    </xsl:for-each>
  </xsl:template>
  <!--

Function to parse a row from an informaltable specifically for an IOD table:
For instance:
Table A.2-1 (CR Image IOD Modules) to A.51-1 (Segmentation IOD Modules).
-->
  <xsl:template match="row" mode="iod">
    <xsl:choose>
      <!--
Some tables have a specific layout, when namest='c2' and nameend='c4', deals with them properly
as they do not repeat the ie name each time:
-->
      <xsl:when test="entry[2]/@namest = 'c2'">
        <xsl:apply-templates select="entry" mode="iod">
          <xsl:with-param name="ie_name" select="entry[1]/para"/>
        </xsl:apply-templates>
      </xsl:when>
      <xsl:when test="entry[5]/para = 'Module Description'">
        <xsl:apply-templates select="entry" mode="iod2"/>
      </xsl:when>
      <!-- Get rid of the first line in the table: IE / Reference / Usage / ... -->
      <xsl:when test="entry[1]/para = 'IE' or entry[1]/para = 'Module'">
      </xsl:when>
      <!--
Most of the IE table simply have an empty entry[1]/para to avoid duplicating the ie name
over and over. We need to get the last ie name we found to fill in the blank:
-->
      <xsl:otherwise>
        <xsl:variable name="ref_joined">
          <xsl:value-of select="entry[3]/para" separator=" "/>
          <!-- actually space is the default separator for value-of -->
        </xsl:variable>
        <xsl:variable name="usage_joined">
          <xsl:value-of select="entry[4]/para" separator=" "/>
        </xsl:variable>
        <xsl:variable name="usage" select="normalize-space(translate($usage_joined,'–','-'))"/>
        <xsl:variable name="usage_required" select="my:normalize-space(replace($usage,'required','Required'))"/>
        <xsl:variable name="ie" select="normalize-space((entry[1]/para[. != ''] , reverse(preceding-sibling::row/entry[1]/para[. != ''])[1])[1])"/>
        <xsl:choose>
          <xsl:when test="count(entry) = 4">
            <xsl:variable name="iefixed" select="normalize-space((entry[1]/para[. != ''] , reverse(preceding-sibling::row[count(entry) = 4]/entry[1]/para[. != ''])[1])[1])"/>
            <entry ie="{$iefixed}" name="{normalize-space(translate(entry[2]/para,'­',''))}" ref="{normalize-space($ref_joined)}" usage="{$usage_required}"/>
          </xsl:when>
          <xsl:when test="count(entry) = 3">
            <xsl:if test="entry[2]/para != ''">
              <xsl:variable name="basic_film" select="normalize-space(translate($ref_joined,'­',''))"/>
              <xsl:choose>
              <xsl:when test="starts-with($basic_film, 'Contains') or starts-with($basic_film, 'References') or starts-with($basic_film, 'Includes') or starts-with($basic_film, 'Identifies')">
                <entry name="{translate($ie,'­','')}" ref="{normalize-space(entry[2]/para)}" description="{normalize-space(translate($ref_joined,'­',''))}"/>
              </xsl:when>
              <xsl:when test="starts-with($basic_film, 'C-') or starts-with($basic_film, 'C -') or $basic_film = 'U' or $basic_film = 'M'">
                <!--xsl:variable name="ie_prev" select="normalize-space(reverse(preceding-sibling::row/entry[1]/para[. != ''])[1])"/-->
                <xsl:variable name="ie_prev" select="preceding-sibling::row[count(entry) = 4 and not(entry/@morerows = '')][1]/entry[1]"/>
                <entry ie="{$ie_prev}" name="{translate($ie,'­','')}" ref="{normalize-space(entry[2]/para)}" usage="{normalize-space(translate($ref_joined,'­',''))}"/>
                <!--xsl:message>Error: FIXME <xsl:value-of select="preceding-sibling::row/entry[1]/para[. != '']"/></xsl:message-->
              </xsl:when>
              <xsl:otherwise>
                <xsl:message>Error: could not MATCH <xsl:value-of select="$basic_film"/></xsl:message>
              </xsl:otherwise>
              </xsl:choose>
            </xsl:if>
          </xsl:when>
          <!-- Table B.18.2 IOD Modules -->
          <xsl:when test="count(entry) = 2">
            <entry name="{$ie}" ref="{normalize-space(entry[2]/para)}"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:message>UNHANDLED</xsl:message>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
  <!--


-->
  <!-- Table C.2-1PATIENT RELATIONSHIP MODULE ATTRIBUTES -->
  <!-- function to extract the table ref (ie: Table C.2-1) -->
  <xsl:template name="get-table-reference">
    <xsl:param name="reference"/>
    <xsl:param name="table_name"/>
    <xsl:variable name="title">
      <xsl:choose>
        <!-- need to do it first, since most of the time $reference is busted and contains garbage misleading us... -->
        <xsl:when test="substring($table_name,1,5) = 'Table'">
          <xsl:value-of select="$table_name"/>
        </xsl:when>
        <xsl:when test="substring($reference,1,5) = 'Table'">
          <xsl:value-of select="$reference"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:text>NO TABLE REF</xsl:text>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <xsl:analyze-string select="$title" regex="Table ([ABFC.]*[0-9a-z\.\-]+)\s*(.*)">
      <xsl:matching-substring>
        <xsl:value-of select="regex-group(1)"/>
      </xsl:matching-substring>
      <xsl:non-matching-substring>
        <xsl:text>ERROR: </xsl:text>
        <xsl:value-of select="$title"/>
      </xsl:non-matching-substring>
    </xsl:analyze-string>
  </xsl:template>
  <!-- function to extract the table ref (ie: Table C.2-1) -->
  <!-- Table C.7-12b -->
  <xsl:variable name="myregex">^([CF]\.[0-9a-z\.]+)\s*(.*)$</xsl:variable>
  <!-- extract a See C.X.Y from a description string -->
  <xsl:template name="get-description-reference">
    <xsl:param name="description"/>
    <!--
<para>See Section C.7.6.4b.1.</para>
-->
    <xsl:variable name="regex2">See Section ([C]\.[0-9\.]+)</xsl:variable>
    <xsl:variable name="regex3">See ([C]\.[0-9a-f\.]+)</xsl:variable>
    <xsl:choose>
      <xsl:when test="matches($description, $regex2)">
        <xsl:analyze-string select="$description" regex="{$regex2}">
          <xsl:matching-substring>
            <match>
              <xsl:value-of select="my:remove-trailing-dot(regex-group(1))"/>
            </match>
          </xsl:matching-substring>
        </xsl:analyze-string>
      </xsl:when>
      <xsl:when test="matches($description, $regex3) and not(matches($description,$regex2))">
        <xsl:analyze-string select="$description" regex="{$regex3}">
          <xsl:matching-substring>
            <match>
              <xsl:value-of select="my:remove-trailing-dot(regex-group(1))"/>
            </match>
          </xsl:matching-substring>
        </xsl:analyze-string>
      </xsl:when>
    </xsl:choose>
  </xsl:template>
  <xsl:template name="get-section-reference">
    <xsl:param name="article"/>
    <xsl:param name="n"/>
    <xsl:variable name="para" select="preceding::para[$n]"/>
    <xsl:choose>
      <xsl:when test="$n &gt; 100">
        <!-- C.8.4.8	NM Multi-frame Module  -->
        <xsl:value-of select="'SECTION ERROR'"/>
      </xsl:when>
      <xsl:when test="matches($para, $myregex)">
        <xsl:analyze-string select="$para" regex="{$myregex}">
          <xsl:matching-substring>
            <match>
              <xsl:value-of select="regex-group(1)"/>
            </match>
          </xsl:matching-substring>
          <!-- no need to non matching-substring case -->
        </xsl:analyze-string>
      </xsl:when>
      <xsl:otherwise>
        <xsl:call-template name="get-section-reference">
          <xsl:with-param name="article" select="."/>
          <xsl:with-param name="n" select="$n+1"/>
        </xsl:call-template>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
  <!--


-->
  <!-- function to extract the table ref (ie: PATIENT RELATIONSHIP MODULE ATTRIBUTES) -->
  <xsl:template name="get-table-name">
    <xsl:param name="reference"/>
    <xsl:param name="table_name"/>
    <xsl:variable name="ret">
      <xsl:analyze-string select="$table_name" regex="(Table [ABFC.]*[0-9a-z\.\-]+)\s*(.+)">
        <xsl:matching-substring>
          <xsl:value-of select="regex-group(2)"/>
        </xsl:matching-substring>
        <xsl:non-matching-substring>
          <xsl:value-of select="$table_name"/>
        </xsl:non-matching-substring>
      </xsl:analyze-string>
    </xsl:variable>
    <xsl:variable name="garbage">—</xsl:variable>
    <xsl:variable name="clean">
      <xsl:value-of select="translate($ret,$garbage,' ')"/>
    </xsl:variable>
    <xsl:call-template name="removedash">
      <xsl:with-param name="text" select="$clean"/>
    </xsl:call-template>
  </xsl:template>
  <!--

Function to remove the dash from a text:
-->
  <xsl:template name="removedash">
    <xsl:param name="text"/>
    <xsl:choose>
      <xsl:when test="starts-with($text, '-')">
        <xsl:value-of select="normalize-space(substring-after($text, '-'))"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="normalize-space($text)"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
  <!--


-->
  <xsl:template match="informaltable">
    <!--xsl:for-each select="//informaltable"-->
    <xsl:variable name="table_ref_raw" select="preceding::para[2]"/>
    <!-- might contain the Table ref or not ... -->
    <xsl:variable name="table_name_raw" select="preceding::para[1]"/>
    <xsl:variable name="section_ref">
      <xsl:call-template name="get-section-reference">
        <xsl:with-param name="article" select="."/>
        <xsl:with-param name="n" select="1"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="table_ref">
      <xsl:call-template name="get-table-reference">
        <xsl:with-param name="reference" select="normalize-space($table_ref_raw)"/>
        <xsl:with-param name="table_name" select="normalize-space($table_name_raw)"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="table_name">
      <xsl:call-template name="get-table-name">
        <xsl:with-param name="reference" select="normalize-space($table_ref_raw)"/>
        <xsl:with-param name="table_name" select="normalize-space($table_name_raw)"/>
      </xsl:call-template>
    </xsl:variable>
    <!-- most of the time it should be equal to 4: -->
    <xsl:variable name="tgroup_cols" select="tgroup/@cols"/>
    <!--xsl:for-each select="tgroup/thead"-->
    <!--xsl:for-each select="tgroup/thead|tgroup/tbody"-->
    <xsl:variable name="attribute_name_head" select="normalize-space(string-join(tgroup/thead/row[1]/entry[1]/para,' '))"/>
    <xsl:for-each select="tgroup/tbody">
      <xsl:variable name="attribute_name" select="normalize-space(string-join(row[1]/entry[1]/para,' '))"/>
      <xsl:choose>
        <!--xsl:when test="$attribute_name = 'Attribute Name' or $attribute_name = 'Attribute name' or (contains($table_name,'MACRO') and ends-with($table_name,'ATTRIBUTES') and not(contains($table_name,'Module')) )"-->
        <xsl:when test="contains(my:camel-case($table_name),'Macro')">
          <!-- macro are referenced by table idx -->
          <macro table="{$table_ref}" name="{my:camel-case($table_name)}">
            <xsl:apply-templates select="row" mode="macro"/>
          </macro>
        </xsl:when>
        <xsl:when test="($attribute_name_head = 'Attribute Name' or $attribute_name = 'Attribute Name' or $attribute_name = 'Attribute name' or $attribute_name = 'Key') and (contains(my:camel-case($table_name),'Module') or ends-with(my:camel-case($table_name),'Keys'))">
          <!-- module are referenced by section idx -->
          <module ref="{$section_ref}" table="{$table_ref}" name="{my:camel-case($table_name)}">
            <xsl:apply-templates select="row" mode="macro"/>
          </module>
        </xsl:when>
        <!--
Table A.2-1 (CR Image IOD Modules) to A.51-1 (Segmentation IOD Modules).
-->
        <xsl:when test="$attribute_name = 'IE' or $attribute_name = 'Module' or contains($table_name,'IOD')">
          <!-- I think we do not need the section number for iod -->
          <iod table="{$table_ref}" name="{my:camel-case($table_name)}">
            <xsl:apply-templates select="row" mode="iod"/>
          </iod>
        </xsl:when>
        <xsl:otherwise>
          <xsl:message>
            <xsl:text>
NOT IOD/Macro or Module ref=</xsl:text>
            <xsl:value-of select="$table_ref_raw"/>
            <xsl:text>
name=</xsl:text>
            <xsl:value-of select="$table_name_raw"/>
            <xsl:text>
att name=</xsl:text>
            <xsl:value-of select="$attribute_name"/>
          </xsl:message>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
    <!--/xsl:for-each-->
  </xsl:template>
  <xsl:template name="extract-section-paragraphs">
    <xsl:param name="article"/>
    <xsl:param name="extractsection"/>
    <xsl:variable name="extract-section" select="$extractsection"/>
    <xsl:variable name="section-number" select="concat($extract-section,' ')"/>
    <xsl:variable name="section-anchor" select="$article/para[starts-with(my:normalize-space(.),$section-number)]"/>
    <xsl:variable name="section-name" select="substring-after(para[starts-with(my:normalize-space(.),$section-number)],$extract-section)"/>
    <!--xsl:message>
<xsl:value-of select="$article/para[1]"/>
</xsl:message-->
    <xsl:choose>
      <xsl:when test="count($section-anchor)=1">
        <xsl:message>Info: section <xsl:value-of select="$extract-section"/> found</xsl:message>
        <xsl:element name="section">
          <xsl:attribute name="ref" select="$extract-section"/>
          <xsl:attribute name="name" select="normalize-space($section-name)"/>
          <xsl:call-template name="copy-section-paragraphs">
            <xsl:with-param name="section-paragraphs" select="$section-anchor/following-sibling::*"/>
          </xsl:call-template>
        </xsl:element>
        <xsl:message>Info: all paragraphs extracted</xsl:message>
      </xsl:when>
      <xsl:when test="count($section-anchor)&gt;1">
        <xsl:message>Error: section <xsl:value-of select="$extract-section"/> found multiple times!</xsl:message>
      </xsl:when>
      <xsl:otherwise>
        <xsl:message>Error: section <xsl:value-of select="$extract-section"/> not found!</xsl:message>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>
  <!-- TEMPLATES -->
  <xsl:template match="para" mode="extract">
    <xsl:value-of select="concat(.,'&#10;')"/>
  </xsl:template>
  <!-- TODO need work on tables to parse defined terms / enumerated-->
  <xsl:template match="informaltable" mode="new">
    <xsl:param name="entry"/>
    <!-- iterate over all rows -->
    <xsl:for-each select="tgroup/tbody/row/entry">
      <!-- output define term and description -->
      <!-- FIXME this is difficult if not impossible to deal both with:
<para>C.7.3.1.1.1	Modality</para>
and
<para>C.8.3.1.1.1	Image Type</para>

FIXME:
See C.8.7.10 and C.8.15.3.9 ... reference a complete module instead of directly defined terms... pffff
-->
      <!-- output defined term only -->
      <xsl:element name="{$entry}">
        <xsl:attribute name="value" select="para"/>
      </xsl:element>
      <!--xsl:value-of select="para"/-->
      <!-- output newline -->
      <!--xsl:if test="not(position()=last())">
        <xsl:value-of select="'&#10;'"/>
      </xsl:if-->
    </xsl:for-each>
  </xsl:template>
  <xsl:template match="informaltable" mode="old">
    <xsl:param name="entry"/>
    <!-- iterate over all rows -->
    <xsl:for-each select="tgroup/tbody/row">
      <xsl:choose>
        <!-- output define term and description -->
        <!-- FIXME this is difficult if not impossible to deal both with:
<para>C.7.3.1.1.1	Modality</para>
and
<para>C.8.3.1.1.1	Image Type</para>

FIXME:
See C.8.7.10 and C.8.15.3.9 ... reference a complete module instead of directly defined terms... pffff
-->
        <xsl:when test="count(entry)&gt;1 and string(entry[2])">
          <xsl:variable name="dummy">
            <xsl:value-of select="concat(entry[1]/para[1],' ')"/>
            <xsl:if test="not(matches(entry[1]/para[1],'= *$') or matches(entry[2]/para[1],'^ *='))">
              <xsl:value-of select="'= '"/>
            </xsl:if>
            <xsl:value-of select="entry[2]/para[1]"/>
          </xsl:variable>
          <xsl:analyze-string select="$dummy" regex="(.*)=(.*)">
            <xsl:matching-substring>
              <xsl:element name="{$entry}">
                <xsl:attribute name="value" select="normalize-space(regex-group(1))"/>
                <xsl:attribute name="meaning" select="normalize-space(regex-group(2))"/>
              </xsl:element>
            </xsl:matching-substring>
            <xsl:non-matching-substring>
              <!--impossible-happen/-->
            </xsl:non-matching-substring>
          </xsl:analyze-string>
        </xsl:when>
        <!-- output defined term only -->
        <xsl:otherwise>
          <xsl:element name="{$entry}">
            <xsl:attribute name="value" select="entry[1]/para[1]"/>
          </xsl:element>
          <!--xsl:value-of select="entry[1]/para[1]"/-->
        </xsl:otherwise>
      </xsl:choose>
      <!-- output newline -->
      <!--xsl:if test="not(position()=last())">
        <xsl:value-of select="'&#10;'"/>
      </xsl:if-->
    </xsl:for-each>
  </xsl:template>
  <xsl:template match="informaltable" mode="extract">
    <xsl:variable name="prevpara" select="preceding::para[1]"/>
    <!--xsl:message>
            PREC PARA:<xsl:value-of select="$prevpara"/>
    </xsl:message-->
    <xsl:variable name="tabletype">
      <xsl:choose>
        <xsl:when test="matches($prevpara,'Retired Defined Terms')">
          <xsl:value-of select="'retired-defined-terms'"/>
        </xsl:when>
        <xsl:when test="matches($prevpara,'Defined Terms') and not(matches($prevpara,'Retired'))">
          <xsl:value-of select="'defined-terms'"/>
        </xsl:when>
        <xsl:when test="matches($prevpara,'Enumerated Values')">
          <xsl:value-of select="'enumerated-values'"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="'unrecognized-rows'"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <!--xsl:message>
            PREC PARA TYPE:<xsl:value-of select="$tabletype"/>
    </xsl:message-->
    <!--xsl:variable name="entryname" select=""/-->
    <xsl:element name="{$tabletype}">
      <xsl:choose>
        <xsl:when test="tgroup/colspec">
          <xsl:apply-templates select="." mode="new">
            <xsl:with-param name="entry" select="'term'"/>
          </xsl:apply-templates>
        </xsl:when>
        <xsl:otherwise>
          <xsl:apply-templates select="." mode="old">
            <xsl:with-param name="entry" select="'term'"/>
          </xsl:apply-templates>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:element>
  </xsl:template>
  <xsl:template name="copy-section-paragraphs">
    <xsl:param name="section-paragraphs"/>
    <xsl:variable name="current-paragraph" select="$section-paragraphs[1]"/>
    <!-- search for next section title -->
    <xsl:if test="($current-paragraph[name()='para' or name()='informaltable'])       and not(matches(normalize-space($current-paragraph),'^([A-F]|[1-9]+[0-9ab]?)(\.[1-9ab]?[0-9ab]+)+ '))">
      <xsl:if test="not(starts-with(normalize-space($current-paragraph),'0M8R4KGxG'))">
        <!-- embedded graphics ? -->
        <xsl:apply-templates select="$current-paragraph" mode="extract"/>
        <xsl:call-template name="copy-section-paragraphs">
          <xsl:with-param name="section-paragraphs" select="$section-paragraphs[position()&gt;1]"/>
        </xsl:call-template>
      </xsl:if>
    </xsl:if>
  </xsl:template>
  <!--
  -->
  <xsl:template match="article">
    <xsl:apply-templates select="informaltable"/>
  </xsl:template>
  <!--
  description post processor to extract defined term
  -->
  <xsl:template name="parse-enum">
    <xsl:param name="text"/>
    <enumerated-values>
      <xsl:analyze-string select="$text" regex="\n">
        <xsl:matching-substring>
          <!--do nothing -->
        </xsl:matching-substring>
        <xsl:non-matching-substring>
          <xsl:element name="term">
            <xsl:analyze-string select="." regex="\s*([A-Z0-9]+)\s*=\s*(.+)\s*">
              <xsl:matching-substring>
                <xsl:attribute name="value">
                  <xsl:value-of select="regex-group(1)"/>
                </xsl:attribute>
                <xsl:attribute name="meaning">
                  <xsl:value-of select="regex-group(2)"/>
                </xsl:attribute>
              </xsl:matching-substring>
              <xsl:non-matching-substring>
                <xsl:attribute name="dummy">
                  <xsl:value-of select="'IMPOSSIBLE ENUM'"/>
                </xsl:attribute>
              </xsl:non-matching-substring>
            </xsl:analyze-string>
          </xsl:element>
        </xsl:non-matching-substring>
      </xsl:analyze-string>
    </enumerated-values>
  </xsl:template>
  <!--

-->
  <xsl:template name="parse-defined">
    <xsl:param name="text"/>
    <defined-terms>
      <xsl:analyze-string select="$text" regex="\n">
        <xsl:matching-substring>
          <!--do nothing -->
        </xsl:matching-substring>
        <xsl:non-matching-substring>
          <xsl:element name="term">
            <xsl:analyze-string select="." regex="\s*([A-Z]+)\s*=\s*(.*)\s*">
              <xsl:matching-substring>
                <xsl:attribute name="value">
                  <xsl:value-of select="regex-group(1)"/>
                </xsl:attribute>
                <xsl:attribute name="meaning">
                  <xsl:value-of select="regex-group(2)"/>
                </xsl:attribute>
              </xsl:matching-substring>
              <xsl:non-matching-substring>
                <xsl:attribute name="value">
                  <xsl:analyze-string select="." regex="\s*([A-Z]+)\s*">
                    <xsl:matching-substring>
                      <xsl:value-of select="regex-group(1)"/>
                    </xsl:matching-substring>
                    <xsl:non-matching-substring>
                      <xsl:value-of select="'IMPOSSIBLE DEFINED'"/>
                    </xsl:non-matching-substring>
                  </xsl:analyze-string>
                </xsl:attribute>
              </xsl:non-matching-substring>
            </xsl:analyze-string>
          </xsl:element>
        </xsl:non-matching-substring>
      </xsl:analyze-string>
    </defined-terms>
  </xsl:template>
  <!--

-->
  <xsl:template name="description-extractor">
    <xsl:param name="desc"/>
    <xsl:variable name="evregex">(.*Enumerated [Vv]alue[s]?\s*(are)*\s*:)(.*)</xsl:variable>
    <xsl:variable name="dtregex">(.*Defined [Tt]erm[s]?\s*(are)*\s*:)(.*)</xsl:variable>
    <description>
      <xsl:choose>
        <xsl:when test="matches($desc,$evregex)">
          <xsl:analyze-string select="$desc" regex="{$evregex}" flags="s">
            <xsl:matching-substring>
              <xsl:value-of select="regex-group(1)"/>
              <xsl:call-template name="parse-enum">
                <xsl:with-param name="text" select="regex-group(3)"/>
              </xsl:call-template>
            </xsl:matching-substring>
          </xsl:analyze-string>
        </xsl:when>
        <xsl:when test="matches($desc,$dtregex)">
          <xsl:analyze-string select="$desc" regex="{$dtregex}" flags="s">
            <xsl:matching-substring>
              <xsl:value-of select="regex-group(1)"/>
              <xsl:call-template name="parse-defined">
                <xsl:with-param name="text" select="regex-group(3)"/>
              </xsl:call-template>
            </xsl:matching-substring>
          </xsl:analyze-string>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="$desc"/>
        </xsl:otherwise>
      </xsl:choose>
    </description>
  </xsl:template>
  <!-- main template -->
  <xsl:template match="/">
    <xsl:processing-instruction name="xml-stylesheet">
type="text/xsl" href="ma2html.xsl"
</xsl:processing-instruction>
    <xsl:comment> to produce output use:
$ xsltproc ma2html.xsl ModuleAttributes.xml
    </xsl:comment>
    <xsl:comment>
  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.
</xsl:comment>
    <tables edition="2008">
      <xsl:apply-templates select="article"/>
    </tables>
  </xsl:template>
</xsl:stylesheet>
