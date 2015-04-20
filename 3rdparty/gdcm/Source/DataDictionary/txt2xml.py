#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re,os

"""
GEMS
This parser parse a table formatted like this:
Attribute Name Tag VR VM
"""
class TextParser:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt  = re.compile("^\s*([A-Za-z0-9&{}=+ «»%;#()./,_:<>-]+)\s+\(?([0-9A-Fa-fn]+),\s?([0-9A-Fa-fyxX]+)\)?\s+([A-Z][A-Z])\s+([0-9Nn-]+)\s*$")
      patt1 = re.compile("^\s*([A-Za-z0-9&{}=+ ;%#\[\]()./,_:<>-]+)\s+\(?([0-9A-Fa-f]+),\s?([0-9A-Fa-fyxX]+)\)?\s+([1-3C]+)\s+([A-Z][A-Z])\s+([0-9Nn-]+)\s*$")
      patt2 = re.compile( "^\s*([Table ]*[A-Z1-9.:-]+)\s+([A-Za-z -]+)\s+\(([A-Z0-9_]+)\)\s*$")
      #patt3 = re.compile( '^\s*Private Creator Identification\s*\((["A-Za-z0-9() ./])\)\s*$' )
      patt3 = re.compile( '^\s*Private Creator Identification\s*\("?(.*)"?\)\)?\s*$' )
      patt4 = re.compile( '^\s*Private Creator Identification\s*([A-Z0-9_]+)\s*$' )
      m = patt.match(line)
      m1 = patt1.match(line)
      m2 = patt2.match(line)
      m3 = patt3.match(line)
      m4 = patt4.match(line)
      #print line
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" name=\"%s\"/>"%(m.group(2),m.group(3),m.group(4),m.group(5),m.group(1).rstrip())
        #dicom = m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + ' ' + m.group(4)
        #print dicom
        outLines.append( dicom )
      elif m1:
        # <entry group="0001" element="0001" vr="LO" vm="1" type="1C"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" type=\"%s\" name=\"%s\"/>"%(m1.group(2),m1.group(3),m1.group(5),m1.group(6),m1.group(4),m1.group(1).rstrip())
        #dicom = m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + ' ' + m.group(4)
        #print dicom
        outLines.append( dicom )
      elif m2:
        # <dict edition="2007" url="http://??" ref="Table A-16" name="Private Creator Identification - Xeleris" owner="GEMS_GENIE_1">
        s = "</dict><dict ref=\"%s\" name=\"%s\" owner=\"%s\">"%(m2.group(1),m2.group(2).rstrip(),m2.group(3))
        s += '\n'
        outLines.append( s )
      elif m3:
        s = "</dict><dict ref=\"%s\" name=\"%s\" owner=\"%s\">"%("??","??",m3.group(1))
        s += '\n'
        outLines.append( s )
      elif m4:
        s = "</dict><dict ref=\"%s\" name=\"%s\" owner=\"%s\">"%("??","??",m4.group(1))
        s += '\n'
        outLines.append( s )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
GEMS
This parser parse a table formatted like this:
Grp Elm VR VM Type Definition
"""
class TextParser2:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^\s*([0-9A-Z]+)\s+([0-9A-Zx]+)\s+([A-Z][A-Z])\s+([1-9SNn-]+)\s+([1-9])\s+([A-Za-z0-9 ()._,/#>-]+)\s*$")
      patt2 = re.compile( "^\s*([A-Z1-9.-]+)\s*([A-Za-z -]+)\s*$")
      m = patt.match(line)
      m2 = patt2.match(line)
      #print line
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" type=\"%s\" name=\"%s\"/>"%(m.group(1),m.group(2),m.group(3),m.group(4),m.group(5),m.group(6).rstrip())
        #dicom = m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + ' ' + m.group(4)
        #print dicom
        outLines.append( dicom )
      elif m2:
        # <dict edition="2007" url="http://??" ref="Table A-16" name="Private Creator Identification - Xeleris" owner="GEMS_GENIE_1">
        s = "<dict ref=\"%s\" name=\"%s\" owner=\"%s\">"%(m2.group(1),m2.group(2).rstrip(),"")
        s += '\n'
        outLines.append( s )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
SIEMENS:
This parser parse a table formatted like this:
Tag Private Owner Code Name VR VM
"""
class TextParser3:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^\s*\(([0-9A-Z]+),([0-9A-Zx]+)\)\s+([A-Za-z0-9./:_ -]+)\s+\|\s+([A-Za-z0-9 ()._,/#>-]+)\s+([A-Z][A-Z]_?O?W?)\s+([0-9n-]+)\s*$")
      patt2 = re.compile( "^\s*([A-Z1-9.-]+)\s*([A-Za-z -]+)\s*$")
      m = patt.match(line)
      m2 = patt2.match(line)
      #print line
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" owner=\"%s\" name=\"%s\"/>"%(m.group(1),m.group(2),m.group(5),m.group(6),m.group(3).rstrip(),m.group(4).rstrip())
        #dicom = m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + ' ' + m.group(4)
        #print dicom
        outLines.append( dicom )
      elif m2:
        # <dict edition="2007" url="http://??" ref="Table A-16" name="Private Creator Identification - Xeleris" owner="GEMS_GENIE_1">
        s = "<dict ref=\"%s\" name=\"%s\" owner=\"%s\">"%(m2.group(1),m2.group(2).rstrip(),"")
        s += '\n'
        outLines.append( s )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
PHILIPS: (see mr91.pdf)
  Diffusion B-Factor                            2001,xx03   VR = FL, VM = 1 Dimension: s/mm2
                                                            Indicates the Diffusion coefficient.
"""
class TextParser4:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^\s*([A-Za-z0-9> -]+)\s+([0-9]+),([0-9A-Fx]+)\s+VR = ([A-Z][A-Z]), VM = ([0-9n-]+)\s+(.*)\s*$")
      patt1 = re.compile("^\s*([A-Za-z0-9()> -]+)\s+([0-9]+),([0-9A-Fx]+)\s+Value Representation = ([A-Z][A-Z]), Multiplicity = ([0-9n-]+)(.*)\s*$")
      patt2 = re.compile("^\s*[STUDYSERIES]+\s+\(([0-9]+),([0-9]+)\)\s+([A-Za-z ]+)\s*$")
      m = patt.match(line)
      m1 = patt1.match(line)
      m2 = patt2.match(line)
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" name=\"%s\"/>"%(m.group(2),m.group(3),m.group(4),m.group(5),m.group(1).rstrip())
        #print dicom
        outLines.append( dicom )
      elif m1:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" name=\"%s\"/>"%(m1.group(2),m1.group(3),m1.group(4),m1.group(5),m1.group(1).rstrip())
        #print dicom
        outLines.append( dicom )
      elif m2:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" name=\"%s\" />"%(m2.group(1),m2.group(2),m2.group(3).rstrip())
        #print dicom
        outLines.append( dicom )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
PHILIPS: (see 453567994381_B.pdf)
                  7053,0010       LO     Private Creator Data element                                                       1
"""
class TextParser5:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^([\s>]*)([0-9]+),([0-9A-Fx]+)\s+([A-Z][A-Z])\s+([A-Za-z0-9.?(,)> -]+)\s+([0-9n-]+)\s*$")
      m = patt.match(line)
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" name=\"%s%s\"/>\n"%(m.group(2),m.group(3),m.group(4),m.group(6),m.group(1).lstrip(),m.group(5).rstrip())
        #print dicom
        outLines.append( dicom )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
PHILIPS: (see 9605_0132RevC.pdf)
                                     Attribute                           Tag          Type       VR        VM
                              ADAC Header Signature                   0019, 0010        3        LO         2
"""
class TextParser6:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^\s*([A-Za-z0-9 #()./,_:>-]+)\s+([0-9A-Z]+),\s?([0-9A-ZxX]+)\s+([1-3C]+)\s+([A-Z][A-Z])\s+([0-9Nn-]+)\s*$")
      m = patt.match(line)
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" type=\"%s\" name=\"%s\" />"%(m.group(2),m.group(3),m.group(5),m.group(6),m.group(4),m.group(1).rstrip())
        #print dicom
        outLines.append( dicom )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
PHILIPS: (see MR_System_R1_5_dcs.pdf
                 Number of PC Directions                    2001,1016      SS       2, USER      -
"""
class TextParser7:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^\s*([A-Za-z0-9'./> -]+)\s+\(?([0-9A-F]+),([0-9A-FxXY]+)\)?\s+([A-Z][A-Z])\s+([1-3C]+)?,?.*\s*$")
      m = patt.match(line)
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" type=\"%s\" name=\"%s\" />"%(m.group(2),m.group(3),m.group(4),m.group(5),m.group(1).rstrip())
        #print dicom
        outLines.append( dicom )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
AGFA
IMPAX object document         (0029,xx00)          OB               1                  Mitra Object Document 1.0
"""
class TextParser8:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^\s*([A-Za-z0-9()> -]+)\s+\(([0-9]+),([0-9A-Fx]+)\)\s+([A-Z][A-Z])\s+([1-9n-]+)\s+([A-Za-z_0-9. ]+)\s*$")
      m = patt.match(line)
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" owner=\"%s\" name=\"%s\" />"%(m.group(2),m.group(3),m.group(4),m.group(5),m.group(6),m.group(1).rstrip())
        #print dicom
        outLines.append( dicom )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
SIEMENS
Parse a diction.pfl file
Pixel Overflow Flag 1 Pixel Overflow,7FE3,SIEMENS MED NM,1B,SS,1
"""
class TextParser9:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^([A-Z0-9a-z()=/:%. -]+),([0-9A-F]+),([A-Za-z0-9. -]+),([0-9A-F][0-9A-F]),([A-Z][A-Z]),([1-9N-]+)$")
      patt1 = re.compile("^[^,]+,([0-9A-F]+),.*$")
      m = patt.match(line)
      m1 = patt1.match(line)
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" vm=\"%s\" owner=\"%s\" name=\"%s\" />"%(m.group(2),m.group(4),m.group(5),m.group(6),m.group(3),m.group(1).rstrip())
        #print dicom
        outLines.append( dicom )
      else:
        #print line
        n = eval( '0x' + m1.group(1) )
        #print m1.group(1)
        if( not (n % 2 == 0) ):
          print n
          print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
Storage.pdf
Attribute Name                 Group     Byte    Type      VR       Attribute Description
"""
class TextParser10:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      patt = re.compile("^\s*([A-Z.a-z -]+[1-2]?)\s+([0-9A-Z]+)\s+([0-9A-Zx]+)\s+([1-3])\s+([A-Z][A-Z])\s+.*$")
      m = patt.match(line)
      #print line
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry group=\"%s\" element=\"%s\" vr=\"%s\" type=\"%s\">"%(m.group(2),m.group(3),m.group(5),m.group(4))
        #dicom = m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + ' ' + m.group(4)
        #print dicom
        dicom += '\n'
        dicom += "<description>%s</description>\n</entry>\n"%m.group(1).rstrip()
        outLines.append( dicom )
      else:
        print line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

"""
S2000_1.5_DCS_public.pdf
Attribute Name                 (Group,Element)    Type      Attribute Description
"""
class TextParser11:
  def __init__(self, inputfilename, outputfilename):
    self._InputFilename = ''
    self._OutputFilename = ''
  def Parse(self):
    infile = file(inputfilename, 'r')
    outLines = []
    for line in infile.readlines():
      #patt = re.compile("^\s*([A-Za-z ]+)\s+\(([0-9A-F]+),([0-9A-F]+)\)\s+([1-3])\s+([A-Za-z0-9=,. ])\s*$")
      patt = re.compile("^\s*([A-Za-z0-9 '/-]+)\s+\(([0-9]+),([0-9A-Fa-fx]+)\)\s+([1-3])\s+.*$")
      m = patt.match(line)
      #print line
      if m:
        # <entry group="0001" element="0001" vr="LO" vm="1" owner="Private Creator"/>
        dicom = "<entry name=\"%s\" group=\"%s\" element=\"%s\" type=\"%s\">"%(m.group(1).rstrip(),m.group(2),m.group(3),m.group(4))
        #dicom = m.group(1) + ' ' + m.group(2) + ' ' + m.group(3) + ' ' + m.group(4)
        #print dicom
        dicom += '\n'
        #dicom += "<description>%s</description>\n</entry>\n"%
        outLines.append( dicom )
      else:
        print "no:",line
      #print self.Reformat(line)
      #outLines.append( self.Reformat(line) + '\n' )
    outfile = file(outputfilename, 'w')
    outfile.writelines( outLines )
    outfile.close()

if __name__ == "__main__":
  argc = len(os.sys.argv )
  if ( argc < 3 ):
    print "Sorry, wrong list of args"
    os.sys.exit(1) #error

  inputfilename = os.sys.argv[1]
  outputfilename = os.sys.argv[2]
  tp = TextParser(inputfilename,outputfilename);
  tp.Parse()
