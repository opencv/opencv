#!/usr/bin/env python
# vim: set fileencoding=iso-8859-1

"""
$ pdftotext -layout -nopgbrk -f 303 -l 305 07_03pu.pdf page303.txt

$ python  ParseAttributes.py page273-1000.txt out.txt > log2
$ grep ADD log2 | grep -v "Notes:" | grep -v "Note:" | grep -v "C.8" | grep -v "C.7"

"""
import re,os

"""

"""
class Attribute:
  # Cstor
  def __init__(self):
    self._Name = ''
    self._Tag = '(0000,0000)'
    self._Type = ''
    self._Description= ''
  def SetInit(self,s):
    # Should be something like:
    # Blue Palette Color Lookup Table       (0028,1103)     1C   Specifies the format of the Blue Palette
    patt = re.compile("^(.*)(\\([0-9A-Fx]+,[0-9A-F]+\\))\s+([1-3C]+)\s+(.*)\s*$")
    m = patt.match(s)
    if not m:
      print s
      assert 0
    self._Name = m.group(1).strip()
    self._Tag = m.group(2).strip()
    self._Type = m.group(3).strip()
    self._Description = m.group(4).strip()
  def SetName(self,s):
    self._Name = s
  def AppendName(self,s):
    self._Name += " "
    self._Name += s.strip()
  def SetTag(self,s):
    self._Tag = s
  def SetType(self,s):
    self._Type = s
  def SetDescription(self,s):
    self._Description = s
  def AppendDescription(self,s):
    self._Description += " "
    self._Description += s.strip()
  def GetAsXML(self):
    description = self._Description.replace('"','&quot;')
    description = description.replace('&','&amp;')
    return "<entry group=\""+self._Tag[1:5]+"\" element=\""+self._Tag[6:10]+"\" name=\""+self._Name.replace('&','&amp;')+"\" type=\""+self._Type+"\" description=\""+description+"\"></entry>"
  def Print(self):
    print self.GetAsXML()

class Part3Parser:
  # Cstor
  def __init__(self):
    self._InputFilename = ''
    self._OutputFilename = ''
    self._Buffer = ''
    self._CurrentAttribute = Attribute()
    self._IsInTable = False
    self._Shift = 0

  def SetInputFileName(self,s):
    self._InputFilename = s

  def SetOutputFileName(self,s):
    self._OutputFilename = s

  def IsComment(self,s):
    if len(s) == 0:
      return True
    patt1 = re.compile("^\s+- Standard -\s*$")
    patt2 = re.compile("^\s*PS 3.3 - 2007\s*")
    patt3 = re.compile("^\s*Page\s+[0-9]+\s*$")
    patt4 = re.compile("^\s*Notes:$")
    m1 = patt1.match(s)
    m2 = patt2.match(s)
    m3 = patt3.match(s)
    m4 = patt4.match(s)
    if(m1 or m2 or m3 or m4):
      print "Comment:", s
      return True
    if self.IsTableDescription(s):
      return True
    return False

  def IsStartTable(self,s):
    #patt = re.compile("^\s+Table C[0-9a-z\.-]+.*\s+$")
    patt = re.compile("^\s+Table\s+C.[0-9A-Za-z-.]+\s*$")
    m = patt.match(s)
    assert self._IsInTable != True
    self._IsInTable = False
    if s.strip() == 'Table C.7-23' or s.strip() == 'Table C.7-24' \
      or s.strip() == 'Table C.7.6.10-1' \
      or s.strip() == 'Table C.7-25' \
      or s.strip() == 'Table C.7-26' \
      or s.strip() == 'Table C.7-27' \
      or s.strip() == 'Table C.8-8'  \
      or s.strip() == 'Table C.8-19' \
      or s.strip() == 'Table C.8-20' \
      or s.strip() == 'Table C.8-21' \
      or s.strip() == 'Table C.8-22' \
      or s.strip() == 'Table C.8-23' \
      or s.strip() == 'Table C.8-80' \
      or s.strip() == 'Table C.8-83' \
      or s.strip() == 'Table C.8-84' \
      or s.strip() == 'Table C.8-85' \
      or s.strip() == 'Table C.8-86' \
      or s.strip() == 'Table C.8-108' \
      or s.strip() == 'Table C.8-109' \
      or s.strip() == 'Table C.8-110' \
      or s.strip() == 'Table C.8-111' \
      or s.strip() == 'Table C.8-112' \
      or s.strip() == 'Table C.8-115' \
      or s.strip() == 'Table C.8-116' \
      or s.strip() == 'Table C.8-127' \
      or s.strip() == 'Table C.8-128' \
      or s.strip() == 'Table C.8-129' \
      or s.strip() == 'Table C.8-130' \
      or s.strip() == 'Table C.8-131' \
      or s.strip() == 'Table C.8-132' \
      or s.strip() == 'Table C.8-133' \
      or s.strip() == 'Table C.8-134' \
      or s.strip() == 'Table C.8.19.2-2' \
      or s.strip() == 'Table C.10-10' \
      or s.strip() == 'Table C.11-4' \
      or s.strip() == 'Table C.12-2' \
      or s.strip() == 'Table C.12-3' \
      or s.strip() == 'Table C.12-4' \
      or s.strip() == 'Table C.12-5' \
      or s.strip() == 'Table C.12-7' \
      or s.strip() == 'Table C.13-1' \
      or s.strip() == 'Table C.13-2' \
      or s.strip() == 'Table C.13-3' \
      or s.strip() == 'Table C.13-4' \
      or s.strip() == 'Table C.13-5' \
      or s.strip() == 'Table C.13-7' \
      or s.strip() == 'Table C.13-8' \
      or s.strip() == 'Table C.13-9' \
      or s.strip() == 'Table C.13-13' \
      or s.strip() == 'Table C.14-1' \
      or s.strip() == 'Table C.17.3-7' \
      or s.strip() == 'Table C.17.3-8' \
      or s.strip() == 'Table C.22.1-1':
      # C.11-4, C.13-*, C.22.1-1:  Does not even comes with column type !!!
      # C.12-7 is difficult to parse
      # C.7.6.16-1 is insane...
      # TODO: Last line of C.19-1...
      return False
    if(m):
      print "Start", s
      self._IsInTable = True
      return True
    # grrrrr: Table C.8-37 - RT SERIES MODULE ATTRIBUTES
    patt = re.compile("^\s+Table\s+C.[0-9A-Za-z-]+\s*[-]*\s*([A-Z/\s-]+)\s*$")
    #patt = re.compile("^\s+Table\s+C.[0-9A-Za-z-]+[-\s]+([A-Z/\s-]+)\s*$")
    m = patt.match(s)
    if(m):
      print "Start", s
      self._IsInTable = True
      return True
    print "IsTable failed with:", s
    return False

  def IsEndTable(self,s):
    assert self._IsInTable == True
    assert not self.IsComment(s)
    self._IsInTable = False
    return True

  def IsTableName(self,s):
    patt = re.compile("^\s*[A-Z/\s-]+ATTRIBUTES\s*$") #MACRO/MODULE
    m = patt.match(s)
    if(m):
      print "Table Name", s
      return True
    patt = re.compile("^\s+[A-Za-z\s]+Attributes\s*$") #MACRO/MODULE
    m = patt.match(s)
    if(m):
      print "Table Name", s
      return True
    # PALETTE COLOR LOOKUP MODULE
    patt = re.compile("^\s+[A-Z\s]+MODULE\s*$") #MACRO/MODULE
    m = patt.match(s)
    if(m):
      print "Table Name", s
      return True
    # MR IMAGE AND SPECTROSCOPY INSTANCE MACRO
    patt = re.compile("^\s+[A-Z\s]+MACRO\s*$") #MACRO/MODULE
    m = patt.match(s)
    if(m):
      print "Table Name", s
      return True
    # Enhanced XA/XRF Image Module Table
    patt = re.compile("^\s+[A-Z/a-z\s]+Module Table\s*$")
    m = patt.match(s)
    if(m):
      print "Table Name", s
      return True
    # Presentation LUT Module
    #patt = re.compile("^\s+Presentation LUT Module\s*$")
    #m = patt.match(s)
    #if(m):
    #  print "Table Name", s
    #  return True
    print "TableName failed with:", s
    return False

  def IsTableName2(self,s):
    # grrrrr: Table C.8-37 - RT SERIES MODULE ATTRIBUTES
    # Table C.8-39--RT DOSE MODULE ATTRIBUTES
    patt = re.compile("^\s+Table\s+C.[0-9A-Za-z-]+\s*[-]*\s*([A-Z/\s-]+)\s*$")
    m = patt.match(s)
    # The previous regex would think : Table C.7-17A
    # is correct...I don't know how to fix the regex, so discard result if
    # len(m.group(1)) <= 1
    if(m and len(m.group(1)) > 1):
      print "Table Name:", m.group(1)
      assert self.IsTableName( m.group(1) )
      return True
    print "TableName2 failed with:", s
    return False

  def IsTableDescription(self,s):
    patt  = re.compile("^\s*Attribute Name\s+Tag\s+Type\s+Attribute Description\s*$")
    m = patt.match(s)
    if(m):
      print "Table Description:", s
      return True
    # Around page 574
    patt  = re.compile("^\s*Attribute [Nn]ame\s+Tag\s+Type\s+Description\s*$")
    m = patt.match(s)
    if(m):
      print "Table Description:", s
      return True
    return False

  def IsFirstLineAttribute(self,s):
    # Line should look like:
    # Bits Stored ... (0028,0101) ... 1 ... Number of bits stored for each pixel
    patt = re.compile("^\s*(.*)\\([0-9A-Fx]+,[0-9A-F]+\\)\s+([1-3C]+).*\s*$") #MACRO/MODULE
    m = patt.match(s)
    if(m):
      s1 = m.group(1).strip()
      if s1 == '':
        return False
      #print "First Line Attribute:", s1, s
      return True
    #print "No:", s
    return False

  def IsIncludeTable(self,s):
    # Need to support : "Include `Image Pixel Macro' Table C.7-11b"
    #assert self._Shift == 0
    #print "Include:", s
    #patt = re.compile("^>*Include `(.*)' Table [A-Z0-9a-z-.]+$")
    #m = patt.match(s)
    #if m:
    #  return True
    #patt = re.compile("^>*Include [`|'](.*)' Table [A-Z0-9a-z-.]+\s+Defined Context ID is.*$")
    #m = patt.match(s)
    #return m
    #print "FALLBACK"
    patt = re.compile("^>*\s*Include [`'\"]*([A-Za-z/ -]*)['\"]* \\(*Table [A-Z0-9a-z-.]+\\)*.*$")
    m = patt.match(s)
    #if not m:
    #  print "FAIL", s
    return m

  def IsNextLineAttribute(self,s):
    if self._Shift == 0:
      print "IsNextLineAttribute failed with", s
      return False
    if len(s) <= self._Shift:
      print "IsNextLineAttribute failed with", s
      return False
    blank = s[0:self._Shift]
    blank = blank.strip()
    #print "BLANK:", blank
    if blank == '':
      self._CurrentAttribute.AppendDescription( s )
      return True
    # The following is really ugly ... need to be fixed
    if blank == 'Descriptor' or blank == 'Data' or blank == 'Center Name' \
      or blank == 'Description' \
      or blank == 'Sequence' \
      or blank == 'Distance' \
      or blank == 'Index' \
      or blank == 'Reordering' \
      or blank == 'Time' \
      or blank == 'Device Number' \
      or blank == 'Justification' \
      or blank == 'Shape' \
      or blank == 'Relationship' \
      or blank == 'in Float' \
      or blank == 'Displacement' \
      or blank == 'Technique Description' \
      or blank == 'Left Vertical Edge' \
      or blank == 'State Sequence' \
      or blank == 'In-plane' \
      or blank == 'Certification Number' \
      or blank == 'Right Vertical Edge' \
      or blank == 'Accumulated' \
      or blank == 'Equivalent Thickness' \
      or blank == 'Distances' \
      or blank == 'Definition' \
      or blank == 'Upper Horizontal Edge' \
      or blank == 'Modification' \
      or blank == 'Power Ratio' \
      or blank == 'Lower Horizontal Edge' \
      or blank == 'Device Distance' \
      or blank == 'Sensing Region' \
      or blank == 'Control Sensing Region' \
      or blank == 'Water Equivalent Thickness' \
      or blank == 'Columns' \
      or blank == 'Rows' \
      or blank == 'Ratio' \
      or blank == 'Display Grayscale Value' \
      or blank == 'Display CIELab Value' \
      or blank == 'UID' \
      or blank == 'Units' \
      or blank == 'Pointer' \
      or blank == 'Value' \
      or blank == 'Annotation' \
      or blank == 'Pointer Private Creator' \
      or blank == 'Creator' \
      or blank == 'Value Mapping Sequence' \
      or blank == 'Performed Procedure' \
      or blank == 'MAC Sequence' \
      or blank == 'Class UID' \
      or blank == 'Instance UID' \
      or blank == 'Syntax UID' \
      or blank == 'Used' \
      or blank == 'Identifier' \
      or blank == 'Datetime' \
      or blank == 'plane Phase Steps' \
      or blank == '(Patient)' \
      or blank == 'Collection Center' \
      or blank == 'Technique' \
      or blank == 'Interpretation' \
      or blank == 'Representation' \
      or blank == 'Configuration' \
      or blank == 'Compression' \
      or blank == 'Reference Code' \
      or blank == 'Encoding Steps' \
      or blank == 'Steps in-plane' \
      or blank == 'Steps out-of-plane' \
      or blank == 'Type' \
      or blank == 'Explanation' \
      or blank == 'Mapped' \
      or blank == 'Calibration' \
      or blank == 'Manufactured' \
      or blank == 'Thickness' \
      or blank == 'Reference Sequence' \
      or blank == 'Reference Number' \
      or blank == 'Transmission' \
      or blank == 'Matrix' \
      or blank == 'Comment' \
      or blank == 'Setup Sequence' \
      or blank == 'Setup Number' \
      or blank == 'Fraction' \
      or blank == 'Tolerance' \
      or blank == 'Number' \
      or blank == 'Day' \
      or blank == 'Parameters' \
      or blank == 'Coefficient' \
      or blank == 'Specification Point' \
      or blank == 'Identification Sequence' \
      or blank == 'Reference UID' \
      or blank == 'Synchronized' \
      or blank == 'Description Code Sequence' \
      or blank == 'Concentration' \
      or blank == 'Procedure Step' \
      or blank == 'Manufacturer' \
      or blank == 'Lookup Table Data' \
      or blank == 'Version' \
      or blank == 'Images' \
      or blank == 'Wavelength' \
      or blank == 'Code Sequence' \
      or blank == 'Housing' \
      or blank == 'Exposure' \
      or blank == 'Beam' \
      or blank == 'Angle' \
      or blank == 'Rotation Angle' \
      or blank == 'Corner' \
      or blank == 'Factor' \
      or blank == 'Product' \
      or blank == "Manufacturer's Model Name" \
      or blank == 'Qualifier Code' \
      or blank == 'Mapping Instance Sequence' \
      or blank == 'Channels' \
      or blank == 'Samples' \
      or blank == 'Transformation Comment' \
      or blank == 'Pixels' \
      or blank == 'Correction Factor' \
      or blank == 'Group' \
      or blank == 'Amount' \
      or blank == 'Priority' \
      or blank == 'Group Name' \
      or blank == 'Frame Rate' \
      or blank == 'Presence' \
      or blank == 'Sequencing' \
      or blank == 'Orientation' \
      or blank == 'Inverted' \
      or blank == 'Numbers' \
      or blank == 'Flag' \
      or blank == 'Annotation Flag' \
      or blank == 'Demographics Flag' \
      or blank == 'Techniques Flag' \
      or blank == 'Group Description' \
      or blank == 'Handling' \
      or blank == 'Initial View Direction' \
      or blank == 'Identification Code Sequence' \
      or blank == 'Identification Code' \
      or blank == 'Category' \
      or blank == 'Spatial Position' \
      or blank == 'Creation Datetime' \
      or blank == 'Grayscale Bit Depth' \
      or blank == 'Bit Depth' \
      or blank == 'Repaint Time' \
      or blank == 'Definition Sequence' \
      or blank == 'Procedure Code' \
      or blank == 'Referenced' \
      or blank == 'Reference' \
      or blank == 'Usage Flag' \
      or blank == 'Horizontal Dimension' \
      or blank == 'Dimension' \
      or blank == 'Direction' \
      or blank == 'Registration Sequence' \
      or blank == 'Transformation Matrix' \
      or blank == 'Transformation Matrix Type' \
      or blank == 'Step Sequence':
      self._CurrentAttribute.AppendName( blank )
      self._CurrentAttribute.AppendDescription( s[self._Shift:] )
      return True
    else:
      print "ADD KEYWORD:", blank
    return False

  def FindShiftValue(self,s):
    # Line should look like:
    # Bits Stored ... (0028,0101) ... 1 ... Number of bits stored for each pixel
    patt = re.compile("^[A-Za-z0-9µ /()'>-]+\s+\\([0-9A-Fx]+,[0-9A-F]+\\)\s+[1-3][C]*\s+(.*)$")
    m = patt.match(s)
    if(m):
      # worse case happen around page 448 with `Required`
      # worse case happen around page 475 with `LOG`...
      self._Shift = s.find( m.group(1) ) - 17
      return self._Shift
    print "OUCH:", s
    return 0

  def Open(self):
    #self._Infile = file(self._InputFilename, 'r')
    #for line in self._Infile.readlines():
    #  line = line[:-1] # remove '\n'
    #  if( self.IsStartTable(line) ):
    #    print line.next()
    cmd_input = open(self._InputFilename,'r')
    outfile = open(self._OutputFilename, 'w')
    # To support some weird output from pdftotext
    outfile.write( '<?xml version="1.0" encoding="ISO-8859-1"?>' )
    outfile.write( '<tables>' )
    for line_ori in cmd_input:
      #while  line.startswith('%') : # skip comment lines
      #print "!!!",line
      #line= cmd_input.next()
      line = line_ori[:-1]
      if( self.IsStartTable(line) ):
        table_name_found = self.IsTableName2(line)
        line2 = line
        # Okay table is on next line:
        if ( not table_name_found ):
          line2 = cmd_input.next()[:-1]
          table_name_found = self.IsTableName(line2)
        # Either way we need to find the table name
        assert table_name_found
        if( table_name_found ):
          line3 = cmd_input.next()[:-1]
          if( self.IsTableDescription(line3) ):
            # Ok we found a table
            outfile.write(
              "<table ref=\""+line.strip()+"\" name=\""+line2.strip()+"\">"
            )
            buffer = ''
            self._CurrentAttribute = Attribute()
            self._Shift = 0
            for subline_ori in cmd_input:
              subline = subline_ori[:-1]
              if( self.IsIncludeTable(subline)):
                # BUG DO NOT SUPPORT MULTI_LINE INCLUDE
                #print "Include Table:", subline
                if( subline != '' ):
                  outfile.write( "<include ref=\""+\
                    subline.replace('"','&quot;')+"\"/>" )
                  outfile.write( '\n' )
              elif( self.IsFirstLineAttribute(subline)):
                #print "Previous Buffer was: ", buffer
                if( buffer != '' ):
                  outfile.write( self._CurrentAttribute.GetAsXML() )
                  outfile.write( '\n' )
                self._CurrentAttribute.SetInit(subline)
                self.FindShiftValue(subline)
                assert self._Shift != 0
                buffer = subline
              else:
                if( not self.IsComment(subline) ):
                  #print "Found Comment: ", subline
                  if( self.IsNextLineAttribute(subline) ):
                    buffer += ' ' + subline.strip()
                  else:
                    print "Wotsit:", subline
                    self._Shift = 0
                    self._IsInTable = False
                    if( buffer != '' ):
                      outfile.write( self._CurrentAttribute.GetAsXML() )
                      outfile.write( '\n' )
                    outfile.write( '</table>' )
                    break
              #print "Working on: ", subline
              if not subline_ori:
                break
        else:
          print "Problem with:", line, line2
      #line = cmd_input.next()
      if not line_ori: break
    cmd_input.close()
    outfile.write( '</tables>' )
    self.Write()

  def Write(self):
   print "Write"

  # Main function to call for parsing
  def Parse(self):
    self.Open()


if __name__ == "__main__":
  argc = len(os.sys.argv )
  if ( argc < 3 ):
    print "Sorry, wrong list of args"
    os.sys.exit(1) #error

  inputfilename = os.sys.argv[1]
  outputfilename = os.sys.argv[2]
  tempfile = "/tmp/mytemp2"

  dp = Part3Parser()
  dp.SetInputFileName( inputfilename )
  dp.SetOutputFileName( tempfile )
  dp.Parse()
