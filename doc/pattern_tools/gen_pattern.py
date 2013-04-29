#!/usr/bin/env python

"""gen_pattern.py
To run:
-c 10 -r 12 -o out.svg
-T type of pattern, circles, acircles, checkerboard
-s --square_size size of squares in pattern
-u --units mm, inches, px, m
-w  page width in units
-h  page height in units
"""

from svgfig import *

import sys
import getopt

class PatternMaker:
  def __init__(self, cols,rows,output,units,square_size,page_width,page_height):
    self.cols = cols
    self.rows = rows
    self.output = output
    self.units = units
    self.square_size = square_size
    self.width = page_width
    self.height = page_height
    self.g = SVG("g") # the svg group container
  def makeCirclesPattern(self):
    spacing = self.square_size
    r = spacing / 5.0 #radius is a 5th of the spacing TODO parameterize
    for x in range(1,self.cols+1):
      for y in range(1,self.rows+1):
        dot = SVG("circle", cx=x * spacing, cy=y * spacing, r=r, fill="black")
        self.g.append(dot)

  def makeACirclesPattern(self):
    spacing = self.square_size
    r = spacing / 5.0
    for i in range(0,self.rows):
      for j in range(0,self.cols):
        dot = SVG("circle", cx= ((j*2 + i%2)*spacing) + spacing, cy=self.height - (i * spacing + spacing), r=r, fill="black")
        self.g.append(dot)

  def makeCheckerboardPattern(self):
    spacing = self.square_size
    r = spacing / 5.0
    for x in range(1,self.cols+1):
      for y in range(1,self.rows+1):
        #TODO make a checkerboard pattern
        dot = SVG("circle", cx=x * spacing, cy=y * spacing, r=r, fill="black")
        self.g.append(dot)
  def save(self):
    c = canvas(self.g,width="%d%s"%(self.width,self.units),height="%d%s"%(self.height,self.units),viewBox="0 0 %d %d"%(self.width,self.height))
    c.inkview(self.output)

def makePattern(cols,rows,output,p_type,units,square_size,page_width,page_height):
    width = page_width
    spacing = square_size
    height = page_height
    r = spacing / 5.0
    g = SVG("g") # the svg group container
    for x in range(1,cols+1):
      for y in range(1,rows+1):
        if "circle" in p_type:
          dot = SVG("circle", cx=x * spacing, cy=y * spacing, r=r, fill="black")
        g.append(dot)
    c = canvas(g,width="%d%s"%(width,units),height="%d%s"%(height,units),viewBox="0 0 %d %d"%(width,height))
    c.inkview(output)


def main():
    # parse command line options, TODO use argparse for better doc
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:c:r:T:u:s:w:h:", ["help","output","columns","rows",
                                                                      "type","units","square_size","page_width",
                                                                      "page_height"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    output = "out.svg"
    columns = 8
    rows = 11
    p_type = "circles"
    units = "mm"
    square_size = 20.0
    page_width = 216    #8.5 inches
    page_height = 279   #11 inches
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
        elif o in ("-r", "--rows"):
            rows = int(a)
        elif o in ("-c", "--columns"):
            columns = int(a)
        elif o in ("-o", "--output"):
            output = a
        elif o in ("-T", "--type"):
            p_type = a
        elif o in ("-u", "--units"):
            units = a
        elif o in ("-s", "--square_size"):
            square_size = float(a)
        elif o in ("-w", "--page_width"):
            page_width = float(a)
        elif o in ("-h", "--page_height"):
            page_height = float(a)
    pm = PatternMaker(columns,rows,output,units,square_size,page_width,page_height)
    #dict for easy lookup of pattern type
    mp = {"circles":pm.makeCirclesPattern,"acircles":pm.makeACirclesPattern,"checkerboard":pm.makeCheckerboardPattern}
    mp[p_type]()
    #this should save pattern to output
    pm.save()

if __name__ == "__main__":
    main()
