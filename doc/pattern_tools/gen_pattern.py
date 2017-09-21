#!/usr/bin/env python

"""gen_pattern.py
Usage example:
python gen_pattern.py -o out.svg -r 11 -c 8 -T circles -s 20.0 -R 5.0 -u mm -w 216 -h 279
-o, --output - output file (default out.svg)
-r, --rows - pattern rows (default 11)
-c, --columns - pattern columns (default 8)
-T, --type - type of pattern, circles, acircles, checkerboard (default circles)
-s, --square_size - size of squares in pattern (default 20.0)
-R, --radius_rate - circles_radius = square_size/radius_rate (default 5.0)
-u, --units - mm, inches, px, m (default mm)
-w, --page_width - page width in units (default 216)
-h, --page_height - page height in units (default 279)
-a, --page_size - page size (default A4), supercedes -h -w arguments
-H, --help - show help
"""

from svgfig import *

import sys
import getopt

class PatternMaker:
  def __init__(self, cols,rows,output,units,square_size,radius_rate,page_width,page_height):
    self.cols = cols
    self.rows = rows
    self.output = output
    self.units = units
    self.square_size = square_size
    self.radius_rate = radius_rate
    self.width = page_width
    self.height = page_height
    self.g = SVG("g") # the svg group container

  def makeCirclesPattern(self):
    spacing = self.square_size
    r = spacing / self.radius_rate
    for x in range(1,self.cols+1):
      for y in range(1,self.rows+1):
        dot = SVG("circle", cx=x * spacing, cy=y * spacing, r=r, fill="black", stroke="none")
        self.g.append(dot)

  def makeACirclesPattern(self):
    spacing = self.square_size
    r = spacing / self.radius_rate
    for i in range(0,self.rows):
      for j in range(0,self.cols):
        dot = SVG("circle", cx= ((j*2 + i%2)*spacing) + spacing, cy=self.height - (i * spacing + spacing), r=r, fill="black", stroke="none")
        self.g.append(dot)

  def makeCheckerboardPattern(self):
    spacing = self.square_size
    xspacing = (self.width - self.cols * self.square_size) / 2.0
    yspacing = (self.height - self.rows * self.square_size) / 2.0
    for x in range(0,self.cols):
      for y in range(0,self.rows):
        if x%2 == y%2:
          square = SVG("rect", x=x * spacing + xspacing, y=y * spacing + yspacing, width=spacing, height=spacing, fill="black", stroke="none")
          self.g.append(square)

  def save(self):
    c = canvas(self.g,width="%d%s"%(self.width,self.units),height="%d%s"%(self.height,self.units),viewBox="0 0 %d %d"%(self.width,self.height))
    c.save(self.output)


def main():
    # parse command line options, TODO use argparse for better doc
    try:
        opts, args = getopt.getopt(sys.argv[1:], "Ho:c:r:T:u:s:R:w:h:a:", ["help","output=","columns=","rows=",
                                                                      "type=","units=","square_size=","radius_rate=",
                                                                      "page_width=","page_height=", "page_size="])
    except getopt.error as msg:
        print(msg)
        print("for help use --help")
        sys.exit(2)
    output = "out.svg"
    columns = 8
    rows = 11
    p_type = "circles"
    units = "mm"
    square_size = 20.0
    radius_rate = 5.0
    page_size = "A4"
    # page size dict (ISO standard, mm) for easy lookup. format - size: [width, height]
    page_sizes = {"A0": [840, 1188], "A1": [594, 840], "A2": [420, 594], "A3": [297, 420], "A4": [210, 297], "A5": [148, 210]}
    page_width = page_sizes[page_size.upper()][0]
    page_height = page_sizes[page_size.upper()][1]
    # process options
    for o, a in opts:
        if o in ("-H", "--help"):
            print(__doc__)
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
        elif o in ("-R", "--radius_rate"):
            radius_rate = float(a)
        elif o in ("-w", "--page_width"):
            page_width = float(a)
        elif o in ("-h", "--page_height"):
            page_height = float(a)
        elif o in ("-a", "--page_size"):
            units = "mm"
            page_size = a.upper()
            page_width = page_sizes[page_size][0]
            page_height = page_sizes[page_size][1]
    pm = PatternMaker(columns,rows,output,units,square_size,radius_rate,page_width,page_height)
    #dict for easy lookup of pattern type
    mp = {"circles":pm.makeCirclesPattern,"acircles":pm.makeACirclesPattern,"checkerboard":pm.makeCheckerboardPattern}
    mp[p_type]()
    #this should save pattern to output
    pm.save()

if __name__ == "__main__":
    main()
