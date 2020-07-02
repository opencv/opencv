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
-a, --page_size - page size (default A4), supersedes -h -w arguments
-H, --help - show help
"""

import argparse

from svgfig import *


class PatternMaker:
    def __init__(self, cols, rows, output, units, square_size, radius_rate, page_width, page_height):
        self.cols = cols
        self.rows = rows
        self.output = output
        self.units = units
        self.square_size = square_size
        self.radius_rate = radius_rate
        self.width = page_width
        self.height = page_height
        self.g = SVG("g")  # the svg group container

    def make_circles_pattern(self):
        spacing = self.square_size
        r = spacing / self.radius_rate
        for x in range(1, self.cols + 1):
            for y in range(1, self.rows + 1):
                dot = SVG("circle", cx=x * spacing, cy=y * spacing, r=r, fill="black", stroke="none")
                self.g.append(dot)

    def make_acircles_pattern(self):
        spacing = self.square_size
        r = spacing / self.radius_rate
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                dot = SVG("circle", cx=((j * 2 + i % 2) * spacing) + spacing, cy=self.height - (i * spacing + spacing),
                          r=r, fill="black", stroke="none")
                self.g.append(dot)

    def make_checkerboard_pattern(self):
        spacing = self.square_size
        xspacing = (self.width - self.cols * self.square_size) / 2.0
        yspacing = (self.height - self.rows * self.square_size) / 2.0
        for x in range(0, self.cols):
            for y in range(0, self.rows):
                if x % 2 == y % 2:
                    square = SVG("rect", x=x * spacing + xspacing, y=y * spacing + yspacing, width=spacing,
                                 height=spacing, fill="black", stroke="none")
                    self.g.append(square)

    def save(self):
        c = canvas(self.g, width="%d%s" % (self.width, self.units), height="%d%s" % (self.height, self.units),
                   viewBox="0 0 %d %d" % (self.width, self.height))
        c.save(self.output)


def main():
    # parse command line options
    parser = argparse.ArgumentParser(description="generate camera-calibration pattern", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-o", "--output", help="output file", default="out.svg", action="store", dest="output")
    parser.add_argument("-c", "--columns", help="pattern columns", default="8", action="store", dest="columns",
                        type=int)
    parser.add_argument("-r", "--rows", help="pattern rows", default="11", action="store", dest="rows", type=int)
    parser.add_argument("-T", "--type", help="type of pattern", default="circles", action="store", dest="p_type",
                        choices=["circles", "acircles", "checkerboard"])
    parser.add_argument("-u", "--units", help="length unit", default="mm", action="store", dest="units",
                        choices=["mm", "inches", "px", "m"])
    parser.add_argument("-s", "--square_size", help="size of squares in pattern", default="20.0", action="store",
                        dest="square_size", type=float)
    parser.add_argument("-R", "--radius_rate", help="circles_radius = square_size/radius_rate", default="5.0",
                        action="store", dest="radius_rate", type=float)
    parser.add_argument("-w", "--page_width", help="page width in units", default="216", action="store",
                        dest="page_width", type=int)
    parser.add_argument("-h", "--page_height", help="page height in units", default="279", action="store",
                        dest="page_width", type=int)
    parser.add_argument("-a", "--page_size", help="page size, supersedes -h -w arguments", default="A4", action="store",
                        dest="page_size", choices=["A0", "A1", "A2", "A3", "A4", "A5"])
    args = parser.parse_args()

    show_help = args.show_help
    if show_help:
        parser.print_help()
        return
    output = args.output
    columns = args.columns
    rows = args.rows
    p_type = args.p_type
    units = args.units
    square_size = args.square_size
    radius_rate = args.radius_rate
    page_size = args.page_size
    # page size dict (ISO standard, mm) for easy lookup. format - size: [width, height]
    page_sizes = {"A0": [840, 1188], "A1": [594, 840], "A2": [420, 594], "A3": [297, 420], "A4": [210, 297],
                  "A5": [148, 210]}
    page_width = page_sizes[page_size.upper()][0]
    page_height = page_sizes[page_size.upper()][1]
    pm = PatternMaker(columns, rows, output, units, square_size, radius_rate, page_width, page_height)
    # dict for easy lookup of pattern type
    mp = {"circles": pm.make_circles_pattern, "acircles": pm.make_acircles_pattern,
          "checkerboard": pm.make_checkerboard_pattern}
    mp[p_type]()
    # this should save pattern to output
    pm.save()


if __name__ == "__main__":
    main()
