#!/usr/bin/env python

"""gen_pattern.py
Usage example:
python gen_pattern.py -o out.svg -r 11 -c 8 -T circles -s 20.0 -R 5.0 -u mm -w 216 -h 279
-o, --output - output file (default out.svg)
-r, --rows - pattern rows (default 11)
-c, --columns - pattern columns (default 8)
-T, --type - type of pattern, circles, acircles, checkerboard, radon_checkerboard (default circles)
-s, --square_size - size of squares in pattern (default 20.0)
-R, --radius_rate - circles_radius = square_size/radius_rate (default 5.0)
-u, --units - mm, inches, px, m (default mm)
-w, --page_width - page width in units (default 216)
-h, --page_height - page height in units (default 279)
-a, --page_size - page size (default A4), supersedes -h -w arguments
-m, --markers - list of cells with markers for the radon checkerboard
-H, --help - show help
"""

import argparse

from svgfig import *


class PatternMaker:
    def __init__(self, cols, rows, output, units, square_size, radius_rate, page_width, page_height, markers):
        self.cols = cols
        self.rows = rows
        self.output = output
        self.units = units
        self.square_size = square_size
        self.radius_rate = radius_rate
        self.width = page_width
        self.height = page_height
        self.markers = markers
        self.g = SVG("g")  # the svg group container

    def make_circles_pattern(self):
        spacing = self.square_size
        r = spacing / self.radius_rate
        pattern_width = ((self.cols - 1.0) * spacing) + (2.0 * r)
        pattern_height = ((self.rows - 1.0) * spacing) + (2.0 * r)
        x_spacing = (self.width - pattern_width) / 2.0
        y_spacing = (self.height - pattern_height) / 2.0
        for x in range(0, self.cols):
            for y in range(0, self.rows):
                dot = SVG("circle", cx=(x * spacing) + x_spacing + r,
                          cy=(y * spacing) + y_spacing + r, r=r, fill="black", stroke="none")
                self.g.append(dot)

    def make_acircles_pattern(self):
        spacing = self.square_size
        r = spacing / self.radius_rate
        pattern_width = ((self.cols-1.0) * 2 * spacing) + spacing + (2.0 * r)
        pattern_height = ((self.rows-1.0) * spacing) + (2.0 * r)
        x_spacing = (self.width - pattern_width) / 2.0
        y_spacing = (self.height - pattern_height) / 2.0
        for x in range(0, self.cols):
            for y in range(0, self.rows):
                dot = SVG("circle", cx=(2 * x * spacing) + (y % 2)*spacing + x_spacing + r,
                          cy=(y * spacing) + y_spacing + r, r=r, fill="black", stroke="none")
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

    @staticmethod
    def _make_round_rect(x, y, diam, corners=("right", "right", "right", "right")):
        rad = diam / 2
        cw_point = ((0, 0), (diam, 0), (diam, diam), (0, diam))
        mid_cw_point = ((0, rad), (rad, 0), (diam, rad), (rad, diam))
        res_str = "M{},{} ".format(x + mid_cw_point[0][0], y + mid_cw_point[0][1])
        n = len(cw_point)
        for i in range(n):
            if corners[i] == "right":
                res_str += "L{},{} L{},{} ".format(x + cw_point[i][0], y + cw_point[i][1],
                                                   x + mid_cw_point[(i + 1) % n][0], y + mid_cw_point[(i + 1) % n][1])
            elif corners[i] == "round":
                res_str += "A{},{} 0,0,1 {},{} ".format(rad, rad, x + mid_cw_point[(i + 1) % n][0],
                                                        y + mid_cw_point[(i + 1) % n][1])
            else:
                raise TypeError("unknown corner type")
        return res_str

    def _get_type(self, x, y):
        corners = ["right", "right", "right", "right"]
        is_inside = True
        if x == 0:
            corners[0] = "round"
            corners[3] = "round"
            is_inside = False
        if y == 0:
            corners[0] = "round"
            corners[1] = "round"
            is_inside = False
        if x == self.cols - 1:
            corners[1] = "round"
            corners[2] = "round"
            is_inside = False
        if y == self.rows - 1:
            corners[2] = "round"
            corners[3] = "round"
            is_inside = False
        return corners, is_inside

    def make_radon_checkerboard_pattern(self):
        spacing = self.square_size
        xspacing = (self.width - self.cols * self.square_size) / 2.0
        yspacing = (self.height - self.rows * self.square_size) / 2.0
        for x in range(0, self.cols):
            for y in range(0, self.rows):
                if x % 2 == y % 2:
                    corner_types, is_inside = self._get_type(x, y)
                    if is_inside:
                        square = SVG("rect", x=x * spacing + xspacing, y=y * spacing + yspacing, width=spacing,
                                     height=spacing, fill="black", stroke="none")
                    else:
                        square = SVG("path", d=self._make_round_rect(x * spacing + xspacing, y * spacing + yspacing,
                                                                     spacing, corner_types), fill="black", stroke="none")
                    self.g.append(square)
        if self.markers is not None:
            r = self.square_size * 0.17
            pattern_width = ((self.cols - 1.0) * spacing) + (2.0 * r)
            pattern_height = ((self.rows - 1.0) * spacing) + (2.0 * r)
            x_spacing = (self.width - pattern_width) / 2.0
            y_spacing = (self.height - pattern_height) / 2.0
            for x, y in self.markers:
                color = "black"
                if x % 2 == y % 2:
                    color = "white"
                dot = SVG("circle", cx=(x * spacing) + x_spacing + r,
                          cy=(y * spacing) + y_spacing + r, r=r, fill=color, stroke="none")
                self.g.append(dot)

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
                        choices=["circles", "acircles", "checkerboard", "radon_checkerboard"])
    parser.add_argument("-u", "--units", help="length unit", default="mm", action="store", dest="units",
                        choices=["mm", "inches", "px", "m"])
    parser.add_argument("-s", "--square_size", help="size of squares in pattern", default="20.0", action="store",
                        dest="square_size", type=float)
    parser.add_argument("-R", "--radius_rate", help="circles_radius = square_size/radius_rate", default="5.0",
                        action="store", dest="radius_rate", type=float)
    parser.add_argument("-w", "--page_width", help="page width in units", default=argparse.SUPPRESS, action="store",
                        dest="page_width", type=float)
    parser.add_argument("-h", "--page_height", help="page height in units", default=argparse.SUPPRESS, action="store",
                        dest="page_height", type=float)
    parser.add_argument("-a", "--page_size", help="page size, superseded if -h and -w are set", default="A4",
                        action="store", dest="page_size", choices=["A0", "A1", "A2", "A3", "A4", "A5"])
    parser.add_argument("-m", "--markers", help="list of cells with markers for the radon checkerboard. Marker "
                                                "coordinates as list of numbers: -m 1 2 3 4 means markers in cells "
                                                "[1, 2] and [3, 4]",
                        default=argparse.SUPPRESS, action="store", dest="markers", nargs="+", type=int)
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
    if 'page_width' and 'page_height' in args:
        page_width = args.page_width
        page_height = args.page_height
    else:
        page_size = args.page_size
        # page size dict (ISO standard, mm) for easy lookup. format - size: [width, height]
        page_sizes = {"A0": [840, 1188], "A1": [594, 840], "A2": [420, 594], "A3": [297, 420], "A4": [210, 297],
                      "A5": [148, 210]}
        page_width = page_sizes[page_size][0]
        page_height = page_sizes[page_size][1]
    markers = None
    if p_type == "radon_checkerboard" and "markers" in args:
        if len(args.markers) % 2 == 1:
            raise ValueError("The length of the markers array={} must be even".format(len(args.markers)))
        markers = set()
        for x, y in zip(args.markers[::2], args.markers[1::2]):
            if x in range(0, columns) and y in range(0, rows):
                markers.add((x, y))
            else:
                raise ValueError("The marker {},{} is outside the checkerboard".format(x, y))

    pm = PatternMaker(columns, rows, output, units, square_size, radius_rate, page_width, page_height, markers)
    # dict for easy lookup of pattern type
    mp = {"circles": pm.make_circles_pattern, "acircles": pm.make_acircles_pattern,
          "checkerboard": pm.make_checkerboard_pattern, "radon_checkerboard": pm.make_radon_checkerboard_pattern}
    mp[p_type]()
    # this should save pattern to output
    pm.save()


if __name__ == "__main__":
    main()
