#!/usr/bin/env python3

"""Generates the code for a sorted unicode range array as used in hb-ot-os2-unicode-ranges.hh
Input is a tab seperated list of unicode ranges from the otspec
(https://docs.microsoft.com/en-us/typography/opentype/spec/os2#ur).
"""

import re
import sys


print ("""static OS2Range _hb_os2_unicode_ranges[] =
{""")

args = sys.argv[1:]
input_file = args[0]

with open (input_file, mode="r", encoding="utf-8") as f:

  all_ranges = []
  current_bit = 0
  while True:
    line = f.readline().strip()
    if not line:
      break
    fields = re.split(r'\t+', line)
    if len(fields) == 3:
      current_bit = fields[0]
      fields = fields[1:]
    elif len(fields) > 3:
      raise Exception("bad input :(.")

    name = fields[0]
    ranges = re.split("-", fields[1])
    if len(ranges) != 2:
      raise Exception("bad input :(.")

    v = tuple((int(ranges[0], 16), int(ranges[1], 16), int(current_bit), name))
    all_ranges.append(v)

all_ranges = sorted(all_ranges, key=lambda t: t[0])

for ranges in all_ranges:
  start = ("0x%X" % ranges[0]).rjust(8)
  end = ("0x%X" % ranges[1]).rjust(8)
  bit = ("%s" % ranges[2]).rjust(3)

  print ("  {%s, %s, %s}, // %s" % (start, end, bit, ranges[3]))

print ("""};""")
