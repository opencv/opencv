# Copyright 2008 Emanuele Rocca <ema@galliera.it>
# Copyright 2008 Marco De Benedetto <debe@galliera.it>
# Copyright (c) 2006-2011 Mathieu Malaterre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DICOM images conversion and manipulation.

This module uses the OFFIS DICOM toolkit command line utilities.
"""

import os

import Image, ImageEnhance
import StringIO

from lib import *

def calc_size(orig_size, max_size):
    tofloat = lambda x: (float(x[0]), float(x[1]))
    toint = lambda x: (int(x[0]), int(x[1]))

    orig_size = tofloat(orig_size)
    max_size = tofloat(max_size)

    h_ratio = orig_size[1] / max_size[1]
    new_w = orig_size[0] / h_ratio

    if new_w > max_size[0]:
        w_ratio = orig_size[0] / max_size[0]
        new_h = orig_size[1] / w_ratio
        new_size = max_size[0], new_h
    else:
        new_size = new_w, max_size[1]

    return toint(new_size)

class Dicom:

    def __init__(self, filename, format):
        """DICOM filename with full path, format can be either jpeg or png"""
        assert format in ('jpeg', 'png', 'dicom')

        self.format = format
        self.filename = filename
        if format != 'dicom':
            converted = "%s.%s" % (filename, format)

            if not os.path.isfile(converted):

                quality = ""
                if format == 'jpeg':
                    quality = '--compr-quality 100'
                try:
                    #trycmd("gdcm2pnm --write-%s %s --use-window 1 %s %s" % (format, quality, filename, converted))
                    print "gdcm2pnm %s %s" % (filename, converted)
                    trycmd("gdcm2pnm %s %s" % (filename, converted))
                except CmdException:
                    #trycmd("gdcm2pnm --write-%s %s %s %s" % (format, quality, filename, converted))
                    print "gdcm2pnm %s %s" % (filename, converted)
                    trycmd("gdcm2pnm %s %s" % (filename, converted))
                os.unlink(filename)

            self.img = Image.open(converted)

    def dump(self):
        """To be called after PIL modifications"""
        destfile = StringIO.StringIO()
        self.img.save(destfile, self.format.upper())

        destfile.seek(0)
        return destfile.read()

    def contrast(self, windowWidth):
        enhancer = ImageEnhance.Brightness(self.img)
        self.img = enhancer.enhance(float(windowWidth))

    def brightness(self, windowCenter):
        enhancer = ImageEnhance.Contrast(self.img)
        self.img = enhancer.enhance(float(windowCenter))

    def resize(self, rows, columns):
        size = (int(rows), int(columns))
        newsize = calc_size(self.img.size, size)

        algorithm = Image.BICUBIC
        if newsize[0] < self.img.size[0]:
            algorithm = Image.ANTIALIAS

        self.img = self.img.resize(newsize, algorithm)

    def crop(self, left, upper, right, lower):
        self.img = self.img.crop((left, upper, right, lower))

    def raw(self):
        return open(self.filename).read()

if __name__ == "__main__":

    img = Dicom('/var/tmp/1.3.76.13.10010.0.5.74.3996.1224513675.10543/CT.1.2.840.113619.2.81.290.3160.35958.3.9.20081020.270228', 'png')
