#!/usr/bin/env python

from optparse import OptionParser
import glob, sys, os, re

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="output", help="output file name", metavar="FILENAME", default=None)
    (options, args) = parser.parse_args()

    if not options.output:
        sys.stderr.write("Error: output file name is not provided")
        exit(-1)

    files = []
    for arg in args:
        if ("*" in arg) or ("?" in arg):
            files.extend([os.path.abspath(f) for f in glob.glob(arg)])
        else:
            files.append(os.path.abspath(arg))

    html = None
    for f in sorted(files):
        try:
            fobj = open(f)
            if not fobj:
                continue
            text = fobj.read()
            if not html:
                html = text
                continue
            idx1 = text.find("<tbody>") + len("<tbody>")
            idx2 = html.rfind("</tbody>")
            html = html[:idx2] + re.sub(r"[ \t\n\r]+", " ", text[idx1:])
        except:
            pass

    if html:
        idx1 = text.find("<title>") + len("<title>")
        idx2 = html.find("</title>")
        html = html[:idx1] + "OpenCV performance testing report" + html[idx2:]
        open(options.output, "w").write(html)
    else:
        sys.stderr.write("Error: no input data")
        exit(-1)
