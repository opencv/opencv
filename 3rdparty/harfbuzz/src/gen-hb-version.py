#!/usr/bin/env python3

"This tool is intended to be used from meson"

import os, sys, shutil

if len (sys.argv) < 5:
	sys.exit(__doc__)

version = sys.argv[1]
major, minor, micro = version.split (".")

OUTPUT = sys.argv[2]
CURRENT_SOURCE_DIR = sys.argv[3]
INPUT = sys.argv[4]

with open (INPUT, "r", encoding='utf-8') as template:
	with open (OUTPUT, "wb") as output:
		output.write (template.read ()
			.replace ("@HB_VERSION_MAJOR@", major)
			.replace ("@HB_VERSION_MINOR@", minor)
			.replace ("@HB_VERSION_MICRO@", micro)
			.replace ("@HB_VERSION@", version)
			.encode ())

# copy it also to src/
shutil.copyfile (OUTPUT, os.path.join (CURRENT_SOURCE_DIR, os.path.basename (OUTPUT)))
