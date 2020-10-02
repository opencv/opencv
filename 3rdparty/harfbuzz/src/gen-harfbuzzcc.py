#!/usr/bin/env python3

"This tool is intended to be used from meson"

import os, sys, shutil

if len (sys.argv) < 3:
	sys.exit (__doc__)

OUTPUT = sys.argv[1]
CURRENT_SOURCE_DIR = sys.argv[2]
sources = sys.argv[3:]

with open (OUTPUT, "wb") as f:
	f.write ("".join ('#include "{}"\n'.format (os.path.basename (x)) for x in sources if x.endswith (".cc")).encode ())

# copy it also to src/
shutil.copyfile (OUTPUT, os.path.join (CURRENT_SOURCE_DIR, os.path.basename (OUTPUT)))
