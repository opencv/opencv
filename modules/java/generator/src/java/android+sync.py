#!/usr/bin/python

import os
import shutil

for f in os.listdir("."):
    shutil.copyfile(f, os.path.join("../../../../../../modules/java/generator/src/java/", "android+" + f));
