#!/usr/bin/python

import os

TARGET_PATH = "img"

pipe = os.popen("which dia")
DiaPath = pipe.readline()
DiaPath = DiaPath.strip("\n");
pipe.close()

if ("" == DiaPath):
    print("Error: Dia tool was not found")
    exit(-1)

print("Dia tool: \"%s\"" % DiaPath)

if (not os.path.exists(TARGET_PATH)):
    os.mkdir("img")

for filename in os.listdir("."):
    if ("dia" == filename[-3:]):
        os.system("%s --export %s %s" % (DiaPath, os.path.join(TARGET_PATH, filename[0:len(filename)-4] + ".png"), filename))
