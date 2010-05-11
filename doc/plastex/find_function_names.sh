#!/bin/bash
grep ".. cfunction::" *.rst -h |
python -c "import sys
print \"opencv_function_names = [\"
for line in sys.stdin.readlines():
    fname = line.split()[3].strip(' (')
    bpos = fname.find('(')
    if bpos >= 0:
        fname = fname[:bpos]
    print \"'%s',\" % fname
print \"]\"" > function_names.py
