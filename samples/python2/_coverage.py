from glob import glob
import cv2
import re

cv2_callable = set(['cv2.'+name for name in dir(cv2) if callable( getattr(cv2, name) )])

found = set()
for fn in glob('*.py'):
    print ' --- ', fn
    code = open(fn).read()
    found |= set(re.findall('cv2?\.\w+', code))

cv2_used = found & cv2_callable

r = 1.0 * len(cv2_used) / len(cv2_callable)
print '\ncv2 api coverage: %d / %d  (%.1f%%)' % ( len(cv2_used), len(cv2_callable), r*100 )

print '\nold (cv) symbols:'
for s in found:
    if s.startswith('cv.'):
        print s
