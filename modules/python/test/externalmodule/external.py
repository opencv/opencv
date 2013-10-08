import unittest, sys

if len(sys.argv) == 2:
  sys.path.append(sys.argv[1])
  sys.argv = [sys.argv[0]]
else:
  print "specify path to compiled module"
  sys.exit(127)

import cv2test, cv2

class TestExternalModule(unittest.TestCase):
  def test_function_call(self):
    img = cv2.imread("../../../../samples/c/lena.jpg", 0)
    self.assertEqual(cv2test.image_height(img), 512)

  def test_method_call(self):
    img      = cv2.imread("../../../../samples/c/lena.jpg", 0)
    test_obj = cv2test.TestKlass(img)
    self.assertEqual(test_obj.image_height(), 512)

if __name__ == '__main__':
  unittest.main()
