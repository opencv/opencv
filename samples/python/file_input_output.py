import cv2 as cv
import numpy as np
import sys

def help(filename):
    print (filename, "shows the usage of the OpenCV serialization functionality.",
            "\n\nusage: ", 
            "\n    python3", filename, "outputfile.yml.gz",
            "\n\nThe output file may be either in XML, YAML or JSON. You can even compress it",
            "\nby specifying this in its extension like xml.gz yaml.gz etc... With",
            "\nFileStorage you can serialize objects in OpenCV.",
            "\n\nFor example: - create a class and have it serialized",
            "\n             - use it to read and write matrices.\n"
            )

class MyData: 
    A = 97
    X = np.pi
    name = "mydata1234" 

    def write(self, fs):
        fs.write("A", self.A)
        fs.write("X", self.X)
        fs.write("name", self.name)

    def read(self, node):
        self.A = node.getNode("A")
        self.X = node.getNode("X")
        self.name = node.getNode("name")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        help(sys.argv[0])
        exit(1);

    # write
    R = np.eye(3,3)
    T = np.zeros((3,1))
    m = MyData()

    filename = sys.argv[1]

    s = cv.FileStorage(filename, 1)

    s.write("iterationNr", 100)

    array = np.array(["image1.jpg", "Awesomeness", "../data/baboon.jpg"],dtype='S18')
    print(repr(array))
    # s.write("strings", array)

    mapping = {'One': 1,'Two': 2}
    names = ['id', 'data']
    formats = ['S3', 'int8']
    dtype = dict(names=names, formats=formats)
    n_mapping = np.array(list(mapping.items()),dtype)
    print(repr(n_mapping))
    s.write ("Mapping", n_mapping)

    s.write("R", R)
    s.write("T", T)

    m.write(s)
    s.release()

    # read
    s = cv.FileStorage()
    s.open(filename, 0)
    n = cv.FileNode()
