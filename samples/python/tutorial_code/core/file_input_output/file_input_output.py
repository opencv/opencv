import cv2 as cv
import numpy as np
import sys

def help(filename):
    print (filename, 'shows the usage of the OpenCV serialization functionality.',
            '\n\nusage: ', 
            '\n    python3', filename, 'outputfile.yml.gz',
            '\n\nThe output file may be either in XML, YAML or JSON. You can even compress it',
            '\nby specifying this in its extension like xml.gz yaml.gz etc... With',
            '\nFileStorage you can serialize objects in OpenCV.',
            '\n\nFor example: - create a class and have it serialized',
            '\n             - use it to read and write matrices.\n'
            )

class MyData: 
    A = 97
    X = np.pi
    name = 'mydata1234' 

    def __repr__(self):
        s = '{ name = ' + self.name + ', X = ' + str(self.X)
        s = s + ', A = ' str(self.A) + '}'
        return s

    def write(self, fs):
        fs.write('MyData','{')
        fs.write('A', self.A)
        fs.write('X', self.X)
        fs.write('name', self.name)
        fs.write('MyData','}')

    def read(self, node):
        if (not node.empty()):
            self.A = int(node.getNode('A').real())
            self.X = node.getNode('X').real()
            self.name = node.getNode('name').str()
        else:
            self.A = self.X = 0
            self.name = ''

if __name__ == '__main__':
    if len(sys.argv) != 2:
        help(sys.argv[0])
        exit(1)

    # write
    R = np.eye(3,3)
    T = np.zeros((3,1))
    m = MyData()

    filename = sys.argv[1]

    s = cv.FileStorage(filename, 1)

    s.write('iterationNr', 100)

    s.write('strings', '[')
    s.write('image1.jpg','Awesomeness')
    s.write('../data/baboon.jpg',']')

    s.write ('Mapping', '{')
    s.write ('One', 1)
    s.write ('Two', 2)
    s.write ('Mapping', '}')

    s.write('R', R)
    s.write('T', T)

    m.write(s)
    s.release()
    print ('Write Done.')

    # read
    print ('\nReading: ')
    s = cv.FileStorage()
    s.open(filename, 0)
    
    n = fs.getNode('iterationNr')
    itNr = int(n.real())
    print (itNr)

    if (not fs.isOpened()):
        print ('Failed to open ', filename, file=sys.stderr)
        help(sys.argv[0])
        exit(1)

    n = fs.getNode('strings')
    if (not n.isSeq()):
        print ('strings is not a sequence! FAIL', file=sys.stderr)
        exit(1)

    for i in range(n.size()):
        print (n.at(i))

    n = fs.getNode('Mapping')
    print ('Two',int(n.getNode('Two').real()),'; ')
    print ('One',int(n.getNode('One').real()),'\n')

    R = fs.getNode('R').mat()
    T = fs.getNode('T').mat()
    m.read(fs.getNode('MyData'))

    print ('\nR =',R)
    print ('T =',T,'\n')
    print ('MyData =','\n',m,'\n')

    print ('Attempt to read NonExisting (should initialize the data structure',
            'with its default).')
    m = (MyData)(fs.getNode('MyData'))
    print ('\nNonExisting =','\n',m)

    print ('\nTip: Open up',filename,'with a text editor to see the serialized data.')
    exit(0)
