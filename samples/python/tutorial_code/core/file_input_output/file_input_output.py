from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

def help(filename):
    print (
        '''
        {0} shows the usage of the OpenCV serialization functionality. \n\n
        usage:\n
            python3 {0} [output file name] (default outputfile.yml.gz)\n\n
        The output file may be XML (xml), YAML (yml/yaml), or JSON (json).\n
        You can even compress it by specifying this in its extension like xml.gz yaml.gz etc...\n
        With FileStorage you can serialize objects in OpenCV.\n\n
        For example: - create a class and have it serialized\n
                     - use it to read and write matrices.\n
        '''.format(filename)
    )

class MyData:
    A = 97
    X = np.pi
    name = 'mydata1234'

    def __repr__(self):
        s = '{ name = ' + self.name + ', X = ' + str(self.X)
        s = s + ', A = ' +  str(self.A) + '}'
        return s

    ## [inside]
    def write(self, fs, name):
        fs.startWriteStruct(name, cv.FileNode_MAP|cv.FileNode_FLOW)
        fs.write('A', self.A)
        fs.write('X', self.X)
        fs.write('name', self.name)
        fs.endWriteStruct()

    def read(self, node):
        if (not node.empty()):
            self.A = int(node.getNode('A').real())
            self.X = node.getNode('X').real()
            self.name = node.getNode('name').string()
        else:
            self.A = self.X = 0
            self.name = ''
    ## [inside]

def main(argv):
    if len(argv) != 2:
        help(argv[0])
        filename = 'outputfile.yml.gz'
    else :
        filename = argv[1]

    # write
    ## [iomati]
    R = np.eye(3,3)
    T = np.zeros((3,1))
    ## [iomati]
    ## [customIOi]
    m = MyData()
    ## [customIOi]

    ## [open]
    s = cv.FileStorage(filename, cv.FileStorage_WRITE)
    # or:
    # s = cv.FileStorage()
    # s.open(filename, cv.FileStorage_WRITE)
    ## [open]

    ## [writeNum]
    s.write('iterationNr', 100)
    ## [writeNum]

    ## [writeStr]
    s.startWriteStruct('strings', cv.FileNode_SEQ)
    for elem in ['image1.jpg', 'Awesomeness', '../data/baboon.jpg']:
        s.write('', elem)
    s.endWriteStruct()
    ## [writeStr]

    ## [writeMap]
    s.startWriteStruct('Mapping', cv.FileNode_MAP)
    s.write('One', 1)
    s.write('Two', 2)
    s.endWriteStruct()
    ## [writeMap]

    ## [iomatw]
    s.write('R_MAT', R)
    s.write('T_MAT', T)
    ## [iomatw]

    ## [customIOw]
    m.write(s, 'MyData')
    ## [customIOw]
    ## [close]
    s.release()
    ## [close]
    print ('Write operation to file:', filename, 'completed successfully.')

    # read
    print ('\nReading: ')
    s = cv.FileStorage()
    s.open(filename, cv.FileStorage_READ)

    ## [readNum]
    n = s.getNode('iterationNr')
    itNr = int(n.real())
    ## [readNum]
    print (itNr)

    if (not s.isOpened()):
        print ('Failed to open ', filename, file=sys.stderr)
        help(argv[0])
        exit(1)

    ## [readStr]
    n = s.getNode('strings')
    if (not n.isSeq()):
        print ('strings is not a sequence! FAIL', file=sys.stderr)
        exit(1)

    for i in range(n.size()):
        print (n.at(i).string())
    ## [readStr]

    ## [readMap]
    n = s.getNode('Mapping')
    print ('Two',int(n.getNode('Two').real()),'; ')
    print ('One',int(n.getNode('One').real()),'\n')
    ## [readMap]

    ## [iomat]
    R = s.getNode('R_MAT').mat()
    T = s.getNode('T_MAT').mat()
    ## [iomat]
    ## [customIO]
    m.read(s.getNode('MyData'))
    ## [customIO]

    print ('\nR =',R)
    print ('T =',T,'\n')
    print ('MyData =','\n',m,'\n')

    ## [nonexist]
    print ('Attempt to read NonExisting (should initialize the data structure',
            'with its default).')
    m.read(s.getNode('NonExisting'))
    print ('\nNonExisting =','\n',m)
    ## [nonexist]

    print ('\nTip: Open up',filename,'with a text editor to see the serialized data.')

if __name__ == '__main__':
    main(sys.argv)
