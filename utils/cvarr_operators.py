#!/usr/bin/python
BINARY_OPERATORS={
    '+':'cvAdd',
    '-':'cvSub', 
    '/':'cvDiv', 
    '*':'cvMul',
    '^':'cvXor',
    '&':'cvAnd',
    '|':'cvOr',
}
SCALAR_OPERATORS={
    '+':'cvAddS',
    '-':'cvSubS', 
    '&':'cvAndS',
    '|':'cvOrS',
    '^':'cvXorS',
}
SCALE_OPERATORS={
    '*':'val',
    '/':'1.0/val'
}
CMP_OPERATORS={
    '==':'CV_CMP_EQ',
    '!=':'CV_CMP_NE',
    '>=':'CV_CMP_GE',
    '>':'CV_CMP_GT',
    '<=':'CV_CMP_LE',
    '<':'CV_CMP_LT',
}
ARR={
    'CvMat':'cvCreateMat(self->rows, self->cols, self->type)',
    'IplImage':'cvCreateImage(cvGetSize(self), self->depth, self->nChannels)'
}
CMP_ARR={
    'CvMat':'cvCreateMat(self->rows, self->cols, CV_8U)',
    'IplImage':'cvCreateImage(cvGetSize(self), IPL_DEPTH_8U, 1)' 
}

def scalar_scale_operator(arr, op, arg):
    print '\t%s * operator %s (double val){' % (arr, op)
    print '\t\t%s * res = %s;' % (arr, ARR[arr] )
    print '\t\tcvScale(self, res, %s);' % arg
    print '\t\treturn res;'
    print '\t}'
    print '\t%s * operator %s (double val){' % (arr, op)
    print '\t\t%s * res = %s;' % (arr, ARR[arr] )
    print '\t\tcvScale(self, res, %s);' % arg
    print '\t\treturn res;'
    print '\t}'

print "/** This file was automatically generated using util/cvarr_operators.py script */"

for arr in ARR:
    print '%%extend %s {' % arr
    for op in BINARY_OPERATORS:
        print '\t%%newobject operator %s;' % (op)
        print '\t%s * operator %s (CvArr * src){' % ( arr, op )
        print '\t\t%s * res = %s;' % ( arr, ARR[arr] )
        print '\t\t%s(self, src, res);' % ( BINARY_OPERATORS[op] )
        print '\t\treturn res;'
        print '\t}'
        print '\t%s * operator %s= (CvArr * src){' % ( arr, op )
        print '\t\t%s(self, src, self);' % ( BINARY_OPERATORS[op] )
        print '\t\treturn self;'
        print '\t}'
    for op in SCALAR_OPERATORS:
        print '\t%%newobject operator %s;' % (op)
        print '\t%s * operator %s (CvScalar val){' % ( arr, op )
        print '\t\t%s * res = %s;' % ( arr, ARR[arr] )
        print '\t\t%s(self, val, res);' % ( SCALAR_OPERATORS[op] )
        print '\t\treturn res;'
        print '\t}'
        print '\t%s * operator %s= (CvScalar val){' % ( arr, op )
        print '\t\t%s(self, val, self);' % ( SCALAR_OPERATORS[op] )
        print '\t\treturn self;'
        print '\t}'
    for op in CMP_OPERATORS:
        print '\t%%newobject operator %s;' % (op)
        print '\t%s * operator %s (CvArr * src){' % ( arr, op )
        print '\t\t%s * res = %s;' % ( arr, CMP_ARR[arr] )
        print '\t\tcvCmp(self, src, res, %s);' % ( CMP_OPERATORS[op] )
        print '\t\treturn res;'
        print '\t}'
        print '\t%s * operator %s (double val){' % ( arr, op )
        print '\t\t%s * res = %s;' % ( arr, CMP_ARR[arr] )
        print '\t\tcvCmpS(self, val, res, %s);' % ( CMP_OPERATORS[op] )
        print '\t\treturn res;'
        print '\t}'

    for op in SCALE_OPERATORS:
        print '\t%%newobject operator %s;' % (op)
        print '\t%s * operator %s (double val){' % (arr, op)
        print '\t\t%s * res = %s;' % (arr, ARR[arr] )
        print '\t\tcvScale(self, res, %s);' % SCALE_OPERATORS[op]
        print '\t\treturn res;'
        print '\t}'
        print '\t%s * operator %s= (double val){' % (arr, op)
        print '\t\tcvScale(self, self, %s);' % SCALE_OPERATORS[op]
        print '\t\treturn self;'
        print '\t}'


    print '} /* extend %s */\n' % arr
