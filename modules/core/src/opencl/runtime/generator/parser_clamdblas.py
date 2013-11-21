#!/bin/python
# usage:
#     cat clAmdBlas.h | $0
import sys, re;

from common import remove_comments, getTokens, getParameters, postProcessParameters

try:
    if len(sys.argv) > 1:
        f = open(sys.argv[1], "r")
    else:
        f = sys.stdin
except:
    sys.exit("ERROR. Can't open input file")

fns = []

while True:
    line = f.readline()
    if len(line) == 0:
        break
    assert isinstance(line, str)
    line = line.strip()
    parts = line.split();
    if (line.startswith('clAmd') or line.startswith('cl_') or line == 'void') and len(line.split()) == 1 and line.find('(') == -1:
        fn = {}
        modifiers = []
        ret = []
        calling = []
        i = 0
        while (i < len(parts)):
            if parts[i].startswith('CL_'):
                modifiers.append(parts[i])
            else:
                break
            i += 1
        while (i < len(parts)):
            if not parts[i].startswith('CL_'):
                ret.append(parts[i])
            else:
                break
            i += 1
        while (i < len(parts)):
            calling.append(parts[i])
            i += 1
        fn['modifiers'] = []  # modifiers
        fn['ret'] = ret
        fn['calling'] = calling

        # print 'modifiers='+' '.join(modifiers)
        # print 'ret='+' '.join(type)
        # print 'calling='+' '.join(calling)

        # read block of lines
        line = f.readline()
        while True:
            nl = f.readline()
            nl = nl.strip()
            nl = re.sub(r'\n', r'', nl)
            if len(nl) == 0:
                break;
            line += ' ' + nl

        line = remove_comments(line)

        parts = getTokens(line)

        i = 0;

        name = parts[i]; i += 1;
        fn['name'] = name
        print 'name=' + name

        params = getParameters(i, parts)

        fn['params'] = params
        # print 'params="'+','.join(params)+'"'

        fns.append(fn)

f.close()

print 'Found %d functions' % len(fns)

postProcessParameters(fns)

from pprint import pprint
pprint(fns)

from common import *

ctx = {}
ctx['CLAMDBLAS_REMAP_ORIGIN'] = generateRemapOrigin(fns)
ctx['CLAMDBLAS_REMAP_DYNAMIC'] = generateRemapDynamic(fns)
ctx['CLAMDBLAS_FN_DECLARATIONS'] = generateFnDeclaration(fns)

sys.stdout = open('../../../include/opencv2/ocl/cl_runtime/clamdblas_runtime.hpp', 'w')
ProcessTemplate('template/clamdblas_runtime.hpp.in', ctx)

ctx['CL_FN_ENUMS'] = generateEnums(fns, 'OPENCLAMDBLAS_FN')
ctx['CL_FN_NAMES'] = generateNames(fns, 'openclamdblas_fn')
ctx['CL_FN_DEFINITIONS'] = generateFnDefinition(fns, 'openclamdblas_fn', 'OPENCLAMDBLAS_FN')
ctx['CL_FN_PTRS'] = generatePtrs(fns, 'openclamdblas_fn')
ctx['CL_FN_SWITCH'] = generateTemplates(23, 'openclamdblas_fn', 'openclamdblas_check_fn', '')

sys.stdout = open('../clamdblas_runtime.cpp', 'w')
ProcessTemplate('template/clamdblas_runtime.cpp.in', ctx)
