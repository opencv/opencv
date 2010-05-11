#!/usr/bin/env python
"""
Usage: check_doc.py > log.txt
The script parses different opencv modules
(that are described by instances of class Comp below) and
checks for typical errors in headers and docs, for consistence and for completeness.
Due to its simplicity, it falsely reports some bugs, that should be
just ignored.
"""

import sys, os, re, glob

comps = []

class Comp:
    def __init__(self,comp_name):
        self.name = comp_name

cxcore = Comp('cxcore')
cxcore.header_path = '../cxcore/include'
cxcore.headers = ['cxcore.h','cxtypes.h']
cxcore.ext_macro = 'CVAPI'
cxcore.inline_macro = 'CV_INLINE'
cxcore.func_prefix = 'cv'
cxcore.doc_path = '../docs/ref'
cxcore.docs = ['opencvref_cxcore.htm']
comps.append(cxcore)

cv = Comp('cv')
cv.header_path = '../cv/include'
cv.headers = ['cv.h','cvtypes.h']
cv.ext_macro = 'CVAPI'
cv.inline_macro = 'CV_INLINE'
cv.func_prefix = 'cv'
cv.doc_path = '../docs/ref'
cv.docs = ['opencvref_cv.htm']
comps.append(cv)


highgui = Comp('highgui')
highgui.header_path = '../otherlibs/highgui'
highgui.headers = ['highgui.h']
highgui.ext_macro = 'CVAPI'
highgui.inline_macro = 'CV_INLINE'
highgui.func_prefix = 'cv'
highgui.doc_path = '../docs/ref'
highgui.docs = ['opencvref_highgui.htm']
comps.append(highgui)


def normalize_decl(decl):
    decl = re.sub( r'^\((.+?)\)', r'\1', decl)
    decl = re.sub( r' CV_DEFAULT\((.+?)\)(,|( *\);))', r'=\1\2', decl)
    decl = re.sub( r'\);', r' );', decl )
    decl = re.sub( r'\(', r'( ', decl )
    decl = re.sub( r'/\*.+?\*/', r'', decl )
    decl = re.sub( r'\binline\b', r'', decl )
    decl = re.sub( r' +', r' ', decl )
    decl = re.sub( r' ?= ?', r'=', decl )
    return decl.strip()

def print_report(filename, line_no, msg):
    print '%s(%d): %s' % (filename,line_no,msg)

for comp in comps:
    print "==================================================="
    print 'Checking %s...' % (comp.name,)
    header_path = comp.header_path
    func_list = {}

    if not header_path.endswith('/') and not header_path.endswith('\\'):
        header_path += '/'
    for header_glob in comp.headers:
        glob_expr = header_path + header_glob
        for header in glob.glob(glob_expr):
            f = open(header,'r')
            func_name = ""
            mode = line_no = 0 # mode - outside func declaration (0) or inside (1)
            for l in f.xreadlines():
                line_no += 1
                ll = ""
                
                #if re.findall(r'\b([abd-z]|([c][a-uw-z]))[a-z]*[A-Z]', l):
                #    print_report(header,line_no,"Bad-style identifier:\n\t"+l) 
                
                if mode == 0:
                    if l.startswith(comp.ext_macro):
                        ll = l[len(comp.ext_macro):]
                        decl = ""
                        mode = 1
                    elif l.startswith(comp.inline_macro):
                        temp_func_name = re.findall( r'^.+?\b(' + comp.func_prefix + '\w+)', l )
                        if temp_func_name and temp_func_name[0] != func_name:
                            ll = l[len(comp.inline_macro):]
                            decl = ""
                            mode = 1
                else:
                    ll = l

                if ll:
                    decl += ll.rstrip('\n') + ' '
                    if ll.find(';') >= 0:
                        mode = 0
                        decl = normalize_decl(decl)
                        func_name = re.findall( r'^.+?\b(' + comp.func_prefix + '\w+)', decl )[0]
                        if func_list.get(func_name,[]):
                            print_report(header,line_no,"Duplicated declaration of " + \
                                         func_name + "... ignored") 
                        else:
                            func_list[func_name] = [decl,header,line_no,0]
                    else:
                        mode = 1
            f.close()
    
    doc_path = comp.doc_path
    if not doc_path.endswith('/') and not doc_path.endswith('\\'):
        doc_path += '/'

    blurb_re = re.compile( r'^<p class="Blurb"' )

    for doc_glob in comp.docs:
        glob_expr = doc_path + doc_glob
        for doc in glob.glob(glob_expr):
            f = open(doc, 'r')
            mode = line_no = 0 # mode - 0 outside function declaration, 2 - inside,
                               # 1 transitional state ('cause <pre> is used not only
                               # for declaring functions)
            for l in f.xreadlines():
                line_no += 1
                #if re.findall(r'\b([abd-z]|([c][a-uw-z]))[a-z]*[A-Z]', l):
                #    print_report(doc,line_no,"Bad-style identifier:\n\t" + l) 
                if mode == 0:
                    if blurb_re.match(l):
                        mode = 1
                elif mode == 1:
                    if l.endswith('<pre>\n'):
                        mode = 2
                        decl = ""
                elif mode == 2:
                    if l.startswith('</pre>'):
                        mode = 0
                        if decl.find('CV_DEFAULT') >= 0:
                            print_report(doc,line_no,'CV_DEFAULT is used in documentation')
                        decl = normalize_decl(decl)
                        decl_list = decl.split(';')
                        for decl in decl_list:
                            decl = decl.strip()
                            if decl:
                                decl = decl + ';'

                                #print '***', decl
                                func_name = re.findall( r'^.+?\b(' + comp.func_prefix + '\w+)\(', decl )
                                if not func_name: continue

                                func_name = func_name[0]
                                decl_info = func_list.get(func_name,[])
                                if decl_info:
                                    if decl_info[3] == 0:
                                        if decl_info[0] != decl:
                                            print_report(doc,line_no,'Incorrect documentation on ' + func_name + ':')
                                            print '  hdr: ' + decl_info[0]
                                            print '  doc: ' + decl
                                        decl_info[3] = 1
                                    else:
                                        print_report(doc,line_no,'Duplicated documentation on ' + func_name)
                                else:
                                    print_report(doc,line_no,'The function '+func_name+' is not declared')
                    elif not l.startswith('#define'):
                        decl += l.rstrip('\n')
            f.close()

    print "---------------------------------------------------"
    keys = func_list.keys()
    undocumented_funcs = []
    for k in keys:
        decl_info = func_list[k]
        if decl_info[3] == 0:
            undocumented_funcs.append((decl_info[1],decl_info[2],k))

    undocumented_funcs.sort()
            
    for decl_info in undocumented_funcs:
        print_report(decl_info[0],decl_info[1],'Undocumented function '+decl_info[2])

