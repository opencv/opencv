#!/usr/bin/env python
"""
Usage: make_index.py <html_ref_file> [ > <output_func_index_file> ]
This script parses html reference file, creates alphabetical list of
functions and list of examples ]
"""

import sys, re, string

f = open(sys.argv[1])
func_list = {}
struct_list = []
func_decl_re = re.compile( r'<a name="decl_(.+?)"' )


for l in f.xreadlines():
    llist = func_decl_re.findall(l)
    if llist:
        ll = llist[0]
        if ll.startswith('Cv'):
            struct_list.append(ll)
        elif ll.startswith('Ipl'):
            struct_list.append(ll)
        elif ll.startswith('cvm'):
            sublist = func_list.get(ll[3], [])
            sublist.append(ll)
            func_list[ll[3]] = sublist
        elif ll.startswith('cv'):
            sublist = func_list.get(ll[2], [])
            sublist.append(ll)
            func_list[ll[2]] = sublist

f.close()

struct_list.sort()
func_letters = func_list.keys()
func_letters.sort()

print "<html><body>"

columns = 3

for letter in func_letters:
    print '<hr><h3>%s</h3>\n<table width="100%%">' % letter
    sublist = func_list[letter]
    sublist.sort()
    col_len = (len(sublist)+columns-1)/columns
    #if col_len*columns > len(sublist):
    #    sublist.append( "" * (col_len*columns - len(sublist)) )
    for i in range(col_len):
        print '<tr>'
        for j in range(columns):
            if i + j*col_len < len(sublist):
                fn = sublist[i+j*col_len]
                fn_short = fn.lstrip(string.lowercase)
                print '<td width="25%%"><a href="#decl_%s">%s</a></td>' % (fn, fn_short)
            else:
                print '<td width="25%%"></td>'
        print '</tr>'
    print "</table>"



