import sys
from latexparser import latexparser, TexCmd
import distutils.dep_util
import os
import cPickle as pickle
import pyparsing as pp
import StringIO
from qfile import QOpen
from string import Template
from random import random
import urllib

# useful things for pyparsing
def returnList(x):
    def listify(s, loc, toks):
        return [toks]
    x.setParseAction(listify)
    return x
def returnTuple(x):
    def listify(s, loc, toks):
        return [tuple(toks)]
    x.setParseAction(listify)
    return x
def CommaList(word):
    return returnList(pp.Optional(word + pp.ZeroOrMore(pp.Suppress(',') + word)))
def sl(s):
    return pp.Suppress(pp.Literal(s))

import pythonapi

python_api = pythonapi.reader("../../modules/python/api")


class SphinxWriter:
    def __init__(self, filename, language, abspath):
        assert language in ['py', 'c', 'cpp']
        self.language = language

        self.abspath = abspath
        os.path.abspath(os.path.dirname(filename))

        self.f_index = QOpen(os.path.join(self.language, filename), 'wt')
        self.f = self.f_index
        self.f_chapter = None
        self.f_section = None
        self.indent = 0
        self.state = None
        self.envstack = []
        self.tags = {}
        self.errors = open('errors.%s' % language, 'wt')
        self.unhandled_commands = set()
        self.freshline = True
        self.function_props = {}
        self.covered = set()        # covered functions, used for error report
        self.description = ""
        self.cur_module = ""

    def write(self, s):
        self.freshline = len(s) > 0 and (s[-1] == '\n')
        self.f.write(s.replace('\n', '\n' + self.indent * "    "))

    def appendspace(self):
        """ append a space to the output - if we're not at the start of a line """
        if not self.freshline:
            self.write(' ')

    def doplain(self, s):
        if (len(s) > 1) and (s[0] == '$' and s[-1] == '$') and self.state != 'math':
            s = ":math:`%s`" % s[1:-1].strip()
        elif self.state != 'math':
            s.replace('\\_', '_')
        if len(s) > 0 and s[-1] == '\n':
            s = s[:-1]
        if self.state == 'fpreamble':
            self.description += s
        else:
            self.write(s)

    def docmd(self, c):
        if self.state == 'math':
            if c.cmd != ']':
                self.default_cmd(c)
            else:
                self.indent -= 1
                self.state = None
                self.write('\n\n')
        else:
            if c.cmd == '\n':
                self.write('\\\n')
            else:
                if c.cmd == '[':
                    meth = self.cmd_gomath
                else:
                    cname = "cmd_" + c.cmd
                    meth = getattr(self, cname, self.unrecognized_cmd)
                meth(c)

    def cmd_gomath(self, c):
        self.state = 'math'
        print >>self, "\n\n.. math::"
        self.indent += 1
        print >>self

    def cmd_chapter(self, c):
        filename = str(c.params[0]).lower().replace(' ', '_').replace('/','_').replace('.','_')
        self.f_index.write("    %s\n" % filename)
        self.f_chapter = QOpen(os.path.join(self.language, filename + '.rst'), 'wt')
        self.f_section = None
        self.f = self.f_chapter
        self.indent = 0
        title = str(c.params[0])
        print >>self, '*' * len(title)
        print >>self, title
        print >>self, '*' * len(title)
        print >>self
        self.chapter_intoc = False

    def cmd_section(self, c):
        filename = str(c.params[0]).lower().replace(' ', '_').replace('/','_')
        if len(self.cur_module) > 0:
            filename = self.cur_module + "_" + filename
        if not self.chapter_intoc:
            self.chapter_intoc = True
            print >>self.f_chapter
            print >>self.f_chapter, '.. toctree::'
            print >>self.f_chapter, '    :maxdepth: 2'
            print >>self.f_chapter
        self.f_chapter.write("    %s\n" % filename)
        self.f_section = QOpen(os.path.join(self.language, filename + '.rst'), 'wt')
        self.f = self.f_section
        self.indent = 0
        title = self.render(c.params[0].str)
        print >>self, title
        print >>self, '=' * len(title)
        print >>self
        print >>self, '.. highlight:: %s' % {'c': 'c', 'cpp': 'cpp', 'py': 'python'}[self.language]
        print >>self

    def cmd_subsection(self, c):
        print >>self
        nm = str(c.params[0])
        print >>self, nm
        print >>self, '-' * len(nm)
        print >>self
        self.function_props = {}
        self.covered.add(nm)

    def cmd_includegraphics(self, c):
        filename = os.path.join('..', '..', str(c.params[0]))
        print >>self, "\n\n.. image:: %s\n\n" % filename

    def cmd_renewcommand(self, c):
        self.indent = 0
        command = self.render(c.params[0].str)
	if command == 'curModule':
		self.cur_module = self.render(c.params[1].str)

    def wikiLink(self, name):
        return '`id=%s Comments from the Wiki <http://opencv.willowgarage.com/wiki/documentation/%s/%s/%s>`__' % (random(), self.language, self.cur_module, urllib.quote(name) )

    def cmd_cvCppCross(self, c):
        self.write(":func:`%s`" % str(c.params[0]))

    def cmd_cvCPyCross(self, c):
        self.write(":ref:`%s`" % str(c.params[0]))

    def cmd_cross(self, c):
        self.write(":ref:`%s`" % str(c.params[0]))

    def cmd_cvCross(self, c):
        self.write(":ref:`%s`" % str(c.params[0]))

    def cmd_cvclass(self, c):
        self.indent = 0
        self.state = None
        nm = self.render(list(c.params[0].str))
        print >>self, "\n.. index:: %s\n" % nm
        print >>self, ".. _%s:\n" % nm
        print >>self, nm
        print >>self, '-' * len(nm)
        print >>self
        print >>self, self.wikiLink(nm)
        print >>self
        if self.language == 'py':
            print >>self, ".. class:: " + nm + "\n"
        else:
            print >>self, ".. ctype:: " + nm + "\n"
        print >>self
        self.addtag(nm, c)
        self.state = 'class'

    def cmd_index(self, c):
        pass

    def cmd_hyperref(self, c):
        pass

    def cmd_footnote(self, c):
        pass

    def cmd_textasciitilde(self, c):
        self.write('~')

    def addtag(self, nm, c):
        if nm == "":
            self.report_error(c, "empty name")
        self.tags[nm] = "%s\t%s\t%d" % (nm, os.path.join(os.getcwd(), c.filename), c.lineno)

    def cmd_cvfunc(self, c):
        self.cmd_cvCPyFunc(c)

    def cmd_cvCPyFunc(self, c):
        self.indent = 0
        nm = self.render(c.params[0].str)
        print >>self, "\n.. index:: %s\n" % nm
        print >>self, ".. _%s:\n" % nm
        print >>self, nm
        print >>self, '-' * len(nm)
        print >>self
        print >>self, self.wikiLink(nm)
        print >>self
        self.state = 'fpreamble'
        if self.description != "":
            self.report_error(c, "overflow - preceding cvfunc (starting %s) not terminated?" % repr(self.description[:30]))
        self.description = ""
        self.addtag(nm, c)

        self.function_props = {'name' : nm}
        self.covered.add(nm)

    def cmd_cvCppFunc(self, c):
        self.indent = 0
        nm = self.render(c.params[0].str)
        print >>self, "\n.. index:: %s\n" % nm
        if 0:
            print >>self, "\n.. _%s:\n" % nm
        print >>self
        print >>self, 'cv::%s' % nm
        print >>self, '-' * (4+len(nm))
        print >>self
        print >>self, self.wikiLink(nm)
        print >>self
        self.state = 'fpreamble'
        if self.description != "":
            self.report_error(c, "overflow - preceding cvfunc (starting %s) not terminated?" % repr(self.description[:30]))
        self.description = ""
        self.addtag(nm, c)

        self.function_props = {'name' : nm}
        self.covered.add(nm)

    def cmd_cvdefC(self, c):
        if self.language != 'c':
            return
        s = str(c.params[0]).replace('\\_', '_')
        s = s.replace('\\par', '')
        s = s.replace('\n', ' ')
        s = s.replace(';', '')
        self.indent = 0
        for proto in s.split('\\newline'):
            if proto.strip() != "":
                print >>self, "\n\n.. cfunction:: " + proto.strip() + "\n"
        # print >>self, "=", repr(c.params[0].str)
        print >>self, '    ' + self.description
        self.description = ""
        print >>self
        self.state = None
        self.function_props['defpy'] = s

    def cmd_cvdefCpp(self, c):
        if self.language != 'cpp':
            return
        s = str(c.params[0]).replace('\\_', '_')
        s = s.replace('\\par', '')
        s = s.replace('\n', ' ')
        s = s.replace(';', '')
        self.indent = 0
        for proto in s.split('\\newline'):
            if proto.strip() != "":
                print >>self, "\n\n.. cfunction:: " + proto.strip() + "\n"
        # print >>self, "=", repr(c.params[0].str)
        if self.description != "":
            print >>self, '    ' + self.description
        else:
            self.report_error(c, 'empty description')
        self.description = ""
        print >>self
        self.state = None
        self.function_props['defpy'] = s

    def cmd_cvdefPy(self, c):
        if self.language != 'py':
            return
        s = str(c.params[0]).replace('\\_', '_')
        self.indent = 0
        print >>self, ".. function:: " + s + "\n"
        # print >>self, "=", repr(c.params[0].str)
        print >>self, '    ' + self.description
        print >>self
        self.description = ""
        self.state = None
        self.function_props['defpy'] = s

        pp.ParserElement.setDefaultWhitespaceChars(" \n\t")

        ident = pp.Word(pp.alphanums + "_.+-")
        ident_or_tuple = ident | (sl('(') + CommaList(ident) + sl(')'))
        initializer = ident_or_tuple
        arg = returnList(ident + pp.Optional(sl('=') + initializer))

        decl = ident + sl('(') + CommaList(arg) + sl(')') + sl("->") + ident_or_tuple + pp.StringEnd()

        try:
            l = decl.parseString(s)
            if str(l[0]) != self.function_props['name']:
                self.report_error(c, 'Decl "%s" does not match function name "%s"' % (str(l[0]), self.function_props['name']))
            self.function_props['signature'] = l
            if l[0] in python_api:
                (ins, outs) = python_api[l[0]]
                ins = [a for a in ins if not 'O' in a.flags]
                if outs != None:
                    outs = outs.split(',')
                if len(ins) != len(l[1]):
                    self.report_error(c, "function %s documented arity %d, code arity %d" % (l[0], len(l[1]), len(ins)))
                if outs == None:
                    if l[2] != 'None':
                        self.report_error(c, "function %s documented None, but code has %s" % (l[0], l[2]))
                else:
                    if isinstance(l[2], str):
                        doc_outs = [l[2]]
                    else:
                        doc_outs = l[2]
                    if len(outs) != len(doc_outs):
                        self.report_error(c, "function %s output code tuple %d, documented %d" % (l[0], len(outs), len(doc_outs)))
            else:
                # self.report_error(c, "function %s documented but not found in code" % l[0])
                pass
        except pp.ParseException, pe:
            self.report_error(c, str(pe))
            print s
            print pe

    def report_error(self, c, msg):
        print >>self.errors, "%s:%d: [%s] Error %s" % (c.filename, c.lineno, self.language, msg)

    def cmd_begin(self, c):
        if len(c.params) == 0:
            self.report_error(c, "Malformed begin")
            return
        self.write('\n')
        s = str(c.params[0])
        self.envstack.append((s, (c.filename, c.lineno)))
        if s == 'description':
            if self.language == 'py' and 'name' in self.function_props and not 'defpy' in self.function_props:
                self.report_error(c, "No cvdefPy for function %s" % self.function_props['name'])
            self.indent += 1
        elif s == 'lstlisting':
            # Set indent to zero while collecting code; so later write will not double-indent
            self.saved_f = self.f
            self.saved_indent = self.indent
            self.f = StringIO.StringIO()
            self.indent = 0
        elif s in ['itemize', 'enumerate']:
            self.indent += 1
        elif s == 'tabular':
            self.f = StringIO.StringIO()
        else:
            self.default_cmd(c)

    def cmd_item(self, c):
        if len(self.ee()) == 0:
            self.report_error(c, "item without environment")
            return
        self.indent -= 1
        markup = {'itemize' : '*', 'enumerate' : '#.', 'description' : '*'}[self.ee()[-1]]
        if len(c.args) > 0:
            markup += " " + self.render([c.args[0].str])
        if len(c.params) > 0:
            markup += " " + self.render(c.params[0].str)
        self.write("\n\n" + markup)
        self.indent += 1

    def cmd_end(self, c):
        if len(c.params) != 1:
            self.report_error(c, "Malformed end")
            return
        if len(self.envstack) == 0:
            self.report_error(c, "end with no env")
            return
        self.write('\n')
        s = str(c.params[0])
        if self.envstack == []:
            print "Cannot pop at", (c.filename, c.lineno)
        if self.envstack[-1][0] != s:
            self.report_error(c, "end{%s} does not match current stack %s" % (s, repr(self.envstack)))
        self.envstack.pop()
        if s == 'description':
            self.indent -= 1
            if self.indent == 0:
                self.function_props['done'] = True
        elif s in ['itemize', 'enumerate']:
            self.indent -= 1
        elif s == 'tabular':
            tabletxt = self.f.getvalue()
            self.f = self.f_section
            self.f.write(self.handle_table(tabletxt))
        elif s == 'lstlisting':
            listing = self.f.getvalue()

            self.f = self.saved_f
            self.indent = self.saved_indent
            print >>self
            if self.language == 'py':
                ckeys = ['#define', 'void', '#include', ';\n']
                found = [repr(k) for k in ckeys if k in listing]
                if len(found) > 0:
                    self.report_error(c, 'listing is probably C, found %s' % ",".join(found))
            if (self.language == 'py') and ('>>>' in listing):
                print >>self, "\n.. doctest::\n"
            else:
                print >>self, "\n::\n"
            self.indent += 1
            print >>self
            self.write(listing)
            self.indent -= 1
            print >>self
            print >>self
            print >>self, ".."      # otherwise a following :param: gets treated as more listing
        elif s == 'document':
            pass
        else:
            self.default_cmd(c)
        
    def cmd_label(self, c):
        pass

    def cmd_lstinputlisting(self, c):
        s = str(c.params[0])
        print >>self.f, ".. include:: %s" % os.path.normpath(os.path.join(self.abspath, s))
        print >>self.f, "    :literal:"
        print >>self.f

    # Conditionals
    def cmd_cvC(self, c):
        self.do_conditional(['c'], c)
    def cmd_cvCpp(self, c):
        self.do_conditional(['cpp'], c)
    def cmd_cvPy(self, c):
        self.do_conditional(['py'], c)
    def cmd_cvCPy(self, c):
        self.do_conditional(['c', 'py'], c)
    def do_conditional(self, langs, c):
        if self.language in langs:
            self.doL(c.params[0].str, False)

    def render(self, L):
        """ return L rendered as a string """
        save = self.f
        self.f = StringIO.StringIO()
        for x in L:
            if isinstance(x, TexCmd):
                self.docmd(x)
            else:
                self.doplain(x)
        r = self.f.getvalue()
        self.f = save
        return r

    def cmd_cvarg(self, c):
        if len(c.params) != 2:
            self.report_error(c, "Malformed cvarg")
            return
        e = self.ee()
        if self.state == 'class':
            nm = self.render(c.params[0].str)
            if '->' in nm:
                print >>self, "\n\n.. method:: %s\n\n" % nm
            else:
                print >>self, "\n\n.. attribute:: %s\n\n" % nm
            self.indent += 1
            print >>self
            self.doL(c.params[1].str, False)
            self.indent -= 1
            print >>self
            return
        is_func_arg = (e == ['description']) and (not 'done' in self.function_props)
        if is_func_arg:
            nm = self.render(c.params[0].str)
            print >>self, '\n:param %s: ' % nm,
            type = None         # Try to figure out the argument type
            # For now, multiple args get a pass
            if (self.language == 'py') and ('signature' in self.function_props) and (not ',' in nm):
                sig = self.function_props['signature']
                argnames = [a[0] for a in sig[1]]
                if isinstance(sig[2], str):
                    resnames = [sig[2]]
                else:
                    resnames = list(sig[2])
                if not nm in argnames + resnames:
                    self.report_error(c, "Argument %s is not mentioned in signature (%s) (%s)" % (nm, ", ".join(argnames), ", ".join(resnames)))

                api = python_api.get(self.function_props['name'], None)
                if api:
                    (ins, outs) = api
                    adict = dict([(a.nm, a) for a in ins])
                    arg = adict.get(nm, None)
                    if arg:
                        type = arg.ty
                    else:
                        self.report_error(c, 'cannot find arg %s in code' % nm)
        elif len(e) > 0 and e[-1] == 'description':
            print >>self, '\n* **%s** ' % self.render(c.params[0].str),
        else:
            self.report_error(c, "unexpected env (%s) for cvarg" % ",".join(e))
        self.indent += 1
        self.doL(c.params[1].str, False)
        self.indent -= 1
        print >>self
        if is_func_arg and type:
            type = type.replace('*', '')
            translate = {
                "ints" : "sequence of int",
                "floats" : "sequence of float",
                "IplImages" : "sequence of :class:`IplImage`",
                "double" : "float",
                "int" : "int",
                "float" : "float",
                "char" : "str",
                "cvarrseq" : ":class:`CvArr` or :class:`CvSeq`",
                "CvPoint2D32fs" : "sequence of (float, float)",
                "pts_npts_contours" : "list of lists of (x,y) pairs",
                "CvSeqOfCvSURFPoint" : ":class:`CvSeq` of :class:`CvSURFPoint`",
                "CvSeqOfCvSURFDescriptor" : ":class:`CvSeq` of list of float",
                "cvpoint2d32f_count" : "int",
                "ranges" : "list of tuples of ints",
                "PyObject" : "object",
                "edgeorpoint" : ":class:`CvSubdiv2DEdge`, :class:`CvSubdiv2DPoint`",
            }
            print >>self, "\n:type %s: %s" % (nm, translate.get(type, ':class:`%s`' % type))

    def cmd_genc(self, c): pass 
    def cmd_genpy(self, c): pass 
    def cmd_author(self, c): pass 
    def cmd_date(self, c): pass
    def cmd_def(self, c): pass
    def cmd_documentclass(self, c): pass
    def cmd_maketitle(self, c): pass
    def cmd_newcommand(self, c): pass
    def cmd_newline(self, c): pass
    def cmd_setcounter(self, c): pass
    def cmd_tableofcontents(self, c): pass
    def cmd_targetlang(self, c): pass
    def cmd_usepackage(self, c): pass
    def cmd_title(self, c): pass
    def cmd_par(self, c): pass
    def cmd_hline(self, c):
        print >>self, "\\hline"

    def cmd_cite(self, c):
        # XXX jcb - these references are crashing Sphinx 0.65
        # self.write("[%s]_" % str(c.params[0]))
        self.write("%s" % str(c.params[0]))

    def cmd_href(self, c):
        if len(c.params) == 2:
            self.write("`%s <%s>`_" % (str(c.params[1]), self.render(c.params[0].str)))
        else:
            self.report_error(c, "href should have two params")

    def cmd_url(self, c):
        self.write(str(c.params[0]))

    def cmd_emph(self, c):
        self.write("*" + self.render(c.params[0].str) + "*")

    def cmd_textit(self, c):
        self.write("*" + self.render(c.params[0].str) + "*")

    def cmd_textbf(self, c):
        self.write("**" + self.render(c.params[0].str) + "**")

    def cmd_texttt(self, c):
        self.write("``" + self.render(c.params[0].str) + "``")

    def cmd_code(self, c):
        self.cmd_texttt(c)

    def default_cmd(self, c):
        if self.f == self.f_section:
            self.write(repr(c))

    def unrecognized_cmd(self, c):
        # if writing the index or chapter heading, anything goes
        if not self.f in [self.f_index, self.f_chapter]:
            self.write(c.cmd)
            if (not 'lstlisting' in self.ee()) and (not c.cmd in "#{}%&*\\_^"):
                if not c.cmd in self.unhandled_commands:
                    self.report_error(c, 'unhandled command %s' % c.cmd)
                    self.unhandled_commands.add(c.cmd)

    def doL(self, L, newlines = True):
        for x in L:
            pos0 = self.f.tell()
            if isinstance(x, TexCmd):
                self.docmd(x)
            else:
                if 'lstlisting' in self.ee() or not newlines:
                    self.doplain(x)
                else:
                    self.doplain(x.lstrip())
            pos1 = self.f.tell()
            if pos0 != pos1:
                if self.state in ['math'] or not newlines:
                    self.appendspace()
                else:
                    if not 'lstlisting' in self.ee():
                        self.write('\n')

    def handle_table(self, s):
        oneline = s.replace('\n', ' ').strip()
        rows = [r.strip() for r in oneline.split('\\hline')]
        tab = []
        for r in rows:
            if r != "":
                cols = [c.strip() for c in r.split('&')]
                tab.append(cols)
        widths = [max([len(r[i]) for r in tab]) for i in range(len(tab[0]))]

        st = ""         # Sphinx table

        if 0:
            sep = "+" + "+".join(["-" * w for w in widths]) + "+"
            st += sep + '\n'
            for r in tab:
                st += "|" + "|".join([c.center(w) for (c, w) in zip(r, widths)]) + "|" + '\n'
                st += sep + '\n'

        st = '.. table::\n\n'
        sep = "  ".join(["=" * w for w in widths])
        st += '    ' + sep + '\n'
        for y,r in enumerate(tab):
            st += '    ' + "  ".join([c.ljust(w) for (c, w) in zip(r, widths)]) + '\n'
            if y == 0:
                st += '    ' + sep + '\n'
        st += '    ' + sep + '\n'
        return st

    def ee(self):
        """ Return tags of the envstack.  envstack[0] is 'document', so skip it """
        return [n for (n,_) in self.envstack[1:]]

    def get_tags(self):
        return self.tags

    def close(self):

        if self.envstack != []:
            print >>self.errors, "Error envstack not empty at end of doc: " + repr(self.envstack)
        print >>self.errors, "Unrecognized commands:"
        for c in sorted(self.unhandled_commands):
            print >>self.errors, "\n    " + c
        print >>self.errors
        if self.language == 'py':
            print >>self.errors, "The following functions are undocumented"
            for f in sorted(set(python_api) - self.covered):
                print >>self.errors, '    ', f

        print >>self.f_index, "    bibliography"
        print >>self.f_index, """

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
"""

# Quick and dirty bibtex parser

def parseBib(filename, language):
    pp.ParserElement.setDefaultWhitespaceChars(" \n\t")
    entry = returnList(pp.Word('@', pp.alphanums) + sl('{') +
        pp.Word(pp.alphanums + "_") + sl(',') +
        CommaList(returnTuple(pp.Word(pp.alphanums) + sl('=') + pp.QuotedString('{', endQuoteChar = '}'))) +
        pp.Suppress(pp.Optional(',')) +
        sl('}'))
    r = (pp.ZeroOrMore(entry) | pp.Suppress('#' + pp.ZeroOrMore(pp.CharsNotIn('\n'))) + pp.StringEnd()).parseFile(filename)

    bibliography = QOpen(os.path.join(language, "bibliography.rst"), 'wt')
    print >>bibliography, "Bibliography"
    print >>bibliography, "============"
    print >>bibliography

    for _,e in sorted([(str(x[1]), x) for x in r]):
        (etype, tag, attrs) = str(e[0][1:]), str(e[1]), dict([(str(a), str(b)) for (a,b) in e[2]])
        
        representations = {
            'article' :         '$author, "$title". $journal $volume $number, pp $pages ($year)',
            'inproceedings' :   '$author "$title", $booktitle, $year',
            'misc' :            '$author "$title", $year',
            'techreport' :      '$author "$title", $edition, $edition ($year)',
        }
        if etype in representations:
            if 0:
                print >>bibliography, tag
                print >>bibliography, "^" * len(tag)
                print >>bibliography

            print >>bibliography, ".. [%s] %s" % (tag, Template(representations[etype]).safe_substitute(attrs))
            print >>bibliography
    bibliography.close()

if 1:
    fulldoc = latexparser(sys.argv[1])

    abspath = os.path.abspath(os.path.dirname(sys.argv[1]))

    raw = open('raw.full', 'w')
    for x in fulldoc:
        print >>raw, repr(x)
    raw.close()

    # Filter on target language
    def preprocess_conditionals(fd, conditionals):
        r = []
        ifstack = []
        for x in fd:
            if isinstance(x, TexCmd):
                ll = x.cmd.rstrip()
                loc = (x.filename, x.lineno)
                if ll.startswith("if"):
                    # print " " * len(ifstack), '{', loc
                    ifstack.append((conditionals.get(ll[2:], False), loc))
                elif ll.startswith("else"):
                    ifstack[-1] = (not ifstack[-1][0], ifstack[-1][1])
                elif ll.startswith("fi"):
                    ifstack.pop()
                    # print " " * len(ifstack), '}', loc
                elif not False in [p for (p,_) in ifstack]:
                    r.append(x)
            else:
                if not False in [p for (p,_) in ifstack]:
                    r.append(x)
        if ifstack != []:
            print "unterminated if", ifstack
            sys.exit(0)
        return r

    tags = {}
    for language in sys.argv[2:]:
        doc = preprocess_conditionals(fulldoc, {
                                              'C' : language=='c',
                                              'Python' : language=='py',
                                              'Py' : language=='py',
                                              'CPy' : (language=='py' or language == 'c'),
                                              'Cpp' : language=='cpp',
                                              'plastex' : True})

        raw = open('raw.%s' % language, 'w')
        for x in doc:
            print >>raw, repr(x)
        raw.close()
        sr = SphinxWriter('index.rst', language, abspath)
        print >>sr, """
OpenCV |version| %s Reference
=================================

The OpenCV Wiki is here: http://opencv.willowgarage.com/

Contents:

.. toctree::
    :maxdepth: 2

""" % {'c': 'C', 'cpp': 'C++', 'py': 'Python'}[language]
        sr.doL(doc)
        sr.close()
        parseBib('../opencv.bib', language)
        tags.update(sr.get_tags())
    open('TAGS', 'w').write("\n".join(sorted(tags.values())) + "\n")

