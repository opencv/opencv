import sys
from pyparsing import Word, CharsNotIn, Optional, OneOrMore, ZeroOrMore, Group, ParseException, Literal, replaceWith

import pyparsing
help(pyparsing)

class Argument:
    def __init__(self, s, loc, toks):
        self.str = toks[1]
    def __repr__(self):
        return "[%s]" % self.str
def argfun(s, loc, toks):
    return Argument(s, loc, toks)

class Parameter:
    def __init__(self, s, loc, toks):
        self.str = toks[1]
    def __repr__(self):
        return "{%s}" % self.str
def paramfun(s, loc, toks):
    return Parameter(s, loc, toks)

class TexCmd:
    def __init__(self, s, loc, toks):
        self.cmd = str(toks[0])[1:]
        #print 'cmd', self.cmd
        self.args = toks[1].asList()
        self.params = toks[2].asList()
    def __repr__(self):
        return self.cmd + "".join([repr(a) for a in self.args]) + "".join([repr(p) for p in self.params])

class ZeroOrMoreAsList(ZeroOrMore):
    def __init__(self, *args):
        ZeroOrMore.__init__(self, *args)
        def listify(s, loc, toks):
            return [toks]
        self.setParseAction(listify)

arg = '[' + CharsNotIn("]") + ']'
arg.setParseAction(argfun)
param = '{' + Optional(CharsNotIn("}")) + '}'
param.setParseAction(paramfun)
texcmd = Word("\\", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + ZeroOrMoreAsList(arg) + ZeroOrMoreAsList(param)
def texcmdfun(s, loc, toks):
    if str(toks[0])[1:] == 'input':
        filename = toks[2].asList()[0].str + "-py.tex"
        print 'Now parsing', filename
        return parsefile(filename)
    else:
        return TexCmd(s, loc, toks)
texcmd.setParseAction(texcmdfun)

legal = "".join([chr(x) for x in set(range(32, 127)) - set("\\")])
document = ZeroOrMore(texcmd | Word(legal)) + Literal(chr(127)).suppress()

def parsefile(filename):
    f = open(filename, "rt")

    lines = list(f)
    def uncomment(s):
        if '%' in s:
            return s[:s.index('%')] + '\n'
        else:
            return s

    lines = [uncomment(l) for l in lines]

    docstr = "".join(lines) + chr(127)
    # document.setFailAction(None)
    return document.parseString(docstr)

for x in parsefile(sys.argv[1]):
    if isinstance(x, TexCmd):
        if x.cmd == 'chapter':
            print repr(x)
