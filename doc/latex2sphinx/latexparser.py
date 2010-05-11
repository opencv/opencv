import sys
import hashlib
import cPickle as pickle
import os

from pyparsing import Word, CharsNotIn, Optional, OneOrMore, ZeroOrMore, Group, Forward, ParseException, Literal, Suppress, replaceWith, StringEnd, lineno, QuotedString, White, NotAny, ParserElement, MatchFirst

class Argument:
    def __init__(self, s, loc, toks):
        self.str = toks[1]
    def __repr__(self):
        return "[%s]" % self.str
    def __str__(self):
        return self.str
def argfun(s, loc, toks):
    return Argument(s, loc, toks)

class Parameter:
    def __init__(self, s, loc, toks):
        self.str = toks[0].asList()
    def __repr__(self):
        return '{' + "".join([str(s) for s in self.str]) + '}'
        return "{%s}" % self.str
    def __str__(self):
        return "".join([str(s) for s in self.str])
def paramfun(s, loc, toks):
    return Parameter(s, loc, toks)

class TexCmd:
    def __init__(self, s, loc, toks):
        self.cmd = str(toks[0])[1:]
        #print 'cmd', self.cmd
        self.args = toks[1].asList()
        self.params = toks[2].asList()
        self.lineno = lineno(loc, s)
        self.filename = None
    def __repr__(self):
        return '\\' + self.cmd + "".join([repr(a) for a in self.args]) + "".join([repr(p) for p in self.params])

class ZeroOrMoreAsList(ZeroOrMore):
    def __init__(self, *args):
        ZeroOrMore.__init__(self, *args)
        def listify(s, loc, toks):
            return [toks]
        self.setParseAction(listify)

ParserElement.setDefaultWhitespaceChars("\n\t")
backslash = chr(92)

texcmd = Forward()
filler = CharsNotIn(backslash + '$')
filler2 = CharsNotIn(backslash + '$' + '{}')

arg = '[' + CharsNotIn("]") + ']'
arg.setParseAction(argfun)

dollarmath = QuotedString('$',  multiline=True, unquoteResults=False)
param = Suppress(Literal('{')) + ZeroOrMoreAsList(dollarmath | filler2 | QuotedString('{', endQuoteChar='}', unquoteResults=False) | texcmd) + Suppress(Literal('}'))
param.setParseAction(paramfun)
def bs(c): return Literal("\\" + c)
singles = bs("[") | bs("]") | bs("{") | bs("}") | bs("\\") | bs("&") | bs("_") | bs(",") | bs("#") | bs("\n") | bs(";") | bs("|") | bs("%") | bs("*") | bs("~") | bs("^")
texcmd << (singles | Word("\\", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", min = 2)) + ZeroOrMoreAsList(arg) + ZeroOrMoreAsList(param)
def texcmdfun(s, loc, toks):
    return TexCmd(s, loc, toks)
texcmd.setParseAction(texcmdfun)

#legal = "".join([chr(x) for x in set(range(32, 127)) - set(backslash)])
#filler = Word(legal)
document = ZeroOrMore(dollarmath | texcmd | filler) + StringEnd().suppress()

if 0:
    s = "This is \\\\ test"
    print s
    for t in document.parseString(s):
        if isinstance(t, TexCmd):
            print '====> cmd=[%s]' % t.cmd, t
        else:
            print '====>', t
    sys.exit(-1)

selfstr = open( __file__).read()        # Own source as a string.  Used as part of hash.
hashbase = hashlib.md5(selfstr)

def tokenize(filename):
    f = open(filename, "rt")

    def uncomment(s):
        if '%' in s and not '\\%' in s:
            return s[:s.index('%')] + '\n'
        else:
            return s

    docstr = "".join([uncomment(l) for l in f])
    hash = hashbase.copy()
    hash.update(docstr)
    cache_filename = os.path.join("parse-cache", hash.hexdigest())
    try:
        return pickle.load(open(cache_filename))
    except IOError:
        print "parsing"
        try:
            r = document.parseString(docstr)
            for x in r:
                if isinstance(x, TexCmd) and not x.filename:
                    x.filename = filename
            pickle.dump(r, open(cache_filename, 'w'))
            return r
        except ParseException, pe:
            print 'Fatal problem at %s line %d col %d' % (filename, pe.lineno, pe.col)
            print pe.line
            sys.exit(1)

def latexparser(filename):
    tokens = tokenize(filename)
    def expand(t):
        if isinstance(t, TexCmd) and t.cmd == "input":
            filename = "../" + str(t.params[0].str[0]) + ".tex"
            print filename
            return latexparser(filename)
        else:
            return [t]
    return sum([expand(t) for t in tokens], [])
