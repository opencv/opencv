# svgfig.py copyright (C) 2008 Jim Pivarski <jpivarski@gmail.com>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
#
# Full licence is in the file COPYING and at http://www.gnu.org/copyleft/gpl.html

import re, codecs, os, platform, copy, itertools, math, cmath, random, sys, copy
_epsilon = 1e-5

if sys.version_info >= (3,0):
  long = int
  basestring = (str,bytes)

# Fix Python 2.x.
try:
    UNICODE_EXISTS = bool(type(unicode))
except NameError:
    unicode = lambda s: str(s)

if re.search("windows", platform.system(), re.I):
    try:
        import _winreg
        _default_directory = _winreg.QueryValueEx(_winreg.OpenKey(_winreg.HKEY_CURRENT_USER,
                             r"Software\Microsoft\Windows\Current Version\Explorer\Shell Folders"), "Desktop")[0]
#   tmpdir = _winreg.QueryValueEx(_winreg.OpenKey(_winreg.HKEY_CURRENT_USER, "Environment"), "TEMP")[0]
#   if tmpdir[0:13] != "%USERPROFILE%":
#     tmpdir = os.path.expanduser("~") + tmpdir[13:]
    except:
        _default_directory = os.path.expanduser("~") + os.sep + "Desktop"

_default_fileName = "tmp.svg"

_hacks = {}
_hacks["inkscape-text-vertical-shift"] = False


def rgb(r, g, b, maximum=1.):
    """Create an SVG color string "#xxyyzz" from r, g, and b.

    r,g,b = 0 is black and r,g,b = maximum is white.
    """
    return "#%02x%02x%02x" % (max(0, min(r*255./maximum, 255)),
                              max(0, min(g*255./maximum, 255)),
                              max(0, min(b*255./maximum, 255)))

def attr_preprocess(attr):
    attrCopy = attr.copy()
    for name in attr.keys():
        name_colon = re.sub("__", ":", name)
        if name_colon != name:
            attrCopy[name_colon] = attrCopy[name]
            del attrCopy[name]
            name = name_colon

        name_dash = re.sub("_", "-", name)
        if name_dash != name:
            attrCopy[name_dash] = attrCopy[name]
            del attrCopy[name]
            name = name_dash

    return attrCopy


class SVG:
    """A tree representation of an SVG image or image fragment.

    SVG(t, sub, sub, sub..., attribute=value)

    t                       required             SVG type name
    sub                     optional list        nested SVG elements or text/Unicode
    attribute=value pairs   optional keywords    SVG attributes

    In attribute names, "__" becomes ":" and "_" becomes "-".

    SVG in XML

    <g id="mygroup" fill="blue">
        <rect x="1" y="1" width="2" height="2" />
        <rect x="3" y="3" width="2" height="2" />
    </g>

    SVG in Python

    >>> svg = SVG("g", SVG("rect", x=1, y=1, width=2, height=2), \
    ...                SVG("rect", x=3, y=3, width=2, height=2), \
    ...           id="mygroup", fill="blue")

    Sub-elements and attributes may be accessed through tree-indexing:

    >>> svg = SVG("text", SVG("tspan", "hello there"), stroke="none", fill="black")
    >>> svg[0]
    <tspan (1 sub) />
    >>> svg[0, 0]
    'hello there'
    >>> svg["fill"]
    'black'

    Iteration is depth-first:

    >>> svg = SVG("g", SVG("g", SVG("line", x1=0, y1=0, x2=1, y2=1)), \
    ...                SVG("text", SVG("tspan", "hello again")))
    ...
    >>> for ti, s in svg:
    ...     print ti, repr(s)
    ...
    (0,) <g (1 sub) />
    (0, 0) <line x2=1 y1=0 x1=0 y2=1 />
    (0, 0, 'x2') 1
    (0, 0, 'y1') 0
    (0, 0, 'x1') 0
    (0, 0, 'y2') 1
    (1,) <text (1 sub) />
    (1, 0) <tspan (1 sub) />
    (1, 0, 0) 'hello again'

    Use "print" to navigate:

    >>> print svg
    None                 <g (2 sub) />
    [0]                      <g (1 sub) />
    [0, 0]                       <line x2=1 y1=0 x1=0 y2=1 />
    [1]                      <text (1 sub) />
    [1, 0]                       <tspan (1 sub) />
    """
    def __init__(self, *t_sub, **attr):
        if len(t_sub) == 0:
            raise TypeError( "SVG element must have a t (SVG type)")

        # first argument is t (SVG type)
        self.t = t_sub[0]
        # the rest are sub-elements
        self.sub = list(t_sub[1:])

        # keyword arguments are attributes
        # need to preprocess to handle differences between SVG and Python syntax
        self.attr = attr_preprocess(attr)

    def __getitem__(self, ti):
        """Index is a list that descends tree, returning a sub-element if
        it ends with a number and an attribute if it ends with a string."""
        obj = self
        if isinstance(ti, (list, tuple)):
            for i in ti[:-1]:
                obj = obj[i]
            ti = ti[-1]

        if isinstance(ti, (int, long, slice)):
            return obj.sub[ti]
        else:
            return obj.attr[ti]

    def __setitem__(self, ti, value):
        """Index is a list that descends tree, returning a sub-element if
        it ends with a number and an attribute if it ends with a string."""
        obj = self
        if isinstance(ti, (list, tuple)):
            for i in ti[:-1]:
                obj = obj[i]
            ti = ti[-1]

        if isinstance(ti, (int, long, slice)):
            obj.sub[ti] = value
        else:
            obj.attr[ti] = value

    def __delitem__(self, ti):
        """Index is a list that descends tree, returning a sub-element if
        it ends with a number and an attribute if it ends with a string."""
        obj = self
        if isinstance(ti, (list, tuple)):
            for i in ti[:-1]:
                obj = obj[i]
            ti = ti[-1]

        if isinstance(ti, (int, long, slice)):
            del obj.sub[ti]
        else:
            del obj.attr[ti]

    def __contains__(self, value):
        """x in svg == True iff x is an attribute in svg."""
        return value in self.attr

    def __eq__(self, other):
        """x == y iff x represents the same SVG as y."""
        if id(self) == id(other):
            return True
        return (isinstance(other, SVG) and
                self.t == other.t and self.sub == other.sub and self.attr == other.attr)

    def __ne__(self, other):
        """x != y iff x does not represent the same SVG as y."""
        return not (self == other)

    def append(self, x):
        """Appends x to the list of sub-elements (drawn last, overlaps
        other primitives)."""
        self.sub.append(x)

    def prepend(self, x):
        """Prepends x to the list of sub-elements (drawn first may be
        overlapped by other primitives)."""
        self.sub[0:0] = [x]

    def extend(self, x):
        """Extends list of sub-elements by a list x."""
        self.sub.extend(x)

    def clone(self, shallow=False):
        """Deep copy of SVG tree.  Set shallow=True for a shallow copy."""
        if shallow:
            return copy.copy(self)
        else:
            return copy.deepcopy(self)

    ### nested class
    class SVGDepthIterator:
        """Manages SVG iteration."""

        def __init__(self, svg, ti, depth_limit):
            self.svg = svg
            self.ti = ti
            self.shown = False
            self.depth_limit = depth_limit

        def __iter__(self):
            return self

        def next(self):
            if not self.shown:
                self.shown = True
                if self.ti != ():
                    return self.ti, self.svg

            if not isinstance(self.svg, SVG):
                raise StopIteration
            if self.depth_limit is not None and len(self.ti) >= self.depth_limit:
                raise StopIteration

            if "iterators" not in self.__dict__:
                self.iterators = []
                for i, s in enumerate(self.svg.sub):
                    self.iterators.append(self.__class__(s, self.ti + (i,), self.depth_limit))
                for k, s in self.svg.attr.items():
                    self.iterators.append(self.__class__(s, self.ti + (k,), self.depth_limit))
                self.iterators = itertools.chain(*self.iterators)

            return self.iterators.next()
    ### end nested class

    def depth_first(self, depth_limit=None):
        """Returns a depth-first generator over the SVG.  If depth_limit
        is a number, stop recursion at that depth."""
        return self.SVGDepthIterator(self, (), depth_limit)

    def breadth_first(self, depth_limit=None):
        """Not implemented yet.  Any ideas on how to do it?

        Returns a breadth-first generator over the SVG.  If depth_limit
        is a number, stop recursion at that depth."""
        raise NotImplementedError( "Got an algorithm for breadth-first searching a tree without effectively copying the tree?")

    def __iter__(self):
        return self.depth_first()

    def items(self, sub=True, attr=True, text=True):
        """Get a recursively-generated list of tree-index, sub-element/attribute pairs.

        If sub == False, do not show sub-elements.
        If attr == False, do not show attributes.
        If text == False, do not show text/Unicode sub-elements.
        """
        output = []
        for ti, s in self:
            show = False
            if isinstance(ti[-1], (int, long)):
                if isinstance(s, basestring):
                    show = text
                else:
                    show = sub
            else:
                show = attr

            if show:
                output.append((ti, s))
        return output

    def keys(self, sub=True, attr=True, text=True):
        """Get a recursively-generated list of tree-indexes.

        If sub == False, do not show sub-elements.
        If attr == False, do not show attributes.
        If text == False, do not show text/Unicode sub-elements.
        """
        return [ti for ti, s in self.items(sub, attr, text)]

    def values(self, sub=True, attr=True, text=True):
        """Get a recursively-generated list of sub-elements and attributes.

        If sub == False, do not show sub-elements.
        If attr == False, do not show attributes.
        If text == False, do not show text/Unicode sub-elements.
        """
        return [s for ti, s in self.items(sub, attr, text)]

    def __repr__(self):
        return self.xml(depth_limit=0)

    def __str__(self):
        """Print (actually, return a string of) the tree in a form useful for browsing."""
        return self.tree(sub=True, attr=False, text=False)

    def tree(self, depth_limit=None, sub=True, attr=True, text=True, tree_width=20, obj_width=80):
        """Print (actually, return a string of) the tree in a form useful for browsing.

        If depth_limit == a number, stop recursion at that depth.
        If sub == False, do not show sub-elements.
        If attr == False, do not show attributes.
        If text == False, do not show text/Unicode sub-elements.
        tree_width is the number of characters reserved for printing tree indexes.
        obj_width is the number of characters reserved for printing sub-elements/attributes.
        """
        output = []

        line = "%s %s" % (("%%-%ds" % tree_width) % repr(None),
                          ("%%-%ds" % obj_width) % (repr(self))[0:obj_width])
        output.append(line)

        for ti, s in self.depth_first(depth_limit):
            show = False
            if isinstance(ti[-1], (int, long)):
                if isinstance(s, basestring):
                    show = text
                else:
                    show = sub
            else:
                show = attr

            if show:
                line = "%s %s" % (("%%-%ds" % tree_width) % repr(list(ti)),
                                  ("%%-%ds" % obj_width) % ("    "*len(ti) + repr(s))[0:obj_width])
                output.append(line)

        return "\n".join(output)

    def xml(self, indent=u"    ", newl=u"\n", depth_limit=None, depth=0):
        """Get an XML representation of the SVG.

        indent      string used for indenting
        newl        string used for newlines
        If depth_limit == a number, stop recursion at that depth.
        depth       starting depth (not useful for users)

        print svg.xml()
        """
        attrstr = []
        for n, v in self.attr.items():
            if isinstance(v, dict):
                v = u"; ".join([u"%s:%s" % (ni, vi) for ni, vi in v.items()])
            elif isinstance(v, (list, tuple)):
                v = u", ".join(v)
            attrstr.append(u" %s=%s" % (n, repr(v)))
        attrstr = u"".join(attrstr)

        if len(self.sub) == 0:
            return u"%s<%s%s />" % (indent * depth, self.t, attrstr)

        if depth_limit is None or depth_limit > depth:
            substr = []
            for s in self.sub:
                if isinstance(s, SVG):
                    substr.append(s.xml(indent, newl, depth_limit, depth + 1) + newl)
                elif isinstance(s, basestring):
                    substr.append(u"%s%s%s" % (indent * (depth + 1), s, newl))
                else:
                    substr.append("%s%s%s" % (indent * (depth + 1), repr(s), newl))
            substr = u"".join(substr)

            return u"%s<%s%s>%s%s%s</%s>" % (indent * depth, self.t, attrstr, newl, substr, indent * depth, self.t)

        else:
            return u"%s<%s (%d sub)%s />" % (indent * depth, self.t, len(self.sub), attrstr)

    def standalone_xml(self, indent=u"    ", newl=u"\n", encoding=u"utf-8"):
        """Get an XML representation of the SVG that can be saved/rendered.

        indent      string used for indenting
        newl        string used for newlines
        """

        if self.t == "svg":
            top = self
        else:
            top = canvas(self)
        return u"""\
<?xml version="1.0" encoding="%s" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

""" % encoding + (u"".join(top.__standalone_xml(indent, newl)))  # end of return statement

    def __standalone_xml(self, indent, newl):
        output = [u"<%s" % self.t]

        for n, v in self.attr.items():
            if isinstance(v, dict):
                v = u"; ".join([u"%s:%s" % (ni, vi) for ni, vi in v.items()])
            elif isinstance(v, (list, tuple)):
                v = u", ".join(v)
            output.append(u' %s="%s"' % (n, v))

        if len(self.sub) == 0:
            output.append(u" />%s%s" % (newl, newl))
            return output

        elif self.t == "text" or self.t == "tspan" or self.t == "style":
            output.append(u">")

        else:
            output.append(u">%s%s" % (newl, newl))

        for s in self.sub:
            if isinstance(s, SVG):
                output.extend(s.__standalone_xml(indent, newl))
            else:
                output.append(unicode(s))

        if self.t == "tspan":
            output.append(u"</%s>" % self.t)
        else:
            output.append(u"</%s>%s%s" % (self.t, newl, newl))

        return output

    def interpret_fileName(self, fileName=None):
        if fileName is None:
            fileName = _default_fileName
        if re.search("windows", platform.system(), re.I) and not os.path.isabs(fileName):
            fileName = _default_directory + os.sep + fileName
        return fileName

    def save(self, fileName=None, encoding="utf-8", compresslevel=None):
        """Save to a file for viewing.  Note that svg.save() overwrites the file named _default_fileName.

        fileName        default=None            note that _default_fileName will be overwritten if
                                                no fileName is specified. If the extension
                                                is ".svgz" or ".gz", the output will be gzipped
        encoding        default="utf-8"         file encoding
        compresslevel   default=None            if a number, the output will be gzipped with that
                                                compression level (1-9, 1 being fastest and 9 most
                                                thorough)
        """
        fileName = self.interpret_fileName(fileName)

        if compresslevel is not None or re.search(r"\.svgz$", fileName, re.I) or re.search(r"\.gz$", fileName, re.I):
            import gzip
            if compresslevel is None:
                f = gzip.GzipFile(fileName, "w")
            else:
                f = gzip.GzipFile(fileName, "w", compresslevel)

            f = codecs.EncodedFile(f, "utf-8", encoding)
            f.write(self.standalone_xml(encoding=encoding))
            f.close()

        else:
            f = codecs.open(fileName, "w", encoding=encoding)
            f.write(self.standalone_xml(encoding=encoding))
            f.close()

    def inkview(self, fileName=None, encoding="utf-8"):
        """View in "inkview", assuming that program is available on your system.

        fileName        default=None            note that any file named _default_fileName will be
                                                overwritten if no fileName is specified. If the extension
                                                is ".svgz" or ".gz", the output will be gzipped
        encoding        default="utf-8"         file encoding
        """
        fileName = self.interpret_fileName(fileName)
        self.save(fileName, encoding)
        os.spawnvp(os.P_NOWAIT, "inkview", ("inkview", fileName))

    def inkscape(self, fileName=None, encoding="utf-8"):
        """View in "inkscape", assuming that program is available on your system.

        fileName        default=None            note that any file named _default_fileName will be
                                                overwritten if no fileName is specified. If the extension
                                                is ".svgz" or ".gz", the output will be gzipped
        encoding        default="utf-8"         file encoding
        """
        fileName = self.interpret_fileName(fileName)
        self.save(fileName, encoding)
        os.spawnvp(os.P_NOWAIT, "inkscape", ("inkscape", fileName))

    def firefox(self, fileName=None, encoding="utf-8"):
        """View in "firefox", assuming that program is available on your system.

        fileName        default=None            note that any file named _default_fileName will be
                                                overwritten if no fileName is specified. If the extension
                                                is ".svgz" or ".gz", the output will be gzipped
        encoding        default="utf-8"         file encoding
        """
        fileName = self.interpret_fileName(fileName)
        self.save(fileName, encoding)
        os.spawnvp(os.P_NOWAIT, "firefox", ("firefox", fileName))

######################################################################

_canvas_defaults = {"width": "400px",
                    "height": "400px",
                    "viewBox": "0 0 100 100",
                    "xmlns": "http://www.w3.org/2000/svg",
                    "xmlns:xlink": "http://www.w3.org/1999/xlink",
                    "version": "1.1",
                    "style": {"stroke": "black",
                              "fill": "none",
                              "stroke-width": "0.5pt",
                              "stroke-linejoin": "round",
                              "text-anchor": "middle",
                             },
                    "font-family": ["Helvetica", "Arial", "FreeSans", "Sans", "sans", "sans-serif"],
                   }

def canvas(*sub, **attr):
    """Creates a top-level SVG object, allowing the user to control the
    image size and aspect ratio.

    canvas(sub, sub, sub..., attribute=value)

    sub                     optional list       nested SVG elements or text/Unicode
    attribute=value pairs   optional keywords   SVG attributes

    Default attribute values:

    width           "400px"
    height          "400px"
    viewBox         "0 0 100 100"
    xmlns           "http://www.w3.org/2000/svg"
    xmlns:xlink     "http://www.w3.org/1999/xlink"
    version         "1.1"
    style           "stroke:black; fill:none; stroke-width:0.5pt; stroke-linejoin:round; text-anchor:middle"
    font-family     "Helvetica,Arial,FreeSans?,Sans,sans,sans-serif"
    """
    attributes = dict(_canvas_defaults)
    attributes.update(attr)

    if sub is None or sub == ():
        return SVG("svg", **attributes)
    else:
        return SVG("svg", *sub, **attributes)

def canvas_outline(*sub, **attr):
    """Same as canvas(), but draws an outline around the drawable area,
    so that you know how close your image is to the edges."""
    svg = canvas(*sub, **attr)
    match = re.match(r"[, \t]*([0-9e.+\-]+)[, \t]+([0-9e.+\-]+)[, \t]+([0-9e.+\-]+)[, \t]+([0-9e.+\-]+)[, \t]*", svg["viewBox"])
    if match is None:
        raise ValueError( "canvas viewBox is incorrectly formatted")
    x, y, width, height = [float(x) for x in match.groups()]
    svg.prepend(SVG("rect", x=x, y=y, width=width, height=height, stroke="none", fill="cornsilk"))
    svg.append(SVG("rect", x=x, y=y, width=width, height=height, stroke="black", fill="none"))
    return svg

def template(fileName, svg, replaceme="REPLACEME"):
    """Loads an SVG image from a file, replacing instances of
    <REPLACEME /> with a given svg object.

    fileName         required                name of the template SVG
    svg              required                SVG object for replacement
    replaceme        default="REPLACEME"     fake SVG element to be replaced by the given object

    >>> print load("template.svg")
    None                 <svg (2 sub) style=u'stroke:black; fill:none; stroke-width:0.5pt; stroke-linejoi
    [0]                      <rect height=u'100' width=u'100' stroke=u'none' y=u'0' x=u'0' fill=u'yellow'
    [1]                      <REPLACEME />
    >>>
    >>> print template("template.svg", SVG("circle", cx=50, cy=50, r=30))
    None                 <svg (2 sub) style=u'stroke:black; fill:none; stroke-width:0.5pt; stroke-linejoi
    [0]                      <rect height=u'100' width=u'100' stroke=u'none' y=u'0' x=u'0' fill=u'yellow'
    [1]                      <circle cy=50 cx=50 r=30 />
    """
    output = load(fileName)
    for ti, s in output:
        if isinstance(s, SVG) and s.t == replaceme:
            output[ti] = svg
    return output

######################################################################

def load(fileName):
    """Loads an SVG image from a file."""
    return load_stream(file(fileName))

def load_stream(stream):
    """Loads an SVG image from a stream (can be a string or a file object)."""

    from xml.sax import handler, make_parser
    from xml.sax.handler import feature_namespaces, feature_external_ges, feature_external_pes

    class ContentHandler(handler.ContentHandler):
        def __init__(self):
            self.stack = []
            self.output = None
            self.all_whitespace = re.compile(r"^\s*$")

        def startElement(self, name, attr):
            s = SVG(name)
            s.attr = dict(attr.items())
            if len(self.stack) > 0:
                last = self.stack[-1]
                last.sub.append(s)
            self.stack.append(s)

        def characters(self, ch):
            if not isinstance(ch, basestring) or self.all_whitespace.match(ch) is None:
                if len(self.stack) > 0:
                    last = self.stack[-1]
                    if len(last.sub) > 0 and isinstance(last.sub[-1], basestring):
                        last.sub[-1] = last.sub[-1] + "\n" + ch
                    else:
                        last.sub.append(ch)

        def endElement(self, name):
            if len(self.stack) > 0:
                last = self.stack[-1]
                if (isinstance(last, SVG) and last.t == "style" and
                    "type" in last.attr and last.attr["type"] == "text/css" and
                    len(last.sub) == 1 and isinstance(last.sub[0], basestring)):
                    last.sub[0] = "<![CDATA[\n" + last.sub[0] + "]]>"

            self.output = self.stack.pop()

    ch = ContentHandler()
    parser = make_parser()
    parser.setContentHandler(ch)
    parser.setFeature(feature_namespaces, 0)
    parser.setFeature(feature_external_ges, 0)
    parser.parse(stream)
    return ch.output

######################################################################
def set_func_name(f, name):
    """try to patch the function name string into a function object"""
    try:
        f.func_name = name
    except TypeError:
        # py 2.3 raises: TypeError: readonly attribute
        pass

def totrans(expr, vars=("x", "y"), globals=None, locals=None):
    """Converts to a coordinate transformation (a function that accepts
    two arguments and returns two values).

    expr       required                  a string expression or a function
                                         of two real or one complex value
    vars       default=("x", "y")        independent variable names; a singleton
                                         ("z",) is interpreted as complex
    globals    default=None              dict of global variables
    locals     default=None              dict of local variables
    """
    if locals is None:
        locals = {}  # python 2.3's eval() won't accept None

    if callable(expr):
        if expr.func_code.co_argcount == 2:
            return expr

        elif expr.func_code.co_argcount == 1:
            split = lambda z: (z.real, z.imag)
            output = lambda x, y: split(expr(x + y*1j))
            set_func_name(output, expr.func_name)
            return output

        else:
            raise TypeError( "must be a function of 2 or 1 variables")

    if len(vars) == 2:
        g = math.__dict__
        if globals is not None:
            g.update(globals)
        output = eval("lambda %s, %s: (%s)" % (vars[0], vars[1], expr), g, locals)
        set_func_name(output, "%s,%s -> %s" % (vars[0], vars[1], expr))
        return output

    elif len(vars) == 1:
        g = cmath.__dict__
        if globals is not None:
            g.update(globals)
        output = eval("lambda %s: (%s)" % (vars[0], expr), g, locals)
        split = lambda z: (z.real, z.imag)
        output2 = lambda x, y: split(output(x + y*1j))
        set_func_name(output2, "%s -> %s" % (vars[0], expr))
        return output2

    else:
        raise TypeError( "vars must have 2 or 1 elements")


def window(xmin, xmax, ymin, ymax, x=0, y=0, width=100, height=100,
           xlogbase=None, ylogbase=None, minusInfinity=-1000, flipx=False, flipy=True):
    """Creates and returns a coordinate transformation (a function that
    accepts two arguments and returns two values) that transforms from
        (xmin, ymin), (xmax, ymax)
    to
        (x, y), (x + width, y + height).

    xlogbase, ylogbase    default=None, None     if a number, transform
                                                 logarithmically with given base
    minusInfinity         default=-1000          what to return if
                                                 log(0 or negative) is attempted
    flipx                 default=False          if true, reverse the direction of x
    flipy                 default=True           if true, reverse the direction of y

    (When composing windows, be sure to set flipy=False.)
    """

    if flipx:
        ox1 = x + width
        ox2 = x
    else:
        ox1 = x
        ox2 = x + width
    if flipy:
        oy1 = y + height
        oy2 = y
    else:
        oy1 = y
        oy2 = y + height
    ix1 = xmin
    iy1 = ymin
    ix2 = xmax
    iy2 = ymax

    if xlogbase is not None and (ix1 <= 0. or ix2 <= 0.):
        raise ValueError ("x range incompatible with log scaling: (%g, %g)" % (ix1, ix2))

    if ylogbase is not None and (iy1 <= 0. or iy2 <= 0.):
        raise ValueError ("y range incompatible with log scaling: (%g, %g)" % (iy1, iy2))

    def maybelog(t, it1, it2, ot1, ot2, logbase):
        if t <= 0.:
            return minusInfinity
        else:
            return ot1 + 1.*(math.log(t, logbase) - math.log(it1, logbase))/(math.log(it2, logbase) - math.log(it1, logbase)) * (ot2 - ot1)

    xlogstr, ylogstr = "", ""

    if xlogbase is None:
        xfunc = lambda x: ox1 + 1.*(x - ix1)/(ix2 - ix1) * (ox2 - ox1)
    else:
        xfunc = lambda x: maybelog(x, ix1, ix2, ox1, ox2, xlogbase)
        xlogstr = " xlog=%g" % xlogbase

    if ylogbase is None:
        yfunc = lambda y: oy1 + 1.*(y - iy1)/(iy2 - iy1) * (oy2 - oy1)
    else:
        yfunc = lambda y: maybelog(y, iy1, iy2, oy1, oy2, ylogbase)
        ylogstr = " ylog=%g" % ylogbase

    output = lambda x, y: (xfunc(x), yfunc(y))

    set_func_name(output, "(%g, %g), (%g, %g) -> (%g, %g), (%g, %g)%s%s" % (
                          ix1, ix2, iy1, iy2, ox1, ox2, oy1, oy2, xlogstr, ylogstr))
    return output


def rotate(angle, cx=0, cy=0):
    """Creates and returns a coordinate transformation which rotates
    around (cx,cy) by "angle" degrees."""
    angle *= math.pi/180.
    return lambda x, y: (cx + math.cos(angle)*(x - cx) - math.sin(angle)*(y - cy), cy + math.sin(angle)*(x - cx) + math.cos(angle)*(y - cy))


class Fig:
    """Stores graphics primitive objects and applies a single coordinate
    transformation to them. To compose coordinate systems, nest Fig
    objects.

    Fig(obj, obj, obj..., trans=function)

    obj     optional list    a list of drawing primitives
    trans   default=None     a coordinate transformation function

    >>> fig = Fig(Line(0,0,1,1), Rect(0.2,0.2,0.8,0.8), trans="2*x, 2*y")
    >>> print fig.SVG().xml()
    <g>
        <path d='M0 0L2 2' />
        <path d='M0.4 0.4L1.6 0.4ZL1.6 1.6ZL0.4 1.6ZL0.4 0.4ZZ' />
    </g>
    >>> print Fig(fig, trans="x/2., y/2.").SVG().xml()
    <g>
        <path d='M0 0L1 1' />
        <path d='M0.2 0.2L0.8 0.2ZL0.8 0.8ZL0.2 0.8ZL0.2 0.2ZZ' />
    </g>
    """

    def __repr__(self):
        if self.trans is None:
            return "<Fig (%d items)>" % len(self.d)
        elif isinstance(self.trans, basestring):
            return "<Fig (%d items) x,y -> %s>" % (len(self.d), self.trans)
        else:
            return "<Fig (%d items) %s>" % (len(self.d), self.trans.func_name)

    def __init__(self, *d, **kwds):
        self.d = list(d)
        defaults = {"trans": None, }
        defaults.update(kwds)
        kwds = defaults

        self.trans = kwds["trans"]; del kwds["trans"]
        if len(kwds) != 0:
            raise TypeError ("Fig() got unexpected keyword arguments %s" % kwds.keys())

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object.

        Coordinate transformations in nested Figs will be composed.
        """

        if trans is None:
            trans = self.trans
        if isinstance(trans, basestring):
            trans = totrans(trans)

        output = SVG("g")
        for s in self.d:
            if isinstance(s, SVG):
                output.append(s)

            elif isinstance(s, Fig):
                strans = s.trans
                if isinstance(strans, basestring):
                    strans = totrans(strans)

                if trans is None:
                    subtrans = strans
                elif strans is None:
                    subtrans = trans
                else:
                    subtrans = lambda x, y: trans(*strans(x, y))

                output.sub += s.SVG(subtrans).sub

            elif s is None:
                pass

            else:
                output.append(s.SVG(trans))

        return output


class Plot:
    """Acts like Fig, but draws a coordinate axis. You also need to supply plot ranges.

    Plot(xmin, xmax, ymin, ymax, obj, obj, obj..., keyword options...)

    xmin, xmax      required        minimum and maximum x values (in the objs' coordinates)
    ymin, ymax      required        minimum and maximum y values (in the objs' coordinates)
    obj             optional list   drawing primitives
    keyword options keyword list    options defined below

    The following are keyword options, with their default values:

    trans           None          transformation function
    x, y            5, 5          upper-left corner of the Plot in SVG coordinates
    width, height   90, 90        width and height of the Plot in SVG coordinates
    flipx, flipy    False, True   flip the sign of the coordinate axis
    minusInfinity   -1000         if an axis is logarithmic and an object is plotted at 0 or
                                  a negative value, -1000 will be used as a stand-in for NaN
    atx, aty        0, 0          the place where the coordinate axes cross
    xticks          -10           request ticks according to the standard tick specification
                                  (see help(Ticks))
    xminiticks      True          request miniticks according to the standard minitick
                                  specification
    xlabels         True          request tick labels according to the standard tick label
                                  specification
    xlogbase        None          if a number, the axis and transformation are logarithmic
                                  with ticks at the given base (10 being the most common)
    (same for y)
    arrows          None          if a new identifier, create arrow markers and draw them
                                  at the ends of the coordinate axes
    text_attr       {}            a dictionary of attributes for label text
    axis_attr       {}            a dictionary of attributes for the axis lines
    """

    def __repr__(self):
        if self.trans is None:
            return "<Plot (%d items)>" % len(self.d)
        else:
            return "<Plot (%d items) %s>" % (len(self.d), self.trans.func_name)

    def __init__(self, xmin, xmax, ymin, ymax, *d, **kwds):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.d = list(d)
        defaults = {"trans": None,
                    "x": 5, "y": 5, "width": 90, "height": 90,
                    "flipx": False, "flipy": True,
                    "minusInfinity": -1000,
                    "atx": 0, "xticks": -10, "xminiticks": True, "xlabels": True, "xlogbase": None,
                    "aty": 0, "yticks": -10, "yminiticks": True, "ylabels": True, "ylogbase": None,
                    "arrows": None,
                    "text_attr": {}, "axis_attr": {},
                   }
        defaults.update(kwds)
        kwds = defaults

        self.trans = kwds["trans"]; del kwds["trans"]
        self.x = kwds["x"]; del kwds["x"]
        self.y = kwds["y"]; del kwds["y"]
        self.width = kwds["width"]; del kwds["width"]
        self.height = kwds["height"]; del kwds["height"]
        self.flipx = kwds["flipx"]; del kwds["flipx"]
        self.flipy = kwds["flipy"]; del kwds["flipy"]
        self.minusInfinity = kwds["minusInfinity"]; del kwds["minusInfinity"]
        self.atx = kwds["atx"]; del kwds["atx"]
        self.xticks = kwds["xticks"]; del kwds["xticks"]
        self.xminiticks = kwds["xminiticks"]; del kwds["xminiticks"]
        self.xlabels = kwds["xlabels"]; del kwds["xlabels"]
        self.xlogbase = kwds["xlogbase"]; del kwds["xlogbase"]
        self.aty = kwds["aty"]; del kwds["aty"]
        self.yticks = kwds["yticks"]; del kwds["yticks"]
        self.yminiticks = kwds["yminiticks"]; del kwds["yminiticks"]
        self.ylabels = kwds["ylabels"]; del kwds["ylabels"]
        self.ylogbase = kwds["ylogbase"]; del kwds["ylogbase"]
        self.arrows = kwds["arrows"]; del kwds["arrows"]
        self.text_attr = kwds["text_attr"]; del kwds["text_attr"]
        self.axis_attr = kwds["axis_attr"]; del kwds["axis_attr"]
        if len(kwds) != 0:
            raise TypeError ("Plot() got unexpected keyword arguments %s" % kwds.keys())

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if trans is None:
            trans = self.trans
        if isinstance(trans, basestring):
            trans = totrans(trans)

        self.last_window = window(self.xmin, self.xmax, self.ymin, self.ymax,
                                  x=self.x, y=self.y, width=self.width, height=self.height,
                                  xlogbase=self.xlogbase, ylogbase=self.ylogbase,
                                  minusInfinity=self.minusInfinity, flipx=self.flipx, flipy=self.flipy)

        d = ([Axes(self.xmin, self.xmax, self.ymin, self.ymax, self.atx, self.aty,
                   self.xticks, self.xminiticks, self.xlabels, self.xlogbase,
                   self.yticks, self.yminiticks, self.ylabels, self.ylogbase,
                   self.arrows, self.text_attr, **self.axis_attr)]
             + self.d)

        return Fig(Fig(*d, **{"trans": trans})).SVG(self.last_window)


class Frame:
    text_defaults = {"stroke": "none", "fill": "black", "font-size": 5, }
    axis_defaults = {}

    tick_length = 1.5
    minitick_length = 0.75
    text_xaxis_offset = 1.
    text_yaxis_offset = 2.
    text_xtitle_offset = 6.
    text_ytitle_offset = 12.

    def __repr__(self):
        return "<Frame (%d items)>" % len(self.d)

    def __init__(self, xmin, xmax, ymin, ymax, *d, **kwds):
        """Acts like Fig, but draws a coordinate frame around the data. You also need to supply plot ranges.

        Frame(xmin, xmax, ymin, ymax, obj, obj, obj..., keyword options...)

        xmin, xmax      required        minimum and maximum x values (in the objs' coordinates)
        ymin, ymax      required        minimum and maximum y values (in the objs' coordinates)
        obj             optional list   drawing primitives
        keyword options keyword list    options defined below

        The following are keyword options, with their default values:

        x, y            20, 5         upper-left corner of the Frame in SVG coordinates
        width, height   75, 80        width and height of the Frame in SVG coordinates
        flipx, flipy    False, True   flip the sign of the coordinate axis
        minusInfinity   -1000         if an axis is logarithmic and an object is plotted at 0 or
                                      a negative value, -1000 will be used as a stand-in for NaN
        xtitle          None          if a string, label the x axis
        xticks          -10           request ticks according to the standard tick specification
                                      (see help(Ticks))
        xminiticks      True          request miniticks according to the standard minitick
                                      specification
        xlabels         True          request tick labels according to the standard tick label
                                      specification
        xlogbase        None          if a number, the axis and transformation are logarithmic
                                      with ticks at the given base (10 being the most common)
        (same for y)
        text_attr       {}            a dictionary of attributes for label text
        axis_attr       {}            a dictionary of attributes for the axis lines
        """

        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.d = list(d)
        defaults = {"x": 20, "y": 5, "width": 75, "height": 80,
                    "flipx": False, "flipy": True, "minusInfinity": -1000,
                    "xtitle": None, "xticks": -10, "xminiticks": True, "xlabels": True,
                    "x2labels": None, "xlogbase": None,
                    "ytitle": None, "yticks": -10, "yminiticks": True, "ylabels": True,
                    "y2labels": None, "ylogbase": None,
                    "text_attr": {}, "axis_attr": {},
                   }
        defaults.update(kwds)
        kwds = defaults

        self.x = kwds["x"]; del kwds["x"]
        self.y = kwds["y"]; del kwds["y"]
        self.width = kwds["width"]; del kwds["width"]
        self.height = kwds["height"]; del kwds["height"]
        self.flipx = kwds["flipx"]; del kwds["flipx"]
        self.flipy = kwds["flipy"]; del kwds["flipy"]
        self.minusInfinity = kwds["minusInfinity"]; del kwds["minusInfinity"]
        self.xtitle = kwds["xtitle"]; del kwds["xtitle"]
        self.xticks = kwds["xticks"]; del kwds["xticks"]
        self.xminiticks = kwds["xminiticks"]; del kwds["xminiticks"]
        self.xlabels = kwds["xlabels"]; del kwds["xlabels"]
        self.x2labels = kwds["x2labels"]; del kwds["x2labels"]
        self.xlogbase = kwds["xlogbase"]; del kwds["xlogbase"]
        self.ytitle = kwds["ytitle"]; del kwds["ytitle"]
        self.yticks = kwds["yticks"]; del kwds["yticks"]
        self.yminiticks = kwds["yminiticks"]; del kwds["yminiticks"]
        self.ylabels = kwds["ylabels"]; del kwds["ylabels"]
        self.y2labels = kwds["y2labels"]; del kwds["y2labels"]
        self.ylogbase = kwds["ylogbase"]; del kwds["ylogbase"]

        self.text_attr = dict(self.text_defaults)
        self.text_attr.update(kwds["text_attr"]); del kwds["text_attr"]

        self.axis_attr = dict(self.axis_defaults)
        self.axis_attr.update(kwds["axis_attr"]); del kwds["axis_attr"]

        if len(kwds) != 0:
            raise TypeError( "Frame() got unexpected keyword arguments %s" % kwds.keys())

    def SVG(self):
        """Apply the window transformation and return an SVG object."""

        self.last_window = window(self.xmin, self.xmax, self.ymin, self.ymax,
                                  x=self.x, y=self.y, width=self.width, height=self.height,
                                  xlogbase=self.xlogbase, ylogbase=self.ylogbase,
                                  minusInfinity=self.minusInfinity, flipx=self.flipx, flipy=self.flipy)

        left = YAxis(self.ymin, self.ymax, self.xmin, self.yticks, self.yminiticks, self.ylabels, self.ylogbase,
                     None, None, None, self.text_attr, **self.axis_attr)
        right = YAxis(self.ymin, self.ymax, self.xmax, self.yticks, self.yminiticks, self.y2labels, self.ylogbase,
                      None, None, None, self.text_attr, **self.axis_attr)
        bottom = XAxis(self.xmin, self.xmax, self.ymin, self.xticks, self.xminiticks, self.xlabels, self.xlogbase,
                       None, None, None, self.text_attr, **self.axis_attr)
        top = XAxis(self.xmin, self.xmax, self.ymax, self.xticks, self.xminiticks, self.x2labels, self.xlogbase,
                    None, None, None, self.text_attr, **self.axis_attr)

        left.tick_start = -self.tick_length
        left.tick_end = 0
        left.minitick_start = -self.minitick_length
        left.minitick_end = 0.
        left.text_start = self.text_yaxis_offset

        right.tick_start = 0.
        right.tick_end = self.tick_length
        right.minitick_start = 0.
        right.minitick_end = self.minitick_length
        right.text_start = -self.text_yaxis_offset
        right.text_attr["text-anchor"] = "start"

        bottom.tick_start = 0.
        bottom.tick_end = self.tick_length
        bottom.minitick_start = 0.
        bottom.minitick_end = self.minitick_length
        bottom.text_start = -self.text_xaxis_offset

        top.tick_start = -self.tick_length
        top.tick_end = 0.
        top.minitick_start = -self.minitick_length
        top.minitick_end = 0.
        top.text_start = self.text_xaxis_offset
        top.text_attr["dominant-baseline"] = "text-after-edge"

        output = Fig(*self.d).SVG(self.last_window)
        output.prepend(left.SVG(self.last_window))
        output.prepend(bottom.SVG(self.last_window))
        output.prepend(right.SVG(self.last_window))
        output.prepend(top.SVG(self.last_window))

        if self.xtitle is not None:
            output.append(SVG("text", self.xtitle, transform="translate(%g, %g)" % ((self.x + self.width/2.), (self.y + self.height + self.text_xtitle_offset)), dominant_baseline="text-before-edge", **self.text_attr))
        if self.ytitle is not None:
            output.append(SVG("text", self.ytitle, transform="translate(%g, %g) rotate(-90)" % ((self.x - self.text_ytitle_offset), (self.y + self.height/2.)), **self.text_attr))
        return output

######################################################################

def pathtoPath(svg):
    """Converts SVG("path", d="...") into Path(d=[...])."""
    if not isinstance(svg, SVG) or svg.t != "path":
        raise TypeError ("Only SVG <path /> objects can be converted into Paths")
    attr = dict(svg.attr)
    d = attr["d"]
    del attr["d"]
    for key in attr.keys():
        if not isinstance(key, str):
            value = attr[key]
            del attr[key]
            attr[str(key)] = value
    return Path(d, **attr)


class Path:
    """Path represents an SVG path, an arbitrary set of curves and
    straight segments. Unlike SVG("path", d="..."), Path stores
    coordinates as a list of numbers, rather than a string, so that it is
    transformable in a Fig.

    Path(d, attribute=value)

    d                       required        path data
    attribute=value pairs   keyword list    SVG attributes

    See http://www.w3.org/TR/SVG/paths.html for specification of paths
    from text.

    Internally, Path data is a list of tuples with these definitions:

        * ("Z/z",): close the current path
        * ("H/h", x) or ("V/v", y): a horizontal or vertical line
          segment to x or y
        * ("M/m/L/l/T/t", x, y, global): moveto, lineto, or smooth
          quadratic curveto point (x, y). If global=True, (x, y) should
          not be transformed.
        * ("S/sQ/q", cx, cy, cglobal, x, y, global): polybezier or
          smooth quadratic curveto point (x, y) using (cx, cy) as a
          control point. If cglobal or global=True, (cx, cy) or (x, y)
          should not be transformed.
        * ("C/c", c1x, c1y, c1global, c2x, c2y, c2global, x, y, global):
          cubic curveto point (x, y) using (c1x, c1y) and (c2x, c2y) as
          control points. If c1global, c2global, or global=True, (c1x, c1y),
          (c2x, c2y), or (x, y) should not be transformed.
        * ("A/a", rx, ry, rglobal, x-axis-rotation, angle, large-arc-flag,
          sweep-flag, x, y, global): arcto point (x, y) using the
          aforementioned parameters.
        * (",/.", rx, ry, rglobal, angle, x, y, global): an ellipse at
          point (x, y) with radii (rx, ry). If angle is 0, the whole
          ellipse is drawn; otherwise, a partial ellipse is drawn.
    """
    defaults = {}

    def __repr__(self):
        return "<Path (%d nodes) %s>" % (len(self.d), self.attr)

    def __init__(self, d=[], **attr):
        if isinstance(d, basestring):
            self.d = self.parse(d)
        else:
            self.d = list(d)

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def parse_whitespace(self, index, pathdata):
        """Part of Path's text-command parsing algorithm; used internally."""
        while index < len(pathdata) and pathdata[index] in (" ", "\t", "\r", "\n", ","):
            index += 1
        return index, pathdata

    def parse_command(self, index, pathdata):
        """Part of Path's text-command parsing algorithm; used internally."""
        index, pathdata = self.parse_whitespace(index, pathdata)

        if index >= len(pathdata):
            return None, index, pathdata
        command = pathdata[index]
        if "A" <= command <= "Z" or "a" <= command <= "z":
            index += 1
            return command, index, pathdata
        else:
            return None, index, pathdata

    def parse_number(self, index, pathdata):
        """Part of Path's text-command parsing algorithm; used internally."""
        index, pathdata = self.parse_whitespace(index, pathdata)

        if index >= len(pathdata):
            return None, index, pathdata
        first_digit = pathdata[index]

        if "0" <= first_digit <= "9" or first_digit in ("-", "+", "."):
            start = index
            while index < len(pathdata) and ("0" <= pathdata[index] <= "9" or pathdata[index] in ("-", "+", ".", "e", "E")):
                index += 1
            end = index

            index = end
            return float(pathdata[start:end]), index, pathdata
        else:
            return None, index, pathdata

    def parse_boolean(self, index, pathdata):
        """Part of Path's text-command parsing algorithm; used internally."""
        index, pathdata = self.parse_whitespace(index, pathdata)

        if index >= len(pathdata):
            return None, index, pathdata
        first_digit = pathdata[index]

        if first_digit in ("0", "1"):
            index += 1
            return int(first_digit), index, pathdata
        else:
            return None, index, pathdata

    def parse(self, pathdata):
        """Parses text-commands, converting them into a list of tuples.
        Called by the constructor."""
        output = []
        index = 0
        while True:
            command, index, pathdata = self.parse_command(index, pathdata)
            index, pathdata = self.parse_whitespace(index, pathdata)

            if command is None and index == len(pathdata):
                break  # this is the normal way out of the loop
            if command in ("Z", "z"):
                output.append((command,))

            ######################
            elif command in ("H", "h", "V", "v"):
                errstring = "Path command \"%s\" requires a number at index %d" % (command, index)
                num1, index, pathdata = self.parse_number(index, pathdata)
                if num1 is None:
                    raise ValueError ( errstring)

                while num1 is not None:
                    output.append((command, num1))
                    num1, index, pathdata = self.parse_number(index, pathdata)

            ######################
            elif command in ("M", "m", "L", "l", "T", "t"):
                errstring = "Path command \"%s\" requires an x,y pair at index %d" % (command, index)
                num1, index, pathdata = self.parse_number(index, pathdata)
                num2, index, pathdata = self.parse_number(index, pathdata)

                if num1 is None:
                    raise ValueError ( errstring)

                while num1 is not None:
                    if num2 is None:
                        raise ValueError ( errstring)
                    output.append((command, num1, num2, False))

                    num1, index, pathdata = self.parse_number(index, pathdata)
                    num2, index, pathdata = self.parse_number(index, pathdata)

            ######################
            elif command in ("S", "s", "Q", "q"):
                errstring = "Path command \"%s\" requires a cx,cy,x,y quadruplet at index %d" % (command, index)
                num1, index, pathdata = self.parse_number(index, pathdata)
                num2, index, pathdata = self.parse_number(index, pathdata)
                num3, index, pathdata = self.parse_number(index, pathdata)
                num4, index, pathdata = self.parse_number(index, pathdata)

                if num1 is None:
                    raise ValueError ( errstring )

                while num1 is not None:
                    if num2 is None or num3 is None or num4 is None:
                        raise ValueError (errstring)
                    output.append((command, num1, num2, False, num3, num4, False))

                    num1, index, pathdata = self.parse_number(index, pathdata)
                    num2, index, pathdata = self.parse_number(index, pathdata)
                    num3, index, pathdata = self.parse_number(index, pathdata)
                    num4, index, pathdata = self.parse_number(index, pathdata)

            ######################
            elif command in ("C", "c"):
                errstring = "Path command \"%s\" requires a c1x,c1y,c2x,c2y,x,y sextuplet at index %d" % (command, index)
                num1, index, pathdata = self.parse_number(index, pathdata)
                num2, index, pathdata = self.parse_number(index, pathdata)
                num3, index, pathdata = self.parse_number(index, pathdata)
                num4, index, pathdata = self.parse_number(index, pathdata)
                num5, index, pathdata = self.parse_number(index, pathdata)
                num6, index, pathdata = self.parse_number(index, pathdata)

                if num1 is None:
                    raise ValueError(errstring)

                while num1 is not None:
                    if num2 is None or num3 is None or num4 is None or num5 is None or num6 is None:
                        raise ValueError(errstring)

                    output.append((command, num1, num2, False, num3, num4, False, num5, num6, False))

                    num1, index, pathdata = self.parse_number(index, pathdata)
                    num2, index, pathdata = self.parse_number(index, pathdata)
                    num3, index, pathdata = self.parse_number(index, pathdata)
                    num4, index, pathdata = self.parse_number(index, pathdata)
                    num5, index, pathdata = self.parse_number(index, pathdata)
                    num6, index, pathdata = self.parse_number(index, pathdata)

            ######################
            elif command in ("A", "a"):
                errstring = "Path command \"%s\" requires a rx,ry,angle,large-arc-flag,sweep-flag,x,y septuplet at index %d" % (command, index)
                num1, index, pathdata = self.parse_number(index, pathdata)
                num2, index, pathdata = self.parse_number(index, pathdata)
                num3, index, pathdata = self.parse_number(index, pathdata)
                num4, index, pathdata = self.parse_boolean(index, pathdata)
                num5, index, pathdata = self.parse_boolean(index, pathdata)
                num6, index, pathdata = self.parse_number(index, pathdata)
                num7, index, pathdata = self.parse_number(index, pathdata)

                if num1 is None:
                    raise ValueError(errstring)

                while num1 is not None:
                    if num2 is None or num3 is None or num4 is None or num5 is None or num6 is None or num7 is None:
                        raise ValueError(errstring)

                    output.append((command, num1, num2, False, num3, num4, num5, num6, num7, False))

                    num1, index, pathdata = self.parse_number(index, pathdata)
                    num2, index, pathdata = self.parse_number(index, pathdata)
                    num3, index, pathdata = self.parse_number(index, pathdata)
                    num4, index, pathdata = self.parse_boolean(index, pathdata)
                    num5, index, pathdata = self.parse_boolean(index, pathdata)
                    num6, index, pathdata = self.parse_number(index, pathdata)
                    num7, index, pathdata = self.parse_number(index, pathdata)

        return output

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if isinstance(trans, basestring):
            trans = totrans(trans)

        x, y, X, Y = None, None, None, None
        output = []
        for datum in self.d:
            if not isinstance(datum, (tuple, list)):
                raise TypeError("pathdata elements must be tuples/lists")

            command = datum[0]

            ######################
            if command in ("Z", "z"):
                x, y, X, Y = None, None, None, None
                output.append("Z")

            ######################
            elif command in ("H", "h", "V", "v"):
                command, num1 = datum

                if command == "H" or (command == "h" and x is None):
                    x = num1
                elif command == "h":
                    x += num1
                elif command == "V" or (command == "v" and y is None):
                    y = num1
                elif command == "v":
                    y += num1

                if trans is None:
                    X, Y = x, y
                else:
                    X, Y = trans(x, y)

                output.append("L%g %g" % (X, Y))

            ######################
            elif command in ("M", "m", "L", "l", "T", "t"):
                command, num1, num2, isglobal12 = datum

                if trans is None or isglobal12:
                    if command.isupper() or X is None or Y is None:
                        X, Y = num1, num2
                    else:
                        X += num1
                        Y += num2
                    x, y = X, Y

                else:
                    if command.isupper() or x is None or y is None:
                        x, y = num1, num2
                    else:
                        x += num1
                        y += num2
                    X, Y = trans(x, y)

                COMMAND = command.capitalize()
                output.append("%s%g %g" % (COMMAND, X, Y))

            ######################
            elif command in ("S", "s", "Q", "q"):
                command, num1, num2, isglobal12, num3, num4, isglobal34 = datum

                if trans is None or isglobal12:
                    if command.isupper() or X is None or Y is None:
                        CX, CY = num1, num2
                    else:
                        CX = X + num1
                        CY = Y + num2

                else:
                    if command.isupper() or x is None or y is None:
                        cx, cy = num1, num2
                    else:
                        cx = x + num1
                        cy = y + num2
                    CX, CY = trans(cx, cy)

                if trans is None or isglobal34:
                    if command.isupper() or X is None or Y is None:
                        X, Y = num3, num4
                    else:
                        X += num3
                        Y += num4
                    x, y = X, Y

                else:
                    if command.isupper() or x is None or y is None:
                        x, y = num3, num4
                    else:
                        x += num3
                        y += num4
                    X, Y = trans(x, y)

                COMMAND = command.capitalize()
                output.append("%s%g %g %g %g" % (COMMAND, CX, CY, X, Y))

            ######################
            elif command in ("C", "c"):
                command, num1, num2, isglobal12, num3, num4, isglobal34, num5, num6, isglobal56 = datum

                if trans is None or isglobal12:
                    if command.isupper() or X is None or Y is None:
                        C1X, C1Y = num1, num2
                    else:
                        C1X = X + num1
                        C1Y = Y + num2

                else:
                    if command.isupper() or x is None or y is None:
                        c1x, c1y = num1, num2
                    else:
                        c1x = x + num1
                        c1y = y + num2
                    C1X, C1Y = trans(c1x, c1y)

                if trans is None or isglobal34:
                    if command.isupper() or X is None or Y is None:
                        C2X, C2Y = num3, num4
                    else:
                        C2X = X + num3
                        C2Y = Y + num4

                else:
                    if command.isupper() or x is None or y is None:
                        c2x, c2y = num3, num4
                    else:
                        c2x = x + num3
                        c2y = y + num4
                    C2X, C2Y = trans(c2x, c2y)

                if trans is None or isglobal56:
                    if command.isupper() or X is None or Y is None:
                        X, Y = num5, num6
                    else:
                        X += num5
                        Y += num6
                    x, y = X, Y

                else:
                    if command.isupper() or x is None or y is None:
                        x, y = num5, num6
                    else:
                        x += num5
                        y += num6
                    X, Y = trans(x, y)

                COMMAND = command.capitalize()
                output.append("%s%g %g %g %g %g %g" % (COMMAND, C1X, C1Y, C2X, C2Y, X, Y))

            ######################
            elif command in ("A", "a"):
                command, num1, num2, isglobal12, angle, large_arc_flag, sweep_flag, num3, num4, isglobal34 = datum

                oldx, oldy = x, y
                OLDX, OLDY = X, Y

                if trans is None or isglobal34:
                    if command.isupper() or X is None or Y is None:
                        X, Y = num3, num4
                    else:
                        X += num3
                        Y += num4
                    x, y = X, Y

                else:
                    if command.isupper() or x is None or y is None:
                        x, y = num3, num4
                    else:
                        x += num3
                        y += num4
                    X, Y = trans(x, y)

                if x is not None and y is not None:
                    centerx, centery = (x + oldx)/2., (y + oldy)/2.
                CENTERX, CENTERY = (X + OLDX)/2., (Y + OLDY)/2.

                if trans is None or isglobal12:
                    RX = CENTERX + num1
                    RY = CENTERY + num2

                else:
                    rx = centerx + num1
                    ry = centery + num2
                    RX, RY = trans(rx, ry)

                COMMAND = command.capitalize()
                output.append("%s%g %g %g %d %d %g %g" % (COMMAND, RX - CENTERX, RY - CENTERY, angle, large_arc_flag, sweep_flag, X, Y))

            elif command in (",", "."):
                command, num1, num2, isglobal12, angle, num3, num4, isglobal34 = datum
                if trans is None or isglobal34:
                    if command == "." or X is None or Y is None:
                        X, Y = num3, num4
                    else:
                        X += num3
                        Y += num4
                        x, y = None, None

                else:
                    if command == "." or x is None or y is None:
                        x, y = num3, num4
                    else:
                        x += num3
                        y += num4
                    X, Y = trans(x, y)

                if trans is None or isglobal12:
                    RX = X + num1
                    RY = Y + num2

                else:
                    rx = x + num1
                    ry = y + num2
                    RX, RY = trans(rx, ry)

                RX, RY = RX - X, RY - Y

                X1, Y1 = X + RX * math.cos(angle*math.pi/180.), Y + RX * math.sin(angle*math.pi/180.)
                X2, Y2 = X + RY * math.sin(angle*math.pi/180.), Y - RY * math.cos(angle*math.pi/180.)
                X3, Y3 = X - RX * math.cos(angle*math.pi/180.), Y - RX * math.sin(angle*math.pi/180.)
                X4, Y4 = X - RY * math.sin(angle*math.pi/180.), Y + RY * math.cos(angle*math.pi/180.)

                output.append("M%g %gA%g %g %g 0 0 %g %gA%g %g %g 0 0 %g %gA%g %g %g 0 0 %g %gA%g %g %g 0 0 %g %g" % (
                              X1, Y1, RX, RY, angle, X2, Y2, RX, RY, angle, X3, Y3, RX, RY, angle, X4, Y4, RX, RY, angle, X1, Y1))

        return SVG("path", d="".join(output), **self.attr)

######################################################################

def funcRtoC(expr, var="t", globals=None, locals=None):
    """Converts a complex "z(t)" string to a function acceptable for Curve.

    expr    required        string in the form "z(t)"
    var     default="t"     name of the independent variable
    globals default=None    dict of global variables used in the expression;
                            you may want to use Python's builtin globals()
    locals  default=None    dict of local variables
    """
    if locals is None:
        locals = {}  # python 2.3's eval() won't accept None
    g = cmath.__dict__
    if globals is not None:
        g.update(globals)
    output = eval("lambda %s: (%s)" % (var, expr), g, locals)
    split = lambda z: (z.real, z.imag)
    output2 = lambda t: split(output(t))
    set_func_name(output2, "%s -> %s" % (var, expr))
    return output2


def funcRtoR2(expr, var="t", globals=None, locals=None):
    """Converts a "f(t), g(t)" string to a function acceptable for Curve.

    expr    required        string in the form "f(t), g(t)"
    var     default="t"     name of the independent variable
    globals default=None    dict of global variables used in the expression;
                            you may want to use Python's builtin globals()
    locals  default=None    dict of local variables
    """
    if locals is None:
        locals = {}  # python 2.3's eval() won't accept None
    g = math.__dict__
    if globals is not None:
        g.update(globals)
    output = eval("lambda %s: (%s)" % (var, expr), g, locals)
    set_func_name(output, "%s -> %s" % (var, expr))
    return output


def funcRtoR(expr, var="x", globals=None, locals=None):
    """Converts a "f(x)" string to a function acceptable for Curve.

    expr    required        string in the form "f(x)"
    var     default="x"     name of the independent variable
    globals default=None    dict of global variables used in the expression;
                            you may want to use Python's builtin globals()
    locals  default=None    dict of local variables
    """
    if locals is None:
        locals = {}  # python 2.3's eval() won't accept None
    g = math.__dict__
    if globals is not None:
        g.update(globals)
    output = eval("lambda %s: (%s, %s)" % (var, var, expr), g, locals)
    set_func_name(output, "%s -> %s" % (var, expr))
    return output


class Curve:
    """Draws a parametric function as a path.

    Curve(f, low, high, loop, attribute=value)

    f                      required         a Python callable or string in
                                            the form "f(t), g(t)"
    low, high              required         left and right endpoints
    loop                   default=False    if True, connect the endpoints
    attribute=value pairs  keyword list     SVG attributes
    """
    defaults = {}
    random_sampling = True
    recursion_limit = 15
    linearity_limit = 0.05
    discontinuity_limit = 5.

    def __repr__(self):
        return "<Curve %s [%s, %s] %s>" % (self.f, self.low, self.high, self.attr)

    def __init__(self, f, low, high, loop=False, **attr):
        self.f = f
        self.low = low
        self.high = high
        self.loop = loop

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    ### nested class Sample
    class Sample:
        def __repr__(self):
            t, x, y, X, Y = self.t, self.x, self.y, self.X, self.Y
            if t is not None:
                t = "%g" % t
            if x is not None:
                x = "%g" % x
            if y is not None:
                y = "%g" % y
            if X is not None:
                X = "%g" % X
            if Y is not None:
                Y = "%g" % Y
            return "<Curve.Sample t=%s x=%s y=%s X=%s Y=%s>" % (t, x, y, X, Y)

        def __init__(self, t):
            self.t = t

        def link(self, left, right):
            self.left, self.right = left, right

        def evaluate(self, f, trans):
            self.x, self.y = f(self.t)
            if trans is None:
                self.X, self.Y = self.x, self.y
            else:
                self.X, self.Y = trans(self.x, self.y)
    ### end Sample

    ### nested class Samples
    class Samples:
        def __repr__(self):
            return "<Curve.Samples (%d samples)>" % len(self)

        def __init__(self, left, right):
            self.left, self.right = left, right

        def __len__(self):
            count = 0
            current = self.left
            while current is not None:
                count += 1
                current = current.right
            return count

        def __iter__(self):
            self.current = self.left
            return self

        def next(self):
            current = self.current
            if current is None:
                raise StopIteration
            self.current = self.current.right
            return current
    ### end nested class

    def sample(self, trans=None):
        """Adaptive-sampling algorithm that chooses the best sample points
        for a parametric curve between two endpoints and detects
        discontinuities.  Called by SVG()."""
        oldrecursionlimit = sys.getrecursionlimit()
        sys.setrecursionlimit(self.recursion_limit + 100)
        try:
            # the best way to keep all the information while sampling is to make a linked list
            if not (self.low < self.high):
                raise ValueError("low must be less than high")
            low, high = self.Sample(float(self.low)), self.Sample(float(self.high))
            low.link(None, high)
            high.link(low, None)

            low.evaluate(self.f, trans)
            high.evaluate(self.f, trans)

            # adaptive sampling between the low and high points
            self.subsample(low, high, 0, trans)

            # Prune excess points where the curve is nearly linear
            left = low
            while left.right is not None:
                # increment mid and right
                mid = left.right
                right = mid.right
                if (right is not None and
                    left.X is not None and left.Y is not None and
                    mid.X is not None and mid.Y is not None and
                    right.X is not None and right.Y is not None):
                    numer = left.X*(right.Y - mid.Y) + mid.X*(left.Y - right.Y) + right.X*(mid.Y - left.Y)
                    denom = math.sqrt((left.X - right.X)**2 + (left.Y - right.Y)**2)
                    if denom != 0. and abs(numer/denom) < self.linearity_limit:
                        # drop mid (the garbage collector will get it)
                        left.right = right
                        right.left = left
                    else:
                        # increment left
                        left = left.right
                else:
                    left = left.right

            self.last_samples = self.Samples(low, high)

        finally:
            sys.setrecursionlimit(oldrecursionlimit)

    def subsample(self, left, right, depth, trans=None):
        """Part of the adaptive-sampling algorithm that chooses the best
        sample points.  Called by sample()."""

        if self.random_sampling:
            mid = self.Sample(left.t + random.uniform(0.3, 0.7) * (right.t - left.t))
        else:
            mid = self.Sample(left.t + 0.5 * (right.t - left.t))

        left.right = mid
        right.left = mid
        mid.link(left, right)
        mid.evaluate(self.f, trans)

        # calculate the distance of closest approach of mid to the line between left and right
        numer = left.X*(right.Y - mid.Y) + mid.X*(left.Y - right.Y) + right.X*(mid.Y - left.Y)
        denom = math.sqrt((left.X - right.X)**2 + (left.Y - right.Y)**2)

        # if we haven't sampled enough or left fails to be close enough to right, or mid fails to be linear enough...
        if (depth < 3 or
            (denom == 0 and left.t != right.t) or
            denom > self.discontinuity_limit or
            (denom != 0. and abs(numer/denom) > self.linearity_limit)):

            # and we haven't sampled too many points
            if depth < self.recursion_limit:
                self.subsample(left, mid, depth+1, trans)
                self.subsample(mid, right, depth+1, trans)

            else:
                # We've sampled many points and yet it's still not a small linear gap.
                # Break the line: it's a discontinuity
                mid.y = mid.Y = None

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        return self.Path(trans).SVG()

    def Path(self, trans=None, local=False):
        """Apply the transformation "trans" and return a Path object in
        global coordinates.  If local=True, return a Path in local coordinates
        (which must be transformed again)."""

        if isinstance(trans, basestring):
            trans = totrans(trans)
        if isinstance(self.f, basestring):
            self.f = funcRtoR2(self.f)

        self.sample(trans)

        output = []
        for s in self.last_samples:
            if s.X is not None and s.Y is not None:
                if s.left is None or s.left.Y is None:
                    command = "M"
                else:
                    command = "L"

                if local:
                    output.append((command, s.x, s.y, False))
                else:
                    output.append((command, s.X, s.Y, True))

        if self.loop:
            output.append(("Z",))
        return Path(output, **self.attr)

######################################################################

class Poly:
    """Draws a curve specified by a sequence of points. The curve may be
    piecewise linear, like a polygon, or a Bezier curve.

    Poly(d, mode, loop, attribute=value)

    d                       required        list of tuples representing points
                                            and possibly control points
    mode                    default="L"     "lines", "bezier", "velocity",
                                            "foreback", "smooth", or an abbreviation
    loop                    default=False   if True, connect the first and last
                                            point, closing the loop
    attribute=value pairs   keyword list    SVG attributes

    The format of the tuples in d depends on the mode.

    "lines"/"L"         d=[(x,y), (x,y), ...]
                                            piecewise-linear segments joining the (x,y) points
    "bezier"/"B"        d=[(x, y, c1x, c1y, c2x, c2y), ...]
                                            Bezier curve with two control points (control points
                                            precede (x,y), as in SVG paths). If (c1x,c1y) and
                                            (c2x,c2y) both equal (x,y), you get a linear
                                            interpolation ("lines")
    "velocity"/"V"      d=[(x, y, vx, vy), ...]
                                            curve that passes through (x,y) with velocity (vx,vy)
                                            (one unit of arclength per unit time); in other words,
                                            (vx,vy) is the tangent vector at (x,y). If (vx,vy) is
                                            (0,0), you get a linear interpolation ("lines").
    "foreback"/"F"      d=[(x, y, bx, by, fx, fy), ...]
                                            like "velocity" except that there is a left derivative
                                            (bx,by) and a right derivative (fx,fy). If (bx,by)
                                            equals (fx,fy) (with no minus sign), you get a
                                            "velocity" curve
    "smooth"/"S"        d=[(x,y), (x,y), ...]
                                            a "velocity" interpolation with (vx,vy)[i] equal to
                                            ((x,y)[i+1] - (x,y)[i-1])/2: the minimal derivative
    """
    defaults = {}

    def __repr__(self):
        return "<Poly (%d nodes) mode=%s loop=%s %s>" % (
               len(self.d), self.mode, repr(self.loop), self.attr)

    def __init__(self, d=[], mode="L", loop=False, **attr):
        self.d = list(d)
        self.mode = mode
        self.loop = loop

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        return self.Path(trans).SVG()

    def Path(self, trans=None, local=False):
        """Apply the transformation "trans" and return a Path object in
        global coordinates.  If local=True, return a Path in local coordinates
        (which must be transformed again)."""
        if isinstance(trans, basestring):
            trans = totrans(trans)

        if self.mode[0] == "L" or self.mode[0] == "l":
            mode = "L"
        elif self.mode[0] == "B" or self.mode[0] == "b":
            mode = "B"
        elif self.mode[0] == "V" or self.mode[0] == "v":
            mode = "V"
        elif self.mode[0] == "F" or self.mode[0] == "f":
            mode = "F"
        elif self.mode[0] == "S" or self.mode[0] == "s":
            mode = "S"

            vx, vy = [0.]*len(self.d), [0.]*len(self.d)
            for i in xrange(len(self.d)):
                inext = (i+1) % len(self.d)
                iprev = (i-1) % len(self.d)

                vx[i] = (self.d[inext][0] - self.d[iprev][0])/2.
                vy[i] = (self.d[inext][1] - self.d[iprev][1])/2.
                if not self.loop and (i == 0 or i == len(self.d)-1):
                    vx[i], vy[i] = 0., 0.

        else:
            raise ValueError("mode must be \"lines\", \"bezier\", \"velocity\", \"foreback\", \"smooth\", or an abbreviation")

        d = []
        indexes = list(range(len(self.d)))
        if self.loop and len(self.d) > 0:
            indexes.append(0)

        for i in indexes:
            inext = (i+1) % len(self.d)
            iprev = (i-1) % len(self.d)

            x, y = self.d[i][0], self.d[i][1]

            if trans is None:
                X, Y = x, y
            else:
                X, Y = trans(x, y)

            if d == []:
                if local:
                    d.append(("M", x, y, False))
                else:
                    d.append(("M", X, Y, True))

            elif mode == "L":
                if local:
                    d.append(("L", x, y, False))
                else:
                    d.append(("L", X, Y, True))

            elif mode == "B":
                c1x, c1y = self.d[i][2], self.d[i][3]
                if trans is None:
                    C1X, C1Y = c1x, c1y
                else:
                    C1X, C1Y = trans(c1x, c1y)

                c2x, c2y = self.d[i][4], self.d[i][5]
                if trans is None:
                    C2X, C2Y = c2x, c2y
                else:
                    C2X, C2Y = trans(c2x, c2y)

                if local:
                    d.append(("C", c1x, c1y, False, c2x, c2y, False, x, y, False))
                else:
                    d.append(("C", C1X, C1Y, True, C2X, C2Y, True, X, Y, True))

            elif mode == "V":
                c1x, c1y = self.d[iprev][2]/3. + self.d[iprev][0], self.d[iprev][3]/3. + self.d[iprev][1]
                c2x, c2y = self.d[i][2]/-3. + x, self.d[i][3]/-3. + y

                if trans is None:
                    C1X, C1Y = c1x, c1y
                else:
                    C1X, C1Y = trans(c1x, c1y)
                if trans is None:
                    C2X, C2Y = c2x, c2y
                else:
                    C2X, C2Y = trans(c2x, c2y)

                if local:
                    d.append(("C", c1x, c1y, False, c2x, c2y, False, x, y, False))
                else:
                    d.append(("C", C1X, C1Y, True, C2X, C2Y, True, X, Y, True))

            elif mode == "F":
                c1x, c1y = self.d[iprev][4]/3. + self.d[iprev][0], self.d[iprev][5]/3. + self.d[iprev][1]
                c2x, c2y = self.d[i][2]/-3. + x, self.d[i][3]/-3. + y

                if trans is None:
                    C1X, C1Y = c1x, c1y
                else:
                    C1X, C1Y = trans(c1x, c1y)
                if trans is None:
                    C2X, C2Y = c2x, c2y
                else:
                    C2X, C2Y = trans(c2x, c2y)

                if local:
                    d.append(("C", c1x, c1y, False, c2x, c2y, False, x, y, False))
                else:
                    d.append(("C", C1X, C1Y, True, C2X, C2Y, True, X, Y, True))

            elif mode == "S":
                c1x, c1y = vx[iprev]/3. + self.d[iprev][0], vy[iprev]/3. + self.d[iprev][1]
                c2x, c2y = vx[i]/-3. + x, vy[i]/-3. + y

                if trans is None:
                    C1X, C1Y = c1x, c1y
                else:
                    C1X, C1Y = trans(c1x, c1y)
                if trans is None:
                    C2X, C2Y = c2x, c2y
                else:
                    C2X, C2Y = trans(c2x, c2y)

                if local:
                    d.append(("C", c1x, c1y, False, c2x, c2y, False, x, y, False))
                else:
                    d.append(("C", C1X, C1Y, True, C2X, C2Y, True, X, Y, True))

        if self.loop and len(self.d) > 0:
            d.append(("Z",))

        return Path(d, **self.attr)

######################################################################

class Text:
    """Draws a text string at a specified point in local coordinates.

    x, y                   required      location of the point in local coordinates
    d                      required      text/Unicode string
    attribute=value pairs  keyword list  SVG attributes
    """

    defaults = {"stroke": "none", "fill": "black", "font-size": 5, }

    def __repr__(self):
        return "<Text %s at (%g, %g) %s>" % (repr(self.d), self.x, self.y, self.attr)

    def __init__(self, x, y, d, **attr):
        self.x = x
        self.y = y
        self.d = unicode(d)
        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if isinstance(trans, basestring):
            trans = totrans(trans)

        X, Y = self.x, self.y
        if trans is not None:
            X, Y = trans(X, Y)
        return SVG("text", self.d, x=X, y=Y, **self.attr)


class TextGlobal:
    """Draws a text string at a specified point in global coordinates.

    x, y                   required      location of the point in global coordinates
    d                      required      text/Unicode string
    attribute=value pairs  keyword list  SVG attributes
    """
    defaults = {"stroke": "none", "fill": "black", "font-size": 5, }

    def __repr__(self):
        return "<TextGlobal %s at (%s, %s) %s>" % (repr(self.d), str(self.x), str(self.y), self.attr)

    def __init__(self, x, y, d, **attr):
        self.x = x
        self.y = y
        self.d = unicode(d)
        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        return SVG("text", self.d, x=self.x, y=self.y, **self.attr)

######################################################################

_symbol_templates = {"dot": SVG("symbol", SVG("circle", cx=0, cy=0, r=1, stroke="none", fill="black"), viewBox="0 0 1 1", overflow="visible"),
                    "box": SVG("symbol", SVG("rect", x1=-1, y1=-1, x2=1, y2=1, stroke="none", fill="black"), viewBox="0 0 1 1", overflow="visible"),
                    "uptri": SVG("symbol", SVG("path", d="M -1 0.866 L 1 0.866 L 0 -0.866 Z", stroke="none", fill="black"), viewBox="0 0 1 1", overflow="visible"),
                    "downtri": SVG("symbol", SVG("path", d="M -1 -0.866 L 1 -0.866 L 0 0.866 Z", stroke="none", fill="black"), viewBox="0 0 1 1", overflow="visible"),
                    }

def make_symbol(id, shape="dot", **attr):
    """Creates a new instance of an SVG symbol to avoid cross-linking objects.

    id                    required         a new identifier (string/Unicode)
    shape                 default="dot"  the shape name from _symbol_templates
    attribute=value list  keyword list     modify the SVG attributes of the new symbol
    """
    output = copy.deepcopy(_symbol_templates[shape])
    for i in output.sub:
        i.attr.update(attr_preprocess(attr))
    output["id"] = id
    return output

_circular_dot = make_symbol("circular_dot")


class Dots:
    """Dots draws SVG symbols at a set of points.

    d                      required               list of (x,y) points
    symbol                 default=None           SVG symbol or a new identifier to
                                                  label an auto-generated symbol;
                                                  if None, use pre-defined _circular_dot
    width, height          default=1, 1           width and height of the symbols
                                                  in SVG coordinates
    attribute=value pairs  keyword list           SVG attributes
    """
    defaults = {}

    def __repr__(self):
        return "<Dots (%d nodes) %s>" % (len(self.d), self.attr)

    def __init__(self, d=[], symbol=None, width=1., height=1., **attr):
        self.d = list(d)
        self.width = width
        self.height = height

        self.attr = dict(self.defaults)
        self.attr.update(attr)

        if symbol is None:
            self.symbol = _circular_dot
        elif isinstance(symbol, SVG):
            self.symbol = symbol
        else:
            self.symbol = make_symbol(symbol)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if isinstance(trans, basestring):
            trans = totrans(trans)

        output = SVG("g", SVG("defs", self.symbol))
        id = "#%s" % self.symbol["id"]

        for p in self.d:
            x, y = p[0], p[1]

            if trans is None:
                X, Y = x, y
            else:
                X, Y = trans(x, y)

            item = SVG("use", x=X, y=Y, xlink__href=id)
            if self.width is not None:
                item["width"] = self.width
            if self.height is not None:
                item["height"] = self.height
            output.append(item)

        return output

######################################################################

_marker_templates = {"arrow_start": SVG("marker", SVG("path", d="M 9 3.6 L 10.5 0 L 0 3.6 L 10.5 7.2 L 9 3.6 Z"), viewBox="0 0 10.5 7.2", refX="9", refY="3.6", markerWidth="10.5", markerHeight="7.2", markerUnits="strokeWidth", orient="auto", stroke="none", fill="black"),
                    "arrow_end": SVG("marker", SVG("path", d="M 1.5 3.6 L 0 0 L 10.5 3.6 L 0 7.2 L 1.5 3.6 Z"), viewBox="0 0 10.5 7.2", refX="1.5", refY="3.6", markerWidth="10.5", markerHeight="7.2", markerUnits="strokeWidth", orient="auto", stroke="none", fill="black"),
                    }

def make_marker(id, shape, **attr):
    """Creates a new instance of an SVG marker to avoid cross-linking objects.

    id                     required         a new identifier (string/Unicode)
    shape                  required         the shape name from _marker_templates
    attribute=value list   keyword list     modify the SVG attributes of the new marker
    """
    output = copy.deepcopy(_marker_templates[shape])
    for i in output.sub:
        i.attr.update(attr_preprocess(attr))
    output["id"] = id
    return output


class Line(Curve):
    """Draws a line between two points.

    Line(x1, y1, x2, y2, arrow_start, arrow_end, attribute=value)

    x1, y1                  required        the starting point
    x2, y2                  required        the ending point
    arrow_start             default=None    if an identifier string/Unicode,
                                            draw a new arrow object at the
                                            beginning of the line; if a marker,
                                            draw that marker instead
    arrow_end               default=None    same for the end of the line
    attribute=value pairs   keyword list    SVG attributes
    """
    defaults = {}

    def __repr__(self):
        return "<Line (%g, %g) to (%g, %g) %s>" % (
               self.x1, self.y1, self.x2, self.y2, self.attr)

    def __init__(self, x1, y1, x2, y2, arrow_start=None, arrow_end=None, **attr):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.arrow_start, self.arrow_end = arrow_start, arrow_end

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""

        line = self.Path(trans).SVG()

        if ((self.arrow_start != False and self.arrow_start is not None) or
            (self.arrow_end != False and self.arrow_end is not None)):
            defs = SVG("defs")

            if self.arrow_start != False and self.arrow_start is not None:
                if isinstance(self.arrow_start, SVG):
                    defs.append(self.arrow_start)
                    line.attr["marker-start"] = "url(#%s)" % self.arrow_start["id"]
                elif isinstance(self.arrow_start, basestring):
                    defs.append(make_marker(self.arrow_start, "arrow_start"))
                    line.attr["marker-start"] = "url(#%s)" % self.arrow_start
                else:
                    raise TypeError("arrow_start must be False/None or an id string for the new marker")

            if self.arrow_end != False and self.arrow_end is not None:
                if isinstance(self.arrow_end, SVG):
                    defs.append(self.arrow_end)
                    line.attr["marker-end"] = "url(#%s)" % self.arrow_end["id"]
                elif isinstance(self.arrow_end, basestring):
                    defs.append(make_marker(self.arrow_end, "arrow_end"))
                    line.attr["marker-end"] = "url(#%s)" % self.arrow_end
                else:
                    raise TypeError("arrow_end must be False/None or an id string for the new marker")

            return SVG("g", defs, line)

        return line

    def Path(self, trans=None, local=False):
        """Apply the transformation "trans" and return a Path object in
        global coordinates.  If local=True, return a Path in local coordinates
        (which must be transformed again)."""
        self.f = lambda t: (self.x1 + t*(self.x2 - self.x1), self.y1 + t*(self.y2 - self.y1))
        self.low = 0.
        self.high = 1.
        self.loop = False

        if trans is None:
            return Path([("M", self.x1, self.y1, not local), ("L", self.x2, self.y2, not local)], **self.attr)
        else:
            return Curve.Path(self, trans, local)


class LineGlobal:
    """Draws a line between two points, one or both of which is in
    global coordinates.

    Line(x1, y1, x2, y2, lcoal1, local2, arrow_start, arrow_end, attribute=value)

    x1, y1                  required        the starting point
    x2, y2                  required        the ending point
    local1                  default=False   if True, interpret first point as a
                                            local coordinate (apply transform)
    local2                  default=False   if True, interpret second point as a
                                            local coordinate (apply transform)
    arrow_start             default=None    if an identifier string/Unicode,
                                            draw a new arrow object at the
                                            beginning of the line; if a marker,
                                            draw that marker instead
    arrow_end               default=None    same for the end of the line
    attribute=value pairs   keyword list    SVG attributes
    """
    defaults = {}

    def __repr__(self):
        local1, local2 = "", ""
        if self.local1:
            local1 = "L"
        if self.local2:
            local2 = "L"

        return "<LineGlobal %s(%s, %s) to %s(%s, %s) %s>" % (
               local1, str(self.x1), str(self.y1), local2, str(self.x2), str(self.y2), self.attr)

    def __init__(self, x1, y1, x2, y2, local1=False, local2=False, arrow_start=None, arrow_end=None, **attr):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.local1, self.local2 = local1, local2
        self.arrow_start, self.arrow_end = arrow_start, arrow_end

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if isinstance(trans, basestring):
            trans = totrans(trans)

        X1, Y1, X2, Y2 = self.x1, self.y1, self.x2, self.y2

        if self.local1:
            X1, Y1 = trans(X1, Y1)
        if self.local2:
            X2, Y2 = trans(X2, Y2)

        line = SVG("path", d="M%s %s L%s %s" % (X1, Y1, X2, Y2), **self.attr)

        if ((self.arrow_start != False and self.arrow_start is not None) or
            (self.arrow_end != False and self.arrow_end is not None)):
            defs = SVG("defs")

            if self.arrow_start != False and self.arrow_start is not None:
                if isinstance(self.arrow_start, SVG):
                    defs.append(self.arrow_start)
                    line.attr["marker-start"] = "url(#%s)" % self.arrow_start["id"]
                elif isinstance(self.arrow_start, basestring):
                    defs.append(make_marker(self.arrow_start, "arrow_start"))
                    line.attr["marker-start"] = "url(#%s)" % self.arrow_start
                else:
                    raise TypeError("arrow_start must be False/None or an id string for the new marker")

            if self.arrow_end != False and self.arrow_end is not None:
                if isinstance(self.arrow_end, SVG):
                    defs.append(self.arrow_end)
                    line.attr["marker-end"] = "url(#%s)" % self.arrow_end["id"]
                elif isinstance(self.arrow_end, basestring):
                    defs.append(make_marker(self.arrow_end, "arrow_end"))
                    line.attr["marker-end"] = "url(#%s)" % self.arrow_end
                else:
                    raise TypeError("arrow_end must be False/None or an id string for the new marker")

            return SVG("g", defs, line)

        return line


class VLine(Line):
    """Draws a vertical line.

    VLine(y1, y2, x, attribute=value)

    y1, y2                  required        y range
    x                       required        x position
    attribute=value pairs   keyword list    SVG attributes
    """
    defaults = {}

    def __repr__(self):
        return "<VLine (%g, %g) at x=%s %s>" % (self.y1, self.y2, self.x, self.attr)

    def __init__(self, y1, y2, x, **attr):
        self.x = x
        self.attr = dict(self.defaults)
        self.attr.update(attr)
        Line.__init__(self, x, y1, x, y2, **self.attr)

    def Path(self, trans=None, local=False):
        """Apply the transformation "trans" and return a Path object in
        global coordinates.  If local=True, return a Path in local coordinates
        (which must be transformed again)."""
        self.x1 = self.x
        self.x2 = self.x
        return Line.Path(self, trans, local)


class HLine(Line):
    """Draws a horizontal line.

    HLine(x1, x2, y, attribute=value)

    x1, x2                  required        x range
    y                       required        y position
    attribute=value pairs   keyword list    SVG attributes
    """
    defaults = {}

    def __repr__(self):
        return "<HLine (%g, %g) at y=%s %s>" % (self.x1, self.x2, self.y, self.attr)

    def __init__(self, x1, x2, y, **attr):
        self.y = y
        self.attr = dict(self.defaults)
        self.attr.update(attr)
        Line.__init__(self, x1, y, x2, y, **self.attr)

    def Path(self, trans=None, local=False):
        """Apply the transformation "trans" and return a Path object in
        global coordinates.  If local=True, return a Path in local coordinates
        (which must be transformed again)."""
        self.y1 = self.y
        self.y2 = self.y
        return Line.Path(self, trans, local)

######################################################################

class Rect(Curve):
    """Draws a rectangle.

    Rect(x1, y1, x2, y2, attribute=value)

    x1, y1                  required        the starting point
    x2, y2                  required        the ending point
    attribute=value pairs   keyword list    SVG attributes
    """
    defaults = {}

    def __repr__(self):
        return "<Rect (%g, %g), (%g, %g) %s>" % (
               self.x1, self.y1, self.x2, self.y2, self.attr)

    def __init__(self, x1, y1, x2, y2, **attr):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        return self.Path(trans).SVG()

    def Path(self, trans=None, local=False):
        """Apply the transformation "trans" and return a Path object in
        global coordinates.  If local=True, return a Path in local coordinates
        (which must be transformed again)."""
        if trans is None:
            return Path([("M", self.x1, self.y1, not local), ("L", self.x2, self.y1, not local), ("L", self.x2, self.y2, not local), ("L", self.x1, self.y2, not local), ("Z",)], **self.attr)

        else:
            self.low = 0.
            self.high = 1.
            self.loop = False

            self.f = lambda t: (self.x1 + t*(self.x2 - self.x1), self.y1)
            d1 = Curve.Path(self, trans, local).d

            self.f = lambda t: (self.x2, self.y1 + t*(self.y2 - self.y1))
            d2 = Curve.Path(self, trans, local).d
            del d2[0]

            self.f = lambda t: (self.x2 + t*(self.x1 - self.x2), self.y2)
            d3 = Curve.Path(self, trans, local).d
            del d3[0]

            self.f = lambda t: (self.x1, self.y2 + t*(self.y1 - self.y2))
            d4 = Curve.Path(self, trans, local).d
            del d4[0]

            return Path(d=(d1 + d2 + d3 + d4 + [("Z",)]), **self.attr)

######################################################################

class Ellipse(Curve):
    """Draws an ellipse from a semimajor vector (ax,ay) and a semiminor
    length (b).

    Ellipse(x, y, ax, ay, b, attribute=value)

    x, y                    required        the center of the ellipse/circle
    ax, ay                  required        a vector indicating the length
                                            and direction of the semimajor axis
    b                       required        the length of the semiminor axis.
                                            If equal to sqrt(ax2 + ay2), the
                                            ellipse is a circle
    attribute=value pairs   keyword list    SVG attributes

    (If sqrt(ax**2 + ay**2) is less than b, then (ax,ay) is actually the
    semiminor axis.)
    """
    defaults = {}

    def __repr__(self):
        return "<Ellipse (%g, %g) a=(%g, %g), b=%g %s>" % (
               self.x, self.y, self.ax, self.ay, self.b, self.attr)

    def __init__(self, x, y, ax, ay, b, **attr):
        self.x, self.y, self.ax, self.ay, self.b = x, y, ax, ay, b

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        return self.Path(trans).SVG()

    def Path(self, trans=None, local=False):
        """Apply the transformation "trans" and return a Path object in
        global coordinates.  If local=True, return a Path in local coordinates
        (which must be transformed again)."""
        angle = math.atan2(self.ay, self.ax) + math.pi/2.
        bx = self.b * math.cos(angle)
        by = self.b * math.sin(angle)

        self.f = lambda t: (self.x + self.ax*math.cos(t) + bx*math.sin(t), self.y + self.ay*math.cos(t) + by*math.sin(t))
        self.low = -math.pi
        self.high = math.pi
        self.loop = True
        return Curve.Path(self, trans, local)

######################################################################

def unumber(x):
    """Converts numbers to a Unicode string, taking advantage of special
    Unicode characters to make nice minus signs and scientific notation.
    """
    output = u"%g" % x

    if output[0] == u"-":
        output = u"\u2013" + output[1:]

    index = output.find(u"e")
    if index != -1:
        uniout = unicode(output[:index]) + u"\u00d710"
        saw_nonzero = False
        for n in output[index+1:]:
            if n == u"+":
                pass # uniout += u"\u207a"
            elif n == u"-":
                uniout += u"\u207b"
            elif n == u"0":
                if saw_nonzero:
                    uniout += u"\u2070"
            elif n == u"1":
                saw_nonzero = True
                uniout += u"\u00b9"
            elif n == u"2":
                saw_nonzero = True
                uniout += u"\u00b2"
            elif n == u"3":
                saw_nonzero = True
                uniout += u"\u00b3"
            elif u"4" <= n <= u"9":
                saw_nonzero = True
                if saw_nonzero:
                    uniout += eval("u\"\\u%x\"" % (0x2070 + ord(n) - ord(u"0")))
            else:
                uniout += n

        if uniout[:2] == u"1\u00d7":
            uniout = uniout[2:]
        return uniout

    return output


class Ticks:
    """Superclass for all graphics primitives that draw ticks,
    miniticks, and tick labels.  This class only draws the ticks.

    Ticks(f, low, high, ticks, miniticks, labels, logbase, arrow_start,
          arrow_end, text_attr, attribute=value)

    f                       required        parametric function along which ticks
                                            will be drawn; has the same format as
                                            the function used in Curve
    low, high               required        range of the independent variable
    ticks                   default=-10     request ticks according to the standard
                                            tick specification (see below)
    miniticks               default=True    request miniticks according to the
                                            standard minitick specification (below)
    labels                  True            request tick labels according to the
                                            standard tick label specification (below)
    logbase                 default=None    if a number, the axis is logarithmic with
                                            ticks at the given base (usually 10)
    arrow_start             default=None    if a new string identifier, draw an arrow
                                            at the low-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    arrow_end               default=None    if a new string identifier, draw an arrow
                                            at the high-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    text_attr               default={}      SVG attributes for the text labels
    attribute=value pairs   keyword list    SVG attributes for the tick marks

    Standard tick specification:

        * True: same as -10 (below).
        * Positive number N: draw exactly N ticks, including the endpoints. To
          subdivide an axis into 10 equal-sized segments, ask for 11 ticks.
        * Negative number -N: draw at least N ticks. Ticks will be chosen with
          "natural" values, multiples of 2 or 5.
        * List of values: draw a tick mark at each value.
        * Dict of value, label pairs: draw a tick mark at each value, labeling
          it with the given string. This lets you say things like {3.14159: "pi"}.
        * False or None: no ticks.

    Standard minitick specification:

        * True: draw miniticks with "natural" values, more closely spaced than
          the ticks.
        * Positive number N: draw exactly N miniticks, including the endpoints.
          To subdivide an axis into 100 equal-sized segments, ask for 101 miniticks.
        * Negative number -N: draw at least N miniticks.
        * List of values: draw a minitick mark at each value.
        * False or None: no miniticks.

    Standard tick label specification:

        * True: use the unumber function (described below)
        * Format string: standard format strings, e.g. "%5.2f" for 12.34
        * Python callable: function that converts numbers to strings
        * False or None: no labels
    """
    defaults = {"stroke-width": "0.25pt", }
    text_defaults = {"stroke": "none", "fill": "black", "font-size": 5, }
    tick_start = -1.5
    tick_end = 1.5
    minitick_start = -0.75
    minitick_end = 0.75
    text_start = 2.5
    text_angle = 0.

    def __repr__(self):
        return "<Ticks %s from %s to %s ticks=%s labels=%s %s>" % (
               self.f, self.low, self.high, str(self.ticks), str(self.labels), self.attr)

    def __init__(self, f, low, high, ticks=-10, miniticks=True, labels=True, logbase=None,
                 arrow_start=None, arrow_end=None, text_attr={}, **attr):
        self.f = f
        self.low = low
        self.high = high
        self.ticks = ticks
        self.miniticks = miniticks
        self.labels = labels
        self.logbase = logbase
        self.arrow_start = arrow_start
        self.arrow_end = arrow_end

        self.attr = dict(self.defaults)
        self.attr.update(attr)

        self.text_attr = dict(self.text_defaults)
        self.text_attr.update(text_attr)

    def orient_tickmark(self, t, trans=None):
        """Return the position, normalized local x vector, normalized
        local y vector, and angle of a tick at position t.

        Normally only used internally.
        """
        if isinstance(trans, basestring):
            trans = totrans(trans)
        if trans is None:
            f = self.f
        else:
            f = lambda t: trans(*self.f(t))

        eps = _epsilon * abs(self.high - self.low)

        X, Y = f(t)
        Xprime, Yprime = f(t + eps)
        xhatx, xhaty = (Xprime - X)/eps, (Yprime - Y)/eps

        norm = math.sqrt(xhatx**2 + xhaty**2)
        if norm != 0:
            xhatx, xhaty = xhatx/norm, xhaty/norm
        else:
            xhatx, xhaty = 1., 0.

        angle = math.atan2(xhaty, xhatx) + math.pi/2.
        yhatx, yhaty = math.cos(angle), math.sin(angle)

        return (X, Y), (xhatx, xhaty), (yhatx, yhaty), angle

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if isinstance(trans, basestring):
            trans = totrans(trans)

        self.last_ticks, self.last_miniticks = self.interpret()
        tickmarks = Path([], **self.attr)
        minitickmarks = Path([], **self.attr)
        output = SVG("g")

        if ((self.arrow_start != False and self.arrow_start is not None) or
            (self.arrow_end != False and self.arrow_end is not None)):
            defs = SVG("defs")

            if self.arrow_start != False and self.arrow_start is not None:
                if isinstance(self.arrow_start, SVG):
                    defs.append(self.arrow_start)
                elif isinstance(self.arrow_start, basestring):
                    defs.append(make_marker(self.arrow_start, "arrow_start"))
                else:
                    raise TypeError("arrow_start must be False/None or an id string for the new marker")

            if self.arrow_end != False and self.arrow_end is not None:
                if isinstance(self.arrow_end, SVG):
                    defs.append(self.arrow_end)
                elif isinstance(self.arrow_end, basestring):
                    defs.append(make_marker(self.arrow_end, "arrow_end"))
                else:
                    raise TypeError("arrow_end must be False/None or an id string for the new marker")

            output.append(defs)

        eps = _epsilon * (self.high - self.low)

        for t, label in self.last_ticks.items():
            (X, Y), (xhatx, xhaty), (yhatx, yhaty), angle = self.orient_tickmark(t, trans)

            if ((not self.arrow_start or abs(t - self.low) > eps) and
                (not self.arrow_end or abs(t - self.high) > eps)):
                tickmarks.d.append(("M", X - yhatx*self.tick_start, Y - yhaty*self.tick_start, True))
                tickmarks.d.append(("L", X - yhatx*self.tick_end, Y - yhaty*self.tick_end, True))

            angle = (angle - math.pi/2.)*180./math.pi + self.text_angle

            ########### a HACK! ############ (to be removed when Inkscape handles baselines)
            if _hacks["inkscape-text-vertical-shift"]:
                if self.text_start > 0:
                    X += math.cos(angle*math.pi/180. + math.pi/2.) * 2.
                    Y += math.sin(angle*math.pi/180. + math.pi/2.) * 2.
                else:
                    X += math.cos(angle*math.pi/180. + math.pi/2.) * 2. * 2.5
                    Y += math.sin(angle*math.pi/180. + math.pi/2.) * 2. * 2.5
            ########### end hack ###########

            if label != "":
                output.append(SVG("text", label, transform="translate(%g, %g) rotate(%g)" %
                                  (X - yhatx*self.text_start, Y - yhaty*self.text_start, angle), **self.text_attr))

        for t in self.last_miniticks:
            skip = False
            for tt in self.last_ticks.keys():
                if abs(t - tt) < eps:
                    skip = True
                    break
            if not skip:
                (X, Y), (xhatx, xhaty), (yhatx, yhaty), angle = self.orient_tickmark(t, trans)

            if ((not self.arrow_start or abs(t - self.low) > eps) and
                (not self.arrow_end or abs(t - self.high) > eps)):
                minitickmarks.d.append(("M", X - yhatx*self.minitick_start, Y - yhaty*self.minitick_start, True))
                minitickmarks.d.append(("L", X - yhatx*self.minitick_end, Y - yhaty*self.minitick_end, True))

        output.prepend(tickmarks.SVG(trans))
        output.prepend(minitickmarks.SVG(trans))
        return output

    def interpret(self):
        """Evaluate and return optimal ticks and miniticks according to
        the standard minitick specification.

        Normally only used internally.
        """

        if self.labels is None or self.labels == False:
            format = lambda x: ""

        elif self.labels == True:
            format = unumber

        elif isinstance(self.labels, basestring):
            format = lambda x: (self.labels % x)

        elif callable(self.labels):
            format = self.labels

        else:
            raise TypeError("labels must be None/False, True, a format string, or a number->string function")

        # Now for the ticks
        ticks = self.ticks

        # Case 1: ticks is None/False
        if ticks is None or ticks == False:
            return {}, []

        # Case 2: ticks is the number of desired ticks
        elif isinstance(ticks, (int, long)):
            if ticks == True:
                ticks = -10

            if self.logbase is None:
                ticks = self.compute_ticks(ticks, format)
            else:
                ticks = self.compute_logticks(self.logbase, ticks, format)

            # Now for the miniticks
            if self.miniticks == True:
                if self.logbase is None:
                    return ticks, self.compute_miniticks(ticks)
                else:
                    return ticks, self.compute_logminiticks(self.logbase)

            elif isinstance(self.miniticks, (int, long)):
                return ticks, self.regular_miniticks(self.miniticks)

            elif getattr(self.miniticks, "__iter__", False):
                return ticks, self.miniticks

            elif self.miniticks == False or self.miniticks is None:
                return ticks, []

            else:
                raise TypeError("miniticks must be None/False, True, a number of desired miniticks, or a list of numbers")

        # Cases 3 & 4: ticks is iterable
        elif getattr(ticks, "__iter__", False):

            # Case 3: ticks is some kind of list
            if not isinstance(ticks, dict):
                output = {}
                eps = _epsilon * (self.high - self.low)
                for x in ticks:
                    if format == unumber and abs(x) < eps:
                        output[x] = u"0"
                    else:
                        output[x] = format(x)
                ticks = output

            # Case 4: ticks is a dict
            else:
                pass

            # Now for the miniticks
            if self.miniticks == True:
                if self.logbase is None:
                    return ticks, self.compute_miniticks(ticks)
                else:
                    return ticks, self.compute_logminiticks(self.logbase)

            elif isinstance(self.miniticks, (int, long)):
                return ticks, self.regular_miniticks(self.miniticks)

            elif getattr(self.miniticks, "__iter__", False):
                return ticks, self.miniticks

            elif self.miniticks == False or self.miniticks is None:
                return ticks, []

            else:
                raise TypeError("miniticks must be None/False, True, a number of desired miniticks, or a list of numbers")

        else:
            raise TypeError("ticks must be None/False, a number of desired ticks, a list of numbers, or a dictionary of explicit markers")

    def compute_ticks(self, N, format):
        """Return less than -N or exactly N optimal linear ticks.

        Normally only used internally.
        """
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        if N == 1:
            raise ValueError("N can be 0 or >1 to specify the exact number of ticks or negative to specify a maximum")

        eps = _epsilon * (self.high - self.low)

        if N >= 0:
            output = {}
            x = self.low
            for i in xrange(N):
                if format == unumber and abs(x) < eps:
                    label = u"0"
                else:
                    label = format(x)
                output[x] = label
                x += (self.high - self.low)/(N-1.)
            return output

        N = -N

        counter = 0
        granularity = 10**math.ceil(math.log10(max(abs(self.low), abs(self.high))))
        lowN = math.ceil(1.*self.low / granularity)
        highN = math.floor(1.*self.high / granularity)

        while lowN > highN:
            countermod3 = counter % 3
            if countermod3 == 0:
                granularity *= 0.5
            elif countermod3 == 1:
                granularity *= 0.4
            else:
                granularity *= 0.5
            counter += 1
            lowN = math.ceil(1.*self.low / granularity)
            highN = math.floor(1.*self.high / granularity)

        last_granularity = granularity
        last_trial = None

        while True:
            trial = {}
            for n in range(int(lowN), int(highN)+1):
                x = n * granularity
                if format == unumber and abs(x) < eps:
                    label = u"0"
                else:
                    label = format(x)
                trial[x] = label

            if int(highN)+1 - int(lowN) >= N:
                if last_trial is None:
                    v1, v2 = self.low, self.high
                    return {v1: format(v1), v2: format(v2)}
                else:
                    low_in_ticks, high_in_ticks = False, False
                    for t in last_trial.keys():
                        if 1.*abs(t - self.low)/last_granularity < _epsilon:
                            low_in_ticks = True
                        if 1.*abs(t - self.high)/last_granularity < _epsilon:
                            high_in_ticks = True

                    lowN = 1.*self.low / last_granularity
                    highN = 1.*self.high / last_granularity
                    if abs(lowN - round(lowN)) < _epsilon and not low_in_ticks:
                        last_trial[self.low] = format(self.low)
                    if abs(highN - round(highN)) < _epsilon and not high_in_ticks:
                        last_trial[self.high] = format(self.high)
                    return last_trial

            last_granularity = granularity
            last_trial = trial

            countermod3 = counter % 3
            if countermod3 == 0:
                granularity *= 0.5
            elif countermod3 == 1:
                granularity *= 0.4
            else:
                granularity *= 0.5
            counter += 1
            lowN = math.ceil(1.*self.low / granularity)
            highN = math.floor(1.*self.high / granularity)

    def regular_miniticks(self, N):
        """Return exactly N linear ticks.

        Normally only used internally.
        """
        output = []
        x = self.low
        for i in xrange(N):
            output.append(x)
            x += (self.high - self.low)/(N-1.)
        return output

    def compute_miniticks(self, original_ticks):
        """Return optimal linear miniticks, given a set of ticks.

        Normally only used internally.
        """
        if len(original_ticks) < 2:
            original_ticks = ticks(self.low, self.high) # XXX ticks is undefined!
        original_ticks = original_ticks.keys()
        original_ticks.sort()

        if self.low > original_ticks[0] + _epsilon or self.high < original_ticks[-1] - _epsilon:
            raise ValueError("original_ticks {%g...%g} extend beyond [%g, %g]" % (original_ticks[0], original_ticks[-1], self.low, self.high))

        granularities = []
        for i in range(len(original_ticks)-1):
            granularities.append(original_ticks[i+1] - original_ticks[i])
        spacing = 10**(math.ceil(math.log10(min(granularities)) - 1))

        output = []
        x = original_ticks[0] - math.ceil(1.*(original_ticks[0] - self.low) / spacing) * spacing

        while x <= self.high:
            if x >= self.low:
                already_in_ticks = False
                for t in original_ticks:
                    if abs(x-t) < _epsilon * (self.high - self.low):
                        already_in_ticks = True
                if not already_in_ticks:
                    output.append(x)
            x += spacing
        return output

    def compute_logticks(self, base, N, format):
        """Return less than -N or exactly N optimal logarithmic ticks.

        Normally only used internally.
        """
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        if N == 1:
            raise ValueError("N can be 0 or >1 to specify the exact number of ticks or negative to specify a maximum")

        eps = _epsilon * (self.high - self.low)

        if N >= 0:
            output = {}
            x = self.low
            for i in xrange(N):
                if format == unumber and abs(x) < eps:
                    label = u"0"
                else:
                    label = format(x)
                output[x] = label
                x += (self.high - self.low)/(N-1.)
            return output

        N = -N

        lowN = math.floor(math.log(self.low, base))
        highN = math.ceil(math.log(self.high, base))
        output = {}
        for n in range(int(lowN), int(highN)+1):
            x = base**n
            label = format(x)
            if self.low <= x <= self.high:
                output[x] = label

        for i in range(1, len(output)):
            keys = output.keys()
            keys.sort()
            keys = keys[::i]
            values = map(lambda k: output[k], keys)
            if len(values) <= N:
                for k in output.keys():
                    if k not in keys:
                        output[k] = ""
                break

        if len(output) <= 2:
            output2 = self.compute_ticks(N=-int(math.ceil(N/2.)), format=format)
            lowest = min(output2)

            for k in output:
                if k < lowest:
                    output2[k] = output[k]
            output = output2

        return output

    def compute_logminiticks(self, base):
        """Return optimal logarithmic miniticks, given a set of ticks.

        Normally only used internally.
        """
        if self.low >= self.high:
            raise ValueError("low must be less than high")

        lowN = math.floor(math.log(self.low, base))
        highN = math.ceil(math.log(self.high, base))
        output = []
        num_ticks = 0
        for n in range(int(lowN), int(highN)+1):
            x = base**n
            if self.low <= x <= self.high:
                num_ticks += 1
            for m in range(2, int(math.ceil(base))):
                minix = m * x
                if self.low <= minix <= self.high:
                    output.append(minix)

        if num_ticks <= 2:
            return []
        else:
            return output

######################################################################

class CurveAxis(Curve, Ticks):
    """Draw an axis with tick marks along a parametric curve.

    CurveAxis(f, low, high, ticks, miniticks, labels, logbase, arrow_start, arrow_end,
    text_attr, attribute=value)

    f                      required         a Python callable or string in
                                            the form "f(t), g(t)", just like Curve
    low, high              required         left and right endpoints
    ticks                  default=-10      request ticks according to the standard
                                            tick specification (see help(Ticks))
    miniticks              default=True     request miniticks according to the
                                            standard minitick specification
    labels                 True             request tick labels according to the
                                            standard tick label specification
    logbase                default=None     if a number, the x axis is logarithmic
                                            with ticks at the given base (10 being
                                            the most common)
    arrow_start            default=None     if a new string identifier, draw an
                                            arrow at the low-end of the axis,
                                            referenced by that identifier; if an
                                            SVG marker object, use that marker
    arrow_end              default=None     if a new string identifier, draw an
                                            arrow at the high-end of the axis,
                                            referenced by that identifier; if an
                                            SVG marker object, use that marker
    text_attr              default={}       SVG attributes for the text labels
    attribute=value pairs  keyword list     SVG attributes
    """
    defaults = {"stroke-width": "0.25pt", }
    text_defaults = {"stroke": "none", "fill": "black", "font-size": 5, }

    def __repr__(self):
        return "<CurveAxis %s [%s, %s] ticks=%s labels=%s %s>" % (
               self.f, self.low, self.high, str(self.ticks), str(self.labels), self.attr)

    def __init__(self, f, low, high, ticks=-10, miniticks=True, labels=True, logbase=None,
                 arrow_start=None, arrow_end=None, text_attr={}, **attr):
        tattr = dict(self.text_defaults)
        tattr.update(text_attr)
        Curve.__init__(self, f, low, high)
        Ticks.__init__(self, f, low, high, ticks, miniticks, labels, logbase, arrow_start, arrow_end, tattr, **attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        func = Curve.SVG(self, trans)
        ticks = Ticks.SVG(self, trans) # returns a <g />

        if self.arrow_start != False and self.arrow_start is not None:
            if isinstance(self.arrow_start, basestring):
                func.attr["marker-start"] = "url(#%s)" % self.arrow_start
            else:
                func.attr["marker-start"] = "url(#%s)" % self.arrow_start.id

        if self.arrow_end != False and self.arrow_end is not None:
            if isinstance(self.arrow_end, basestring):
                func.attr["marker-end"] = "url(#%s)" % self.arrow_end
            else:
                func.attr["marker-end"] = "url(#%s)" % self.arrow_end.id

        ticks.append(func)
        return ticks


class LineAxis(Line, Ticks):
    """Draws an axis with tick marks along a line.

    LineAxis(x1, y1, x2, y2, start, end, ticks, miniticks, labels, logbase,
    arrow_start, arrow_end, text_attr, attribute=value)

    x1, y1                  required        starting point
    x2, y2                  required        ending point
    start, end              default=0, 1    values to start and end labeling
    ticks                   default=-10     request ticks according to the standard
                                            tick specification (see help(Ticks))
    miniticks               default=True    request miniticks according to the
                                            standard minitick specification
    labels                  True            request tick labels according to the
                                            standard tick label specification
    logbase                 default=None    if a number, the x axis is logarithmic
                                            with ticks at the given base (usually 10)
    arrow_start             default=None    if a new string identifier, draw an arrow
                                            at the low-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    arrow_end               default=None    if a new string identifier, draw an arrow
                                            at the high-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    text_attr               default={}      SVG attributes for the text labels
    attribute=value pairs   keyword list    SVG attributes
    """
    defaults = {"stroke-width": "0.25pt", }
    text_defaults = {"stroke": "none", "fill": "black", "font-size": 5, }

    def __repr__(self):
        return "<LineAxis (%g, %g) to (%g, %g) ticks=%s labels=%s %s>" % (
               self.x1, self.y1, self.x2, self.y2, str(self.ticks), str(self.labels), self.attr)

    def __init__(self, x1, y1, x2, y2, start=0., end=1., ticks=-10, miniticks=True, labels=True,
                 logbase=None, arrow_start=None, arrow_end=None, exclude=None, text_attr={}, **attr):
        self.start = start
        self.end = end
        self.exclude = exclude
        tattr = dict(self.text_defaults)
        tattr.update(text_attr)
        Line.__init__(self, x1, y1, x2, y2, **attr)
        Ticks.__init__(self, None, None, None, ticks, miniticks, labels, logbase, arrow_start, arrow_end, tattr, **attr)

    def interpret(self):
        if self.exclude is not None and not (isinstance(self.exclude, (tuple, list)) and len(self.exclude) == 2 and
                                             isinstance(self.exclude[0], (int, long, float)) and isinstance(self.exclude[1], (int, long, float))):
            raise TypeError("exclude must either be None or (low, high)")

        ticks, miniticks = Ticks.interpret(self)
        if self.exclude is None:
            return ticks, miniticks

        ticks2 = {}
        for loc, label in ticks.items():
            if self.exclude[0] <= loc <= self.exclude[1]:
                ticks2[loc] = ""
            else:
                ticks2[loc] = label

        return ticks2, miniticks

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        line = Line.SVG(self, trans) # must be evaluated first, to set self.f, self.low, self.high

        f01 = self.f
        self.f = lambda t: f01(1. * (t - self.start) / (self.end - self.start))
        self.low = self.start
        self.high = self.end

        if self.arrow_start != False and self.arrow_start is not None:
            if isinstance(self.arrow_start, basestring):
                line.attr["marker-start"] = "url(#%s)" % self.arrow_start
            else:
                line.attr["marker-start"] = "url(#%s)" % self.arrow_start.id

        if self.arrow_end != False and self.arrow_end is not None:
            if isinstance(self.arrow_end, basestring):
                line.attr["marker-end"] = "url(#%s)" % self.arrow_end
            else:
                line.attr["marker-end"] = "url(#%s)" % self.arrow_end.id

        ticks = Ticks.SVG(self, trans) # returns a <g />
        ticks.append(line)
        return ticks


class XAxis(LineAxis):
    """Draws an x axis with tick marks.

    XAxis(xmin, xmax, aty, ticks, miniticks, labels, logbase, arrow_start, arrow_end,
    exclude, text_attr, attribute=value)

    xmin, xmax              required        the x range
    aty                     default=0       y position to draw the axis
    ticks                   default=-10     request ticks according to the standard
                                            tick specification (see help(Ticks))
    miniticks               default=True    request miniticks according to the
                                            standard minitick specification
    labels                  True            request tick labels according to the
                                            standard tick label specification
    logbase                 default=None    if a number, the x axis is logarithmic
                                            with ticks at the given base (usually 10)
    arrow_start             default=None    if a new string identifier, draw an arrow
                                            at the low-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    arrow_end               default=None    if a new string identifier, draw an arrow
                                            at the high-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    exclude                 default=None    if a (low, high) pair, don't draw text
                                            labels within this range
    text_attr               default={}      SVG attributes for the text labels
    attribute=value pairs   keyword list    SVG attributes for all lines

    The exclude option is provided for Axes to keep text from overlapping
    where the axes cross. Normal users are not likely to need it.
    """
    defaults = {"stroke-width": "0.25pt", }
    text_defaults = {"stroke": "none", "fill": "black", "font-size": 5, "dominant-baseline": "text-before-edge", }
    text_start = -1.
    text_angle = 0.

    def __repr__(self):
        return "<XAxis (%g, %g) at y=%g ticks=%s labels=%s %s>" % (
               self.xmin, self.xmax, self.aty, str(self.ticks), str(self.labels), self.attr) # XXX self.xmin/xmax undefd!

    def __init__(self, xmin, xmax, aty=0, ticks=-10, miniticks=True, labels=True, logbase=None,
                 arrow_start=None, arrow_end=None, exclude=None, text_attr={}, **attr):
        self.aty = aty
        tattr = dict(self.text_defaults)
        tattr.update(text_attr)
        LineAxis.__init__(self, xmin, aty, xmax, aty, xmin, xmax, ticks, miniticks, labels, logbase, arrow_start, arrow_end, exclude, tattr, **attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        self.y1 = self.aty
        self.y2 = self.aty
        return LineAxis.SVG(self, trans)


class YAxis(LineAxis):
    """Draws a y axis with tick marks.

    YAxis(ymin, ymax, atx, ticks, miniticks, labels, logbase, arrow_start, arrow_end,
    exclude, text_attr, attribute=value)

    ymin, ymax              required        the y range
    atx                     default=0       x position to draw the axis
    ticks                   default=-10     request ticks according to the standard
                                            tick specification (see help(Ticks))
    miniticks               default=True    request miniticks according to the
                                            standard minitick specification
    labels                  True            request tick labels according to the
                                            standard tick label specification
    logbase                 default=None    if a number, the y axis is logarithmic
                                            with ticks at the given base (usually 10)
    arrow_start             default=None    if a new string identifier, draw an arrow
                                            at the low-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    arrow_end               default=None    if a new string identifier, draw an arrow
                                            at the high-end of the axis, referenced by
                                            that identifier; if an SVG marker object,
                                            use that marker
    exclude                 default=None    if a (low, high) pair, don't draw text
                                            labels within this range
    text_attr               default={}      SVG attributes for the text labels
    attribute=value pairs   keyword list    SVG attributes for all lines

    The exclude option is provided for Axes to keep text from overlapping
    where the axes cross. Normal users are not likely to need it.
    """
    defaults = {"stroke-width": "0.25pt", }
    text_defaults = {"stroke": "none", "fill": "black", "font-size": 5, "text-anchor": "end", "dominant-baseline": "middle", }
    text_start = 2.5
    text_angle = 90.

    def __repr__(self):
        return "<YAxis (%g, %g) at x=%g ticks=%s labels=%s %s>" % (
               self.ymin, self.ymax, self.atx, str(self.ticks), str(self.labels), self.attr) # XXX self.ymin/ymax undefd!

    def __init__(self, ymin, ymax, atx=0, ticks=-10, miniticks=True, labels=True, logbase=None,
                 arrow_start=None, arrow_end=None, exclude=None, text_attr={}, **attr):
        self.atx = atx
        tattr = dict(self.text_defaults)
        tattr.update(text_attr)
        LineAxis.__init__(self, atx, ymin, atx, ymax, ymin, ymax, ticks, miniticks, labels, logbase, arrow_start, arrow_end, exclude, tattr, **attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        self.x1 = self.atx
        self.x2 = self.atx
        return LineAxis.SVG(self, trans)


class Axes:
    """Draw a pair of intersecting x-y axes.

    Axes(xmin, xmax, ymin, ymax, atx, aty, xticks, xminiticks, xlabels, xlogbase,
    yticks, yminiticks, ylabels, ylogbase, arrows, text_attr, attribute=value)

    xmin, xmax               required       the x range
    ymin, ymax               required       the y range
    atx, aty                 default=0, 0   point where the axes try to cross;
                                            if outside the range, the axes will
                                            cross at the closest corner
    xticks                   default=-10    request ticks according to the standard
                                            tick specification (see help(Ticks))
    xminiticks               default=True   request miniticks according to the
                                            standard minitick specification
    xlabels                  True           request tick labels according to the
                                            standard tick label specification
    xlogbase                 default=None   if a number, the x axis is logarithmic
                                            with ticks at the given base (usually 10)
    yticks                   default=-10    request ticks according to the standard
                                            tick specification
    yminiticks               default=True   request miniticks according to the
                                            standard minitick specification
    ylabels                  True           request tick labels according to the
                                            standard tick label specification
    ylogbase                 default=None   if a number, the y axis is logarithmic
                                            with ticks at the given base (usually 10)
    arrows                   default=None   if a new string identifier, draw arrows
                                            referenced by that identifier
    text_attr                default={}     SVG attributes for the text labels
    attribute=value pairs    keyword list   SVG attributes for all lines
    """
    defaults = {"stroke-width": "0.25pt", }
    text_defaults = {"stroke": "none", "fill": "black", "font-size": 5, }

    def __repr__(self):
        return "<Axes x=(%g, %g) y=(%g, %g) at (%g, %g) %s>" % (
               self.xmin, self.xmax, self.ymin, self.ymax, self.atx, self.aty, self.attr)

    def __init__(self, xmin, xmax, ymin, ymax, atx=0, aty=0,
                 xticks=-10, xminiticks=True, xlabels=True, xlogbase=None,
                 yticks=-10, yminiticks=True, ylabels=True, ylogbase=None,
                 arrows=None, text_attr={}, **attr):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.atx, self.aty = atx, aty
        self.xticks, self.xminiticks, self.xlabels, self.xlogbase = xticks, xminiticks, xlabels, xlogbase
        self.yticks, self.yminiticks, self.ylabels, self.ylogbase = yticks, yminiticks, ylabels, ylogbase
        self.arrows = arrows

        self.text_attr = dict(self.text_defaults)
        self.text_attr.update(text_attr)

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        atx, aty = self.atx, self.aty
        if atx < self.xmin:
            atx = self.xmin
        if atx > self.xmax:
            atx = self.xmax
        if aty < self.ymin:
            aty = self.ymin
        if aty > self.ymax:
            aty = self.ymax

        xmargin = 0.1 * abs(self.ymin - self.ymax)
        xexclude = atx - xmargin, atx + xmargin

        ymargin = 0.1 * abs(self.xmin - self.xmax)
        yexclude = aty - ymargin, aty + ymargin

        if self.arrows is not None and self.arrows != False:
            xarrow_start = self.arrows + ".xstart"
            xarrow_end = self.arrows + ".xend"
            yarrow_start = self.arrows + ".ystart"
            yarrow_end = self.arrows + ".yend"
        else:
            xarrow_start = xarrow_end = yarrow_start = yarrow_end = None

        xaxis = XAxis(self.xmin, self.xmax, aty, self.xticks, self.xminiticks, self.xlabels, self.xlogbase, xarrow_start, xarrow_end, exclude=xexclude, text_attr=self.text_attr, **self.attr).SVG(trans)
        yaxis = YAxis(self.ymin, self.ymax, atx, self.yticks, self.yminiticks, self.ylabels, self.ylogbase, yarrow_start, yarrow_end, exclude=yexclude, text_attr=self.text_attr, **self.attr).SVG(trans)
        return SVG("g", *(xaxis.sub + yaxis.sub))

######################################################################

class HGrid(Ticks):
    """Draws the horizontal lines of a grid over a specified region
    using the standard tick specification (see help(Ticks)) to place the
    grid lines.

    HGrid(xmin, xmax, low, high, ticks, miniticks, logbase, mini_attr, attribute=value)

    xmin, xmax              required        the x range
    low, high               required        the y range
    ticks                   default=-10     request ticks according to the standard
                                            tick specification (see help(Ticks))
    miniticks               default=False   request miniticks according to the
                                            standard minitick specification
    logbase                 default=None    if a number, the axis is logarithmic
                                            with ticks at the given base (usually 10)
    mini_attr               default={}      SVG attributes for the minitick-lines
                                            (if miniticks != False)
    attribute=value pairs   keyword list    SVG attributes for the major tick lines
    """
    defaults = {"stroke-width": "0.25pt", "stroke": "gray", }
    mini_defaults = {"stroke-width": "0.25pt", "stroke": "lightgray", "stroke-dasharray": "1,1", }

    def __repr__(self):
        return "<HGrid x=(%g, %g) %g <= y <= %g ticks=%s miniticks=%s %s>" % (
               self.xmin, self.xmax, self.low, self.high, str(self.ticks), str(self.miniticks), self.attr)

    def __init__(self, xmin, xmax, low, high, ticks=-10, miniticks=False, logbase=None, mini_attr={}, **attr):
        self.xmin, self.xmax = xmin, xmax

        self.mini_attr = dict(self.mini_defaults)
        self.mini_attr.update(mini_attr)

        Ticks.__init__(self, None, low, high, ticks, miniticks, None, logbase)

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        self.last_ticks, self.last_miniticks = Ticks.interpret(self)

        ticksd = []
        for t in self.last_ticks.keys():
            ticksd += Line(self.xmin, t, self.xmax, t).Path(trans).d

        miniticksd = []
        for t in self.last_miniticks:
            miniticksd += Line(self.xmin, t, self.xmax, t).Path(trans).d

        return SVG("g", Path(d=ticksd, **self.attr).SVG(), Path(d=miniticksd, **self.mini_attr).SVG())


class VGrid(Ticks):
    """Draws the vertical lines of a grid over a specified region
    using the standard tick specification (see help(Ticks)) to place the
    grid lines.

    HGrid(ymin, ymax, low, high, ticks, miniticks, logbase, mini_attr, attribute=value)

    ymin, ymax              required        the y range
    low, high               required        the x range
    ticks                   default=-10     request ticks according to the standard
                                            tick specification (see help(Ticks))
    miniticks               default=False   request miniticks according to the
                                            standard minitick specification
    logbase                 default=None    if a number, the axis is logarithmic
                                            with ticks at the given base (usually 10)
    mini_attr               default={}      SVG attributes for the minitick-lines
                                            (if miniticks != False)
    attribute=value pairs   keyword list    SVG attributes for the major tick lines
    """
    defaults = {"stroke-width": "0.25pt", "stroke": "gray", }
    mini_defaults = {"stroke-width": "0.25pt", "stroke": "lightgray", "stroke-dasharray": "1,1", }

    def __repr__(self):
        return "<VGrid y=(%g, %g) %g <= x <= %g ticks=%s miniticks=%s %s>" % (
               self.ymin, self.ymax, self.low, self.high, str(self.ticks), str(self.miniticks), self.attr)

    def __init__(self, ymin, ymax, low, high, ticks=-10, miniticks=False, logbase=None, mini_attr={}, **attr):
        self.ymin, self.ymax = ymin, ymax

        self.mini_attr = dict(self.mini_defaults)
        self.mini_attr.update(mini_attr)

        Ticks.__init__(self, None, low, high, ticks, miniticks, None, logbase)

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        self.last_ticks, self.last_miniticks = Ticks.interpret(self)

        ticksd = []
        for t in self.last_ticks.keys():
            ticksd += Line(t, self.ymin, t, self.ymax).Path(trans).d

        miniticksd = []
        for t in self.last_miniticks:
            miniticksd += Line(t, self.ymin, t, self.ymax).Path(trans).d

        return SVG("g", Path(d=ticksd, **self.attr).SVG(), Path(d=miniticksd, **self.mini_attr).SVG())


class Grid(Ticks):
    """Draws a grid over a specified region using the standard tick
    specification (see help(Ticks)) to place the grid lines.

    Grid(xmin, xmax, ymin, ymax, ticks, miniticks, logbase, mini_attr, attribute=value)

    xmin, xmax              required        the x range
    ymin, ymax              required        the y range
    ticks                   default=-10     request ticks according to the standard
                                            tick specification (see help(Ticks))
    miniticks               default=False   request miniticks according to the
                                            standard minitick specification
    logbase                 default=None    if a number, the axis is logarithmic
                                            with ticks at the given base (usually 10)
    mini_attr               default={}      SVG attributes for the minitick-lines
                                            (if miniticks != False)
    attribute=value pairs   keyword list    SVG attributes for the major tick lines
    """
    defaults = {"stroke-width": "0.25pt", "stroke": "gray", }
    mini_defaults = {"stroke-width": "0.25pt", "stroke": "lightgray", "stroke-dasharray": "1,1", }

    def __repr__(self):
        return "<Grid x=(%g, %g) y=(%g, %g) ticks=%s miniticks=%s %s>" % (
               self.xmin, self.xmax, self.ymin, self.ymax, str(self.ticks), str(self.miniticks), self.attr)

    def __init__(self, xmin, xmax, ymin, ymax, ticks=-10, miniticks=False, logbase=None, mini_attr={}, **attr):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

        self.mini_attr = dict(self.mini_defaults)
        self.mini_attr.update(mini_attr)

        Ticks.__init__(self, None, None, None, ticks, miniticks, None, logbase)

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        self.low, self.high = self.xmin, self.xmax
        self.last_xticks, self.last_xminiticks = Ticks.interpret(self)
        self.low, self.high = self.ymin, self.ymax
        self.last_yticks, self.last_yminiticks = Ticks.interpret(self)

        ticksd = []
        for t in self.last_xticks.keys():
            ticksd += Line(t, self.ymin, t, self.ymax).Path(trans).d
        for t in self.last_yticks.keys():
            ticksd += Line(self.xmin, t, self.xmax, t).Path(trans).d

        miniticksd = []
        for t in self.last_xminiticks:
            miniticksd += Line(t, self.ymin, t, self.ymax).Path(trans).d
        for t in self.last_yminiticks:
            miniticksd += Line(self.xmin, t, self.xmax, t).Path(trans).d

        return SVG("g", Path(d=ticksd, **self.attr).SVG(), Path(d=miniticksd, **self.mini_attr).SVG())

######################################################################

class XErrorBars:
    """Draws x error bars at a set of points. This is usually used
    before (under) a set of Dots at the same points.

    XErrorBars(d, attribute=value)

    d                       required        list of (x,y,xerr...) points
    attribute=value pairs   keyword list    SVG attributes

    If points in d have

        * 3 elements, the third is the symmetric error bar
        * 4 elements, the third and fourth are the asymmetric lower and
          upper error bar. The third element should be negative,
          e.g. (5, 5, -1, 2) is a bar from 4 to 7.
        * more than 4, a tick mark is placed at each value. This lets
          you nest errors from different sources, correlated and
          uncorrelated, statistical and systematic, etc.
    """
    defaults = {"stroke-width": "0.25pt", }

    def __repr__(self):
        return "<XErrorBars (%d nodes)>" % len(self.d)

    def __init__(self, d=[], **attr):
        self.d = list(d)

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if isinstance(trans, basestring):
            trans = totrans(trans) # only once

        output = SVG("g")
        for p in self.d:
            x, y = p[0], p[1]

            if len(p) == 3:
                bars = [x - p[2], x + p[2]]
            else:
                bars = [x + pi for pi in p[2:]]

            start, end = min(bars), max(bars)
            output.append(LineAxis(start, y, end, y, start, end, bars, False, False, **self.attr).SVG(trans))

        return output


class YErrorBars:
    """Draws y error bars at a set of points. This is usually used
    before (under) a set of Dots at the same points.

    YErrorBars(d, attribute=value)

    d                       required        list of (x,y,yerr...) points
    attribute=value pairs   keyword list    SVG attributes

    If points in d have

        * 3 elements, the third is the symmetric error bar
        * 4 elements, the third and fourth are the asymmetric lower and
          upper error bar. The third element should be negative,
          e.g. (5, 5, -1, 2) is a bar from 4 to 7.
        * more than 4, a tick mark is placed at each value. This lets
          you nest errors from different sources, correlated and
          uncorrelated, statistical and systematic, etc.
    """
    defaults = {"stroke-width": "0.25pt", }

    def __repr__(self):
        return "<YErrorBars (%d nodes)>" % len(self.d)

    def __init__(self, d=[], **attr):
        self.d = list(d)

        self.attr = dict(self.defaults)
        self.attr.update(attr)

    def SVG(self, trans=None):
        """Apply the transformation "trans" and return an SVG object."""
        if isinstance(trans, basestring):
            trans = totrans(trans) # only once

        output = SVG("g")
        for p in self.d:
            x, y = p[0], p[1]

            if len(p) == 3:
                bars = [y - p[2], y + p[2]]
            else:
                bars = [y + pi for pi in p[2:]]

            start, end = min(bars), max(bars)
            output.append(LineAxis(x, start, x, end, start, end, bars, False, False, **self.attr).SVG(trans))

        return output
