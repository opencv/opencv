# -*- coding: utf-8 -*-
"""
    jinja2.utils
    ~~~~~~~~~~~~

    Utility functions.

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import re
import sys
import errno
try:
    from urllib.parse import quote_from_bytes as url_quote
except ImportError:
    from urllib import quote as url_quote
try:
    from thread import allocate_lock
except ImportError:
    from dummy_thread import allocate_lock
from collections import deque
from itertools import imap


_word_split_re = re.compile(r'(\s+)')
_punctuation_re = re.compile(
    '^(?P<lead>(?:%s)*)(?P<middle>.*?)(?P<trail>(?:%s)*)$' % (
        '|'.join(imap(re.escape, ('(', '<', '&lt;'))),
        '|'.join(imap(re.escape, ('.', ',', ')', '>', '\n', '&gt;')))
    )
)
_simple_email_re = re.compile(r'^\S+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9._-]+$')
_striptags_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
_entity_re = re.compile(r'&([^;]+);')
_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_digits = '0123456789'

# special singleton representing missing values for the runtime
missing = type('MissingType', (), {'__repr__': lambda x: 'missing'})()

# internal code
internal_code = set()


# concatenate a list of strings and convert them to unicode.
# unfortunately there is a bug in python 2.4 and lower that causes
# unicode.join trash the traceback.
_concat = u''.join
try:
    def _test_gen_bug():
        raise TypeError(_test_gen_bug)
        yield None
    _concat(_test_gen_bug())
except TypeError, _error:
    if not _error.args or _error.args[0] is not _test_gen_bug:
        def concat(gen):
            try:
                return _concat(list(gen))
            except Exception:
                # this hack is needed so that the current frame
                # does not show up in the traceback.
                exc_type, exc_value, tb = sys.exc_info()
                raise exc_type, exc_value, tb.tb_next
    else:
        concat = _concat
    del _test_gen_bug, _error


# for python 2.x we create ourselves a next() function that does the
# basics without exception catching.
try:
    next = next
except NameError:
    def next(x):
        return x.next()


# if this python version is unable to deal with unicode filenames
# when passed to encode we let this function encode it properly.
# This is used in a couple of places.  As far as Jinja is concerned
# filenames are unicode *or* bytestrings in 2.x and unicode only in
# 3.x because compile cannot handle bytes
if sys.version_info < (3, 0):
    def _encode_filename(filename):
        if isinstance(filename, unicode):
            return filename.encode('utf-8')
        return filename
else:
    def _encode_filename(filename):
        assert filename is None or isinstance(filename, str), \
            'filenames must be strings'
        return filename

from keyword import iskeyword as is_python_keyword


# common types.  These do exist in the special types module too which however
# does not exist in IronPython out of the box.  Also that way we don't have
# to deal with implementation specific stuff here
class _C(object):
    def method(self): pass
def _func():
    yield None
FunctionType = type(_func)
GeneratorType = type(_func())
MethodType = type(_C.method)
CodeType = type(_C.method.func_code)
try:
    raise TypeError()
except TypeError:
    _tb = sys.exc_info()[2]
    TracebackType = type(_tb)
    FrameType = type(_tb.tb_frame)
del _C, _tb, _func


def contextfunction(f):
    """This decorator can be used to mark a function or method context callable.
    A context callable is passed the active :class:`Context` as first argument when
    called from the template.  This is useful if a function wants to get access
    to the context or functions provided on the context object.  For example
    a function that returns a sorted list of template variables the current
    template exports could look like this::

        @contextfunction
        def get_exported_names(context):
            return sorted(context.exported_vars)
    """
    f.contextfunction = True
    return f


def evalcontextfunction(f):
    """This decorator can be used to mark a function or method as an eval
    context callable.  This is similar to the :func:`contextfunction`
    but instead of passing the context, an evaluation context object is
    passed.  For more information about the eval context, see
    :ref:`eval-context`.

    .. versionadded:: 2.4
    """
    f.evalcontextfunction = True
    return f


def environmentfunction(f):
    """This decorator can be used to mark a function or method as environment
    callable.  This decorator works exactly like the :func:`contextfunction`
    decorator just that the first argument is the active :class:`Environment`
    and not context.
    """
    f.environmentfunction = True
    return f


def internalcode(f):
    """Marks the function as internally used"""
    internal_code.add(f.func_code)
    return f


def is_undefined(obj):
    """Check if the object passed is undefined.  This does nothing more than
    performing an instance check against :class:`Undefined` but looks nicer.
    This can be used for custom filters or tests that want to react to
    undefined variables.  For example a custom default filter can look like
    this::

        def default(var, default=''):
            if is_undefined(var):
                return default
            return var
    """
    from jinja2.runtime import Undefined
    return isinstance(obj, Undefined)


def consume(iterable):
    """Consumes an iterable without doing anything with it."""
    for event in iterable:
        pass


def clear_caches():
    """Jinja2 keeps internal caches for environments and lexers.  These are
    used so that Jinja2 doesn't have to recreate environments and lexers all
    the time.  Normally you don't have to care about that but if you are
    messuring memory consumption you may want to clean the caches.
    """
    from jinja2.environment import _spontaneous_environments
    from jinja2.lexer import _lexer_cache
    _spontaneous_environments.clear()
    _lexer_cache.clear()


def import_string(import_name, silent=False):
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If the `silent` is True the return value will be `None` if the import
    fails.

    :return: imported object
    """
    try:
        if ':' in import_name:
            module, obj = import_name.split(':', 1)
        elif '.' in import_name:
            items = import_name.split('.')
            module = '.'.join(items[:-1])
            obj = items[-1]
        else:
            return __import__(import_name)
        return getattr(__import__(module, None, None, [obj]), obj)
    except (ImportError, AttributeError):
        if not silent:
            raise


def open_if_exists(filename, mode='rb'):
    """Returns a file descriptor for the filename if that file exists,
    otherwise `None`.
    """
    try:
        return open(filename, mode)
    except IOError, e:
        if e.errno not in (errno.ENOENT, errno.EISDIR):
            raise


def object_type_repr(obj):
    """Returns the name of the object's type.  For some recognized
    singletons the name of the object is returned instead. (For
    example for `None` and `Ellipsis`).
    """
    if obj is None:
        return 'None'
    elif obj is Ellipsis:
        return 'Ellipsis'
    # __builtin__ in 2.x, builtins in 3.x
    if obj.__class__.__module__ in ('__builtin__', 'builtins'):
        name = obj.__class__.__name__
    else:
        name = obj.__class__.__module__ + '.' + obj.__class__.__name__
    return '%s object' % name


def pformat(obj, verbose=False):
    """Prettyprint an object.  Either use the `pretty` library or the
    builtin `pprint`.
    """
    try:
        from pretty import pretty
        return pretty(obj, verbose=verbose)
    except ImportError:
        from pprint import pformat
        return pformat(obj)


def urlize(text, trim_url_limit=None, nofollow=False):
    """Converts any URLs in text into clickable links. Works on http://,
    https:// and www. links. Links can have trailing punctuation (periods,
    commas, close-parens) and leading punctuation (opening parens) and
    it'll still do the right thing.

    If trim_url_limit is not None, the URLs in link text will be limited
    to trim_url_limit characters.

    If nofollow is True, the URLs in link text will get a rel="nofollow"
    attribute.
    """
    trim_url = lambda x, limit=trim_url_limit: limit is not None \
                         and (x[:limit] + (len(x) >=limit and '...'
                         or '')) or x
    words = _word_split_re.split(unicode(escape(text)))
    nofollow_attr = nofollow and ' rel="nofollow"' or ''
    for i, word in enumerate(words):
        match = _punctuation_re.match(word)
        if match:
            lead, middle, trail = match.groups()
            if middle.startswith('www.') or (
                '@' not in middle and
                not middle.startswith('http://') and
                len(middle) > 0 and
                middle[0] in _letters + _digits and (
                    middle.endswith('.org') or
                    middle.endswith('.net') or
                    middle.endswith('.com')
                )):
                middle = '<a href="http://%s"%s>%s</a>' % (middle,
                    nofollow_attr, trim_url(middle))
            if middle.startswith('http://') or \
               middle.startswith('https://'):
                middle = '<a href="%s"%s>%s</a>' % (middle,
                    nofollow_attr, trim_url(middle))
            if '@' in middle and not middle.startswith('www.') and \
               not ':' in middle and _simple_email_re.match(middle):
                middle = '<a href="mailto:%s">%s</a>' % (middle, middle)
            if lead + middle + trail != word:
                words[i] = lead + middle + trail
    return u''.join(words)


def generate_lorem_ipsum(n=5, html=True, min=20, max=100):
    """Generate some lorem impsum for the template."""
    from jinja2.constants import LOREM_IPSUM_WORDS
    from random import choice, randrange
    words = LOREM_IPSUM_WORDS.split()
    result = []

    for _ in xrange(n):
        next_capitalized = True
        last_comma = last_fullstop = 0
        word = None
        last = None
        p = []

        # each paragraph contains out of 20 to 100 words.
        for idx, _ in enumerate(xrange(randrange(min, max))):
            while True:
                word = choice(words)
                if word != last:
                    last = word
                    break
            if next_capitalized:
                word = word.capitalize()
                next_capitalized = False
            # add commas
            if idx - randrange(3, 8) > last_comma:
                last_comma = idx
                last_fullstop += 2
                word += ','
            # add end of sentences
            if idx - randrange(10, 20) > last_fullstop:
                last_comma = last_fullstop = idx
                word += '.'
                next_capitalized = True
            p.append(word)

        # ensure that the paragraph ends with a dot.
        p = u' '.join(p)
        if p.endswith(','):
            p = p[:-1] + '.'
        elif not p.endswith('.'):
            p += '.'
        result.append(p)

    if not html:
        return u'\n\n'.join(result)
    return Markup(u'\n'.join(u'<p>%s</p>' % escape(x) for x in result))


def unicode_urlencode(obj, charset='utf-8'):
    """URL escapes a single bytestring or unicode string with the
    given charset if applicable to URL safe quoting under all rules
    that need to be considered under all supported Python versions.

    If non strings are provided they are converted to their unicode
    representation first.
    """
    if not isinstance(obj, basestring):
        obj = unicode(obj)
    if isinstance(obj, unicode):
        obj = obj.encode(charset)
    return unicode(url_quote(obj))


class LRUCache(object):
    """A simple LRU Cache implementation."""

    # this is fast for small capacities (something below 1000) but doesn't
    # scale.  But as long as it's only used as storage for templates this
    # won't do any harm.

    def __init__(self, capacity):
        self.capacity = capacity
        self._mapping = {}
        self._queue = deque()
        self._postinit()

    def _postinit(self):
        # alias all queue methods for faster lookup
        self._popleft = self._queue.popleft
        self._pop = self._queue.pop
        if hasattr(self._queue, 'remove'):
            self._remove = self._queue.remove
        self._wlock = allocate_lock()
        self._append = self._queue.append

    def _remove(self, obj):
        """Python 2.4 compatibility."""
        for idx, item in enumerate(self._queue):
            if item == obj:
                del self._queue[idx]
                break

    def __getstate__(self):
        return {
            'capacity':     self.capacity,
            '_mapping':     self._mapping,
            '_queue':       self._queue
        }

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._postinit()

    def __getnewargs__(self):
        return (self.capacity,)

    def copy(self):
        """Return a shallow copy of the instance."""
        rv = self.__class__(self.capacity)
        rv._mapping.update(self._mapping)
        rv._queue = deque(self._queue)
        return rv

    def get(self, key, default=None):
        """Return an item from the cache dict or `default`"""
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key, default=None):
        """Set `default` if the key is not in the cache otherwise
        leave unchanged. Return the value of this key.
        """
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default

    def clear(self):
        """Clear the cache."""
        self._wlock.acquire()
        try:
            self._mapping.clear()
            self._queue.clear()
        finally:
            self._wlock.release()

    def __contains__(self, key):
        """Check if a key exists in this cache."""
        return key in self._mapping

    def __len__(self):
        """Return the current size of the cache."""
        return len(self._mapping)

    def __repr__(self):
        return '<%s %r>' % (
            self.__class__.__name__,
            self._mapping
        )

    def __getitem__(self, key):
        """Get an item from the cache. Moves the item up so that it has the
        highest priority then.

        Raise a `KeyError` if it does not exist.
        """
        rv = self._mapping[key]
        if self._queue[-1] != key:
            try:
                self._remove(key)
            except ValueError:
                # if something removed the key from the container
                # when we read, ignore the ValueError that we would
                # get otherwise.
                pass
            self._append(key)
        return rv

    def __setitem__(self, key, value):
        """Sets the value for an item. Moves the item up so that it
        has the highest priority then.
        """
        self._wlock.acquire()
        try:
            if key in self._mapping:
                try:
                    self._remove(key)
                except ValueError:
                    # __getitem__ is not locked, it might happen
                    pass
            elif len(self._mapping) == self.capacity:
                del self._mapping[self._popleft()]
            self._append(key)
            self._mapping[key] = value
        finally:
            self._wlock.release()

    def __delitem__(self, key):
        """Remove an item from the cache dict.
        Raise a `KeyError` if it does not exist.
        """
        self._wlock.acquire()
        try:
            del self._mapping[key]
            try:
                self._remove(key)
            except ValueError:
                # __getitem__ is not locked, it might happen
                pass
        finally:
            self._wlock.release()

    def items(self):
        """Return a list of items."""
        result = [(key, self._mapping[key]) for key in list(self._queue)]
        result.reverse()
        return result

    def iteritems(self):
        """Iterate over all items."""
        return iter(self.items())

    def values(self):
        """Return a list of all values."""
        return [x[1] for x in self.items()]

    def itervalue(self):
        """Iterate over all values."""
        return iter(self.values())

    def keys(self):
        """Return a list of all keys ordered by most recent usage."""
        return list(self)

    def iterkeys(self):
        """Iterate over all keys in the cache dict, ordered by
        the most recent usage.
        """
        return reversed(tuple(self._queue))

    __iter__ = iterkeys

    def __reversed__(self):
        """Iterate over the values in the cache dict, oldest items
        coming first.
        """
        return iter(tuple(self._queue))

    __copy__ = copy


# register the LRU cache as mutable mapping if possible
try:
    from collections import MutableMapping
    MutableMapping.register(LRUCache)
except ImportError:
    pass


class Cycler(object):
    """A cycle helper for templates."""

    def __init__(self, *items):
        if not items:
            raise RuntimeError('at least one item has to be provided')
        self.items = items
        self.reset()

    def reset(self):
        """Resets the cycle."""
        self.pos = 0

    @property
    def current(self):
        """Returns the current item."""
        return self.items[self.pos]

    def next(self):
        """Goes one item ahead and returns it."""
        rv = self.current
        self.pos = (self.pos + 1) % len(self.items)
        return rv


class Joiner(object):
    """A joining helper for templates."""

    def __init__(self, sep=u', '):
        self.sep = sep
        self.used = False

    def __call__(self):
        if not self.used:
            self.used = True
            return u''
        return self.sep


# try markupsafe first, if that fails go with Jinja2's bundled version
# of markupsafe.  Markupsafe was previously Jinja2's implementation of
# the Markup object but was moved into a separate package in a patchlevel
# release
try:
    from markupsafe import Markup, escape, soft_unicode
except ImportError:
    from jinja2._markupsafe import Markup, escape, soft_unicode


# partials
try:
    from functools import partial
except ImportError:
    class partial(object):
        def __init__(self, _func, *args, **kwargs):
            self._func = _func
            self._args = args
            self._kwargs = kwargs
        def __call__(self, *args, **kwargs):
            kwargs.update(self._kwargs)
            return self._func(*(self._args + args), **kwargs)
