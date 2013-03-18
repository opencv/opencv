# -*- coding: utf-8 -*-
"""
    jinja2.testsuite.api
    ~~~~~~~~~~~~~~~~~~~~

    Tests the public API and related stuff.

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import unittest

from jinja2.testsuite import JinjaTestCase

from jinja2 import Environment, Undefined, DebugUndefined, \
     StrictUndefined, UndefinedError, meta, \
     is_undefined, Template, DictLoader
from jinja2.utils import Cycler

env = Environment()


class ExtendedAPITestCase(JinjaTestCase):

    def test_item_and_attribute(self):
        from jinja2.sandbox import SandboxedEnvironment

        for env in Environment(), SandboxedEnvironment():
            # the |list is necessary for python3
            tmpl = env.from_string('{{ foo.items()|list }}')
            assert tmpl.render(foo={'items': 42}) == "[('items', 42)]"
            tmpl = env.from_string('{{ foo|attr("items")()|list }}')
            assert tmpl.render(foo={'items': 42}) == "[('items', 42)]"
            tmpl = env.from_string('{{ foo["items"] }}')
            assert tmpl.render(foo={'items': 42}) == '42'

    def test_finalizer(self):
        def finalize_none_empty(value):
            if value is None:
                value = u''
            return value
        env = Environment(finalize=finalize_none_empty)
        tmpl = env.from_string('{% for item in seq %}|{{ item }}{% endfor %}')
        assert tmpl.render(seq=(None, 1, "foo")) == '||1|foo'
        tmpl = env.from_string('<{{ none }}>')
        assert tmpl.render() == '<>'

    def test_cycler(self):
        items = 1, 2, 3
        c = Cycler(*items)
        for item in items + items:
            assert c.current == item
            assert c.next() == item
        c.next()
        assert c.current == 2
        c.reset()
        assert c.current == 1

    def test_expressions(self):
        expr = env.compile_expression("foo")
        assert expr() is None
        assert expr(foo=42) == 42
        expr2 = env.compile_expression("foo", undefined_to_none=False)
        assert is_undefined(expr2())

        expr = env.compile_expression("42 + foo")
        assert expr(foo=42) == 84

    def test_template_passthrough(self):
        t = Template('Content')
        assert env.get_template(t) is t
        assert env.select_template([t]) is t
        assert env.get_or_select_template([t]) is t
        assert env.get_or_select_template(t) is t

    def test_autoescape_autoselect(self):
        def select_autoescape(name):
            if name is None or '.' not in name:
                return False
            return name.endswith('.html')
        env = Environment(autoescape=select_autoescape,
                          loader=DictLoader({
            'test.txt':     '{{ foo }}',
            'test.html':    '{{ foo }}'
        }))
        t = env.get_template('test.txt')
        assert t.render(foo='<foo>') == '<foo>'
        t = env.get_template('test.html')
        assert t.render(foo='<foo>') == '&lt;foo&gt;'
        t = env.from_string('{{ foo }}')
        assert t.render(foo='<foo>') == '<foo>'


class MetaTestCase(JinjaTestCase):

    def test_find_undeclared_variables(self):
        ast = env.parse('{% set foo = 42 %}{{ bar + foo }}')
        x = meta.find_undeclared_variables(ast)
        assert x == set(['bar'])

        ast = env.parse('{% set foo = 42 %}{{ bar + foo }}'
                        '{% macro meh(x) %}{{ x }}{% endmacro %}'
                        '{% for item in seq %}{{ muh(item) + meh(seq) }}{% endfor %}')
        x = meta.find_undeclared_variables(ast)
        assert x == set(['bar', 'seq', 'muh'])

    def test_find_refererenced_templates(self):
        ast = env.parse('{% extends "layout.html" %}{% include helper %}')
        i = meta.find_referenced_templates(ast)
        assert i.next() == 'layout.html'
        assert i.next() is None
        assert list(i) == []

        ast = env.parse('{% extends "layout.html" %}'
                        '{% from "test.html" import a, b as c %}'
                        '{% import "meh.html" as meh %}'
                        '{% include "muh.html" %}')
        i = meta.find_referenced_templates(ast)
        assert list(i) == ['layout.html', 'test.html', 'meh.html', 'muh.html']

    def test_find_included_templates(self):
        ast = env.parse('{% include ["foo.html", "bar.html"] %}')
        i = meta.find_referenced_templates(ast)
        assert list(i) == ['foo.html', 'bar.html']

        ast = env.parse('{% include ("foo.html", "bar.html") %}')
        i = meta.find_referenced_templates(ast)
        assert list(i) == ['foo.html', 'bar.html']

        ast = env.parse('{% include ["foo.html", "bar.html", foo] %}')
        i = meta.find_referenced_templates(ast)
        assert list(i) == ['foo.html', 'bar.html', None]

        ast = env.parse('{% include ("foo.html", "bar.html", foo) %}')
        i = meta.find_referenced_templates(ast)
        assert list(i) == ['foo.html', 'bar.html', None]


class StreamingTestCase(JinjaTestCase):

    def test_basic_streaming(self):
        tmpl = env.from_string("<ul>{% for item in seq %}<li>{{ loop.index "
                               "}} - {{ item }}</li>{%- endfor %}</ul>")
        stream = tmpl.stream(seq=range(4))
        self.assert_equal(stream.next(), '<ul>')
        self.assert_equal(stream.next(), '<li>1 - 0</li>')
        self.assert_equal(stream.next(), '<li>2 - 1</li>')
        self.assert_equal(stream.next(), '<li>3 - 2</li>')
        self.assert_equal(stream.next(), '<li>4 - 3</li>')
        self.assert_equal(stream.next(), '</ul>')

    def test_buffered_streaming(self):
        tmpl = env.from_string("<ul>{% for item in seq %}<li>{{ loop.index "
                               "}} - {{ item }}</li>{%- endfor %}</ul>")
        stream = tmpl.stream(seq=range(4))
        stream.enable_buffering(size=3)
        self.assert_equal(stream.next(), u'<ul><li>1 - 0</li><li>2 - 1</li>')
        self.assert_equal(stream.next(), u'<li>3 - 2</li><li>4 - 3</li></ul>')

    def test_streaming_behavior(self):
        tmpl = env.from_string("")
        stream = tmpl.stream()
        assert not stream.buffered
        stream.enable_buffering(20)
        assert stream.buffered
        stream.disable_buffering()
        assert not stream.buffered


class UndefinedTestCase(JinjaTestCase):

    def test_stopiteration_is_undefined(self):
        def test():
            raise StopIteration()
        t = Template('A{{ test() }}B')
        assert t.render(test=test) == 'AB'
        t = Template('A{{ test().missingattribute }}B')
        self.assert_raises(UndefinedError, t.render, test=test)

    def test_undefined_and_special_attributes(self):
        try:
            Undefined('Foo').__dict__
        except AttributeError:
            pass
        else:
            assert False, "Expected actual attribute error"

    def test_default_undefined(self):
        env = Environment(undefined=Undefined)
        self.assert_equal(env.from_string('{{ missing }}').render(), u'')
        self.assert_raises(UndefinedError,
                           env.from_string('{{ missing.attribute }}').render)
        self.assert_equal(env.from_string('{{ missing|list }}').render(), '[]')
        self.assert_equal(env.from_string('{{ missing is not defined }}').render(), 'True')
        self.assert_equal(env.from_string('{{ foo.missing }}').render(foo=42), '')
        self.assert_equal(env.from_string('{{ not missing }}').render(), 'True')

    def test_debug_undefined(self):
        env = Environment(undefined=DebugUndefined)
        self.assert_equal(env.from_string('{{ missing }}').render(), '{{ missing }}')
        self.assert_raises(UndefinedError,
                           env.from_string('{{ missing.attribute }}').render)
        self.assert_equal(env.from_string('{{ missing|list }}').render(), '[]')
        self.assert_equal(env.from_string('{{ missing is not defined }}').render(), 'True')
        self.assert_equal(env.from_string('{{ foo.missing }}').render(foo=42),
                          u"{{ no such element: int object['missing'] }}")
        self.assert_equal(env.from_string('{{ not missing }}').render(), 'True')

    def test_strict_undefined(self):
        env = Environment(undefined=StrictUndefined)
        self.assert_raises(UndefinedError, env.from_string('{{ missing }}').render)
        self.assert_raises(UndefinedError, env.from_string('{{ missing.attribute }}').render)
        self.assert_raises(UndefinedError, env.from_string('{{ missing|list }}').render)
        self.assert_equal(env.from_string('{{ missing is not defined }}').render(), 'True')
        self.assert_raises(UndefinedError, env.from_string('{{ foo.missing }}').render, foo=42)
        self.assert_raises(UndefinedError, env.from_string('{{ not missing }}').render)

    def test_indexing_gives_undefined(self):
        t = Template("{{ var[42].foo }}")
        self.assert_raises(UndefinedError, t.render, var=0)

    def test_none_gives_proper_error(self):
        try:
            Environment().getattr(None, 'split')()
        except UndefinedError, e:
            assert e.message == "'None' has no attribute 'split'"
        else:
            assert False, 'expected exception'

    def test_object_repr(self):
        try:
            Undefined(obj=42, name='upper')()
        except UndefinedError, e:
            assert e.message == "'int object' has no attribute 'upper'"
        else:
            assert False, 'expected exception'


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ExtendedAPITestCase))
    suite.addTest(unittest.makeSuite(MetaTestCase))
    suite.addTest(unittest.makeSuite(StreamingTestCase))
    suite.addTest(unittest.makeSuite(UndefinedTestCase))
    return suite
