# -*- coding: utf-8 -*-
"""
    jinja2.testsuite.imports
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Tests the import features (with includes).

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import unittest

from jinja2.testsuite import JinjaTestCase

from jinja2 import Environment, DictLoader
from jinja2.exceptions import TemplateNotFound, TemplatesNotFound


test_env = Environment(loader=DictLoader(dict(
    module='{% macro test() %}[{{ foo }}|{{ bar }}]{% endmacro %}',
    header='[{{ foo }}|{{ 23 }}]',
    o_printer='({{ o }})'
)))
test_env.globals['bar'] = 23


class ImportsTestCase(JinjaTestCase):

    def test_context_imports(self):
        t = test_env.from_string('{% import "module" as m %}{{ m.test() }}')
        assert t.render(foo=42) == '[|23]'
        t = test_env.from_string('{% import "module" as m without context %}{{ m.test() }}')
        assert t.render(foo=42) == '[|23]'
        t = test_env.from_string('{% import "module" as m with context %}{{ m.test() }}')
        assert t.render(foo=42) == '[42|23]'
        t = test_env.from_string('{% from "module" import test %}{{ test() }}')
        assert t.render(foo=42) == '[|23]'
        t = test_env.from_string('{% from "module" import test without context %}{{ test() }}')
        assert t.render(foo=42) == '[|23]'
        t = test_env.from_string('{% from "module" import test with context %}{{ test() }}')
        assert t.render(foo=42) == '[42|23]'

    def test_trailing_comma(self):
        test_env.from_string('{% from "foo" import bar, baz with context %}')
        test_env.from_string('{% from "foo" import bar, baz, with context %}')
        test_env.from_string('{% from "foo" import bar, with context %}')
        test_env.from_string('{% from "foo" import bar, with, context %}')
        test_env.from_string('{% from "foo" import bar, with with context %}')

    def test_exports(self):
        m = test_env.from_string('''
            {% macro toplevel() %}...{% endmacro %}
            {% macro __private() %}...{% endmacro %}
            {% set variable = 42 %}
            {% for item in [1] %}
                {% macro notthere() %}{% endmacro %}
            {% endfor %}
        ''').module
        assert m.toplevel() == '...'
        assert not hasattr(m, '__missing')
        assert m.variable == 42
        assert not hasattr(m, 'notthere')


class IncludesTestCase(JinjaTestCase):

    def test_context_include(self):
        t = test_env.from_string('{% include "header" %}')
        assert t.render(foo=42) == '[42|23]'
        t = test_env.from_string('{% include "header" with context %}')
        assert t.render(foo=42) == '[42|23]'
        t = test_env.from_string('{% include "header" without context %}')
        assert t.render(foo=42) == '[|23]'

    def test_choice_includes(self):
        t = test_env.from_string('{% include ["missing", "header"] %}')
        assert t.render(foo=42) == '[42|23]'

        t = test_env.from_string('{% include ["missing", "missing2"] ignore missing %}')
        assert t.render(foo=42) == ''

        t = test_env.from_string('{% include ["missing", "missing2"] %}')
        self.assert_raises(TemplateNotFound, t.render)
        try:
            t.render()
        except TemplatesNotFound, e:
            assert e.templates == ['missing', 'missing2']
            assert e.name == 'missing2'
        else:
            assert False, 'thou shalt raise'

        def test_includes(t, **ctx):
            ctx['foo'] = 42
            assert t.render(ctx) == '[42|23]'

        t = test_env.from_string('{% include ["missing", "header"] %}')
        test_includes(t)
        t = test_env.from_string('{% include x %}')
        test_includes(t, x=['missing', 'header'])
        t = test_env.from_string('{% include [x, "header"] %}')
        test_includes(t, x='missing')
        t = test_env.from_string('{% include x %}')
        test_includes(t, x='header')
        t = test_env.from_string('{% include x %}')
        test_includes(t, x='header')
        t = test_env.from_string('{% include [x] %}')
        test_includes(t, x='header')

    def test_include_ignoring_missing(self):
        t = test_env.from_string('{% include "missing" %}')
        self.assert_raises(TemplateNotFound, t.render)
        for extra in '', 'with context', 'without context':
            t = test_env.from_string('{% include "missing" ignore missing ' +
                                     extra + ' %}')
            assert t.render() == ''

    def test_context_include_with_overrides(self):
        env = Environment(loader=DictLoader(dict(
            main="{% for item in [1, 2, 3] %}{% include 'item' %}{% endfor %}",
            item="{{ item }}"
        )))
        assert env.get_template("main").render() == "123"

    def test_unoptimized_scopes(self):
        t = test_env.from_string("""
            {% macro outer(o) %}
            {% macro inner() %}
            {% include "o_printer" %}
            {% endmacro %}
            {{ inner() }}
            {% endmacro %}
            {{ outer("FOO") }}
        """)
        assert t.render().strip() == '(FOO)'


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ImportsTestCase))
    suite.addTest(unittest.makeSuite(IncludesTestCase))
    return suite
