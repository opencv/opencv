# -*- coding: utf-8 -*-
"""
    jinja2.testsuite.filters
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Tests for the jinja filters.

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import unittest
from jinja2.testsuite import JinjaTestCase

from jinja2 import Markup, Environment

env = Environment()


class FilterTestCase(JinjaTestCase):

    def test_capitalize(self):
        tmpl = env.from_string('{{ "foo bar"|capitalize }}')
        assert tmpl.render() == 'Foo bar'

    def test_center(self):
        tmpl = env.from_string('{{ "foo"|center(9) }}')
        assert tmpl.render() == '   foo   '

    def test_default(self):
        tmpl = env.from_string(
            "{{ missing|default('no') }}|{{ false|default('no') }}|"
            "{{ false|default('no', true) }}|{{ given|default('no') }}"
        )
        assert tmpl.render(given='yes') == 'no|False|no|yes'

    def test_dictsort(self):
        tmpl = env.from_string(
            '{{ foo|dictsort }}|'
            '{{ foo|dictsort(true) }}|'
            '{{ foo|dictsort(false, "value") }}'
        )
        out = tmpl.render(foo={"aa": 0, "b": 1, "c": 2, "AB": 3})
        assert out == ("[('aa', 0), ('AB', 3), ('b', 1), ('c', 2)]|"
                       "[('AB', 3), ('aa', 0), ('b', 1), ('c', 2)]|"
                       "[('aa', 0), ('b', 1), ('c', 2), ('AB', 3)]")

    def test_batch(self):
        tmpl = env.from_string("{{ foo|batch(3)|list }}|"
                               "{{ foo|batch(3, 'X')|list }}")
        out = tmpl.render(foo=range(10))
        assert out == ("[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]|"
                       "[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 'X', 'X']]")

    def test_slice(self):
        tmpl = env.from_string('{{ foo|slice(3)|list }}|'
                               '{{ foo|slice(3, "X")|list }}')
        out = tmpl.render(foo=range(10))
        assert out == ("[[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]|"
                       "[[0, 1, 2, 3], [4, 5, 6, 'X'], [7, 8, 9, 'X']]")

    def test_escape(self):
        tmpl = env.from_string('''{{ '<">&'|escape }}''')
        out = tmpl.render()
        assert out == '&lt;&#34;&gt;&amp;'

    def test_striptags(self):
        tmpl = env.from_string('''{{ foo|striptags }}''')
        out = tmpl.render(foo='  <p>just a small   \n <a href="#">'
                          'example</a> link</p>\n<p>to a webpage</p> '
                          '<!-- <p>and some commented stuff</p> -->')
        assert out == 'just a small example link to a webpage'

    def test_filesizeformat(self):
        tmpl = env.from_string(
            '{{ 100|filesizeformat }}|'
            '{{ 1000|filesizeformat }}|'
            '{{ 1000000|filesizeformat }}|'
            '{{ 1000000000|filesizeformat }}|'
            '{{ 1000000000000|filesizeformat }}|'
            '{{ 100|filesizeformat(true) }}|'
            '{{ 1000|filesizeformat(true) }}|'
            '{{ 1000000|filesizeformat(true) }}|'
            '{{ 1000000000|filesizeformat(true) }}|'
            '{{ 1000000000000|filesizeformat(true) }}'
        )
        out = tmpl.render()
        self.assert_equal(out, (
            '100 Bytes|1.0 kB|1.0 MB|1.0 GB|1.0 TB|100 Bytes|'
            '1000 Bytes|976.6 KiB|953.7 MiB|931.3 GiB'
        ))

    def test_filesizeformat_issue59(self):
        tmpl = env.from_string(
            '{{ 300|filesizeformat }}|'
            '{{ 3000|filesizeformat }}|'
            '{{ 3000000|filesizeformat }}|'
            '{{ 3000000000|filesizeformat }}|'
            '{{ 3000000000000|filesizeformat }}|'
            '{{ 300|filesizeformat(true) }}|'
            '{{ 3000|filesizeformat(true) }}|'
            '{{ 3000000|filesizeformat(true) }}'
        )
        out = tmpl.render()
        self.assert_equal(out, (
            '300 Bytes|3.0 kB|3.0 MB|3.0 GB|3.0 TB|300 Bytes|'
            '2.9 KiB|2.9 MiB'
        ))


    def test_first(self):
        tmpl = env.from_string('{{ foo|first }}')
        out = tmpl.render(foo=range(10))
        assert out == '0'

    def test_float(self):
        tmpl = env.from_string('{{ "42"|float }}|'
                               '{{ "ajsghasjgd"|float }}|'
                               '{{ "32.32"|float }}')
        out = tmpl.render()
        assert out == '42.0|0.0|32.32'

    def test_format(self):
        tmpl = env.from_string('''{{ "%s|%s"|format("a", "b") }}''')
        out = tmpl.render()
        assert out == 'a|b'

    def test_indent(self):
        tmpl = env.from_string('{{ foo|indent(2) }}|{{ foo|indent(2, true) }}')
        text = '\n'.join([' '.join(['foo', 'bar'] * 2)] * 2)
        out = tmpl.render(foo=text)
        assert out == ('foo bar foo bar\n  foo bar foo bar|  '
                       'foo bar foo bar\n  foo bar foo bar')

    def test_int(self):
        tmpl = env.from_string('{{ "42"|int }}|{{ "ajsghasjgd"|int }}|'
                               '{{ "32.32"|int }}')
        out = tmpl.render()
        assert out == '42|0|32'

    def test_join(self):
        tmpl = env.from_string('{{ [1, 2, 3]|join("|") }}')
        out = tmpl.render()
        assert out == '1|2|3'

        env2 = Environment(autoescape=True)
        tmpl = env2.from_string('{{ ["<foo>", "<span>foo</span>"|safe]|join }}')
        assert tmpl.render() == '&lt;foo&gt;<span>foo</span>'

    def test_join_attribute(self):
        class User(object):
            def __init__(self, username):
                self.username = username
        tmpl = env.from_string('''{{ users|join(', ', 'username') }}''')
        assert tmpl.render(users=map(User, ['foo', 'bar'])) == 'foo, bar'

    def test_last(self):
        tmpl = env.from_string('''{{ foo|last }}''')
        out = tmpl.render(foo=range(10))
        assert out == '9'

    def test_length(self):
        tmpl = env.from_string('''{{ "hello world"|length }}''')
        out = tmpl.render()
        assert out == '11'

    def test_lower(self):
        tmpl = env.from_string('''{{ "FOO"|lower }}''')
        out = tmpl.render()
        assert out == 'foo'

    def test_pprint(self):
        from pprint import pformat
        tmpl = env.from_string('''{{ data|pprint }}''')
        data = range(1000)
        assert tmpl.render(data=data) == pformat(data)

    def test_random(self):
        tmpl = env.from_string('''{{ seq|random }}''')
        seq = range(100)
        for _ in range(10):
            assert int(tmpl.render(seq=seq)) in seq

    def test_reverse(self):
        tmpl = env.from_string('{{ "foobar"|reverse|join }}|'
                               '{{ [1, 2, 3]|reverse|list }}')
        assert tmpl.render() == 'raboof|[3, 2, 1]'

    def test_string(self):
        x = [1, 2, 3, 4, 5]
        tmpl = env.from_string('''{{ obj|string }}''')
        assert tmpl.render(obj=x) == unicode(x)

    def test_title(self):
        tmpl = env.from_string('''{{ "foo bar"|title }}''')
        assert tmpl.render() == "Foo Bar"
        tmpl = env.from_string('''{{ "foo's bar"|title }}''')
        assert tmpl.render() == "Foo's Bar"
        tmpl = env.from_string('''{{ "foo   bar"|title }}''')
        assert tmpl.render() == "Foo   Bar"
        tmpl = env.from_string('''{{ "f bar f"|title }}''')
        assert tmpl.render() == "F Bar F"
        tmpl = env.from_string('''{{ "foo-bar"|title }}''')
        assert tmpl.render() == "Foo-Bar"
        tmpl = env.from_string('''{{ "foo\tbar"|title }}''')
        assert tmpl.render() == "Foo\tBar"

    def test_truncate(self):
        tmpl = env.from_string(
            '{{ data|truncate(15, true, ">>>") }}|'
            '{{ data|truncate(15, false, ">>>") }}|'
            '{{ smalldata|truncate(15) }}'
        )
        out = tmpl.render(data='foobar baz bar' * 1000,
                          smalldata='foobar baz bar')
        assert out == 'foobar baz barf>>>|foobar baz >>>|foobar baz bar'

    def test_upper(self):
        tmpl = env.from_string('{{ "foo"|upper }}')
        assert tmpl.render() == 'FOO'

    def test_urlize(self):
        tmpl = env.from_string('{{ "foo http://www.example.com/ bar"|urlize }}')
        assert tmpl.render() == 'foo <a href="http://www.example.com/">'\
                                'http://www.example.com/</a> bar'

    def test_wordcount(self):
        tmpl = env.from_string('{{ "foo bar baz"|wordcount }}')
        assert tmpl.render() == '3'

    def test_block(self):
        tmpl = env.from_string('{% filter lower|escape %}<HEHE>{% endfilter %}')
        assert tmpl.render() == '&lt;hehe&gt;'

    def test_chaining(self):
        tmpl = env.from_string('''{{ ['<foo>', '<bar>']|first|upper|escape }}''')
        assert tmpl.render() == '&lt;FOO&gt;'

    def test_sum(self):
        tmpl = env.from_string('''{{ [1, 2, 3, 4, 5, 6]|sum }}''')
        assert tmpl.render() == '21'

    def test_sum_attributes(self):
        tmpl = env.from_string('''{{ values|sum('value') }}''')
        assert tmpl.render(values=[
            {'value': 23},
            {'value': 1},
            {'value': 18},
        ]) == '42'

    def test_sum_attributes_nested(self):
        tmpl = env.from_string('''{{ values|sum('real.value') }}''')
        assert tmpl.render(values=[
            {'real': {'value': 23}},
            {'real': {'value': 1}},
            {'real': {'value': 18}},
        ]) == '42'

    def test_abs(self):
        tmpl = env.from_string('''{{ -1|abs }}|{{ 1|abs }}''')
        assert tmpl.render() == '1|1', tmpl.render()

    def test_round_positive(self):
        tmpl = env.from_string('{{ 2.7|round }}|{{ 2.1|round }}|'
                               "{{ 2.1234|round(3, 'floor') }}|"
                               "{{ 2.1|round(0, 'ceil') }}")
        assert tmpl.render() == '3.0|2.0|2.123|3.0', tmpl.render()

    def test_round_negative(self):
        tmpl = env.from_string('{{ 21.3|round(-1)}}|'
                               "{{ 21.3|round(-1, 'ceil')}}|"
                               "{{ 21.3|round(-1, 'floor')}}")
        assert tmpl.render() == '20.0|30.0|20.0',tmpl.render()

    def test_xmlattr(self):
        tmpl = env.from_string("{{ {'foo': 42, 'bar': 23, 'fish': none, "
                               "'spam': missing, 'blub:blub': '<?>'}|xmlattr }}")
        out = tmpl.render().split()
        assert len(out) == 3
        assert 'foo="42"' in out
        assert 'bar="23"' in out
        assert 'blub:blub="&lt;?&gt;"' in out

    def test_sort1(self):
        tmpl = env.from_string('{{ [2, 3, 1]|sort }}|{{ [2, 3, 1]|sort(true) }}')
        assert tmpl.render() == '[1, 2, 3]|[3, 2, 1]'

    def test_sort2(self):
        tmpl = env.from_string('{{ "".join(["c", "A", "b", "D"]|sort) }}')
        assert tmpl.render() == 'AbcD'

    def test_sort3(self):
        tmpl = env.from_string('''{{ ['foo', 'Bar', 'blah']|sort }}''')
        assert tmpl.render() == "['Bar', 'blah', 'foo']"

    def test_sort4(self):
        class Magic(object):
            def __init__(self, value):
                self.value = value
            def __unicode__(self):
                return unicode(self.value)
        tmpl = env.from_string('''{{ items|sort(attribute='value')|join }}''')
        assert tmpl.render(items=map(Magic, [3, 2, 4, 1])) == '1234'

    def test_groupby(self):
        tmpl = env.from_string('''
        {%- for grouper, list in [{'foo': 1, 'bar': 2},
                                  {'foo': 2, 'bar': 3},
                                  {'foo': 1, 'bar': 1},
                                  {'foo': 3, 'bar': 4}]|groupby('foo') -%}
            {{ grouper }}{% for x in list %}: {{ x.foo }}, {{ x.bar }}{% endfor %}|
        {%- endfor %}''')
        assert tmpl.render().split('|') == [
            "1: 1, 2: 1, 1",
            "2: 2, 3",
            "3: 3, 4",
            ""
        ]

    def test_groupby_tuple_index(self):
        tmpl = env.from_string('''
        {%- for grouper, list in [('a', 1), ('a', 2), ('b', 1)]|groupby(0) -%}
            {{ grouper }}{% for x in list %}:{{ x.1 }}{% endfor %}|
        {%- endfor %}''')
        assert tmpl.render() == 'a:1:2|b:1|'

    def test_groupby_multidot(self):
        class Date(object):
            def __init__(self, day, month, year):
                self.day = day
                self.month = month
                self.year = year
        class Article(object):
            def __init__(self, title, *date):
                self.date = Date(*date)
                self.title = title
        articles = [
            Article('aha', 1, 1, 1970),
            Article('interesting', 2, 1, 1970),
            Article('really?', 3, 1, 1970),
            Article('totally not', 1, 1, 1971)
        ]
        tmpl = env.from_string('''
        {%- for year, list in articles|groupby('date.year') -%}
            {{ year }}{% for x in list %}[{{ x.title }}]{% endfor %}|
        {%- endfor %}''')
        assert tmpl.render(articles=articles).split('|') == [
            '1970[aha][interesting][really?]',
            '1971[totally not]',
            ''
        ]

    def test_filtertag(self):
        tmpl = env.from_string("{% filter upper|replace('FOO', 'foo') %}"
                               "foobar{% endfilter %}")
        assert tmpl.render() == 'fooBAR'

    def test_replace(self):
        env = Environment()
        tmpl = env.from_string('{{ string|replace("o", 42) }}')
        assert tmpl.render(string='<foo>') == '<f4242>'
        env = Environment(autoescape=True)
        tmpl = env.from_string('{{ string|replace("o", 42) }}')
        assert tmpl.render(string='<foo>') == '&lt;f4242&gt;'
        tmpl = env.from_string('{{ string|replace("<", 42) }}')
        assert tmpl.render(string='<foo>') == '42foo&gt;'
        tmpl = env.from_string('{{ string|replace("o", ">x<") }}')
        assert tmpl.render(string=Markup('foo')) == 'f&gt;x&lt;&gt;x&lt;'

    def test_forceescape(self):
        tmpl = env.from_string('{{ x|forceescape }}')
        assert tmpl.render(x=Markup('<div />')) == u'&lt;div /&gt;'

    def test_safe(self):
        env = Environment(autoescape=True)
        tmpl = env.from_string('{{ "<div>foo</div>"|safe }}')
        assert tmpl.render() == '<div>foo</div>'
        tmpl = env.from_string('{{ "<div>foo</div>" }}')
        assert tmpl.render() == '&lt;div&gt;foo&lt;/div&gt;'

    def test_urlencode(self):
        env = Environment(autoescape=True)
        tmpl = env.from_string('{{ "Hello, world!"|urlencode }}')
        assert tmpl.render() == 'Hello%2C%20world%21'
        tmpl = env.from_string('{{ o|urlencode }}')
        assert tmpl.render(o=u"Hello, world\u203d") == "Hello%2C%20world%E2%80%BD"
        assert tmpl.render(o=(("f", 1),)) == "f=1"
        assert tmpl.render(o=(('f', 1), ("z", 2))) == "f=1&amp;z=2"
        assert tmpl.render(o=((u"\u203d", 1),)) == "%E2%80%BD=1"
        assert tmpl.render(o={u"\u203d": 1}) == "%E2%80%BD=1"
        assert tmpl.render(o={0: 1}) == "0=1"


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FilterTestCase))
    return suite
