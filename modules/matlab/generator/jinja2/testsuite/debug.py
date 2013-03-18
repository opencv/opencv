# -*- coding: utf-8 -*-
"""
    jinja2.testsuite.debug
    ~~~~~~~~~~~~~~~~~~~~~~

    Tests the debug system.

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import sys
import unittest

from jinja2.testsuite import JinjaTestCase, filesystem_loader

from jinja2 import Environment, TemplateSyntaxError

env = Environment(loader=filesystem_loader)


class DebugTestCase(JinjaTestCase):

    if sys.version_info[:2] != (2, 4):
        def test_runtime_error(self):
            def test():
                tmpl.render(fail=lambda: 1 / 0)
            tmpl = env.get_template('broken.html')
            self.assert_traceback_matches(test, r'''
  File ".*?broken.html", line 2, in (top-level template code|<module>)
    \{\{ fail\(\) \}\}
  File ".*?debug.pyc?", line \d+, in <lambda>
    tmpl\.render\(fail=lambda: 1 / 0\)
ZeroDivisionError: (int(eger)? )?division (or modulo )?by zero
''')

    def test_syntax_error(self):
        # XXX: the .*? is necessary for python3 which does not hide
        # some of the stack frames we don't want to show.  Not sure
        # what's up with that, but that is not that critical.  Should
        # be fixed though.
        self.assert_traceback_matches(lambda: env.get_template('syntaxerror.html'), r'''(?sm)
  File ".*?syntaxerror.html", line 4, in (template|<module>)
    \{% endif %\}.*?
(jinja2\.exceptions\.)?TemplateSyntaxError: Encountered unknown tag 'endif'. Jinja was looking for the following tags: 'endfor' or 'else'. The innermost block that needs to be closed is 'for'.
    ''')

    def test_regular_syntax_error(self):
        def test():
            raise TemplateSyntaxError('wtf', 42)
        self.assert_traceback_matches(test, r'''
  File ".*debug.pyc?", line \d+, in test
    raise TemplateSyntaxError\('wtf', 42\)
(jinja2\.exceptions\.)?TemplateSyntaxError: wtf
  line 42''')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(DebugTestCase))
    return suite
