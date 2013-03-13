# -*- coding: utf-8 -*-
"""
    jinja2.testsuite.loader
    ~~~~~~~~~~~~~~~~~~~~~~~

    Test the loaders.

    :copyright: (c) 2010 by the Jinja Team.
    :license: BSD, see LICENSE for more details.
"""
import os
import sys
import tempfile
import shutil
import unittest

from jinja2.testsuite import JinjaTestCase, dict_loader, \
     package_loader, filesystem_loader, function_loader, \
     choice_loader, prefix_loader

from jinja2 import Environment, loaders
from jinja2.loaders import split_template_path
from jinja2.exceptions import TemplateNotFound


class LoaderTestCase(JinjaTestCase):

    def test_dict_loader(self):
        env = Environment(loader=dict_loader)
        tmpl = env.get_template('justdict.html')
        assert tmpl.render().strip() == 'FOO'
        self.assert_raises(TemplateNotFound, env.get_template, 'missing.html')

    def test_package_loader(self):
        env = Environment(loader=package_loader)
        tmpl = env.get_template('test.html')
        assert tmpl.render().strip() == 'BAR'
        self.assert_raises(TemplateNotFound, env.get_template, 'missing.html')

    def test_filesystem_loader(self):
        env = Environment(loader=filesystem_loader)
        tmpl = env.get_template('test.html')
        assert tmpl.render().strip() == 'BAR'
        tmpl = env.get_template('foo/test.html')
        assert tmpl.render().strip() == 'FOO'
        self.assert_raises(TemplateNotFound, env.get_template, 'missing.html')

    def test_choice_loader(self):
        env = Environment(loader=choice_loader)
        tmpl = env.get_template('justdict.html')
        assert tmpl.render().strip() == 'FOO'
        tmpl = env.get_template('test.html')
        assert tmpl.render().strip() == 'BAR'
        self.assert_raises(TemplateNotFound, env.get_template, 'missing.html')

    def test_function_loader(self):
        env = Environment(loader=function_loader)
        tmpl = env.get_template('justfunction.html')
        assert tmpl.render().strip() == 'FOO'
        self.assert_raises(TemplateNotFound, env.get_template, 'missing.html')

    def test_prefix_loader(self):
        env = Environment(loader=prefix_loader)
        tmpl = env.get_template('a/test.html')
        assert tmpl.render().strip() == 'BAR'
        tmpl = env.get_template('b/justdict.html')
        assert tmpl.render().strip() == 'FOO'
        self.assert_raises(TemplateNotFound, env.get_template, 'missing')

    def test_caching(self):
        changed = False
        class TestLoader(loaders.BaseLoader):
            def get_source(self, environment, template):
                return u'foo', None, lambda: not changed
        env = Environment(loader=TestLoader(), cache_size=-1)
        tmpl = env.get_template('template')
        assert tmpl is env.get_template('template')
        changed = True
        assert tmpl is not env.get_template('template')
        changed = False

        env = Environment(loader=TestLoader(), cache_size=0)
        assert env.get_template('template') \
               is not env.get_template('template')

        env = Environment(loader=TestLoader(), cache_size=2)
        t1 = env.get_template('one')
        t2 = env.get_template('two')
        assert t2 is env.get_template('two')
        assert t1 is env.get_template('one')
        t3 = env.get_template('three')
        assert 'one' in env.cache
        assert 'two' not in env.cache
        assert 'three' in env.cache

    def test_split_template_path(self):
        assert split_template_path('foo/bar') == ['foo', 'bar']
        assert split_template_path('./foo/bar') == ['foo', 'bar']
        self.assert_raises(TemplateNotFound, split_template_path, '../foo')


class ModuleLoaderTestCase(JinjaTestCase):
    archive = None

    def compile_down(self, zip='deflated', py_compile=False):
        super(ModuleLoaderTestCase, self).setup()
        log = []
        self.reg_env = Environment(loader=prefix_loader)
        if zip is not None:
            self.archive = tempfile.mkstemp(suffix='.zip')[1]
        else:
            self.archive = tempfile.mkdtemp()
        self.reg_env.compile_templates(self.archive, zip=zip,
                                       log_function=log.append,
                                       py_compile=py_compile)
        self.mod_env = Environment(loader=loaders.ModuleLoader(self.archive))
        return ''.join(log)

    def teardown(self):
        super(ModuleLoaderTestCase, self).teardown()
        if hasattr(self, 'mod_env'):
            if os.path.isfile(self.archive):
                os.remove(self.archive)
            else:
                shutil.rmtree(self.archive)
            self.archive = None

    def test_log(self):
        log = self.compile_down()
        assert 'Compiled "a/foo/test.html" as ' \
               'tmpl_a790caf9d669e39ea4d280d597ec891c4ef0404a' in log
        assert 'Finished compiling templates' in log
        assert 'Could not compile "a/syntaxerror.html": ' \
               'Encountered unknown tag \'endif\'' in log

    def _test_common(self):
        tmpl1 = self.reg_env.get_template('a/test.html')
        tmpl2 = self.mod_env.get_template('a/test.html')
        assert tmpl1.render() == tmpl2.render()

        tmpl1 = self.reg_env.get_template('b/justdict.html')
        tmpl2 = self.mod_env.get_template('b/justdict.html')
        assert tmpl1.render() == tmpl2.render()

    def test_deflated_zip_compile(self):
        self.compile_down(zip='deflated')
        self._test_common()

    def test_stored_zip_compile(self):
        self.compile_down(zip='stored')
        self._test_common()

    def test_filesystem_compile(self):
        self.compile_down(zip=None)
        self._test_common()

    def test_weak_references(self):
        self.compile_down()
        tmpl = self.mod_env.get_template('a/test.html')
        key = loaders.ModuleLoader.get_template_key('a/test.html')
        name = self.mod_env.loader.module.__name__

        assert hasattr(self.mod_env.loader.module, key)
        assert name in sys.modules

        # unset all, ensure the module is gone from sys.modules
        self.mod_env = tmpl = None

        try:
            import gc
            gc.collect()
        except:
            pass

        assert name not in sys.modules

    def test_byte_compilation(self):
        log = self.compile_down(py_compile=True)
        assert 'Byte-compiled "a/test.html"' in log
        tmpl1 = self.mod_env.get_template('a/test.html')
        mod = self.mod_env.loader.module. \
            tmpl_3c4ddf650c1a73df961a6d3d2ce2752f1b8fd490
        assert mod.__file__.endswith('.pyc')

    def test_choice_loader(self):
        log = self.compile_down(py_compile=True)
        assert 'Byte-compiled "a/test.html"' in log

        self.mod_env.loader = loaders.ChoiceLoader([
            self.mod_env.loader,
            loaders.DictLoader({'DICT_SOURCE': 'DICT_TEMPLATE'})
        ])

        tmpl1 = self.mod_env.get_template('a/test.html')
        self.assert_equal(tmpl1.render(), 'BAR')
        tmpl2 = self.mod_env.get_template('DICT_SOURCE')
        self.assert_equal(tmpl2.render(), 'DICT_TEMPLATE')

    def test_prefix_loader(self):
        log = self.compile_down(py_compile=True)
        assert 'Byte-compiled "a/test.html"' in log

        self.mod_env.loader = loaders.PrefixLoader({
            'MOD':      self.mod_env.loader,
            'DICT':     loaders.DictLoader({'test.html': 'DICT_TEMPLATE'})
        })

        tmpl1 = self.mod_env.get_template('MOD/a/test.html')
        self.assert_equal(tmpl1.render(), 'BAR')
        tmpl2 = self.mod_env.get_template('DICT/test.html')
        self.assert_equal(tmpl2.render(), 'DICT_TEMPLATE')


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(LoaderTestCase))
    suite.addTest(unittest.makeSuite(ModuleLoaderTestCase))
    return suite
