#!/usr/bin/env python

"""
    This script can generate XLS reports from OpenCV tests' XML output files.

    To use it, first, create a directory for each machine you ran tests on.
    Each such directory will become a sheet in the report. Put each XML file
    into the corresponding directory.

    Then, create your configuration file(s). You can have a global configuration
    file (specified with the -c option), and per-sheet configuration files, which
    must be called sheet.conf and placed in the directory corresponding to the sheet.
    The settings in the per-sheet configuration file will override those in the
    global configuration file, if both are present.

    A configuration file must consist of a Python dictionary. The following keys
    will be recognized:

    * 'comparisons': [{'from': string, 'to': string}]
        List of configurations to compare performance between. For each item,
        the sheet will have a column showing speedup from configuration named
        'from' to configuration named "to".

    * 'configuration_matchers': [{'properties': {string: object}, 'name': string}]
        Instructions for matching test run property sets to configuration names.

        For each found XML file:

        1) All attributes of the root element starting with the prefix 'cv_' are
           placed in a dictionary, with the cv_ prefix stripped and the cv_module_name
           element deleted.

        2) The first matcher for which the XML's file property set contains the same
           keys with equal values as its 'properties' dictionary is searched for.
           A missing property can be matched by using None as the value.

           Corollary 1: you should place more specific matchers before less specific
           ones.

           Corollary 2: an empty 'properties' dictionary matches every property set.

        3) If a matching matcher is found, its 'name' string is presumed to be the name
           of the configuration the XML file corresponds to. A warning is printed if
           two different property sets match to the same configuration name.

        4) If a such a matcher isn't found, if --include-unmatched was specified, the
           configuration name is assumed to be the relative path from the sheet's
           directory to the XML file's containing directory. If the XML file isinstance
           directly inside the sheet's directory, the configuration name is instead
           a dump of all its properties. If --include-unmatched wasn't specified,
           the XML file is ignored and a warning is printed.

    * 'configurations': [string]
        List of names for compile-time and runtime configurations of OpenCV.
        Each item will correspond to a column of the sheet.

    * 'module_colors': {string: string}
        Mapping from module name to color name. In the sheet, cells containing module
        names from this mapping will be colored with the corresponding color. You can
        find the list of available colors here:
        <http://www.simplistix.co.uk/presentations/python-excel.pdf>.

    * 'sheet_name': string
        Name for the sheet. If this parameter is missing, the name of sheet's directory
        will be used.

    * 'sheet_properties': [(string, string)]
        List of arbitrary (key, value) pairs that somehow describe the sheet. Will be
        dumped into the first row of the sheet in string form.

    Note that all keys are optional, although to get useful results, you'll want to
    specify at least 'configurations' and 'configuration_matchers'.

    Finally, run the script. Use the --help option for usage information.
"""

from __future__ import division

import ast
import errno
import fnmatch
import logging
import numbers
import os, os.path
import re

from argparse import ArgumentParser
from glob import glob
from itertools import ifilter

import xlwt

from testlog_parser import parseLogFile

re_image_size = re.compile(r'^ \d+ x \d+$', re.VERBOSE)
re_data_type = re.compile(r'^ (?: 8 | 16 | 32 | 64 ) [USF] C [1234] $', re.VERBOSE)

time_style = xlwt.easyxf(num_format_str='#0.00')
no_time_style = xlwt.easyxf('pattern: pattern solid, fore_color gray25')
failed_style = xlwt.easyxf('pattern: pattern solid, fore_color red')
noimpl_style = xlwt.easyxf('pattern: pattern solid, fore_color orange')
style_dict = {"failed": failed_style, "noimpl":noimpl_style}

speedup_style = time_style
good_speedup_style = xlwt.easyxf('font: color green', num_format_str='#0.00')
bad_speedup_style = xlwt.easyxf('font: color red', num_format_str='#0.00')
no_speedup_style = no_time_style
error_speedup_style = xlwt.easyxf('pattern: pattern solid, fore_color orange')
header_style = xlwt.easyxf('font: bold true; alignment: horizontal centre, vertical top, wrap True')
subheader_style = xlwt.easyxf('alignment: horizontal centre, vertical top')

class Collector(object):
    def __init__(self, config_match_func, include_unmatched):
        self.__config_cache = {}
        self.config_match_func = config_match_func
        self.include_unmatched = include_unmatched
        self.tests = {}
        self.extra_configurations = set()

    # Format a sorted sequence of pairs as if it was a dictionary.
    # We can't just use a dictionary instead, since we want to preserve the sorted order of the keys.
    @staticmethod
    def __format_config_cache_key(pairs, multiline=False):
        return (
          ('{\n' if multiline else '{') +
          (',\n' if multiline else ', ').join(
             ('  ' if multiline else '') + repr(k) + ': ' + repr(v) for (k, v) in pairs) +
          ('\n}\n' if multiline else '}')
        )

    def collect_from(self, xml_path, default_configuration):
        run = parseLogFile(xml_path)

        module = run.properties['module_name']

        properties = run.properties.copy()
        del properties['module_name']

        props_key = tuple(sorted(properties.iteritems())) # dicts can't be keys

        if props_key in self.__config_cache:
            configuration = self.__config_cache[props_key]
        else:
            configuration = self.config_match_func(properties)

            if configuration is None:
                if self.include_unmatched:
                    if default_configuration is not None:
                        configuration = default_configuration
                    else:
                        configuration = Collector.__format_config_cache_key(props_key, multiline=True)

                    self.extra_configurations.add(configuration)
                else:
                    logging.warning('failed to match properties to a configuration: %s',
                        Collector.__format_config_cache_key(props_key))

            else:
                same_config_props = [it[0] for it in self.__config_cache.iteritems() if it[1] == configuration]
                if len(same_config_props) > 0:
                    logging.warning('property set %s matches the same configuration %r as property set %s',
                        Collector.__format_config_cache_key(props_key),
                        configuration,
                        Collector.__format_config_cache_key(same_config_props[0]))

            self.__config_cache[props_key] = configuration

        if configuration is None: return

        module_tests = self.tests.setdefault(module, {})

        for test in run.tests:
            test_results = module_tests.setdefault((test.shortName(), test.param()), {})
            new_result = test.get("gmean") if test.status == 'run' else test.status
            test_results[configuration] = min(
              test_results.get(configuration), new_result,
              key=lambda r: (1, r) if isinstance(r, numbers.Number) else
                            (2,) if r is not None else
                            (3,)
            ) # prefer lower result; prefer numbers to errors and errors to nothing

def make_match_func(matchers):
    def match_func(properties):
        for matcher in matchers:
            if all(properties.get(name) == value
                   for (name, value) in matcher['properties'].iteritems()):
                return matcher['name']

        return None

    return match_func

def main():
    arg_parser = ArgumentParser(description='Build an XLS performance report.')
    arg_parser.add_argument('sheet_dirs', nargs='+', metavar='DIR', help='directory containing perf test logs')
    arg_parser.add_argument('-o', '--output', metavar='XLS', default='report.xls', help='name of output file')
    arg_parser.add_argument('-c', '--config', metavar='CONF', help='global configuration file')
    arg_parser.add_argument('--include-unmatched', action='store_true',
        help='include results from XML files that were not recognized by configuration matchers')
    arg_parser.add_argument('--show-times-per-pixel', action='store_true',
        help='for tests that have an image size parameter, show per-pixel time, as well as total time')

    args = arg_parser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if args.config is not None:
        with open(args.config) as global_conf_file:
            global_conf = ast.literal_eval(global_conf_file.read())
    else:
        global_conf = {}

    wb = xlwt.Workbook()

    for sheet_path in args.sheet_dirs:
        try:
            with open(os.path.join(sheet_path, 'sheet.conf')) as sheet_conf_file:
                sheet_conf = ast.literal_eval(sheet_conf_file.read())
        except IOError as ioe:
            if ioe.errno != errno.ENOENT: raise
            sheet_conf = {}
            logging.debug('no sheet.conf for %s', sheet_path)

        sheet_conf = dict(global_conf.items() + sheet_conf.items())

        config_names = sheet_conf.get('configurations', [])
        config_matchers = sheet_conf.get('configuration_matchers', [])

        collector = Collector(make_match_func(config_matchers), args.include_unmatched)

        for root, _, filenames in os.walk(sheet_path):
            logging.info('looking in %s', root)
            for filename in fnmatch.filter(filenames, '*.xml'):
                if os.path.normpath(sheet_path) == os.path.normpath(root):
                  default_conf = None
                else:
                  default_conf = os.path.relpath(root, sheet_path)
                collector.collect_from(os.path.join(root, filename), default_conf)

        config_names.extend(sorted(collector.extra_configurations - set(config_names)))

        sheet = wb.add_sheet(sheet_conf.get('sheet_name', os.path.basename(os.path.abspath(sheet_path))))

        sheet_properties = sheet_conf.get('sheet_properties', [])

        sheet.write(0, 0, 'Properties:')

        sheet.write(0, 1,
          'N/A' if len(sheet_properties) == 0 else
          ' '.join(str(k) + '=' + repr(v) for (k, v) in sheet_properties))

        sheet.row(2).height = 800
        sheet.panes_frozen = True
        sheet.remove_splits = True

        sheet_comparisons = sheet_conf.get('comparisons', [])

        row = 2

        col = 0

        for (w, caption) in [
                (2500, 'Module'),
                (10000, 'Test'),
                (2000, 'Image\nwidth'),
                (2000, 'Image\nheight'),
                (2000, 'Data\ntype'),
                (7500, 'Other parameters')]:
            sheet.col(col).width = w
            if args.show_times_per_pixel:
                sheet.write_merge(row, row + 1, col, col, caption, header_style)
            else:
                sheet.write(row, col, caption, header_style)
            col += 1

        for config_name in config_names:
            if args.show_times_per_pixel:
                sheet.col(col).width = 3000
                sheet.col(col + 1).width = 3000
                sheet.write_merge(row, row, col, col + 1, config_name, header_style)
                sheet.write(row + 1, col, 'total, ms', subheader_style)
                sheet.write(row + 1, col + 1, 'per pixel, ns', subheader_style)
                col += 2
            else:
                sheet.col(col).width = 4000
                sheet.write(row, col, config_name, header_style)
                col += 1

        col += 1 # blank column between configurations and comparisons

        for comp in sheet_comparisons:
            sheet.col(col).width = 4000
            caption = comp['to'] + '\nvs\n' + comp['from']
            if args.show_times_per_pixel:
                sheet.write_merge(row, row + 1, col, col, caption, header_style)
            else:
                sheet.write(row, col, caption, header_style)
            col += 1

        row += 2 if args.show_times_per_pixel else 1

        sheet.horz_split_pos = row
        sheet.horz_split_first_visible = row

        module_colors = sheet_conf.get('module_colors', {})
        module_styles = {module: xlwt.easyxf('pattern: pattern solid, fore_color {}'.format(color))
                         for module, color in module_colors.iteritems()}

        for module, tests in sorted(collector.tests.iteritems()):
            for ((test, param), configs) in sorted(tests.iteritems()):
                sheet.write(row, 0, module, module_styles.get(module, xlwt.Style.default_style))
                sheet.write(row, 1, test)

                param_list = param[1:-1].split(', ') if param.startswith('(') and param.endswith(')') else [param]

                image_size = next(ifilter(re_image_size.match, param_list), None)
                if image_size is not None:
                    (image_width, image_height) = map(int, image_size.split('x', 1))
                    sheet.write(row, 2, image_width)
                    sheet.write(row, 3, image_height)
                    del param_list[param_list.index(image_size)]

                data_type = next(ifilter(re_data_type.match, param_list), None)
                if data_type is not None:
                    sheet.write(row, 4, data_type)
                    del param_list[param_list.index(data_type)]

                sheet.row(row).write(5, ' | '.join(param_list))

                col = 6

                for c in config_names:
                    if c in configs:
                        sheet.write(row, col, configs[c], style_dict.get(configs[c], time_style))
                    else:
                        sheet.write(row, col, None, no_time_style)
                    col += 1
                    if args.show_times_per_pixel:
                        sheet.write(row, col,
                          xlwt.Formula('{0} * 1000000 / ({1} * {2})'.format(
                              xlwt.Utils.rowcol_to_cell(row, col - 1),
                              xlwt.Utils.rowcol_to_cell(row, 2),
                              xlwt.Utils.rowcol_to_cell(row, 3)
                          )),
                          time_style
                        )
                        col += 1

                col += 1 # blank column

                for comp in sheet_comparisons:
                    cmp_from = configs.get(comp["from"])
                    cmp_to = configs.get(comp["to"])

                    if isinstance(cmp_from, numbers.Number) and isinstance(cmp_to, numbers.Number):
                        try:
                            speedup = cmp_from / cmp_to
                            sheet.write(row, col, speedup, good_speedup_style if speedup > 1.1 else
                                                           bad_speedup_style  if speedup < 0.9 else
                                                           speedup_style)
                        except ArithmeticError as e:
                            sheet.write(row, col, None, error_speedup_style)
                    else:
                        sheet.write(row, col, None, no_speedup_style)

                    col += 1

                row += 1
                if row % 1000 == 0: sheet.flush_row_data()

    wb.save(args.output)

if __name__ == '__main__':
    main()
