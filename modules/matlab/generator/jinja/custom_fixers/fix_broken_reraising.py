from lib2to3 import fixer_base, pytree
from lib2to3.fixer_util import Name, BlankLine, Name, Attr, ArgList


class FixBrokenReraising(fixer_base.BaseFix):
    PATTERN = """
    raise_stmt< 'raise' any ',' val=any ',' tb=any >
    """

    # run before the broken 2to3 checker with the same goal
    # tries to rewrite it with a rule that does not work out for jinja
    run_order = 1

    def transform(self, node, results):
        tb = results['tb'].clone()
        tb.prefix = ''
        with_tb = Attr(results['val'].clone(), Name('with_traceback')) + \
                  [ArgList([tb])]
        new = pytree.Node(self.syms.simple_stmt, [Name("raise")] + with_tb)
        new.prefix = node.prefix
        return new
