from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, BlankLine


# whyever this is necessary..

class FixXrange2(fixer_base.BaseFix):
    PATTERN = "'xrange'"

    def transform(self, node, results):
        node.replace(Name('range', prefix=node.prefix))
