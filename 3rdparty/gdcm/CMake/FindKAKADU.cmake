#
# this module looks for KAKADu
# http://www.kakadusoftware.com/
#
#
#  Copyright (c) 2006-2011 Mathieu Malaterre <mathieu.malaterre@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

find_program(KDU_EXPAND_EXECUTABLE
  kdu_expand
  )

mark_as_advanced(
  KDU_EXPAND_EXECUTABLE
  )
