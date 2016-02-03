# Copyright 2008 Emanuele Rocca <ema@galliera.it>
# Copyright 2008 Marco De Benedetto <debe@galliera.it>
# Copyright (c) 2006-2011 Mathieu Malaterre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import commands

class CmdException(Exception):
    """Exception representing a command line error"""
    pass

def trycmd(cmd):
    """Try to execute the given command, raising an Exception on errors"""
    (exitstatus, outtext) = commands.getstatusoutput(cmd)
    if exitstatus != 0:
        raise CmdException, "cmd: %s\noutput: %s" % (cmd, outtext)
