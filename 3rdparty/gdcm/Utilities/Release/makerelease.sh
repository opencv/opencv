#!/bin/sh

# http://www.winehq.org/docs/winedev-guide/dbg-control
WINEDEBUG=-all wineconsole --backend=curses cmd /c release.bat
