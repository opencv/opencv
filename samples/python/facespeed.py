# Copyright 2017 Muzi Li marlonli@bu.edu
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 10:57:07 2017

@author: ec602
"""

ME=5.972*10**27
aim=6.022*10**23
EI=32.1/100*ME/55.84*26*aim
EO=30.1/100*ME/15.99*8*aim
ES=15.1/100*ME/28.08*aim*14
EM=13.9/100*ME/24.3*aim*12
ESU=2.9/100*ME/32.06*aim*16
EN=1.8/100*ME/58.69*aim*28
EC=1.5/100*ME/40.07*aim*20
EA=1.4/100*ME/26.998*aim*13
EOther=1.2/100*ME/75*aim*60

Ecount=EI+EO+ES+EM+ESU+EN+EC+EA+EOther;
TB=Ecount/(10**13)
lb=1.0*10**38
hb=9.0*10**38

print(TB)
print(lb)
print(hb)



