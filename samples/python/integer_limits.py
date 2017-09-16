
# Copyright 2017 Muzi Li marlonli@bu.edu
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:35:04 2017

@author: ec602
"""
Table="{:<6} {:<22} {:<22} {:<22}"
print(Table.format('Bytes','Largest Unsigned Int','Minimum Signed Int','Maximum Signed Int'))
print(Table.format('1', 2**8-1, -2**7, 2**7-1))
print(Table.format('2', 2**16-1, -2**15, 2**15-1))
print(Table.format('3', 2**24-1, -2**23, 2**23-1))
print(Table.format('4', 2**32-1, -2**31, 2**31-1))
print(Table.format('5', 2**40-1, -2**39, 2**39-1))
print(Table.format('6', 2**48-1, -2**47, 2**47-1))
print(Table.format('7', 2**56-1, -2**55, 2**55-1))
print(Table.format('8', 2**64-1, -2**63, 2**63-1))