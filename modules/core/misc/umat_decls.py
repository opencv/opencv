UMAT_DECLS = [
    ['class cv.UMat', '', [], [['size_t', 'offset', '', ['/RW']]], None, ''],
    ['cv.UMat.UMat', '', [], [['UMatUsageFlags', 'usageFlags', 'USAGE_DEFAULT', []]], None, ''],
    ['cv.UMat.UMat', '', [], [['int', 'rows', '', []], ['int', 'cols', '', []], ['int', 'type', '', []], ['UMatUsageFlags', 'usageFlags', 'USAGE_DEFAULT', []]], None, ''],
    ['cv.UMat.UMat', '', [], [['Size', 'size', '', []], ['int', 'type', '', []], ['UMatUsageFlags', 'usageFlags', 'USAGE_DEFAULT', []]], None, ''],
    ['cv.UMat.UMat', '', [], [['int', 'rows', '', []], ['int', 'cols', '', []], ['int', 'type', '', []], ['Scalar', 's', '', ['/C', '/Ref']], ['UMatUsageFlags', 'usageFlags', 'USAGE_DEFAULT', []]], None, ''],
    ['cv.UMat.UMat', '', [], [['Size', 'size', '', []], ['int', 'type', '', []], ['Scalar', 's', '', ['/C', '/Ref']], ['UMatUsageFlags', 'usageFlags', 'USAGE_DEFAULT', []]], None, ''],
    ['UMat.UMat', None, ['/mappable=Ptr<Mat>'], [], None, None],
    ['cv.UMat.queue', 'void*', ['/phantom', '/S'], [], 'void*', ''],
    ['cv.UMat.context', 'void*', ['/phantom', '/S'], [], 'void*', ''],
    ['cv.UMat.UMat', '', [], [['UMat', 'm', '', ['/C', '/Ref']]], None, ''],
    ['cv.UMat.UMat', '', [], [['UMat', 'm', '', ['/C', '/Ref']], ['Range', 'rowRange', '', ['/C', '/Ref']], ['Range', 'colRange', 'Range::all()', ['/C', '/Ref']]], None, ''],
    ['cv.UMat.UMat', '', [], [['UMat', 'm', '', ['/C', '/Ref']], ['Rect', 'roi', '', ['/C', '/Ref']]], None, ''],
    ['cv.UMat.UMat', '', [], [['UMat', 'm', '', ['/C', '/Ref']], ['vector_Range', 'ranges', '', ['/C', '/Ref']]], None, ''],
    ['cv.UMat.get', 'Mat', ['/phantom', '/C'], [], 'Mat', ''],
    ['cv.UMat.isContinuous', 'bool', ['/C'], [], 'bool', ''],
    ['cv.UMat.isSubmatrix', 'bool', ['/C'], [], 'bool', ''],
    ['cv.UMat.handle', 'void*', ['/C'], [['AccessFlag', 'accessFlags', '', []]], 'void*', '']
]
