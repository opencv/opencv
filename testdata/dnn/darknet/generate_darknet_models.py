# export PYTHONPATH=/path/to/darknet/python/:$PYTHONPATH
# export LD_LIBRARY_PATH=/path/to/darknet/:$LD_LIBRARY_PATH
import darknet as dn
import numpy as np

np.random.seed(324)

def genTestData(name, inpShape, outShape):
    net = dn.load_net((name + '.cfg').encode(), b'', 0)

    inp = np.random.standard_normal(inpShape).astype(np.float32)

    inpData = dn.c_array(dn.c_float, inp.flatten())
    pred = dn.predict(net, inpData)

    total = np.prod(outShape)
    out = np.zeros([total], np.float32)
    for i in range(total):
        out[i] = pred[i]
    out = out.reshape(outShape)

    np.save(name + '_in.npy', inp)
    np.save(name + '_out.npy', out)

genTestData('shortcut', [1, 2, 3, 4], [1, 2, 3, 4])
genTestData('upsample', [2, 3, 4, 5], [2, 3, 8, 10])
genTestData('avgpool_softmax',  [2, 10, 4, 5], [2, 10, 1, 1])
genTestData('shortcut_leaky', [1, 2, 3, 4], [1, 2, 3, 4])
genTestData('shortcut_unequal', [1, 2, 3, 5], [1, 4, 3, 5])
genTestData('shortcut_unequal_2', [1, 2, 3, 5], [1, 2, 3, 5])
genTestData('convolutional', [1, 3, 4, 6], [1, 5, 2, 3])
genTestData('connected', [1, 3, 4, 6], [1, 2])
genTestData('maxpool', [1, 3, 3, 6], [1, 3, 2, 3])
genTestData('scale_channels', [1, 3, 3, 3], [1, 3, 3, 3])
genTestData('mish', [1, 3, 4, 6], [1, 4, 4, 6])
genTestData('route', [1, 4, 3, 6], [1, 2, 2, 3])
genTestData('route_multi', [1, 6, 3, 6], [1, 4, 2, 3])
