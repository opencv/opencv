EMD-L1
======
Computes the "minimal work" distance between two weighted point configurations base on the papers "EMD-L1: An efficient and Robust Algorithm
for comparing histogram-based descriptors", by Haibin Ling and Kazunori Okuda; and "The Earth Mover's Distance is the Mallows Distance:
Some Insights from Statistics", by Elizaveta Levina and Peter Bickel.

.. ocv:function:: float EMDL1( InputArray signature1, InputArray signature2 )

    :param signature1: First signature, a single column floating-point matrix. Each row is the value of the histogram in each bin.

    :param signature2: Second signature of the same format and size as  ``signature1``.
