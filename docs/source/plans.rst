=====
Plans
=====

A variety of plans for your use.  These are built out of Bluesky 
plans_ and `plan stubs`_, and customized to be very specific to operations 
at SSRLDeNovX.  If you'd like to build
your own plans out of the existing plans, see this `tutorial section`_.

.. _plans:  https://blueskyproject.io/bluesky/plans.html#pre-assembled-plans
.. _plan stubs: https://blueskyproject.io/bluesky/plans.html#stub-plans
.. _tutorial section: https://nsls-ii.github.io/bluesky/tutorial.html#plans-in-series

Helper functions, plans
=======================

``show_table()``:
-----------------
Shows data from the last run in tablular from.  Providing an index (i) 
will return data from the 'i-th' run.  This is simply a wrapper for ``db[i].table()``

.. code:: ipython

    In [33]: RE(bp.count([det]))


    Transient Scan ID: 9     Time: 2020-09-03 14:32:59
    Persistent Unique Scan ID: 'a70862bd-782f-44f6-98aa-bf5328854a67'
    New stream: 'primary'
    +-----------+------------+------------+
    |   seq_num |       time |        det |
    +-----------+------------+------------+
    |         1 | 14:32:59.1 |      0.989 |
    +-----------+------------+------------+
    generator count ['a70862bd'] (scan num: 9)