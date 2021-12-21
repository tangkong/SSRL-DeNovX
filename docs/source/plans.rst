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


``find_coords(dets,motor1,motor2,guess,delt=1,num=50)``:
--------------------------------------------------------
Function for calibrating the position of a DeNovX cassette based on a pinhole. 

Cassette Plans
===============

``cassette_scan()``:
--------------------
Scans 48 samples in a DeNovX cassette along with two calibrant materials.

``opt_cassette_scan()``:
--------------------
Scans 48 samples in a DeNovX cassette along with two calibrant materials using adaptive optimization to set the
attenuating filters and detector exposure time.


Adaptive Optimization
=====================
Plans for measuring, calculating, and setting various acquisition parameters like attenuation, exposure time, and
sample surface area coverage.

``max_pixel_count(det, sat_count)``:
------------------------------------
Takes a detector ``det`` and determines a new acquisition time based on the desired saturation count ``sat_count``.

``stub_filter_opt_count(det, motor, ranges, target_count=1000, det_key='dexela_image' ,md={})``:
-----------------------------------
Stub plan for finding the optimal combination of attenuators for the current sample in the path of the beam.
