===================================
Beamline Startup, EPICS and Bluesky
===================================

Currently the beamline devices are all controlled via EPICS_, and the data 
collection orchestrated via the Bluesky_ Ecosystem.  Each of these systems must 
be started to render the beamline operational.  

.. _EPICS: https://epics.anl.gov/
.. _Bluesky: https://blueskyproject.io/

.. Note:: This system is still under development at SSRL.  Many tasks may seem 
    tedious, but there is significant room for streamlining the process.  Please
    view this list of startup items in that lens, and be patient.  

To start up EPICS, the IOCs (Input-Output Controllers) must be started.  Each 
set of devices has its own IOC