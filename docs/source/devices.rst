=======
Devices
=======

A variety of devices are available for use at SSRL-DeNovX.  Currently only 
High Throughput (HiTp) diffraction is supported, but more functionality will
be added in the future.  

While most PV's are accessible from the ipython session, it's generally 
advised to use the GUI elements to adjust parameters like exposure time or 
frame rates.

A fleshed-out profile has been built for beamline 10-2a at Stanford Synchrotron Radiation Laboratory (SSRL)
with a specific set of devices and instruments.

As of December 2021 the 10-2a profile includes the following devices:
- An xy motor stage with sample mount
- A z motor for detector motion
- A Dexela 2923 detector
- A photodiode beamstop
- A filter box with 4 interchangeable slots for filters of varying thickness
- The shutter to the hutch

Dexela 2923
===========
The Dexela detector is currently defined as the variable ``dexDet``. The ``dexDet`` variable is passed to various
Bluesky commands and wrappers for data acquisition.

Some useful functions:

* ``dexDet.cam.acquire_time.get()`` -- Print the current detector exposure time.
* ``dexDet.cam.acquire_time.set(t)`` -- Set the detector exposure time to ``t`` seconds.

Cassette Motor Stage
===============
As of December 2021 the profile is configured for a 2-axis motion stage that moves in a plane perpendicular to the
beam direction (the x and y directions, in the laboratory coordinate system). The stage is defined as the variable
``c_stage`` and this variable is passed to various Bluesky commands for motor movement.

The two motors can be accessed individually:

* ``c_stage.cx`` -- The x motor
* ``c_stage.cy`` -- the y motor

The variables ``cx`` and ``cy`` have also been defined to directly access each motor for convenience. There are also two
convenience functions defined:

* wmx() -- returns the current position (float) of the x motor
* wmy() -- returns the current position (float) of the y motor

These functions can be passed into Bluesky plans as ranges or locations to scan over.

The sample stage currently has 51 sample positions predefined. There are 8 columns labeled A through F and 6 rows
labeled 1 through 6. There are also two calibrants mounted on the sample holder. Finally, there is a Beryllium screen
a pinhole mounted near the base of the sample holder.

Sample locations can be accessed through the function ``c_stage.loc([sample_name])``. This function will return the position
of a given sample (sample_name) in motor coordinates (if the stage has been aligned). For example, you can find the
coordinates of a sample in column A row 4 by ``loc = c_stage.loc(['A4'])``. ``loc`` will be a list with two coordinates
``[x_coordinate,y_coordinate]`` corresponding to the sample location.

When the profile is booted for the first time it automatically loads all sample positions in a general coordinate system.
The origin of the coordinate system (0,0) is defined as the pinhole location. However, these locations do not necessarily
correspond to the location of samples in the motor coordinate system initially. The ``c_stage`` class has a method called
``correct`` that can align the two coordinate systems. To correct for an offset between the two systems, you can pass
offsets in the x and y direction by ``c_stage.correct([x_offset,y_offset])``.

The stage object also has a motor called ``c_stage.detz`` that is used for setting the sample-to-detector distance. This
distance is set on startup and should not be changed unless the user is absolutely sure. This functionality may change
and be expanded upon in future builds.

Beamstop
========
Beamline 10-2a is equipped with a photodiode beamstop that can read the direct intensity of the beam. It is generally
used for aligning the stage and can also monitor beam intensity/flux. The variable ``bstop`` can be passed to
various Bluesky plans as an instrument.

Filter Box
==========
Beamline 10-2a has a filter box with slots for 4 filter foils. The filter box can be accessed through the variable
``filt``. The state of each filter is defined as ``1`` if the filter is in the path of the beam and ``0`` if the
filter is out of the path of the beam.

Some useful functions:

* filt.set([filter1,filter2,filter3,filter4]) -- set the state of each filter. Example: filter1-0,filter2=1,filter3=0,filter4=0
* filt.all() -- set all filters to a state of 1 (in the path of the beam)
* filt.none() -- set all filters to a state of 0 (out of the path of the beam)

Shutter
======
The shutter is defined by a variable ``shutter``. The shutter has two states: closed (0) and open (5).


Ion Chambers
============
There are two ion chambers in the path of the beam -- one is upstream of the shutter and one is downstream of the shutter
and filter box. These sensors are defined by the variables ``I0`` and ``I1`` respectively. They can be passed to
Bluesky plans as metadata for purposes of monitoring the beam intensity and normalizing images.
