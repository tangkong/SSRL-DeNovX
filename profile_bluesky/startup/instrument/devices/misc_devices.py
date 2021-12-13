"""
Miscellaneous devices
"""
__all__ = ['filt','shutter', 'I1', 'I0', 'lrf', 'table_trigger', 'table_busy',
            'filter1', 'filter2', 'filter3', 'filter4','bstop']

from ..framework import sd
from ophyd import EpicsSignalRO, EpicsSignal, Device, Component as Cpt

# fast shutter
shutter = EpicsSignal('TXRD:RIO.AO0', name='FastShutter')

# ion chambers
I1 = EpicsSignalRO('TXRD:RIO.AI0', name='I1')
I0 = EpicsSignalRO('TXRD:RIO.AI1', name='I0')


table_trigger = EpicsSignal('TXRD:RIO.DO01', name='tablev_scan trigger')
table_busy = EpicsSignalRO('TXRD:RIO.AI3', name='tablev_scan busy')

# filter box import
filter1 = EpicsSignal('TXRD:RIO.DO08', name='filter1') # high (4.9V) = filter out
filter2 = EpicsSignal('TXRD:RIO.DO09', name='filter2') 
filter3 = EpicsSignal('TXRD:RIO.DO10', name='filter3') 
filter4 = EpicsSignal('TXRD:RIO.DO11', name='filter4') 

# photodiode beamstop
bstop = EpicsSignal('TXRD:RIO.AI2',kind='hinted',name='bstop')

# filter box class
class FilterBox(Device):
    filter1 = Cpt(EpicsSignal, 'TXRD:RIO.DO08')
    filter2 = Cpt(EpicsSignal, 'TXRD:RIO.DO09')
    filter3 = Cpt(EpicsSignal, 'TXRD:RIO.DO10')
    filter4 = Cpt(EpicsSignal, 'TXRD:RIO.DO11')

    ### etymology for filters
    ### in == in the path of the beam
    ### out == not in the path of the beam

    _filter_list = [filter1, filter2, filter3, filter4]

    in_value = 1
    out_value = 0

    # time is short, hardcoding this
    def set(self,num,state):
        if state != 0 and state != 1:
            print('Not a valid filter state. Please use 0 (out) or 1 (in).')
        else:
            if num == 1:
                self.filter1.set(state)
            if num == 2:
                self.filter2.set(state)
            if num == 3:
                self.filter3.set(state)
            if num == 4:
                self.filter4.set(state)

    def none(self):
        self.filter1.set(0)
        self.filter2.set(0)
        self.filter3.set(0)
        self.filter4.set(0)

    def all(self):
        self.filter1.set(1)
        self.filter2.set(1)
        self.filter3.set(1)
        self.filter4.set(1)

# define the filter box object
filt = FilterBox('',name='filt')

# append the filters to the datastream
sd.baseline.append(filt)
