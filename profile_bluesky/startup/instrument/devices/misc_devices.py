"""
Miscellaneous devices
"""
__all__ = ['shutter', 'I1', 'I0', 'lrf', 'table_trigger', 'table_busy',
            'filter1', 'filter2', 'filter3', 'filter4','bstop']

from ophyd import EpicsSignalRO, EpicsSignal, Device, Component as Cpt

shutter = EpicsSignal('TXRD:RIO.AO0', name='FastShutter')
I1 = EpicsSignalRO('TXRD:RIO.AI0', name='I1')
I0 = EpicsSignalRO('TXRD:RIO.AI1', name='I0')

lrf = EpicsSignalRO('TXRD:RIO.AI4', name='lrf')

table_trigger = EpicsSignal('TXRD:RIO.DO01', name='tablev_scan trigger')
table_busy = EpicsSignalRO('TXRD:RIO.AI3', name='tablev_scan busy')

filter1 = EpicsSignal('TXRD:RIO.DO08', name='filter1') # high (4.9V) = filter out
filter2 = EpicsSignal('TXRD:RIO.DO09', name='filter2') 
filter3 = EpicsSignal('TXRD:RIO.DO10', name='filter3') 
filter4 = EpicsSignal('TXRD:RIO.DO11', name='filter4') 

bstop = EpicsSignal('TXRD:RIO.AI2',kind='hinted',name='bstop')

#class FilterBox(Device):
#    filter1 = Cpt(EpicsSignal, 'TXRD:RIO.DO08')
#    filter2 = Cpt(EpicsSignal, 'TXRD:RIO.DO09')
#    filter3 = Cpt(EpicsSignal, 'TXRD:RIO.DO10')
#    filter4 = Cpt(EpicsSignal, 'TXRD:RIO.DO11')

#    _filter_list = [filter1, filter2, filter3, filter4]
#    valid_open_values = ['open', 'opened']
#    valid_close_values = ['close', 'closed']

#    open_value = 0
#    close_value = 1

#    def __init__(self, prefix, *, config=[1,1,1,1], **kwargs):
#        """ Register components and initialize with configuration 
#        filter_box = FilterBox()
#        """
#        self.config = config
#        self.thicks = [0.025, 0.051, 0.127, 0.399]

#        # set filters
#        for i in range(4):
#            self.set_filter(i, 'close')

#        super().__init__(prefix, **kwargs)
        

#    def set_filter(self, num, state):
#        """ set filter in or out"""

#        if state in self.valid_open_values:
#            self._filter_list[num].set(self.open_value)
#        elif state in self.valid_close_values:
#            self._filter_list[num].set(self.close_value)
#        else: 
#            raise ValueError('setting not a valid open or close state')
    
#    def status(self):
#        print('status:.....')

#    def step_up(self):
#        print('filters incremented')
    
#    def step_down(self):
#        print('filters decremented')

#fbox = FilterBox('',name='fbox')
