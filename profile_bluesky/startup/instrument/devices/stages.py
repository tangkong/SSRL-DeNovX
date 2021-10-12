import pandas as pd
import numpy as np
"""
stages 
"""

__all__ = ['c_stage']

from ..framework import sd
from ..session_logs import logger
logger.info(__file__)

from ophyd import Component as Cpt, MotorBundle, EpicsMotor

class cassetteStage(MotorBundle):
    """DeNovX Cassette Holder Sample Stage"""
    cx = Cpt(EpicsMotor,'BL22:IMS:MOTOR1',kind='hinted',labels=('sample',))
    cy = Cpt(EpicsMotor,'BL22:IMS:MOTOR2',kind='hinted',labels=('sample',))

    # TODO: can't find the .csv file in this directory, need to fix 
    # import the absolute sample positions
    casslocs =pd.read_csv('/home/b_spec/.ipython/profile_DeNovX/startup/instrument/devices/casslocs.csv',header=0)
    xlocs = -1*np.array(casslocs['x']) # !!! NOTE THE NEGATIVE 1 !!!
    ylocs = np.array(casslocs['y'])
    ids = np.array(casslocs['ID'])
    def loc(self,keys):
        """ 
        function that returns motor positions for samples based on string IDs
        if a .csv file with ID:position pairs is provided
        : param key: a string id with an associate position
        : type key: a variable length string
        """
        pos = []
        for key in keys:
            print(key)
            m1 = self.xlocs[np.where(self.ids == key)[0]][0]
            m2 = self.ylocs[np.where(self.ids == key)[0]][0]
            pos.append([m1,m2])
        
        return pos

    def correct(self,cLocs):
        """
        function that takes a set of motor offsets and adjusts sample positions
        accordingly
        : param cLocs: motor offset positions
        : type cLocs: list of length 2
        """
        self.xlocs = self.xlocs + cLocs[0] #plus based on motor directions
        self.ylocs = self.ylocs + cLocs[1] #plus based on motor directions
        return self

c_stage = cassetteStage('', name='c_stage')

# measure stage status at beginning of every plan
sd.baseline.append(c_stage)

