"""
stages 
"""

__all__ = ['s_stage','c_stage', 'px', 'py', 'pz', 'th', 'vx', 'vy']

from ..framework import sd
from ..session_logs import logger
logger.info(__file__)

from ophyd import Component as Cpt, MotorBundle, EpicsMotor

class HiTpStage(MotorBundle):
    """HiTp Sample Stage"""
    #stage x, y
    px = Cpt(EpicsMotor, 'BL00:IMS:MOTOR3', kind='hinted', labels=('sample',))
    py = Cpt(EpicsMotor, 'BL00:IMS:MOTOR4', kind='hinted', labels=('sample',))
    pz = Cpt(EpicsMotor, 'BL00:IMS:MOTOR2', kind='hinted', labels=('sample',))

    # plate vert adjust motor 1, 2
    vx = Cpt(EpicsMotor, 'BL00:PICOD1:MOTOR3', labels=('sample',))
    vy = Cpt(EpicsMotor, 'BL00:PICOD1:MOTOR2', labels=('sample',))

    th = Cpt(EpicsMotor, 'BL00:IMS:MOTOR1', labels=('sample',))

class cassetteStage(MotorBundle):
    """DeNovX Cassette Holder Sample Stage"""
    cx = Cpt(EpicsMotor,'BL22:IMS:MOTOR1',kind='hinted',labels=('sample',))
    cy = Cpt(EpicsMotor,'BL22:IMS:MOTOR2',kind='hinted',labels=('sample',))

c_stage = cassetteStage('', name='c_stage')
s_stage = HiTpStage('',name='s_stage')

# measure stage status at beginning of every plan
sd.baseline.append(c_stage)

# convenience definitions 
px = s_stage.px
py = s_stage.py
pz = s_stage.pz

vx = s_stage.vx
vy = s_stage.vy

th = s_stage.th
