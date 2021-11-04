from ophyd import EpicsSignal, EpicsSignalWithRBV as SignalWithRBV, Component as Cpt
from ophyd.signal import SignalRO
from ophyd import ADComponent as ADCpt
from ophyd.areadetector import cam
from ssrltools.devices.areadetectors import SSRLDexelaDet, DexelaTiffPlugin

from ..session_logs import logger
logger.info(__file__)

__all__ = ['dexDet', ]

class HackedCam(cam.DexelaDetectorCam):
    port_name = Cpt(SignalRO, value='DEX1')
    acquire_gain = ADCpt(EpicsSignal, 'AcquireGain')
    acquire_offset = ADCpt(EpicsSignal, 'AcquireOffset')
    binning_mode = ADCpt(SignalWithRBV, 'BinningMode')
    corrections_dir = ADCpt(EpicsSignal, 'CorrectionsDir', string=True)
    current_gain_frame = ADCpt(EpicsSignal, 'CurrentGainFrame')
    current_offset_frame = ADCpt(EpicsSignal, 'CurrentOffsetFrame')
    defect_map_available = ADCpt(EpicsSignal, 'DefectMapAvailable')
    defect_map_file = ADCpt(EpicsSignal, 'DefectMapFile', string=True)
    full_well_mode = ADCpt(SignalWithRBV, 'FullWellMode')
    gain_available = ADCpt(EpicsSignal, 'GainAvailable')
    gain_file = ADCpt(EpicsSignal, 'GainFile', string=True)
    load_defect_map_file = ADCpt(EpicsSignal, 'LoadDefectMapFile')
    load_gain_file = ADCpt(EpicsSignal, 'LoadGainFile')
    load_offset_file = ADCpt(EpicsSignal, 'LoadOffsetFile')
    num_gain_frames = ADCpt(EpicsSignal, 'NumGainFrames')
    num_offset_frames = ADCpt(EpicsSignal, 'NumOffsetFrames')
    offset_available = ADCpt(EpicsSignal, 'OffsetAvailable')
    offset_constant = ADCpt(SignalWithRBV, 'OffsetConstant')
    offset_file = ADCpt(EpicsSignal, 'OffsetFile', string=True)
    save_gain_file = ADCpt(EpicsSignal, 'SaveGainFile')
    save_offset_file = ADCpt(EpicsSignal, 'SaveOffsetFile')
    serial_number = ADCpt(EpicsSignal, 'SerialNumber')
    software_trigger = ADCpt(EpicsSignal, 'SoftwareTrigger')
    use_defect_map = ADCpt(EpicsSignal, 'UseDefectMap')
    use_gain = ADCpt(EpicsSignal, 'UseGain')
    use_offset = ADCpt(EpicsSignal, 'UseOffset')

class DexelaDet15(SSRLDexelaDet):
    """
    Final class for Dexela Detector on SSRL BL DeNovX
    - add Plugins (TIFF plugin, etc)
    det = DexelaDet15(prefix, name='name')
    """
    # DexelaDetector from ophyd pulls in all Dexela specific PV's
    write_path = 'E:\\dexela_images\\'
    cam = ADCpt(HackedCam, '' ) #cam.DexelaDetectorCam, '') 
    # In case where TIFF plugin is being used
    tiff = Cpt(DexelaTiffPlugin, 'TIFF:',
                       read_attrs=[], configuration_attrs=[],
                       write_path_template=write_path,
                       read_path_template='/dexela_images/',
                       path_semantics='windows')
    # Else there should be an NDArrayData PV
    image = Cpt(EpicsSignal, 'IMAGE1:ArrayData')
    highest_pixel = Cpt(EpicsSignal, 'HighestPixel', labels=('point_det',))

    def trigger(self):
        ret = super().trigger()
        #self.cam.image_mode.put(0) # Set image mode to single...
        return ret
        
    # Could add more attributes to file_plugin
    # could add stage behavior


# Connect PV's to Ophyd objects

dexDet = DexelaDet15('SSRL:DEX2923:', name='dexela', 
                        read_attrs=['highest_pixel', 'tiff'])

dexDet.configuration_attrs.append('cam.num_images')
dexDet.configuration_attrs.append('cam.binning_mode')
dexDet.configuration_attrs.append('cam.full_well_mode')
