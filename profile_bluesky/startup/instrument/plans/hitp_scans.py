""" 
scans for high throughput stage
"""

from ..devices.stages import c_stage
from ..devices.misc_devices import shutter as fs, lrf, I0, I1, table_busy, table_trigger
from ..devices.misc_devices import filt,filter1,filter2,filter3,filter4
from ..framework import db
from .helpers import show_image,inscribe,generate_rocking_range, filters, box,run_summary
from .adapt_opt import int_to_bool_list

import time 
import matplotlib.pyplot as plt
import bluesky.plans as bp
from bluesky.preprocessors import inject_md_decorator 
import bluesky.plan_stubs as bps
from bluesky_live.bluesky_run import BlueskyRun, DocumentCache
from ssrltools.plans import meshcirc, nscan, level_stage_single
import numpy as np

__all__ = ['find_coords', 'sample_scan','cassette_scan', 'dark_light_plan', 'exp_time_plan', 'gather_plot_ims',
            'plot_dark_corrected', 'multi_acquire_plan', 'level_stage_single','rock','opt_rock_scan',
    # open the shutter
           'opt_cassette_scan','max_pixel_rock','survey','test_stats','opt_survey']



# center the cassette sample coordinates on the motor stage
@inject_md_decorator({'macro_name':'find_coords'})
def find_coords(dets,motor1,motor2,guess,delt=1,num=50):
    """
    find_coords takes a guess for the pinhole center, performs a scan,
    then assigns the motor coordinates of each sample

    :param dets: photodiode detector to be used
    :type dets: ophyd det object
    :param motor1: motor for x axis
    :type motor1: ophyd EPICS motor
    :param motor2: motor for y axis
    :type motor2: ophyd EPICS motor
    :param center: inital guess for the center of the pinhole
    :type center: list with two motor position entires [motor1, motor2]
    """

    # move the motor to the guess positions
    yield from bps.mv(motor1,guess[0],motor2,guess[1])

    # do the scan twice
    count = 0
    while count < 2:

        # first take a scan in the motor1 direction
        xid = yield from bp.scan([dets],motor1,guess[0] - delt,guess[0]+delt,num=num)

        # grab the photodiode data and find the max
        xhdr = db[xid].table(fill=True) # generate data table
        xarr = xhdr[dets.name] #grab the detector data
        xPos = xhdr[motor1.name] # grab the motor positions
        xInd = np.where(xarr == np.max(xarr))[0] # find the max value of the det
        if len(xInd) > 1:
            xInd = np.take(xInd,len(xInd)//2) # check for multiple maxs: choose middle value
        if type(xInd) != np.int64:
            xInd = xInd[0]
        xLoc = xPos[xInd]

        # now move the sample stage to that location and take the other scan
        yield from bps.mv(motor1,xLoc)

        # now take the scan in the motor2 direction
        yid = yield from bp.scan([dets],motor2,guess[1]-delt, guess[1]+delt,num=num)

        # grab the photodiode data and find the max
        yhdr = db[yid].table(fill=True) # generate data table
        yarr = yhdr[dets.name] # get the detector data
        yPos = yhdr[motor2.name] # motor positions
        yInd = np.where(yarr == np.max(yarr))[0] # find the max value
        if len (yInd) > 1:
            yInd = np.take(yInd,len(yInd)//2)
        if type(yInd) != np.int64:
            yInd = yInd[0]
        yLoc = yPos[yInd]

        # move the motor to the max values
        yield from bps.mv(motor1,xLoc,motor2,yLoc)

        # set the new guess to be your current max values
        guess = [xLoc,yLoc]

        # index the count
        count+=1

    # now pass the offsets to the stage object  to reset the positions
    c_stage.correct([xLoc,yLoc])

# scan a single sample in a DeNovX cassette based on its center location
@inject_md_decorator({'macro_name':'sample_scan'})
def sample_scan(dets,motor1,motor2,center):
    """
    sample_scan starts at the center of a a single DeNovX sample and executes a list_scan within the boundaries of the sample geometry

    :param dets: detectors to be used
    :type dets: list
    :param motor1: first motor to be used
    # open the shutter
    :type motor1: ophyd EPICS motor
    :param motor2: second motor to be used
    :type motor2: ophyd EPICS motor
    :param center: two positions defining center of sample
    :type center: list of floats

    :yield: results from list_scan
    """

    ### TODO: make boundaries an arg, calculate from arbitrary boundaries

    # define the boundaries on the sample
    xWid = 10
    yWid = 10

    xHigh = center[0] + xwid/2
    xLow = center[0] - xWid/2

    yhigh = center[1] + yWid/2
    yLow = center[1] - yWid/2
    
    # get all scan locations for list_scan
    scan_locs = np.vstack((np.linspace(xLow,xHigh),np.linspace(yLow,yHigh)))

    yield from list_scan([dets],motor1,list(scan_locs[0]),motor2,list(scan_locs[1]))

# scan all samples in a DeNovX cassette
@inject_md_decorator({'macro_name':'cassette_scan'})
def cassette_scan(dets,motor1,motor2,corr_locs,skip=0,md={}):
    """ cassette_scan moves to the center of each point on a DeNovX and executesa pre-defined scan in each location

    :param dets: detectors to be used
    :type dets: list
    :param motor1: first stage motor to be used
    :type motor1: ophyd EPICS motor
    :param motor2: second stage to be used
    :type motor2: ophyd EPICS motor
    :param skip: number of datapoints to skip
    :yield: results from sample_scan
    """

    # iterate through each center location and execute a sample scan
    for center in corr_locs:
        yield from sample_scan(dets,motor1,motor2,center)


# collection plans
# Basic dark > light collection plan
@inject_md_decorator({'macro_name':'dark_light_plan'})
def dark_light_plan(dets, shutter=fs, md={}):
    '''
        Simple acquisition plan:
        - Close shutter, take image, open shutter, take image, close shutter
        dets : detectors to read from
        motors : motors to take readings from (not fully implemented yet)
        sample_name : the sample name
        Example usage:
    '''
    if I0 not in dets:
        dets.append(I0)
    if I1 not in dets:
        dets.append(I1)

    start_time = time.time()
    uids = []

    #close fast shutter, take a dark
    yield from bps.mv(fs, 1)
    mdd = md.copy()
    mdd.update(im_type='dark')
    uid = yield from bp.count(dets, md=mdd)
    uids.append(uid)


    # open fast shutter, take light
    yield from bps.mv(fs, 0)
    mdl = md.copy()
    mdl.update(im_type='primary')
    uid = yield from bp.count(dets, md=mdl)
    uids.append(uid)

    end_time = time.time()
    print(f'Duration: {end_time - start_time:.3f} sec')

    plot_dark_corrected(db[uids])

    return(uids)


# Plan for meshgrid + dark/light?...

# image time series
@inject_md_decorator({'macro_name':'exposure_time_series'})
def exp_time_plan(det, timeList=[1]):
    '''
    Run count with each exposure time in time list.  
    Specific to Dexela Detector, only run with single detector
    return uids from each count

    # TO-DO: Fix so both are under the same start document
    '''
    primary_det = det
    dets = []
    if I0 not in dets:
        dets.append(I0)
    if I1 not in dets:
        dets.append(I1)

    dets.append(primary_det)
    md = {'acquire_time': 0, 'plan_name': 'exp_time_plan'}
    uids = []
    for time in timeList:
        # set acquire time
        yield from bps.mov(primary_det.cam.acquire_time, time)
        md['acquire_time'] = time
        uid = yield from bp.count(dets, md=md)

        yield from bps.sleep(1)
        uids.append(uid)
   
    return uids

#helper functions, probably should go in a separate file

import datetime
fts = datetime.datetime.fromtimestamp
def gather_plot_ims(hdrs):
    '''
    helper function for plotting images. 
    '''
    plots=[]
    times=[]
    # gather arrs from db
    for hdr in hdrs:
        arr = hdr.table(fill=True)['dexela_image'][1]
        plots.append(arr)
        time = hdr.start['time']
        times.append(time)
        

    global curr_pos
    curr_pos = 0
    # Register key event
    def key_event(e): 
        global curr_pos 
        if e.key == 'right': 
            curr_pos=curr_pos + 1 
        elif e.key == 'left': 
            curr_pos = curr_pos - 1  
        else: 
            return 
        curr_pos = curr_pos % len(plots) 
        ax.cla()
        curr_arr = plots[curr_pos]
        ax.imshow(curr_arr, vmax=curr_arr.mean() + 3*curr_arr.std()) 

        dt = fts(times[curr_pos])
        ax.set_title(f'{dt.month}/{dt.day}, {dt.hour}:{dt.minute}')
        fig.canvas.draw() 

    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', key_event)
    ax = fig.add_subplot(111)
    ax.imshow(plots[0], vmax=plots[0].mean() + 3*plots[0].std())
    dt = fts(times[0])
    ax.set_title(f'{dt.month}/{dt.day}, {dt.hour}:{dt.minute}')

    plt.show()


def plot_dark_corrected(hdrs):
    '''
    looks for name='dark' or 'primary'
    otherwise assumes dark comes first?
    '''

    for hdr in hdrs:
        if hdr.start['im_type']=='dark':
            darkarr = hdr.table(fill=True)['dexela_image'][1]
        elif hdr.start['im_type'] == 'primary':
            lightarr = hdr.table(fill=True)['dexela_image'][1]
        else:
            print('mislabeled data... ignoring for now')
            return

    bkgd_subbed = lightarr.astype(int) - darkarr.astype(int)
    bkgd_subbed[ bkgd_subbed < 0 ] = 0
    plt.imshow(bkgd_subbed, vmax = bkgd_subbed.mean() + 3*bkgd_subbed.std())

@inject_md_decorator({'macro_name':'multi_acquire_plan'})
def multi_acquire_plan(det, acq_time, reps): 
    '''multiple acquisition run.  Single dark image, multiple light
    '''
    yield from bps.mv(det.cam.acquire_time, acq_time) 
    print(det.cam.acquire_time.read()) 
    yield from bps.mv(fs, 1) 
    dark_uid = yield from bp.count([det], md={'name':'dark'}) 
    yield from bps.mv(fs, 0) 
    light_uids = [] 
    
    for _ in range(reps): 
        light_uid = yield from bp.count([det], md={'name':'primary'}) 
        light_uids.append(light_uid) 
    
    return (dark_uid, light_uids) 


@inject_md_decorator({'marco_name': 'tablev_scan'})
def tablev_scan():
    '''
    send signal to DO01 to trigger tablev_scan
    '''
    yield from bps.mv(table_trigger, 0) # start tablev_scan
    yield from bps.sleep(1)

    if I0.get() < 0.01: # If we don't have beam
        print('No beam, aborting...')
        yield from bps.mv(table_trigger, 1) # end tablev_scan
        return

    # sleep until we see the busy signal go high
    cnt = 0
    while (table_busy.read()[table_busy.name]['value'] < 0.5) and cnt < 100: 
        print('waiting for busy signal...') 
        yield from bps.sleep(2)
        cnt += 1 

    # Turn off trigger signal
    yield from bps.mv(table_trigger, 1)

def tuned_mesh_grid_scan(AD, mot1, s1, f1, int1, mot2, s2, f2, int2, 
            radius, ROI=1, pin=None, skip=0, md=None):
    '''
    Attempt to be intelligent about which points on a grid are scanned
    Tune within per_step plan based on signal from xsp3 roi.  
        - scan with xsp3
        - find peak location
        - move to peak location
        - collect AD
    AD must be some area detector 
    '''
    raise NotImplementedError

    # Add other relevant detectors
    if I0 not in dets:
        dets.append(I0)
    if I1 not in dets:
        dets.append(I1)

    # Determine points for meshgrid 
    # Verification (check non-negative, motors are motors, non-zero steps?)
    if (s1 >= f1) or (s2 >= f2):
        raise ValueError('starting bounds must be less than end bounds')
    # Basic plan logic
    ## Define new bounds
    if not pin: # no pinning tuple provided
        pin = (mot1.position, mot2.position) 

    if (pin[0] < s1) or (f1 < pin[0]) or (pin[1] < s2) or (f2 < pin[1]):
        raise ValueError('pin location not inside defined grid')

    ## subtract fraction of interval to account for edges
    s1_new = np.arange(pin[0], s1-int1/2, -int1)[-1]
    f1_new = np.arange(pin[0], f1+int1/2, int1)[-1]

    s2_new = np.arange(pin[1], s2-int2/2, -int2)[-1]
    f2_new = np.arange(pin[1], f2+int2/2, int2)[-1]

    ## add half of interval to include endpoints if interval is perfect
    num1 = len(np.arange(s1_new, f1_new+int1/2, int1))
    num2 = len(np.arange(s2_new, f2_new+int2/2, int2))
    
    center = (s1+(f1-s1)/2, s2+(f2-s2)/2)

    motor_args = list([mot1, s1_new, f1_new, num1, 
                        mot2, s2_new, f2_new, num2, False])

    # metadata addition
    _md = {'radius': radius}
    _md.update(md or {})

    # parameters for _refine_scan
    peak_stats = PeakStats(x=motor.name, y=signal.name)

    # start plans
    @subs_decorator(peak_stats)
    def _refine_scan(md=None):

        yield from bp.rel_scan([signal], motor, -width/2, width/2, num)

        # Grab final position from subscribed peak_stats
        valid_peak = peak_stats.max[-1] >= (4 * peak_stats.min[-1])
        if peak_stats.max is None:
            valid_peak = False

        final_position = 0 
        if valid_peak: # Condition for finding a peak
            if peak_choice == 'cen':
                final_position = peak_stats.cen
            elif peak_choice == 'com':
                final_position = peak_stats.com

        yield from bps.mv(motor, final_position)


    # inject logic via per_step 
    class stateful_per_step:
        
        def __init__(self, skip):
            self.skip = skip
            self.cnt = 0
            #print(self.skip, self.cnt)

        def __call__(self, detectors, step, pos_cache):
            """
            has signature of bps.one_and_step, but with added logic of skipping 
            a point if it is outside of provided radius
            """
            if self.cnt < self.skip: # if not enough skipped
                self.cnt += 1
                pass
            else:
                # rel_scan and refine
                yield from _refine_scan()
                # acquire
                yield from bps.one_nd_step(detectors, step, pos_cache)

    per_stepper = stateful_per_step(skip)

    # Skip points if need
    newGen = bp.grid_scan(detectors, *motor_args, 
                            per_step=per_stepper, md=_md)
    return (yield from newGen)

@inject_md_decorator({'macro_name':'powder_check'})
def powder_check(dets):
    """
    plan to check the coverage of peaks on the detector. this plan takes a scan, applies some metrics to check the q/chi coverage of peaks on the detector, and reports a value.
    """

@inject_md_decorator({'macro_name':'rock'})
def rock(det, motor, ranges, *, stage=None, md=None):
    """
    based on Robert's rocking_scan code to include a motor staging option
    it also accepts an arbitrary number of ranges to rock across
    :param det: detector object
    :type det: subclass of ophyd.areadetector
    :param motor: motor to rock during exposure
    :type motor: EpicsMotor
    :param ranges: min and max values to rock the motor across
    :type minn: list of N length lists containing [min,max] pairs
    :param stage: optional staging positions to execute before each rocking command, defaults to None
    :type stage: dictionary containing N key-value pairs; each key is an integer 1...N and each value
                 is a dictionary; the inner dictionaries contain key-value pairs for a motor to stage
                 and the staging position; ex. stage={1:{motor1,pos1},2:{motor1,pos2}}
    :param md: metadata dictionary, defaults to None
    :type md: dict, optional
    :return: uid
    :rtype: string
    :yield: plan messages
    :rtype: generator
    """
    uid = yield from bps.open_run(md)

    # assume dexela detector trigger time PV
    exposure_time = det.cam.acquire_time.get()
    start_pos = motor.user_readback.get()

    yield from bps.stage(det)
    
    for ind,ranger in enumerate(ranges):
        # get the motor limits
        minn = ranger[0]
        maxx = ranger[1]

        # stage the motors/detectors in the correct position
        if stage:
    # open the shutter
            # stage is a dictionary with n many dictionaries inside
            # each inner diction has a key-value pair of motor-position
            # iterate through each motor and set the position
            for m in stage:
                print('Motor to stage:' + str(m.name))
                print('Staging position:' + str(stage[m][ind]))
                yield from bps.mv(m,stage[m][ind])

        trig_status = (yield from bps.trigger(det,wait=False))
        print(trig_status)
        start = time.time()
        now = time.time()

        #while ((now - start ) < exposure_time) or (not trig_status.done):
        while not trig_status.done:
            yield from bps.mv(motor, maxx)
            yield from bps.mv(motor, minn)
            now = time.time()
        
        print(trig_status) 
        yield from bps.create('primary')
        reading = (yield from bps.read(det))
        yield from bps.save()
        #print(reading)
        #print('sleep once')
        yield from bps.sleep(1)
    
    yield from bps.close_run()
    yield from bps.unstage(det)

    # reset position
    yield from bps.mv(motor, start_pos)

    return uid

def rock_stub(det, motor, ranges):
    '''
    only for use (currently) with filter_opt_count.  
    Takes a list of detectors, unlike rock
    '''

    # assume dexela detector trigger time PV
    for d in det: 
        if hasattr(d, 'cam'):
            exposure_time = d.cam.acquire_time.get()
    start_pos = motor.user_readback.get()
    
    for ind,ranger in enumerate(ranges):
        # get the motor limits
        minn = ranger[0]
        maxx = ranger[1]

        for d in det:
            if hasattr(d, 'cam'): # This is awful find a better way to filter devices
                trig_status = (yield from bps.trigger(d,wait=False))
        print(trig_status)
        start = time.time()
        now = time.time()

        #while ((now - start ) < exposure_time) or (not trig_status.done):
        while not trig_status.done:
            yield from bps.mv(motor, maxx)
            yield from bps.mv(motor, minn)
            now = time.time()
        
        print(trig_status) 
        yield from bps.create('primary')
        
        ret = {}
        for d in det:
            reading = (yield from bps.read(d))
             # open the shutter
        if reading is not None:
                ret.update(reading)
        yield from bps.save()
        #print(reading)
        #print('sleep once')
        yield from bps.sleep(1)
    
    yield from bps.mv(motor,start_pos)


@inject_md_decorator({'macro_name':'opt_rock_scan'})
def opt_rock_scan(det,motors,center,prms,*,sat_count=1e4,md=None):
    """
    performs a check for on detector acquisition time based on a rocking scan range,
    sets the detector acquisition time, and computes the scan
    :param det: detector to be used
    :type det: epics ophyd object
    :param motors: motors to be used NOTE: first motor must be scanning motor, all other motors are for staging
    :type motor: EpicsMotor signal
    :param center: center of the sample to be scann [x,y]
    :type center: list of two floats
    :param prms: list of instrument and sample parameters prms = [dia,box,res]
    :param prms: where dia is the sample diameter, box is the box beam size, and res is the scanning resolution
    :type prms: list of lists
    """

    # move to the sample center
    for ind,motor in enumerate(motors):
        yield from bps.mv(motor,center[ind])

    # pull out parameters
    dia = prms[0]
    box = prms[1]
    res = prms[2]
    # inscribe everything
    mmask, mpos = inscribe(motors[0], motors[1], center, dia, box, res)
    # get the range to rock across
    r,s = generate_rocking_range(motors[1],mmask,mpos,transpose=True)
    # pull out the longest scan
    
    #cRange = [center[1],center[1]]

    # find the longest rock scan in the ranges
    # lol logic
    #for rr in r:
    #    minn = min(rr)
    #    maxx = max(rr)
    #    if minn < cRange[0]:
    #        cRange[0] = minn
    #    if maxx > cRange[1]:
    #        cRange[1] = maxx

    # optimize acquisition time based on a test scan in this range
    #yield from max_pixel_rock([det],motors[0],cRange,sat_count=1e4)

    #print('Detector Acq Time: ' + str(det.cam.acquire_time.get()))

    # now perform the rocking scan
    # TODO: need to set up the databroker so that this experiment is recorded
    yield from rock(det,motors[1],r,stage=s,md=md)

# scan with adaptive optimization
@inject_md_decorator({'macro_name':'opt_cassette_scan'})
def opt_cassette_scan(dets,motors,centers,prms,*,md=None):
    """
    wrapper for performing multiple opt_rock_scans on a series of samples
    performs a check for on detector acquisition time based on a rocking scan range,
    sets the detector acquisition time, and computes the scan
    :param det: detector to be used
    :type det: epics ophyd object
    :param motors: motors to be used rocked
    :type motor: E'A1','A2','A3','A4','A5','A6','A7','A8',
            'calib1',
picsMotor signal
    :param center: center of the sample to be scann [x,y]
    :type center: list of two floats
    :param prms: list of instrument and sample parameters prms = [dia,box,res]
    :param prms: where dia is the sample diameter, box is the box beam size, and res is the scanning resolution
    :type prms: list of lists
    """
    # performs a series of opt_rock_scans based on sample centers and diameters

    for center in centers:
        yield from opt_rock_scan(dets,motors,center,prms,md=md)

@inject_md_decorator({'macro_name':'max_pixel_rock'})
def max_pixel_rock(dets, motor,ranger, sat_count=1e4, md={},img_key='dexela_image'):
    """max_pixel_count

    Adjust acquisition time based on max pixel count
    Assume each det in dets has an attribute det.max_count.
    Assume counts are linear with time.
    Scale acquisition time to make det.max_count.get()=sat_count
    """

    for det in dets:
        n = 0

        det.cam.acquire_time.set(1)

        curr_max_counts = 1e10
        while curr_max_counts > sat_count:
            # get a dark image
            yield from bps.mv(fs,0)
            did = yield from rock(det,motor,[ranger])
            dhdr = db[did].table(fill=True)
            dark = dhdr[img_key][1][0]
            ndark = dark.astype(int)
            yield from bps.mv(fs,5)

            # perform a rocking scan
            uid = yield from rock(det,motor,[ranger])
            ahdr = db[uid].table(fill=True)
            # grab the image
            arr = ahdr['dexela_image'][1][0]
            narr = arr.astype(int)
            sarr = arr - dark

            #check, zero negative values
            vals = np.where(sarr < 0)
            sarr[vals] = 0
            
            # count "saturated" pixels, max 1% of detector saturated
            vals = np.where(sarr > 0.9*sat_count)
            sarr[vals] = 0
            ## if over allowed amount, reduce time

            # 

            plt.figure()
            plt.imshow(sarr,vmax = 1e4)
            plt.show()
            # find the max pixel count on the image
            curr_max_counts = np.max(sarr)
            print('Current Max Counts; ' + str(curr_max_counts))
            # get the current detector acquisition time
            curr_acq_time = det.cam.acquire_time.get()
            # calculate a new acquisition time based on the desired max counts
            new_acq_time = round((curr_acq_time * sat_counts)/curr_max_counts, 4)

            if new_acq_time > 600:
                print('Acquisition time too long!! Resetting to 1.')
                new_acq_time = 1

            # set the detector to the new value
            yield from bps.mv(det.cam.acquire_time, new_acq_time)

            n+=1

            if n == 5:
                break

        print('Final Acquisition Time: ' + str(new_acq_time))


@inject_md_decorator({'macro_name':'survey'})
def survey(f,stage,det,stop,motors,pinguess,name,*,md=None):

    
    prms = [5,[0.548,0.156],[20,20]]

    # clear the filters
    f.none()

    # move to the pin and calibrate the position
    yield from find_coords(stop,motors[0],motors[1],pinguess)

    f.set(2,1)
    f.set(1,1)

    # define all the locations you want to scan
    allpos = ['calib1','calib2',
            'A1','A2','A3','A4','A5','A6','A7','A8',
            'calib1','calib2',
            'B1','B2','B3','B4','B5','B6','B7','B8',
            'calib1','calib2',
            'C1','C2','C3','C4','C5','C6','C7','C8',
            'calib1','calib2',
            'D1','D2','D3','D4','D5','D6','D7','D8',
            'calib1','calib2',
            'E1','E2','E3','E4','E5','E6','E7','E8',
            'calib1','calib2',
            'F1','F2','F3','F4','F5','F6','F7','F8',
            'calib1','calib2']
     

    for pos in allpos:
        

        center = stage.loc([pos])[0]

        #move to the sample locations
        yield from bps.mv(motors[0],center[0],motors[1],center[1])

        #take a dark image
        yield from bps.mv(fs,0)
        yield from rock(det,motors[0],[[center[0]-1,center[0]+1]],md = {name + 'scan':str(pos) + '_dark'})
        yield from bps.mv(fs,5)


        # take a single exposure
        yield from bp.count([det],
                md = {name + 'scan':str(pos) + '_single_exposure',
                'I0':I0.get(),'I1':I1.get(),'bstop':stop.get(),'samp_center':center})


        # take a single rocking scan
        yield from rock(det,motors[0],[[center[0]-2.5,center[0]+2.5]],
                md={name + 'scan':str(pos) + '_single_rock',
                'I0':I0.get(),'I1':I1.get(),'bstop':stop.get(),'samp_center':center})


        # take rocking scans across the whole sample surface area
        yield from opt_rock_scan(det,motors,center,prms,
                md={name + 'scan':str(pos) + '_full_rock_res10',
                'I0':I0.get(),'I1':I1.get(),'bstop':stop.get(),'samp_center':center})
        print('SAMPLE ' + str(pos) + ' FINISHED!!!!')
    yield from bps.mv(fs,0)
    f.all()

@inject_md_decorator({'macro_name':'test_stats'})
def test_stats(det,motor,ranger):
    
    # get the starting acquisition time
    estart = det.cam.acquire_time.get()

    etimes = [0,0.1,0.2,0.5,0.75,1,1.5,2,3,5,7,10]

    for e in etimes:
        # set the detector acquisition time
        det.cam.acquire_time.set(e)
        count = 0
        while count < len(etimes):
            yield from rock(det,motor,ranger,md = {'etime':e})
            count+=1

    # reset the acquisition time
    det.cam.acquire_time.set(estart)

def filter_opt_count(det, motor, ranges, target_count=1000, det_key='dexela_image' ,md={}):
    """ filter_opt_count
    OPtimize counts using filters 

    Only takes one detector, since we are optimizing based on it alone
    aims to only reduce saturated pixels
    automatically collects filters as well
    """
    dc = DocumentCache()
    token = yield from bps.subscribe('all', dc)
    yield from bps.stage(det)

    yield from bps.open_run(md=md)
    # set current filter status @ midpoint (7)
    int_curr = 7
    filters(int_to_bool_list(7))
    # BlueskyRun object allows interaction with documents similar to db.v2, 
    # but documents are in memory
    run = BlueskyRun(dc)
    yield from rock_stub([det, filt], motor, ranges)
    #yield from bps.trigger_and_read([det,filt])
    data = run.primary.read()[det_key][-1].values
    sat_count = np.sum(data > 16380)

    int_max = 15 # lowest counts at all filters in
    int_min = 0 # highest counts at all filters out
    max_iter = 5
    curr_iter = 0
    sat_count_record = [0 for i in range(16)]
    sat_count_record[int_curr] = sat_count

    # gather filter info and binary search to get to within 200 below target
    while (  not ((sat_count<target_count) and (sat_count>(target_count-200)))
             and (curr_iter < max_iter)   ):
        if sat_count > target_count:
            print('too high')
            # restrict search range to (int_min, curr_filt)
            int_min = int(int_max - (int_max - int_min)/2)
        elif sat_count < (target_count - 200):
            print('too low')
            # restrict search range to (curr_filt, int_max)
            int_max = int((int_max - int_min)/2 + int_min)

        # new setting is midpoint of new range
        int_curr = int((int_max - int_min)/2 + int_min)
        new_filters = int_to_bool_list(int_curr)

        curr_filters = [e>0 for e in filt.get()]
        # set new filter configuration
        print(f'search range: ({int_min}, {int_max})')
        print(f'sat_count: {sat_count}, filters: {curr_filters} -> {new_filters}')
        if sat_count_record[int_curr] > 0:
            break
        filters(new_filters)

        # grab new sat_count
        run = BlueskyRun(dc)
        # or yield from rock_stub(...), then yield from bps.trigger_and_read([filt])
        #yield from bps.trigger_and_read([det, filt])
        yield from rock_stub([det, filt], motor, ranges)
        data = run.primary.read()[det_key][-1].values
        sat_count = np.sum(data > 16380)
        sat_count_record[int_curr] = sat_count
        curr_iter += 1

    # Final acquisition
    # find setting just below target_count 
    counts = np.array(sat_count_record) - target_count
    index = np.argmin(np.abs(counts))
    if sat_count_record[index] > target_count:
        # move one below
        index += 1
    filters(int_to_bool_list(index))

    # scale up acquisition time to get close to target_count
    old_time = det.cam.acquire_time.get()
    old_counts = sat_count_record[index]
    new_time = np.min([old_time * target_count / old_counts * 0.8,2])
    print(f'old time: {old_time} -> new_time: {new_time}')
    print(sat_count_record)
    yield from bps.mv(det.cam.acquire_time, new_time)
    #yield from bps.trigger_and_read([det,filt])
    yield from rock_stub([det, filt], motor, ranges)

    yield from bps.close_run()
    yield from bps.unsubscribe(token)
    yield from bps.unstage(det)


@inject_md_decorator({'macro_name':'opt_survey'})
def opt_survey(filt,det,bstop,stage,motors,pinguess,mesh,name,*,md=None):
    # a survey scan for an entire cassette using Robert's adaptive optimization for
    # filters and exposure time

    # filt -- filter box class
    # det -- dexela detector
    # bstop -- beamstop
    # stage -- the stage class object
    # motors -- [x motor, y motor]
    # pinguess -- [guess x, guess y] guess for the pinhole location
    # prms -- parameters for inscribing rocks

    # this scan rocks in y, which should be motor[1]i

    # open the shutter
    yield from bps.mv(fs,5)

    # clear the filters
    filt.none()

    # move to the pin and calibrate the position
    yield from find_coords(bstop,motors[0],motors[1],pinguess)

    # define all the locations you want to scan
    allpos = ['calib1','calib2',
            'A1','A2','A3','A4','A5','A6','A7','A8',
            'calib1','calib2',
            'B1','B2','B3','B4','B5','B6','B7','B8',
            'calib1','calib2',
            'C1','C2','C3','C4','C5','C6','C7','C8',
            'calib1','calib2',
            'D1','D2','D3','D4','D5','D6','D7','D8',
            'calib1','calib2',
            'E1','E2','E3','E4','E5','E6','E7','E8',
            'calib1','calib2',
            'F1','F2','F3','F4','F5','F6','F7','F8',
            'calib1','calib2']


    for pos in allpos:
        # get the sample position
        c = stage.loc([pos])[0]

        # move to the sample position
        yield from bps.mv(motors[0],c[0],motors[1],c[1])

        # set the exposure time before doing filter_opt_count
        det.cam.acquire_time.set(2)

        # first we want to test the filters and exposure time
        # aim for fewer than 1000 saturated pixels
        yield from filter_opt_count(det,motors[1],[[c[1]-2,c[1]+2]],target_count=1000)

        # now do the full rocking scan
        # first inscribe the motors
        # define the params

        m,p = inscribe(motors[0],motors[1], c, 5, box, [mesh,mesh])

        # generate the ranges
        r,s = generate_rocking_range(motors[1],m,p,transpose=True)

        yield from rock(det,motors[1],r,stage=s,md={name:str(pos),'I0':I0.get(),'I1':I1.get(),'bstop':bstop.get(),'samp_center':c,'mesh':mesh,'filter1':filter1.get(),'filter2':filter2.get(),'filter3':filter3.get(),'filter4':filter4.get()})
        
        #close the filters so you don't burn the detector
        filt.none()

        # generate a run summary
        run_summary(db[-1],name + '_' + str(pos))
        
    # close the shutter    
    yield from bps.mv(fs,0)
