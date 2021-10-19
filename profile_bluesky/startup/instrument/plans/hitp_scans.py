""" 
scans for high throughput stage
"""

from ..devices.stages import c_stage
from ..devices.misc_devices import shutter as fs, lrf, I0, I1, table_busy, table_trigger
from ..framework import db

import time 
import matplotlib.pyplot as plt
import bluesky.plans as bp
from bluesky.preprocessors import inject_md_decorator 
import bluesky.plan_stubs as bps
from ssrltools.plans import meshcirc, nscan, level_stage_single

__all__ = ['find_coords', 'sample_scan','cassette_scan', 'dark_light_plan', 'exp_time_plan', 'gather_plot_ims',
            'plot_dark_corrected', 'multi_acquire_plan', 'level_stage_single','rock','opt_rock_scan',
           'opt_cassette_scan']



# center the cassette sample coordinates on the motor stage
@inject_md_decorator({'macro_name':'find_coords'})
def find_coords(dets,motor1,motor2,guess, detKey):
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

    # TODO:
    # grab pinhole bounds from casslocs.csv and scan based on that
    # determine how many points to scan in each direction

    # first take a scan in the motor1 direction
    xid = yield from bp.scan([dets],motor1,guess[0] - 0.65,guess[0]+0.65,num=25)

    # grab the photodiode data and find the max
    xhdr = db[xid].table(fill=True)
    xarr = xhdr[detKey] # xhdr[det.name]
    xLoc = xarr[xarr == xarr.max()]

    # now move the sample stage to that location and take the other scan
    yield from bps.mv(motor1,xLoc)

    # now take the scan in the motor2 direction
    yid = yield from bp.scan([dets],motor2,guess[1]-0.65, guess[1]+0.65,num=25)

    # grab the photodiode data and find the max
    yhdr = db[yid].table(fill=True)
    yarr = yhdr[detKey]
    yLoc = yarr[yarr == yarr.max()] #assuming a 1d array here

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
        fs : Fast shutter, high is closed
        sample_name : the sample name
        Example usage:
        >>> RE(dark_light_plan())
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

    for ind,range in enumerate(ranges):
        # get the motor limits
        minn = range[0]
        maxx = range[1]

        # stage the motors/detectors in the correct position
        if stage:
            # stage is a dictionary with n many dictionaries inside
            # each inner diction has a key-value pair of motor-position
            # iterate through each motor and set the position
            for k,v in stage[ind].items():
                yield from bps.mv(k,v)

        yield from bps.stage(det)
        yield from bps.trigger(det, wait=False)

        start = time.time()
        now = time.time()

        while (now - start) < exposure_time:
            yield from bps.mvr(motor, maxx)
            yield from bps.mvr(motor, minn)
            now = time.time()

        yield from bps.create('primary')
        reading = (yield from bps.read(det))
        yield from bps.save()
        yield from bps.close_run()

        # reset position
        yield from bps.mv(motor, start_pos)

        return uid

@inject_md_decorator({'macro_name':'opt_rock_scan'})
def opt_rock_scan(det,motors,center,prms,*,md=None):
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
    mmask, mpos = inscribe(motor[0], motor[1], calib1, dia, box, res)
    # update the mask
    mask.update_mask(mask,mmask,mpos[motor[0]],mpos[motor[1]])
    # get the range to rock across
    r,s = generate_rocking_range(mask,motor[0])
    # pull out the longest scan
    cRange = [min(r),max(r)]
    # optimize acquisition time based on a test scan in this range
    yield from max_pixel_rock(det,motor[0],cRange,sat_count=6*10e6)

    # now perform the rocking scan
    # TODO: need to set up the databroker so that this experiment is recorded
    yield from rock(det,motor[0],r,stage=s,md=md)

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
    :type motor: EpicsMotor signal
    :param center: center of the sample to be scann [x,y]
    :type center: list of two floats
    :param prms: list of instrument and sample parameters prms = [dia,box,res]
    :param prms: where dia is the sample diameter, box is the box beam size, and res is the scanning resolution
    :type prms: list of lists
    """
    # performs a series of opt_rock_scans based on sample centers and diameters

    for center in centers:
        yield from opt_rock_Scan(dets,motors,center,prms,*,md=None)





