"""
Plans for more automated optimization acquisition plans, w.r.t area detectors
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import bluesky.plan_stubs as bps
import bluesky.plans as bp
from bluesky_live.bluesky_run import BlueskyRun, DocumentCache

from .helpers import sum_images, filters, cossim, imthresh
#from .hitp_scans import rock, rock_stub
from ..devices.misc_devices import filter1, filter2, filter3, filter4, filt
from ..framework import db

__all__ = ['check_coverage','check_sample','max_pixel_count','old_max_pixel_rock','int_to_bool_list','stub_filter_opt_count', 'solve_filter_setup','filter_thicks', 'filter_hist']

# dataframe to record intensity and filter information.  
filter_hist = pd.DataFrame(columns=['time','filter_i','filter_f','I_i', 'I_f', 
                                    'I_f/I_i', 'mu', 'signal'])

filter_thicks = [0.025, 0.051, 0.127, 0.399]

def check_sample(runs):
    # takes a series of scans and measures cosine similarity between them

    # list for storing sim scores
    sims = []

    # iterate through the images in the run
    for ind,trun in enumerate(list(runs)):
        # image to test (test array)
        tarr = []

        # grab all the images in this scan and sum them
        for timg in trun.table(fill=True)['dexela_image']:
            tarr.append(timg[0])
        tarr = np.sum(tarr,axis=0)

        # list to store the similarity measured againt tarr
        sim = []

        # iterate through the images in the run
        for crun in list(runs):
            # image to check against (check array)
            carr = []

            # grab all the images in this scan and sum them
            for cimg in crun.table(fill=True)['dexela_image']:
                carr.append(cimg[0])
            carr = np.sum(carr,axis=0)

            # measure similarity between test and check
            sim.append(cossim(tarr,carr))

        # append the similarities 
        sims.append(sim)

    # create a confusion matrix plot
    fig,ax = plt.subplots(1,1)
    im = ax.imshow(sims)
    ax.set_xlabel('Run #')
    ax.set_ylabel('Run #')
    fig.colorbar(im)

    return(sims)

def check_coverage(runs):
    check = False

    sarr = sum_images()
    sarr = sarr/np.max(sarr)
    arr = np.zeros(sarr.shape)
    sims = []
    for img in list(runs)[0].table(fill=True)['dexela_image']:
        arr += img[0]/np.max(img[0])

        # compute the cosine similarity
        sim = cossim(sarr,arr)
        sims.append(sim)
        #
        if sim > 0.998:
            check = True
            break
        else:
            continue
             #
    sims.append(1)
    plt.figure()
    plt.plot(sims)
    plt.xlabel('Number of Rocking Scans')
    plt.ylabel('Similarity')

    return check


def max_pixel_count(dets, sat_count=60000, img_Key = 'dexela_image',md={}):
    """max_pixel_count 

    Adjust acquisition time based on max pixel count
    Assume each det in dets has an attribute det.max_count.
    Assume counts are linear with time.
    Scale acquisition time to make det.max_count.get()=sat_count
    """

    for det in dets:
        uid = yield from bp.count([det])
        curr_acq_time = det.cam.acquire_time.get()
        hdr = db[uid].table(fill=True)
        arr = hdr[img_Key][1]
        curr_max_counts = np.max(arr)
        new_acq_time = round(sat_count / curr_max_counts * curr_acq_time, 2)

        yield from bps.mv(det.cam.acquire_time, new_acq_time)


def old_max_pixel_rock(dets, motor,ranger, sat_count=60000, md={},img_key='dexela_image'):
    """max_pixel_count

    Adjust acquisition time based on max pixel count
    Assume each det in dets has an attribute det.max_count.
    Assume counts are linear with time.
    Scale acquisition time to make det.max_count.get()=sat_count
    """

    for det in dets:
        # perform a rocking scan
        uid = yield from rock(det,motor,[ranger])
        # grab the image
        sarr = sum_images(ind=-1,img_key = img_key)
        # find the max pixel count on the image
        curr_max_counts = np.max(sarr)
        print(curr_max_counts)
        # get the current detector acquisition time
        curr_acq_time = det.cam.acquire_time.get()
        # calculate a new acquisition time based on the desired max counts
        new_acq_time = round(sat_count / curr_max_counts * curr_acq_time, 4)
        print(new_acq_time) 
        # set the detector to the new value
        yield from bps.mv(det.cam.acquire_time, new_acq_time)

def stub_filter_opt_count(det, motor, ranges, target_count=1000, det_key='dexela_image' ,md={}):
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
    
    # close out run
    yield from bps.close_run()
    yield from bps.unsubscribe(token)
    yield from bps.unstage(det)


# testing out some acquisition time tuning
def tune_acq_time(det,motor,ranges,target_counts):
    # copying what robert did for filter_opt_count first
    old_time = det.cam.acquire_time.get()
    
    # take a rocking scan
    yield from rock(dexDet,motor,ranges)
    data = run.primary.read()[det_key][-1].values

    # grab the number of saturated pixels
    sat_count = np.sum(data>16380)

    new_time = old_time * target_count / old_counts
    print(f'old time: {old_time} -> new_time: {new_time}')

    yield from bps.mv(det.cam.acquire_time, new_time)

    yield from rock_stub([det], motor, ranges)

    # grab the new image and count the saturated pixels
    data = run.primary.read()[det_key][-1].values
    sat_count = np.sum(data>16380)
    print(f'Saturated Pixels: {sat_count}')

def old_filter_opt_count(det, target_count=100000, det_key='dexela_image' ,md={}):
    """ filter_opt_count
    OPtimize counts using filters 
    Assumes mu=0.2, x = [0.89, 2.52, 3.83, 10.87]
    I = I_o \exp(-mu*x) 

    Only takes one detector, since we are optimizing based on it alone
    target is mean+2std
    """
    dc = DocumentCache()
    token = yield from bps.subscribe('all', dc)
    yield from bps.stage(det)

    md = {}
    
    yield from bps.open_run(md=md)    
    # BlueskyRun object allows interaction with documents similar to db.v2, 
    # but documents are in memory
    run = BlueskyRun(dc)
    yield from bps.trigger_and_read([det, filter1, filter2, filter3, filter4])

    data = run.primary.read()[det_key]
    mean = data[-1].mean().values.item() # xarray.DataArray methods
    std = data[-1].std().values.item() # xarray.DataArray methods
    curr_counts = mean + 2*std
    print(curr_counts)


    # gather filter information and solve 
    filter_status = [round(filter1.get()/5), round(filter2.get()/5), 
                        round(filter3.get()/5), round(filter4.get()/5)]
    print(filter_status)
    filter_status = [e for e in filter_status]
    new_filters = solve_filter_setup(filter_status, curr_counts, target_count,x = [0.025,0.051,0.127,0.399])
    # invert again to reflect filter status
    print(new_filters)
    # set new filters and read.  For some reason hangs on bps.mv when going high
    filter1.put(new_filters[0]*4.9)
    filter2.put(new_filters[1]*4.9)
    filter3.put(new_filters[2]*4.9)
    filter4.put(new_filters[3]*4.9)
    
    yield from bps.trigger_and_read([det, filter1, filter2, filter3, filter4])

    # close out runl
    yield from bps.close_run()
    yield from bps.unsubscribe(token)
    yield from bps.unstage(det)


def solve_filter_setup(curr_filters, curr_counts, target_counts, 
                        x=filter_thicks, mu=0.8):
    """solve_filter_setup 
    
    Return filter setup given filter thicknesses (x), and attenuation coeff (mu)
    curr_filters: boolean array

    Operates with 1 = filter in
    ... boolean values are flipped.  signal: 1=filter out, math: 1=filter in
    """
    # get I_o from detector
    x_curr = np.dot(curr_filters, x)
    I0 = curr_counts / np.exp(-mu * x_curr)

    # solve for desired x
    x_new = -np.log(target_counts / I0) / mu

    # return closest arrangement of filters
    # due to small number of possible combinations, can just enumerate
    # max=2^4-1=15

    filter_cfgs = list(range(16))
    filter_cfgs = [int_to_bool_list(k) for k in filter_cfgs]

    I_new = [I0*np.exp(-mu * np.dot(x, k)) for k in filter_cfgs]

    index = np.argmin(np.abs(np.array(I_new)-target_counts))

    return filter_cfgs[index]

def int_to_bool_list(num):
    """int_to_bool_list 
    credits to https://stackoverflow.com/questions/33608280/convert-4-bit-integer-into-boolean-list
    """
    bin_string = format(num, '04b')
    return [x=='1' for x in bin_string[::-1]]

def bool_list_to_int(filt_list):
    """bool_list_to_int
    for our specific backwards filter list
    """
    return 1*filt_list[0] + 2*filt_list[1] + 4*filt_list[2] + 8*filt_list[3]


def opt_filters():
    """
    make decorator? or just run as plan prior to each collection? 
    """


    pass

def solve_mu(hist=filter_hist):
    """
    Return average mu from most recent hour of measurements?  
    ? how to account for filter box changes?  Zero out measurements?  

    --> assume all measurements are from same filter box, shouldn't see any big jumps
    """
    
    mu = hist['mu'].mean()

    if mu is np.nan:
        raise ValueError('no measurements taken with current filters')
    
    return mu



def SNR_opt():
    """ ???? Maybe?
    """
