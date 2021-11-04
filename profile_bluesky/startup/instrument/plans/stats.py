## this is a test development script for looking at pixel statistics on the dexela detector

import bluesky.plans as bp
import bluesky.plan_stubs as bps
from ..framework import db

from .helpers import data_reduction, calibration,dead_mask
from ..devices.misc_devices import shutter
from ..devices.misc_devices import filter1,filter2,filter3,filter4

import matplotlib.pyplot as plt
import numpy as np


# take a bunch of dark counts, integrate them, and plot the results
def count_dark(det,num,time):
    # close the shutter and take a bunch of dark counts 
    yield from bps.mv(shutter,0)

    # set the detector time
    det.cam.acquire_time.set(time)

    # list for storing the runs
    runs = []

    n = 0
    while n < num:
        run = yield from bp.count([det],md={'did':'dark'})
        runs.append(run)
        n+=1

    return runs


# integrates and plots the dark count sample
def plot_count_dark(imgs):
    
    fig,ax = plt.subplots(1,1)
    
    # also want to store the sum of all the images to get means/stddev
    sums = []

    #plots the results of count_dark
    for img in imgs:
        arr = db[img].table(fill=True)['dexela_image'][1][0]
        arr = dead_mask(arr)
        ax.hist(arr.flatten(),bins=256,stacked=True,histtype='barstacked')
        print(np.max(arr))
    
    plt.show()
