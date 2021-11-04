"""
Helper plans, functions

"""
from ..framework.initialize import db
from ..devices.misc_devices import filter1, filter2, filter3, filter4
from ..devices.stages import c_stage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import pyFAI
import cv2

__all__ = ['peak_id','qchi','show_table', 'show_image', 'show_stats','show_sum_stats', 'show_scan','show_sum_images', 'avg_images','sum_images', 'filters','inscribe','generate_rocking_range',
           'hthresh','data_reduction','wmx','wmy','dist','box','calibration','get_stats','cossim','imthresh','run_summary','hough_bkg']

def show_table(ind=-1):
    return db[ind].table()

def show_image(ind=-1, data_pt=1, img_key='dexela_image', max_val=16000):
    """show_image attempts to plot area detector data, plot a vertical slice, 
    and calculates number of pixels above the provided threshold.  

    :param ind: Index of run, -1 referring to most recent run. defaults to -1
    :type ind: int, optional
    :param data_pt: row of run to plot, defaults to 1
    :type data_pt: int, optional
    :param img_key: column name holding image data, defaults to 'marCCD_image'
    :type img_key: str, optional
    :param max_val: max pixel threshold, defaults to 60000
    :type max_val: int, optional
    """
    # Try databroker v2 maybe, at some point
    #if img_key in ['pilatus300k_image']:
    #    horizontal=True
    #else:
    #    horizontal=False

    try:
        hdr = db[ind].table(fill=True)
        arr = hdr[img_key][data_pt][0]
        #if horizontal:
        #    arr = np.rot90(arr, -1)
    except KeyError:
        print(f'{img_key} not found in run: {ind}, data point: {data_pt}')
        return

    fig, axes = plt.subplots(1,2, sharey=True, figsize=(7, 4.9),
                            gridspec_kw={'width_ratios': [3,1]})
    
    vmax = np.mean(arr)+3*np.std(arr)
    n_max = np.sum(arr>max_val)
    
    axes[0].imshow(arr, vmax=vmax)
    axes[0].text(100,100, f'{n_max} pixels > {max_val}', 
                    backgroundcolor='w')
    
    scan_no = db[ind].start['scan_id']
    axes[0].set_title(f'{img_key}, Scan #{scan_no}, data point: {data_pt}')

    height, width = arr.shape
        
    sl = arr[:, int(0.45*width):int(0.55*width)]
    axes[1].plot(sl.sum(axis=1), list(range(height)))
    plt.tight_layout()

def show_stats(ind=-1, data_pt=1, img_key='dexela_image', max_val=16000):
    try:
        hdr = db[ind].table(fill=True)
        arr = hdr[img_key][data_pt][0]
        #if horizontal:
        #    arr = np.rot90(arr, -1)
    except KeyError:
        print(f'{img_key} not found in run: {ind}, data point: {data_pt}')
        return

    fig, axes = plt.subplots(1,3, figsize=(12, 4.9))

    vmax = np.mean(arr)+3*np.std(arr)
    n_max = np.sum(arr>max_val)
    axes[0].imshow(arr, vmax=vmax)
    axes[0].text(100,100, f'{n_max} pixels > {max_val}', 
                    backgroundcolor='w')
    
    scan_no = db[ind].start['scan_id']
    axes[0].set_title(f'{img_key}, Scan #{scan_no}, data point: {data_pt}')


    axes[1].boxplot(arr.flatten())
    axes[1].set_ylabel('Intensity')
    axes[2].hist(arr.flatten(), bins=256)
    axes[2].set_ylabel('counts')
    axes[2].set_xlabel('intensity bin')
    plt.tight_layout() 

def show_sum_stats(ind=-1,data_pt=1,img_key='dexela_image',max_val=16000):
    try:
        arr = []
        imgs = db[ind].table(fill=True)
        for img in imgs[img_key]:
            arr.append(img[0])
        arr = np.sum(arr,axis=0)
    except KeyError:
        print(f'{img_key} not found in run: {ind}')

    # remove dead pixels (set to 0)
    arr = dead_mask(arr)

    fig,axes = plt.subplots(1,3,figsize=(12,5))

    vmax = np.mean(arr)+3*np.std(arr)
    n_max = np.sum(arr>max_val)
    axes[0].imshow(arr,vmax=vmax)
    axes[0].text(100,100,f'{n_max} pixels > {max_val}',
            backgroundcolor='w')

    scan_no = db[ind].start['scan_id']
    axes[0].set_title(f'{img_key}, Scan #{scan_no}')
    axes[1].boxplot(arr.flatten())
    axes[1].set_ylabel('Intensity')
    axes[2].hist(arr.flatten(),bins=256)
    axes[2].set_ylabel('counts')
    axes[2].set_xlabel('intesnity bin')
    plt.tight_layout()


def show_scan(ind=-1, dep_subkey='channel1_rois_', indep_subkey='s_stage'):
    """show_scan attempts to plot tabular data.  Looks for dependent and 
    independent variables based on provided subkeys

    :param ind: Index of run, -1 referring to most recent run., defaults to -1
    :type ind: int, optional
    :param dep_subkey: dependent variable search term, defaults to 'channel1_rois_roi01'
    :type dep_subkey: str, optional
    :param indep_subkey: independent variable search term, defaults to 's_stage'
    :type indep_subkey: str, optional
    """
    df = db[ind].table()
    scan_no = db[ind].start['scan_id']

    # grab relevant keys
    dep_keylist = df.columns[[dep_subkey in x for x in df.columns]]
    dep_key = dep_keylist[0]
    indep_keylist = df.columns[[indep_subkey in x for x in df.columns]]
    indep_key = indep_keylist[0]
    
    try:
        fig, ax = plt.subplots()
        ax.set_title(f'Scan #{scan_no}')
        
        for key in indep_keylist:
            df.plot(indep_key, dep_key, marker='o', figsize=(8,5))

    except KeyError:
        print(e)
        return


def avg_images(ind=-1,img_key='Dexela'):
    """avg_images [summary]

    [extended_summary]

    :param ind: Run index, defaults to -1.  
                If negative integer, counts backward from most recent run (-1=most recent, -2=second most recent)
                If positive integer, matches 'scan_id'
                If string, interprets as the start of a UID (ex: '8ee443d')
    :type ind: int, optional
    :param img_key: [description], defaults to 'marCCD_image'
    :type img_key: str, optional
    :return: [description]
    :rtype: [type]
    """
    ''' Tries to average images inside a run.  
    Currently assumes MarCCD format 
    returns the array after.'''

    df = db[ind].table(fill=True)

    #gather images
    arr = []
    for i in range(len(df)):
        arr.append(df[img_key][i+1][0])

    avg_arr = np.sum(arr, axis=0) / len(arr)

    # plot
    fig, axes = plt.subplots(1,2, sharey=True, figsize=(7, 4.9),
                            gridspec_kw={'width_ratios': [3,1]})
    
    vmax = np.mean(avg_arr)+3*np.std(avg_arr)

    axes[0].imshow(avg_arr, vmax=vmax)
    
    scan_no = db[ind].start['scan_id']
    axes[0].set_title(f'Scan #{scan_no}, averaged over {len(arr)} images')

    sl = avg_arr[:, 900:1100]
    axes[1].plot(sl.sum(axis=1), list(range(2048)))
    plt.tight_layout()

    return avg_arr


def sum_images(ind=-1, img_key='dexela_image'):
    """sum_images [summary]

    [extended_sum5       2021-10-30 02:05:02.915573359  466b5958-8ad2-45a9-b3e8-23a4b72d3b5f/4                     0             0             0             4             0
mary]

    :param ind: Run index, defaults to -1.
                If negative integer, counts backward from most recent run (-1=most recent, -2=second most recent)
                If positive integer, matches 'scan_id'
                If string, interprets as the start of a UID (ex: '8ee443d')
    :type ind: int, optional
    :param img_key: [description], defaults to 'marCCD_image'
    :type img_key: str, optional
    :return: [description]
    :rtype: [type]
    """
    ''' Tries to sum images inside a run.  
    Currently assumes Dexela format 
    returns the array after.'''

    df = db[ind].table(fill=True)

    # gather images
    arr = []
    for i in range(len(df)):
        arr.append(df[img_key][i + 1][0])

    arr = np.sum(arr,axis=0,dtype='uint16')

    return arr

def show_sum_images(ind=-1,img_key='dexela_image',max_val=16000):

    # get the image summ
    sarr = sum_images(ind,img_key)

    # create a plot object
    fig, axes = plt.subplots(1,2, sharey=True, figsize=(7, 4.9),
                            gridspec_kw={'width_ratios': [3,1]})

    # set the colormap max 
    vmax = np.mean(sarr)+3*np.std(sarr)

    # and plot
    axes[0].imshow(sarr, vmax=vmax)

    plt.tight_layout()



# function for setting filter box
def filters(new_vals=None):
    """
    :param new_vals: new vals
    :type new_vals: list?
    """
    if new_vals:
        filter1.put(new_vals[0]*4.9)
        filter2.put(new_vals[1]*4.9)
        filter3.put(new_vals[2]*4.9)
        filter4.put(new_vals[3]*4.9)
    print(f'filter1: {filter1.get():.1f}')
    print(f'filter2: {filter2.get():.1f}')
    print(f'filter3: {filter3.get():.1f}')
    print(f'filter4: {filter4.get():.1f}')

# calculating the boundaries on a sample and inscribing motor movement
# inside of that 2D area
# currently this only works for a circular sample and a box beam
def inscribe(motor1,motor2,C,dia,box,res):
   
    """
    TODO: write up an explanation of your math/logic
    :param center: list of sample center coordinates
    :type center: list
    :param dia: sample diameter
    :type dia: float
    :param box: width and height of box beam
    :type box: list
    :param res: number of points to generate along each direction
    :type res: list of size for each dimension
    :return mmask: mask of valid motor positions
    :rtype mmask: 2d array
    :return mpos: motor positions corresponding to mmask
    :rtype mpos: dict
    """

    # convert the C list to an array because
    C = np.array(C)

    # calculate how many boxes it takes to cover the area of the circle
    # if the box is w wide and h tall then to cover a circle of
    # diameter d you will need n*w boxes wide and m*h boxes tall (english?)
    # calculate n=d/w and m=d/h
    # keep integer values at the end
    w = box[0]
    h = box[1]
    n = np.int(np.ceil(dia/w))
    m = np.int(np.ceil(dia/h))
    # we want to center the box on the circle center so keep it even
    if n % 2 != 0:
        n += 1
    if m % 2 != 0:
        m += 1

    # this is not going to be pretty
    # generate an array for the inscribing box
    # calculate start and end points
    xRange = [C[0]-(w*n)/2,C[0]+(w*n)/2]
    yRange = [C[1]-(h*m)/2,C[1]+(h*m)/2]

    # create points along the ranges
    xarr = np.linspace(xRange[0],xRange[1],num=res[0],endpoint=True)
    yarr = np.linspace(yRange[0],yRange[1],num=res[1],endpoint=True)

    # array to store all of the (in)valid motor positions
    mask = np.zeros([res[0],res[1]])
    xpos = np.zeros([res[0],res[1]])
    ypos = np.zeros([res[0], res[1]])

    # now step through the points on the grid and check if the corners of the box fall outside the circle radius
    for ind1,ii in enumerate(xarr):
        for ind2,jj in enumerate(yarr):
            # place the motor positions at that location
            xpos[ind2,ind1] = ii
            ypos[ind2,ind1] = jj

            # find the corner of the box
            bnds = [C - [ii-w/2,jj-h/2], # bottom left corner
                    C - [ii-w/2,jj+h/2], # bottom right corner
                    C - [ii+w/2,jj-h/2], #top left corner
                    C - [ii+w/2,jj+h/2] #top right corner
                    ]

            test = [0,0,0,0] # so stupid, find a better way plz
            for ind,bnd in enumerate(bnds):
                # check if the corners of the box are outside the radius
                if np.linalg.norm(bnd) > dia/2:
                    test[ind] = 1

            if 1 in test:
                mask[ind1,ind2] = 0
            else:
                mask[ind1,ind2] = 1


    # create a dictionary with the motors and positions
    motorpos = {motor1:xpos,motor2:ypos}

    return mask, motorpos

# take an inscribed area and generate rocking scan positions
def generate_rocking_range(rmotor, mask,mpos,transpose = False):
    """
    this function takes an array of motor positions and calculates ranges for a rocking scan
    it can also optionally generate a dictionary for staging the motors before each scan
    :param mask: valid motor positions for the sample
    :type mask: l    try:
        hdr = db[ind].table(fill=True)
        arr = hdr[img_key][data_pt][0]
        #if horizontal:
        #    arr = np.rot90(arr, -1)
    except KeyError:
        print(f'{img_key} not found in run: {ind}, data point: {data_pt}')
        return

    fig, axes = plt.subplots(1,2, sharey=True, figsize=(7, 4.9),
                            gridspec_kw={'width_ratios': [3,1]})

    vmax = np.mean(arr)+3*np.std(arr)
    n_max = np.sum(arr>max_val)

    axes[0].imshow(arr, vmax=vmax)
    axes[0].text(100,100, f'{n_max} pixels > {max_val}',
                    backgroundcolor='w')

    scan_no = db[ind].start['scan_id']
    axes[0].set_title(f'{img_key}, Scan #{scan_no}, data point: {data_pt}')

    height, width = arr.shape

    sl = arr[:, int(0.45*width):int(0.55*width)]
    axes[1].plot(sl.sum(axis=1), list(range(height)))
    plt.tight_layout()
ist created by inscribe()
    :return ranges,stage: list of [min.max] values to rock across, dictionary of staging positions
    """

    ranger = []
    stage  = {}
    for motor in mpos:
        stage[motor] = []

    # take transpose if asked
    if transpose:
        mask = np.transpose(mask)
        for motor in mpos:
            mpos[motor] = np.transpose(mpos[motor])

    # go through rows (columns) of the mask
    for ind,row in enumerate(mask):

        # find valid indices that equal 1
        valid = np.where(row == 1)[0]
        # if no valid points, move to the next row
        if len(valid) == 0:
            continue

        # create a list to store all the valid motor positions
        # use the indices from the mask to get the motor position values from mpos
        # find the range to rock across
        minn = min(mpos[rmotor][ind][valid])
        maxx = max(mpos[rmotor][ind][valid])

        # update the range list
        ranger.append([minn,maxx])
        
        for motor in mpos:
            #get index for staring motor value
            lind = np.where(mpos[rmotor][ind] == minn)[0]
            stage[motor].append(mpos[motor][ind][lind])

    return ranger,stage

def data_reduction(imArray, calib,
                   QRange=None, ChiRange=None):
    """
    @author: fangren
    The input is the raw file's name and calibration parameters
    return Q-chi (2D array) and a spectrum (1D array)
    :param imArray: image array
    :type imArray: 2d image array
    :param d: detector distance
    :param Rot: detector rotation
    :param tilt: detector tilt
    :param lamda: wavelength (angstroms)
    :param x0: x pixel center
    :param y0: y pixel center
    :param PP: polarization factor
    :param pixelsize: detector pixelsize
    :type all above: float
    """

    # get the calibration parameters
    # defaults to the calibration defined in helpers.py
    ### calibration list:
    ### 0--pixelsize
    ### 1--center x
    ### 2--center y
    ### 3--lambda
    ### 4--distance
    ### 5--tilt
    ### 6-- rot
    d = calib[4]
    Rot = calib[6]
    tilt = calib[5]
    lamda = calib[3]
    x0 = calib[1]
    y0 = calib[2]
    PP = 0 # not refined in calibration
    pixelsize = calib[0]


    s1 = int(imArray.shape[0])
    s2 = int(imArray.shape[1])
    imArray = signal.medfilt(imArray, kernel_size=5)

    #detector_mask = np.ones((s1, s2)) * (imArray <= 0)
    p = pyFAI.AzimuthalIntegrator(wavelength=lamda)

    # refer to http://pythonhosted.org/pyFAI/api/pyFAI.html for pyFAI parameters
    p.setFit2D(d, x0, y0, tilt, Rot, pixelsize, pixelsize)

    # the output unit for Q is angstrom-1.  Always integrate all in 2D
    cake, Q, chi = p.integrate2d(imArray, 1000, 1000)

    # pyFAI output unit for Fit2D gemoetry incorrect. Multiply by 10e8 for correction
    Q = Q * 10e8

    # create azimuthal range from chi values found in 2D integrate
    # modify ranges to fit with detector geometry
    centerChi = (np.max(chi) + np.min(chi)) / 2
    if (QRange is not None) and (ChiRange is not None):
        azRange = (centerChi + ChiRange[0], centerChi + ChiRange[1])
        radRange = tuple([y / 10E8 for y in QRange])
        print(azRange, radRange)
    else:
        azRange, radRange = None, None

    Qlist, IntAve = p.integrate1d(imArray, 1000)
                                  #azimuth_range=azRange, radial_range=radRange,
                                  #mask=detector_mask, polarization_factor=PP)

    # the output unit for Q is angstrom-1
    Qlist = Qlist * 10e9

    # shift chi from 2D integrate
    chi = chi - centerChi

    return cake, Q, chi, Qlist, IntAve

# function that takes an image array and returns an image array with dead pixelsmasked out
# right now it's hardcoded because I only have 1 dead pixel that I know of
def dead_mask(arr):
    # assumes the shape 3888 X 3072
    narr = np.copy(arr)
    narr[851,1998] = 1
    narr[3704,2246] =1
    return narr


# plotting up statistics from get_stats in hitp_scans
def get_stats(inds = range(1,5*12+1)):
    # grabs some image statistics from the last n runs
    # right now defaults to the last thirty five  runs to match test_stats()

    # define the exposure times -- this is hardcoded in test_stats() atm
    etime = []
    etimes = [5*[10],5*[7],5*[5],5*[3],5*[2],5*[1],5*[0.75],5*[0.5],5*[0.2],5*[0.1],5*[0]]
    for test in etimes:
        for time in test:
            etime.append(time)

    # some of the stats we want to calculate
    max_pixel = []
    mean_pixel = []
    min_pixel = []
    med_pixel = []
    
    # iterate through the header and grab the images
    for ind in inds:
        img = db[-ind].table(fill=True)['dexela_image'][1][0]
        max_pixel.append(np.max(dead_mask(img))) #I'm using dead_mask() for the max pixel so that we're not biased by dead pixels
        mean_pixel.append(np.mean(img))
        min_pixel.append(np.min(img))
        med_pixel.append(np.median(img[0]))

    # generate some plots
    fig,ax = plt.subplots(1,2,figsize=(10,5))


    ax[0].plot(etime,mean_pixel,'*')
    ax[0].set_xlabel('Exposure time (s)')
    ax[0].set_ylabel('Mean Pixel Intensity')

    ax[1].plot(etime,med_pixel,'*')
    ax[1].set_xlabel('Exposure time (s)')
    ax[1].set_ylabel('Median Pixel Intensity')
    plt.tight_layout()

    plt.show()

def cossim(A,B):
    # takes the cosine similarity of two vectors A and B
    # wikipedia has a good page on it for reference
    # cos(A,B) = ||A.B||^2 / (||A|| ||B||)
    numer = np.sum(np.multiply(A,B))
    normA = round(np.sqrt(np.sum(np.multiply(A,A))),5)
    normB = round(np.sqrt(np.sum(np.multiply(B,B))),5)
    return round(numer/(normA*normB),5)

def imthresh(img,thresh):
    # takes an image, performs a binary threshold around thresh, returns the binary array
    ret,thim = cv2.threshold(img,thresh,16383,cv2.THRESH_BINARY)
    return thim

def qchi(ind=-1,img_key='dexela_image'):
    
    # if given a run with multiple image files, it will compute the sum
    # gather images
    df = db[ind].table(fill=True)

    arr = []
    for i in range(len(df)):
        arr.append(df[img_key][i + 1][0])
    sarr = np.sum(arr,axis=0)

    # integrate
    qc = data_reduction(sarr,calibration)

    vmax = np.mean(qc[0]) + 3*np.std(qc[0])

    # make a plot
    fig,ax = plt.subplots(1,2)
    ax[0].pcolormesh(qc[1],qc[2],qc[0],vmax=vmax)
    ax[0].set_xlabel('q (' + r'$\AA^{-1]}$)')
    ax[0].set_ylabel(r'$\chi$')

    ax[1].plot(qc[3],qc[4])
    ax[1].set_xlabel('q (' + r'$\AA^{-1]}$)')
    ax[1].set_ylabel(r'$\chi$')
    plt.show()

    return qc

def hough(q,I):
    norm = np.zeros(len(q))
    ang = np.zeros(len(I))
    # compute the hough transform
    for ii in np.arange(len(q)):
        norm[ii] = np.sqrt(q[ii]**2 + I[ii]**2)
        ang[ii] = np.sin(I[ii]/norm[ii])


    # creating an array for storing values of the lower bound
    store = np.zeros(len(q))
    store[0] = ang[0]
    store[-1] = ang[-1]
    count = 0

    # if the angle of the vector increases then we've moved off the boundary
    for ii, item in enumerate(ang[1:-2]):

        if item < store[count]:
            store[ii] = item
            count = ii

    # fill in the zeros
    z = np.where(store != 0)[0]
    # store the q values of the boundary and their angles
    lbound = np.array([q[z],store[z]])

    # fit a curve to the line
    f = interp1d(lbound[0],lbound[1])

    # create an array of len(x) that is fit with the spline
    xnew = np.array(q)
    bound = f(xnew)

    # transform back to q vs Intensity space
    nI = np.zeros(len(xnew))
    for ii in range(0,len(xnew)):
        nI[ii] = np.sqrt( ((xnew[ii]**2) * np.arcsin(bound[ii])**2) / (1- np.arcsin(bound[ii])**2))

    return(nI)

def hough_bkg(q,I):
    # compute and re-compute the hough transform at multiple fixed points
    # and combine the best fit from each transform
    # we will be moving our fixed point along the q axis

    # determine how many points to sample
    # if you sample every point in Q, you will overfit
    # so this is very hacky at the moment
    start = 0
    stop = len(q)
    step = 10

    # scale the intensity
    I = I + 1000

    # create a matrix computing the transform at each point
    houghim = np.zeros([len(q[start:stop:step]),len(q)])

    # compute and store the hough transform for each fixed point
    for ind,qq in enumerate(q[start:stop:step]):
        # first compute the hough trasnform from (0,0)
        houghim[ind,:] = hough(q-qq,I)

    # find the upper bound of each hough transform
    bound = np.max(houghim,axis=0)

    return(I - bound - 1000)

def hthresh(I,thresh):
    # take a histogram, define a threshold value, perform a binary threshold, and return the new array

    # find where the intensities are greater than thresh
    tind = np.where(I > thresh)[0]

    tI = np.zeros(len(I))
    tI[tind] = 16383

    return(tI)

def peak_id(run,thresh):
    # grab the images
    imgs = run.table(fill=True)['dexela_image']
    arr = []
    for img in imgs:
        arr.append(img[0])
    arr = np.sum(arr,axis=0)

    #get the qc image
    qc = data_reduction(arr,calibration)
    
    # compute the background fit
    nI = hough_bkg(qc[3],qc[4])
    
    # now threshold hte histogram
    tI = hthresh(nI,thresh)
    
    # find the peaks
    pinds = np.where(tI == 16383)[0]
    
    # now return the q values
    return (qc[3][pinds])

# pull data from a run, grab a bunch of plots, plot them,save into a pdf
def run_summary(runs,name, outputPath='/bluedata/b_mehta/export_DeNovX/summary/'):

    # first grab the uid
    uid = runs.start['uid']

    with PdfPages(str(name) + '_summary.pdf') as pdf:
        firstPage = plt.figure(figsize=(11.69,8.27))
        firstPage.clf()
        # grab all relevant metadata
        start = runs.start
        ind = 0
        if 'uid' in start:
            firstPage.text(1,ind,
                    'uid: ' + str(start['uid']),
                    transform=firstPage.transFigure, size=24)
            ind+=1
        
        if 'scan_id' in start:
            firstPage.text(1,ind,
                    'scan_id: ' + str(start['scan_id']),
                    transform=firstPage.transFigure, size=24)
            ind+=1
        if 'proposal_id' in start:
            firstPage.text(1,ind,
                'proposal_id: ' + str(start['proposal_id']),
                transform=firstPage.transFigure, size=24)
            
            ind+=1
        if 'plan_type' in start:
            firstPage.text(1,ind,
                'plan_type: ' + str(start['plan_type']),
                transform=firstPage.transFigure, size=24)
            ind+=1
        if 'plan_name' in start:
            firstPage.text(1,ind,
                'plan_name: ' + str(start['plan_name']),
                transform=firstPage.transFigure, size=24)
            ind+=1
        if 'macro_name' in start:
            firstPage.text(1,ind,
                'macro_name: ' + str(start['macro_name']),
                transform=firstPage.transFigure, size=24)
            ind+=1
            
        #close the first page
        pdf.savefig(firstPage)
        plt.close()
        
        # now start generating plots
        
        # plot the images
        if 'dexela_image' in runs.table(fill=True):
            #open a new page
            page,ax = plt.subplots(1,3,figsize=(11.69,8.27))
            
            # get the sum of all images in the run
            sarr = sum_images(ind=uid)
            ax[0].imshow(sarr,vmax = np.mean(sarr)+3*np.std(sarr))
            ax[0].set_title('Sum Image')

            # get the q-chi transformation
            qc = data_reduction(sarr,calibration)

            vmax = np.mean(sarr)+3*np.std(sarr)

            # q-chi plot
            ax[1].pcolormesh(qc[1],qc[2],qc[0])
            ax[1].set_xlabel('q (' + r'$\AA^{-1}$)')
            ax[1].set_ylabel(r'$\chi$')

            # plot the histogram
            ax[2].plot(qc[3],qc[4],'k-')
            ax[2].set_xlabel('q (' + r'$\AA^{-1}$)')
            ax[2].set_ylabel('Intensity')
            plt.close()
            pdf.savefig(page)

            # now plot some statistics
            
            page,ax = plt.subplots(1,2,figsize=(11.69,8.27))

            vmax = np.mean(sarr)+3*np.std(sarr)
            n_max = np.sum(sarr>16383)

            if 'scan_no' in start:
                scan_no = start['scan_no']
            else:
                scan_no = ''
            ax[0].set_title(f'Scan #{scan_no}')
            ax[0].boxplot(sarr.flatten())
            ax[0].set_ylabel('Intensity')
            ax[1].hist(sarr.flatten(),bins=256)
            ax[1].set_ylabel('counts')
            ax[1].set_xlabel('intesnity bin')
            #plt.tight_layout()
            plt.close()
            pdf.savefig(page)





#spec wmx
def wmx():
    return c_stage.cx.user_readback.get()

def wmy():
    return c_stage.cy.user_readback.get()

def dist():
    return c_stage.detz.user_readback.get()

box = [0.548,0.156]
### detector calibration parameters
### calibration = [pixelsize,center x, center y, lambda, distance, tilt, rot]
#calibration = [0.0748,120.022/0.0748,144.963/0.0748,0.72553,434.513,-1.076,317.08]

calibration = [0.0748,120.08/0.0748,152.856/0.0748,0.7293,633.725,-0.648,312.76]
