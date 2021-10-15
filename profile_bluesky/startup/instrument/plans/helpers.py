"""
Helper plans, functions

"""
from ..framework.initialize import db
from ..devices.misc_devices import filter1, filter2, filter3, filter4
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['show_table', 'show_image', 'show_scan', 'avg_images', 'filters','inscribe','generate_rocking_range']

def show_table(ind=-1):
    return db[ind].table()

def show_image(ind=-1, data_pt=1, img_key='pilatus300k_image', max_val=500000):
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
    if img_key in ['pilatus300k_image']:
        horizontal=True
    else:
        horizontal=False

    try:
        hdr = db[ind].table(fill=True)
        arr = hdr[img_key][data_pt][0]
        if horizontal:
            arr = np.rot90(arr, -1)
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


def avg_images(ind=-1,img_key='marCCD_image'):
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
    :return xacc,yacc: mask of valid motor positions
    :type xacc,yacc: lists of floats
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
            ypos[ind1,ind2] = ii
            xpos[ind1,ind2] = jj

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
def generate_rocking_range(mask,motor):
    """
    this function takes an array of motor positions and calculates ranges for a rocking scan
    it can also optionally generate a dictionary for staging the motors before each scan
    :param mask: valid motor positions for the sample
    :type mask: list created by inscribe()
    :return ranges,stage: list of [min.max] values to rock across, dictionary of staging positions
    """

    range = []
    stage = {'1':[],'2':[]}

    for ind,row in enumerate(mask.mask):
        # find valid indices
        valid = np.where(row == 1)[0]

        if len(valid) == 0:
            continue

        mval = []
        for val in valid:
            mval.append(mask.cx[ind][val])

        # find the range to rock across
        minn = min(mval)
        maxx = max(mval)

        range.append([minn,maxx])
        
        # get indices for start location
        lind = np.where(mask.cx[ind] == minn)[0]
        print(lind)
        
        # now update the stage object
        stage['1'].append(mask.cx[ind][lind])
        stage['2'].append(mask.cy[ind][lind])

      
    return range,stage
