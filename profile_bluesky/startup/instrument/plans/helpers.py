"""
Helper plans, functions for use in plan orchestration

"""

from ..devices.misc_devices import filter1, filter2, filter3, filter4
from ..devices.stages import c_stage

__all__ = ['filters', 'inscribe', 'generate_rocking_range', 'wmx', 'wmy',
           'dist','box','calibration']



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

#spec wmx
def wmx():
    return c_stage.cx.user_readback.get()

def wmy():
    return c_stage.cy.user_readback.get()

def wm():
    return [c_stage.cx.user_readback.get(), c_stage.cy.user_readback.get()]

def dist():
    return c_stage.detz.user_readback.get()

box = [0.548,0.156]
### detector calibration parameters
### calibration = [pixelsize,center x, center y, lambda, distance, tilt, rot]
#calibration = [0.0748,120.022/0.0748,144.963/0.0748,0.72553,434.513,-1.076,317.08]

calibration = [0.0748,120.08/0.0748,152.856/0.0748,0.7293,633.725,-0.648,312.76]
