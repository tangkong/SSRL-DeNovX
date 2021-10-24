# test script for setting filter box based on observed images
# move to a location, take a calibrant scan, plot the data, and set the filters accordingly

def set_filter(det,center,cx,cy,instprm,*,md=None):

    # move to the sample center location
    yield from bps.mv(cx,center[0],cy,center[1])

    # get the current acquisition time
    acqt = det.cam.acquire_time.get()

    # take a standardized scan
    fid = yield from

    # set up a rocking scan based on the sample geometry
    dia = instprm[0] # sample diameter
    box = instprm[1] # box beam size
    res = instprm[2] # xy resolution

    # inscribe the xy positions into the motor
    mmask,mpos = inscribe(motors[0],motors[1],center,dia,box,res)
    # update the mask
    mask.update_mask(mask,mmask,mpos[cx],mpos[cy]) # need to change syntax on this to be dict
    # generate ranges to rock across
    r,s = generate_rocking_range(mmask,cx)

    # do the rock scan
    uid = yield from rock(cx, cy, r[ii], stage=s[ii], md=md)

    # get the data from the databroker
    hdr = db[uid].table(fill=True)

    # going to take the sum of all images
    arr = []

    # set the imae key
    img_key = 'Dexela'

    # get the number of scans based on the length of r
    for ii in len(r):
        # plot the image
        show_image(ind=ii,img_key=img_key) # find the actual image key
        arr.append(hdr[img_key][ii][0])

    # take the sum of all images
    sarr = np.sum(arr,axis=0)

    # integrate the summed image
