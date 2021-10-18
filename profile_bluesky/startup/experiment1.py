# this is a development script for testing out functionalities that have been written
# you can think of it as virtually stepping through a real diffraction experiment

# first we want to get the stage aligned
# make some simple definitions
cx = c_stage.cx
cy = c_stage.cy

# give a rough estimate for where the cradle pinhole is
guess = [0.1,0.1]

# now use find_coords to align the frame based on your guess location
# need to assign a detector key for the databroker
detKey = 'Dexela'
yield from find_coords(dexDet,cx,cy,guess,detKey) # stage locations should be aligned now

# move to the first calibrant location
calib1 = c_stage.loc('calib1')[0]
yield from bps.mv(cx,calib1[0],cy,calib1[1])

# set the filter box based on the calibrant
# define a target count value
tVal = 10e6
yield from filter_opt_count(dexDet,target_count=tVal)

# set the acquisition time based on the calibrant
# define a saturation count
sVal = 6*10e6
yield from max_pixel_count(dexDet,sat_count=sVal)

# inscribe the calibrant material and take a rocking scan
# use calib1 for the sample center
dia = 10 # sample diameter
box = [0.1,0.2] #box beam size
res = [10,20] # desired mesh resolution

# inscribe the motor locations
mmask,mpos = inscribe(cx,cy,calib1,dia,box,res)

# update the motor sample mask
mask.update_mask(mask,mmask,mpos[cx],mpos[cy])

# create ranges to rock across
r,s = generate_rocking_range(mask,cx)

# # perform the rocking scan
# uid = yield from rock([dexDet],cx,r,stage=s)
#
# # grab all the data from the rocking scan
# hdr = db[uid].table(fill=True)
#
# # take the sum of all the data you just collected
# sarr = sum_images(ind=-1,img_key = imgKey)
#
# # integrate the image and plot it up
# # will need to provide calibration parameters on our own
# Q, chi, cake, Qlist, IntAve = data_reduction[sarr,d,Rot,tilt,x0,y0,PP,pixelsize]
#
# # okay now take a slice out of the Q-chi range and plot it
# # define a qrange
# qval = 4
# qind = np.where(Q == qval)[0]
# qd = 0.1 # step size
# qRange = [qind-10,qind+10] # find better way to define index range
# fig,ax = plt.subplots(1)
# ax[0].plot(Q[qRange[0]:qRange[1]],chi[qRange[0]:qRange[1]])
# ax.set_xlabel('Q')
# ax.set_ylabel('chi')





