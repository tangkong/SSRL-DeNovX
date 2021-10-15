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
    :param stage: optional staging positgit ions to execute before each rocking command, defaults to None
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