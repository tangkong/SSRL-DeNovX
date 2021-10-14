def rocking_scan(det, motor, minn, maxx, *, md=None):
    """rocking_scan moves motor between (minn, maxx) continuously during a
    single exposure
    Rocking scan averages a single exposure over several spatial locations.
    This type of plan typically sees use in highly polycrystalline samples during
    calibration scans.
    :param det: detector object
    :type det: subclass of ophyd.areadetector
    :param motor: motor to rock during exposure
    :type motor: EpicsMotor
    :param minn: min value to move to while rocking
    :type minn: float
    :param maxx: max value to move to while rocking
    :type maxx: float
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
    yield from bps.trigger(det, wait=False)
    start = time.time()
    now = time.time()

    while (now-start) < exposure_time:
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