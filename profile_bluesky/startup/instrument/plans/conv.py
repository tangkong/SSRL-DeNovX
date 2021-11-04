__all__ = ['dist','wmx','wmy','pin1','calib1','guess','calibration']

# convenience functions and definitions for DeNovX beamtime Oct2021
dist = c_stage.detz.user_readback.get() # current detector distance from diffraction center
wmx = c_stage.cx.user_readback.get() # current x motor position
wmy = c_stage.cy.user_readback.get() # current y motor position

pin1 = c_stage.loc(['pin1'])[0]
calib1 = c_stage.loc(['calib1'])[0]


def guess(*vals):
    guess = []
    for val in vals:
        guess.append(val)

# calibration = [center x, center y, wavelength, distance,tilt, rotation]
calibration = [119.43,44.674,0.7293,454.41,0.514,280.99]
