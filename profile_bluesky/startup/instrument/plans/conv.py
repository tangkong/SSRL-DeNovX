__all__ = ['dist','wmx','wmy','pin1','calib1','guess',]

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

