"""
start Bluesky in Ipython Session
"""
print('starting denovx profile')
from instrument.collection import *
print('imported')

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
