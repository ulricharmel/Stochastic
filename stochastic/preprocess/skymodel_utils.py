import sys
import os

def lsmconvert(skymodel):
	"""
	Converting textfile to lsm.html format
	"""
	os.system("tigger-convert %s -f"%(skymodel))
	return "%s.lsm.html"%(skymodel[:-4])
