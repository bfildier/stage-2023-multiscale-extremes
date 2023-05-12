# -*- coding: utf-8 -*-

##########################################################################
# Histogrammes permettant de choisir les variables les plus pertinentes  #
# pour ensuite les utiliser dans les methodes de clustering              #
##########################################################################

### Importation des bibliotheques
import sys
import matplotlib as mpl

mpl.use("agg")
import time
from math import *
import numpy as np
import gzip
import subprocess

from struct import unpack
from struct import *

# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.basemap import addcyclic

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.gridspec as gridspec
import gzip

from datetime import *
from math import *
from mpl_toolkits.mplot3d import Axes3D
from xml.dom import minidom
from random import randint
from datetime import datetime

##################################################################################
#
#  Class convective system_IntParameters   :
#
#
#
##################################################################################


class MCS_IntParameters(object):
   
    def __init__(self):
        self.label = 0 # Label of the convective system in the segmented images
        self.qc_MCS = 0 # Quality control of the convective system
        self.duration = 0 # duration of the convective system (slot)
        self.classif = 0 # Quality control of the convective system (?duplicate)
        self.Tmax = 0.0 
        self.Utime_Init = 0  # time TU of initiation of the convective system
        self.lonInit = 0 # longitude of the center of mass at inititiation
        self.latInit = 0 # latitude of the center of mass at inititiation
        self.Utime_End = 0 # time TU of dissipation of the convective system
        self.lonEnd = 0 # longitude of the center of mass at dissipation
        self.latEnd = 0 # latitude of the center of mass at dissipation
        self.lonmin = 0 # longitude min of the center of mass during its life cycle
        self.latmin = 0 # latitude min of the center of mass during its life cycle
        self.lonmax = 0 # longitude max of the center of mass during its life cycle
        self.latmax = 0 # latitude max of the center of mass during its life cycle
        self.vavg = 0 # average velocity during its life cycle(m/s)
        self.dist = 0 # distance covered by the convective system during its life cycle(km)
        self.olrmin = 0 # minimum Brigthness temperature (K)
        self.surfmaxPix_172Wm2 = 0 # maximum surface for a 235K threshold of the convective system during its life cycle (pixel) !!!String no match
        self.surfmaxkm2_172Wm2 = 0 # maximum surfacefor a 235K threshold of the convective system during its life cycle (km2) !!!String no match
        self.surfmaxkm2_132Wm2 = 0 # maximum surfacefor a 220K threshold of the convective system during its life cycle (km2) !!!String no match
        self.surfmaxkm2_110Wm2 = 0 # maximum surfacefor a 210K threshold of the convective system during its life cycle (km2) !!!String no match
        self.surfmaxkm2_90Wm2 = 0 # maximum surfacefor a 200K threshold of the convective system during its life cycle (km2) !!!String no match
        self.surfcumkm2_172Wm2 = 0
        self.surfcumkm2_132Wm2 = 0
        self.surfcumkm2_110Wm2 = 0
        self.surfcumkm2_90Wm2 = 0
        self.precip_total = 0 
        self.precip_max = 0 
        self.maxSurf00mmh_km2 = 0
        self.maxSurf02mmh_km2 = 0
        self.maxSurf05mmh_km2 = 0
        self.maxSurf10mmh_km2 = 0
        self.classif_JIRAK = 0 # no description

        ## labels description from previous func not found in class : 
           # local time of inititiation ; 
           # local hour of dissipation ; 

## Ajout B. Fildier ##

    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the
        attribute value when its string fits is small enough."""

        out = '< MCS_IntParameters object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

## Fin ajout B. Fildier ##


class MCS_Lifecycle(object):
    def __init__(self):
        #self.qc_im = [] # quality control on the Infrared image
        self.olrmin = [] # min brightness temperature of the convective system at day TU (K)
        #self.olravg_172Wm2 = [] #  average brightness temperature of the convective system at day TU (K)
        #self.olravg_110Wm2 = [] #  min brightness temperature of the convective system at day TU (K) !!! I think the description was wrong 
        #self.olravg_90Wm2 = [] #  min brightness temperature of the convective system at day TU (K)
        ## MJC start
        self.olravg = []  
        self.slot = [] # I have no clue what it can mean, it wasn't lower_cased from the file header
        self.jcm = []
        self.icm = []
        self.sminor_220k = []
        self.smajor_220k = []
        self.e_220k = []
        self.angle_220k = []
        self.sminor_235k = []
        self.smajor_235k = []
        self.e_235k = []
        self.angle_235k = []
        self.surf235k_pix = []
        self.surf210k_pix = []
        self.surf235k_km2 = []
        self.surf220k_km2 = []
        self.surf210k_km2 = []
        self.surf200k_km2 = []
        self.pr_mean = []
        self.pr_max = []
        self.condpr = []
        self.condpr01mmh = []
        self.surf00mmh_km2 = []
        self.surf01mmh_km2 = []
        self.surf02mmh_km2 = []
        self.surf05mmh_km2 = []
        self.surf10mmh_km2 = []
        ## MJC end
        self.olr_90th = []  #  min brightness temperature of the convective system at day TU (K)
        #self.surfPix_172Wm2 = [] # surface of the convective system at time day TU (pixel)
        #self.surfPix_110Wm2 = [] # surface of the convective system at time day TU (pixel)
        #self.surfKm2 = []
        #self.Utime = [] # day TU
        #self.Localtime = [] # local hour (h)
        #self.lon = [] # longitude of the center of mass (°)
        #self.lat = [] # latitude of the center of mass (°)
        #self.x = [] # column of the center of mass (pixel)
        #self.y = [] # line of the center of mass(pixel)
        self.velocity = [] # instantaneous velocity of the center of mass (m/s)
        #self.semiminor_132Wm2 = [] # no desc
        #self.semimajor_132Wm2 = [] # no desc
        #self.orientation_132Wm2 = [] # no desc
        #self.excentricity_132Wm2 = [] # no desc
        #self.semiminor_172Wm2 = [] # no desc
        #self.semimajor_172Wm2 = [] # no desc
        #self.orientation_172Wm2 = [] # no desc
        #self.excentricity_172Wm2 = [] # no desc
        #self.surfkm2_172Wm2 = [] # surface of the convective system for a 235K threshold
        #self.surfkm2_132Wm2 = [] # surface of the convective system for a 220K threshold
        #self.surfkm2_110Wm2 = [] # surface of the convective system for a 210K threshold
        #self.surfkm2_90Wm2 = [] # surface of the convective system for a 200K threshold

## Ajout B. Fildier ##

    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the
        attribute value when its string fits is small enough."""

        out = '< MCS_Lifecycle object:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if sys.getsizeof(getattr(self,k).__str__()) < 80:
                # show value
                out = out+'%s\n'%str(getattr(self,k))
            else:
                # show type
                out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

## Fin ajout B. Fildier ##

## MJC start ## 
def convert_to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Could not convert {s} to number")
## MJC end ##

def load_TOOCAN_DYAMOND(FileTOOCAN, verbose = False):

    lunit = gzip.open(FileTOOCAN, "rt")
    #print(FileTOOCAN)

    #
    # Read the Header
    ##########################
    header1 = lunit.readline()
    header2 = lunit.readline()
    header3 = lunit.readline()
    header4 = lunit.readline()
    header5 = lunit.readline()
    header6 = lunit.readline()
    header7 = lunit.readline()
    header8 = lunit.readline()
    header9 = lunit.readline()
    header10 = lunit.readline()
    header11 = lunit.readline()
    header12 = lunit.readline()
    header13 = lunit.readline()
    header14 = lunit.readline()
    header15 = lunit.readline()

    header16 = lunit.readline()
    header17 = lunit.readline()
    header18 = lunit.readline()
    header19 = lunit.readline()
    header20 = lunit.readline()
    header21 = lunit.readline()
    header22 = lunit.readline()
    header23 = lunit.readline()
    
    Labels_MCS_IntParameters = header21.split()[:]
    Labels_MCS_Lifecycle = header22.split()[:]
    

    data = []
    iMCS = -1
    lines = lunit.readlines()
        
    ## print all the headers
    if verbose : 
        for i in range(1, 24): print(f"header{i} = {eval(f'header{i}')}")

    for iline in lines:
        Values = iline.split()

        #
        # Read the integrated parameters of the convective systems
        ###########################################################


        if Values[0] == '==>':

            ## print values in front of their label
            if verbose :
                for label, v in zip(Labels_MCS_IntParameters, Values): print( label, ":", v)

            data.append(MCS_IntParameters())
            iMCS = iMCS + 1
            
            for i in range(1, len(Values)):
                setattr(
                        data[iMCS], 
                        Labels_MCS_IntParameters[i],
                        convert_to_number(Values[i])
                        )
            # Instantiate Lifecycle object
            data[iMCS].clusters = MCS_Lifecycle()

        #
        # Read the parameters of the convective systems
        # along their life cycles
        ##################################################
        

        
        else:
            ## print values in front of their label
            if verbose : 
                for label, v in zip(Labels_MCS_Lifecycle, Values): print( label, ":", v)

            for i in range(len(Values)):
                getattr(
                        data[iMCS].clusters, 
                        Labels_MCS_Lifecycle[i].lower()
                        ).append(
                                convert_to_number(Values[i])
                                )            

    # data = MCS_Classif(data)
    # data = Compute_Tmax(data)

    return data