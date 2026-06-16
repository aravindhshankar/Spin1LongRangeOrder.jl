import numpy as np 
import cmath

def ret_Vpm_crit(t, U, Vpp):
    '''returns critical value of Vpm from bosonization'''
    vacrit = (0.5 * U) - np.pi * np.abs(t)  #approximate value from bosonization
    Vpmcrit = Vpp - vacrit 
    return Vpmcrit

def ret_Va_crit(t, U, Vpp=None):
    '''returns dV_crit from bosonization, does not depend on Vpp '''
    vacrit = (0.5 * U) - np.pi * np.abs(t)  #approximate value from bosonization
    return vacrit

def compute_Luttinger_parameter(t, U, Vpp, Vpm):
    '''returns Kc, Ks'''
    Vs = Vpp + Vpm
    Va = Vpp - Vpm
    pivf = np.pi * 2.0 * np.abs(t)
    kcinv2 = 1 + U/pivf + 2.0 * Vs/pivf
    ksinv2 = 1 - U/pivf + 2.0 * Va/pivf
    Kc = 1.0/cmath.sqrt(kcinv2)
    Ks = 1.0/cmath.sqrt(ksinv2)

    return Kc, Ks
