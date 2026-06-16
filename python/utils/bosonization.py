import numpy as np 

def ret_Vpm_crit(t, U, Vpp):
    '''returns critical value of Vpm from bosonization'''
    vacrit = (0.5 * U) - np.pi * np.abs(t)  #approximate value from bosonization
    Vpmcrit = Vpp - vacrit 
    return Vpmcrit

def ret_Va_crit(t, U, Vpp=None):
    '''returns dV_crit from bosonization, does not depend on Vpp '''
    vacrit = (0.5 * U) - np.pi * np.abs(t)  #approximate value from bosonization
    return vacrit