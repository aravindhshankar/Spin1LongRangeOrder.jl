import numpy as np
import tenpy

def SzSz_corr(psi, length=10, connected=False):
    corrfn = psi.correlation_function("Sz", "Sz", [0], length)[0]
    if connected : 
        magz = psi.expectation_value("Sz")[0]
        conn_corr = np.abs(corrfn - magz**2)
        return conn_corr
    
    return corrfn

def charge_corr(psi, length=10, connected=False):
    charge_corr_fn = psi.correlation_function("Ntot", "Ntot", [0], length)[0]
    if connected : 
        avcharge = psi.expectation_value("Ntot")[0]
        conn_charge_corr = np.abs(charge_corr_fn - avcharge**2)
        return conn_charge_corr
    return charge_corr_fn

def electron_corr(psi, length=10, whichspin="tot"):
    ''' whichspin can be d, u, tot'''
    match whichspin: 
        case "d":
            electron_corr_fn = psi.correlation_function("Cdd", "Cd", [0], length)[0]
        case "u":
            electron_corr_fn = psi.correlation_function("Cdu", "Cu", [0], length)[0]
        case "tot":
            electron_corr_fn = (psi.correlation_function("Cdu", "Cu", [0], length)[0] 
                + psi.correlation_function("Cdd", "Cd", [0], length)[0])
        case _:
            raise Exception("Unavailble electron type : allowed options d | u | tot")
    return electron_corr_fn

