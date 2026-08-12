import numpy as np
import tenpy
from scipy.fftpack import fft

def SzSz_corr(psi, length=10, connected=False):
    corrfn = psi.correlation_function("Sz", "Sz", [0], length)[0]
    if connected : 
        magz = psi.expectation_value("Sz")[0]
        conn_corr = np.abs(corrfn - magz**2)
        return conn_corr
    
    return corrfn

def SpSm_corr(psi, length=10, connected=False):
    corrfn = psi.correlation_function("Sp", "Sm", [0], length)[0]
    if connected: 
        print("CONNECTED CORRELATION NOT IMPLEMENTED FOR SpSm_corr")
    
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


def denoise_rolling(y, window=10):
    """
    Smooth a 1D signal using a centered rolling average.
    Supports both even and odd window sizes.
    
    Returns a list of same length as input.
    """
    y = np.asarray(y, dtype=float)

    if window < 1:
        raise ValueError("window must be >= 1")

    # Symmetric padding (keeps length stable)
    pad_left = window // 2
    pad_right = window - 1 - pad_left

    y_padded = np.pad(y, (pad_left, pad_right), mode='edge')

    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_padded, kernel, mode='valid')

    return y_smooth.tolist()

import numpy as np

def op_correlation_idmrg(psi, opA=("Sp", "Sp"), opAdag=("Sm", "Sm"),
                          i0=0, xs=None, autoJW=True, connected=False):
    L = psi.L
    if xs is None:
        xs = np.arange(1, 10 * L)
    xs = np.asarray(xs)

    term_Odag = [(opAdag[0], i0 + 1), (opAdag[1], i0)]   # O_dag(i0)

    corr = np.zeros(len(xs), dtype=complex)
    for k, x in enumerate(xs):
        term_O = [(opA[0], i0 + x), (opA[1], i0 + x + 1)]   # O(i0+x)
        term = term_O + term_Odag                            # product, left-to-right = operator order
        val = psi.expectation_value_term(term, autoJW=autoJW)
        corr[k] = val

    if connected:
        O_val = psi.expectation_value_term([(opA[0], i0), (opA[1], i0 + 1)], autoJW=autoJW)
        Odag_val = psi.expectation_value_term(term_Odag, autoJW=autoJW)
        corr = corr - O_val * Odag_val

    return xs, corr


def ret_Kc_Ks_from_seldf(seldf, fitsliceKc=slice(1,4), fitsliceKs=slice(1,4)):
    '''This is a specific helper function for the julia generated data, obtained from 
        seldf = julio.select_N_dv_from_df(N, dV, df)
        Note : Ks has a lot of spread, and can change by up to 0.5,
               Kc has lower spread, and can change by up to 0.1 in the worst case
    ''' 
    N = seldf["N"].values[0]
    charge_conn = seldf["chargemat"].values[0][N//2, :] - seldf["ntot_expect"].values[0][N//2] * seldf["ntot_expect"].values[0]
    fft_chargecorr = fft(charge_conn)
    omega = 2 * np.pi * np.arange(N) / (N)
    gradKc = np.gradient(np.abs(fft_chargecorr)[fitsliceKc], omega[fitsliceKc]).mean()
    Kc = np.pi * gradKc

    spinzconn = seldf["szmat"].values[0][N//2, :] - seldf["sz_expect"].values[0][N//2] * seldf["sz_expect"].values[0]
    fft_spinzcorr = fft(spinzconn)
    gradKs = np.gradient(np.abs(fft_spinzcorr)[fitsliceKs], omega[fitsliceKs]).mean()
    Ks = 4.0 * np.pi * gradKs

    return Kc, Ks