import numpy as np

def connected_charge_correlations_avg(psi, length):
    """
    Connected density-density correlations averaged over the unit cell.
    """
    L = psi.L
    n = psi.expectation_value("Ntot")

    C = np.zeros(length)

    for i in range(L):
        nn = psi.correlation_function(
            "Ntot", "Ntot",
            sites1=[i],
            sites2=list(range(i, i + length))
        )[0]

        nj = np.array([n[(i + r) % L] for r in range(length)])
        C += nn - n[i] * nj

    C /= L
    return np.arange(length), C

def connected_sz_correlations_avg(psi, length):
    """
    Connected spin-spin correlations

        C(r) = <Sz_i Sz_j> - <Sz_i><Sz_j>

    averaged over the unit cell.
    """
    L = psi.L
    sz = psi.expectation_value("Sz")

    C = np.zeros(length)

    for i in range(L):
        ssz = psi.correlation_function(
            "Sz", "Sz",
            sites1=[i],
            sites2=list(range(i, i + length))
        )[0]

        szj = np.array([sz[(i + r) % L] for r in range(length)])
        C += ssz - sz[i] * szj

    C /= L
    return np.arange(length), C


def sp_sm_correlations_avg(psi, length):
    """
    Transverse spin correlations

        <S+_i S-_j>

    averaged over the unit cell.
    """
    L = psi.L

    C = np.zeros(length, dtype=complex)

    for i in range(L):
        corr = psi.correlation_function(
            "Sp", "Sm",
            sites1=[i],
            sites2=list(range(i, i + length))
        )[0]

        C += corr

    C /= L
    return np.arange(length), C


def green_function_up_avg(psi, length):
    """
    Spin-up single-particle Green's function

        G(r) = <Cd_up(i) C_up(j)>

    averaged over the unit cell.
    """
    L = psi.L

    G = np.zeros(length, dtype=complex)

    for i in range(L):
        corr = psi.correlation_function(
            "Cdu", "Cu",
            sites1=[i],
            sites2=list(range(i, i + length))
        )[0]

        G += corr

    G /= L
    return np.arange(length), G


def green_function_down_avg(psi, length):
    """
    Spin-down single-particle Green's function

        G(r) = <Cd_down(i) C_down(j)>
    """
    L = psi.L

    G = np.zeros(length, dtype=complex)

    for i in range(L):
        corr = psi.correlation_function(
            "Cdd", "Cd",
            sites1=[i],
            sites2=list(range(i, i + length))
        )[0]

        G += corr

    G /= L
    return np.arange(length), G