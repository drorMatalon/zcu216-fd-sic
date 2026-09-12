# The beamforming self interference cancellation algorithms. The dictionary at the
# bottom is what a notebook picks one of, by name.

# =========================================
# Imports and defines
# =========================================
import numpy as np

# =========================================
# BSIC - non joint max sinr
# =========================================
def non_joint_max_sinr_bsic(h_si, h_downlink, h_uplink, noise_power):
    """Most wanted power per unit of leakage, transmit and receive solved separately.

        transmit   max_x  x^H H10^H H10 x / x^H (H00^H H00 + noise I) x
        receive    max_y  y^H H01* H01^T y / y^H (H00* H00^T + noise I) y

    The two sides see different matrices because a weight meets its channel as w^T
    going out and as w^T on the rows coming in.
    """
    tx_signal_matrix = h_downlink.conj().T @ h_downlink
    tx_interference_matrix = h_si.conj().T @ h_si
    tx_noise_matrix = noise_power * np.eye(len(tx_signal_matrix))
    rx_signal_matrix = h_uplink.conj() @ h_uplink.T
    rx_interference_matrix = h_si.conj() @ h_si.T
    rx_noise_matrix = noise_power * np.eye(len(rx_signal_matrix))
    tx = find_max_generalized_eigenvector(tx_signal_matrix,
                                          tx_interference_matrix + tx_noise_matrix)
    nf_rx = find_max_generalized_eigenvector(rx_signal_matrix,
                                             rx_interference_matrix + rx_noise_matrix)
    ff_rx = solve_matched_direction(h_downlink @ tx)
    return {"tx": tx, "nf_rx": nf_rx, "ff_rx": ff_rx}

def find_max_generalized_eigenvector(a_mat, b_mat):
    """The top solution of A v = lambda B v, the strongest eigenvector of B^-1 A.

        max_v  v^H A v / v^H B v      solved as   A v = lambda B v

    B^-1 A is not hermitian, so the eigenvalues come back complex and only their
    real part decides which one is strongest.
    """
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.solve(b_mat, a_mat))
    strongest = eigenvectors[:, np.argmax(eigenvalues.real)]
    normalized = strongest / np.linalg.norm(strongest)
    return normalized

# =========================================
# BSIC - non joint zero forcing
# =========================================
def non_joint_zf_bsic(h_si, h_downlink, h_uplink, noise_power):
    """Transmit into the best of the downlink, then null what leaks back in.

        transmit   max_x  x^H H10^H H10 x
        receive    max_y  y^H H01* H01^T y   subject to   y^T H00 x = 0

    With the transmit weight fixed the leakage arrives from one direction only, so
    the null is exact, taken once, and the noise floor never enters it.
    """
    tx = find_max_eigenvector(h_downlink.conj().T @ h_downlink)
    wanted = find_max_eigenvector(h_uplink.conj() @ h_uplink.T)
    nf_rx = project_out_leakage(wanted, h_si @ tx)
    ff_rx = solve_matched_direction(h_downlink @ tx)
    return {"tx": tx, "nf_rx": nf_rx, "ff_rx": ff_rx}

def find_max_eigenvector(matrix):
    """The strongest eigenvector of a hermitian matrix, with nothing to reject.

        max_v  v^H M v / v^H v        solved as   M v = lambda v

    eigh and not eig: everything reaching this is a channel gram matrix, so the
    eigenvalues are real and come back sorted.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    strongest = eigenvectors[:, -1]
    normalized = strongest / np.linalg.norm(strongest)
    return normalized

def project_out_leakage(weights, leakage):
    """The part of the weights that hears none of a leakage direction, w^T leak = 0.

        w <- w - a* (a*^H w) / |a|^2      with   a = leakage

    The conjugate is what gets removed, because the constraint is an inner product
    against it.
    """
    norm = np.linalg.norm(leakage)
    if norm < 1e-12:
        return weights
    direction = np.conj(leakage) / norm
    nulled = weights - direction * (np.conj(direction) @ weights)
    normalized = nulled / np.linalg.norm(nulled)
    return normalized

# =========================================
# BSIC - non joint soft null, max sinr based
# =========================================
def non_joint_softnull_max_sinr_based_bsic(h_si, h_downlink, h_uplink, noise_power):
    """Transmit where the self interference channel is quietest, then max sinr on top.

        transmit   min_x  ||H00 x||                       the quietest direction
        receive    max_y  y^H H01* H01^T y / y^H (a* a^T + noise I) y,  a = H00 x

    The leakage is held down at the antenna and not only after combining, so the
    receiver has less to compress on before any of this reaches the digital side.
    """
    tx = solve_quietest_direction(h_si)
    leakage = h_si @ tx
    rx_signal_matrix = h_uplink.conj() @ h_uplink.T
    rx_interference_matrix = np.outer(leakage.conj(), leakage)
    rx_noise_matrix = noise_power * np.eye(len(rx_signal_matrix))
    nf_rx = find_max_generalized_eigenvector(rx_signal_matrix,
                                             rx_interference_matrix + rx_noise_matrix)
    ff_rx = solve_matched_direction(h_downlink @ tx)
    return {"tx": tx, "nf_rx": nf_rx, "ff_rx": ff_rx}

def solve_quietest_direction(channel):
    """The transmit weights that put the least power into a channel, min ||H w||.

    That is the right singular vector of the smallest singular value, conjugated
    because the weight meets the channel as a plain H w. The full singular set is
    asked for: an array with more transmitters than the far side has receivers has
    directions of exact silence, and they live in the rows the economy form drops.
    """
    _, _, right_vectors = np.linalg.svd(channel, full_matrices = True)
    quietest = np.conj(right_vectors[-1, :])
    normalized = quietest / np.linalg.norm(quietest)
    return normalized

# =========================================
# Shared by more than one algorithm
# =========================================

def solve_matched_direction(wanted):
    """The match to a single direction, maximizing |w^T wanted|^2.

        max_w  |w^T wanted|^2         solved as   w = wanted* / |wanted|

    Conjugated because the weights meet their channel as w^T and not as w^H.
    """
    matched = np.conj(wanted)
    normalized = matched / np.linalg.norm(matched)
    return normalized

# =========================================
# The algorithms, by the name a notebook asks for
# =========================================

BSIC_ALGORITHMS = {"non_joint_max_sinr": non_joint_max_sinr_bsic,
                   "non_joint_zf": non_joint_zf_bsic,
                   "non_joint_softnull_max_sinr_based":
                       non_joint_softnull_max_sinr_based_bsic}
