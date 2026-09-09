# The self interference cancellation algorithms the one sided bench can run. The
# two dictionaries at the bottom are what a notebook picks one of, by name.

# =========================================
# Imports and defines
# =========================================
import numpy as np

ODD_ORDERS = 3 # 1st, 3rd and 5th order - the even ones land off carrier

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
# DSIC - fitting a model to the interference
# =========================================

def linear_wiener_dsic(tx_signal, received, n_taps, step_size):
    """The least squares channel in one shot, what the LMS fits converge towards.

        H = Y X* / (|X|^2 + loading)      in the frequency domain

    The loading in the denominator stands in for the noise and stops a quiet
    frequency bin from blowing the division up.
    """
    tx_spectrum = np.fft.fft(tx_signal)
    received_spectrum = np.fft.fft(received)
    loading = 1e-6 * np.mean(np.abs(tx_spectrum) ** 2)
    channel_spectrum = (received_spectrum * np.conj(tx_spectrum)
                        / (np.abs(tx_spectrum) ** 2 + loading))
    channel = np.fft.ifft(channel_spectrum)
    return channel[:n_taps]

def linear_lms_dsic(tx_signal, received, n_taps, step_size):
    """Walk the taps downhill on the squared error, one sample at a time.

        e[n] = y[n] - h^T x[n]        h <- h + mu e[n] x[n]*

    They start at zero and not at one: behind a working beamformer the leakage is
    already tiny, so doing nothing is the closer starting point.
    """
    taps = np.zeros(n_taps, dtype=complex)
    for index in range(n_taps, len(tx_signal)):
        window = tx_signal[index - n_taps + 1:index + 1][::-1]
        estimate = np.dot(taps, window)
        error = received[index] - estimate
        taps = taps + step_size * error * np.conj(window)
    return taps

def non_linear_lms_dsic(tx_signal, received, n_taps, step_size):
    """The same walk with an odd order basis in front of it.

        basis[n] = [x[n], x[n]|x[n]|^2, x[n]|x[n]|^4]     h <- h + mu e[n] basis*

    That lets the fit rebuild what the amplifier did to the leakage and not only
    what the multipath did, at n_taps * ODD_ORDERS taps.
    """
    taps = np.zeros(n_taps * ODD_ORDERS, dtype=complex)
    for index in range(n_taps, len(tx_signal)):
        window = tx_signal[index - n_taps + 1:index + 1][::-1]
        basis = np.concatenate([window * np.abs(window) ** (2 * order)
                                for order in range(ODD_ORDERS)])
        estimate = np.dot(taps, basis)
        error = received[index] - estimate
        taps = taps + step_size * error * np.conj(basis)
    return taps

def non_linear_wiener_dsic(tx_signal, received, n_taps, step_size):
    """A multipath filter with a memoryless amplifier behind it, fitted in that order.

        f = h * x     y ~ sum_k c_k f |f|^(2k)      h by wiener, c by least squares

    The linear step comes out as the right filter times a wrong constant, which
    normalizing removes; the taps are never refined afterwards.
    """
    fitted = linear_wiener_dsic(tx_signal, received, n_taps, step_size)
    taps = fitted / np.linalg.norm(fitted)
    filtered = apply_taps(taps, tx_signal)
    basis = np.column_stack([filtered * np.abs(filtered) ** (2 * order)
                             for order in range(ODD_ORDERS)])
    coeffs, *_ = np.linalg.lstsq(basis, received, rcond=None)
    return taps, coeffs

# =========================================
# DSIC - rebuilding the interference from a fit
# =========================================

def apply_taps(taps, signal):
    """Run a list of taps over a signal, one delay per tap.

        y[n] = sum_d h[d] x[n - d]

    It wraps around the end of the block, so the channel making the interference
    and the canceller rebuilding it stay the same operation.
    """
    output = np.zeros(signal.shape, dtype=complex)
    for delay, tap in enumerate(taps):
        output = output + tap * np.roll(signal, delay)
    return output

def apply_non_linear_taps(taps, signal):
    """One filtered copy of the signal per odd order, summed.

        y[n] = sum_k sum_d h_k[d] b_k[n - d]      b_k[n] = x[n] |x[n]|^(2k)

    The taps arrive as one long vector and are cut back into a block of delays per
    order.
    """
    n_taps = len(taps) // ODD_ORDERS
    output = np.zeros(signal.shape, dtype=complex)
    for order in range(ODD_ORDERS):
        branch = signal * np.abs(signal) ** (2 * order)
        branch_taps = taps[order * n_taps:(order + 1) * n_taps]
        output = output + apply_taps(branch_taps, branch)
    return output

def apply_wiener_model(model, signal):
    """The filter first, then the amplifier on what it produced - the fitted order."""
    taps, coeffs = model
    filtered = apply_taps(taps, signal)
    output = np.zeros(signal.shape, dtype=complex)
    for order, coeff in enumerate(coeffs):
        output = output + coeff * filtered * np.abs(filtered) ** (2 * order)
    return output

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
                   "non_joint_zf": non_joint_zf_bsic}

DSIC_ALGORITHMS = {"linear_wiener": (linear_wiener_dsic, apply_taps),
                   "linear_lms": (linear_lms_dsic, apply_taps),
                   "non_linear_lms": (non_linear_lms_dsic, apply_non_linear_taps),
                   "non_linear_wiener": (non_linear_wiener_dsic, apply_wiener_model)}
