# The digital self interference cancellation algorithms. The dictionary at the
# bottom is what a notebook picks one of, by name - each entry a fit and the
# rebuild that goes with it.

# =========================================
# Imports and defines
# =========================================
import numpy as np

ODD_ORDERS = 3 # 1st, 3rd and 5th order - the even ones land off carrier

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
# The algorithms, by the name a notebook asks for
# =========================================

DSIC_ALGORITHMS = {"linear_wiener": (linear_wiener_dsic, apply_taps),
                   "linear_lms": (linear_lms_dsic, apply_taps),
                   "non_linear_lms": (non_linear_lms_dsic, apply_non_linear_taps),
                   "non_linear_wiener": (non_linear_wiener_dsic, apply_wiener_model)}
