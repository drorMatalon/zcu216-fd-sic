# The joint beamforming self interference cancellation algorithms - both nodes
# designed together, each side's weight solved against the beam the other side is
# pointing at. Ported from si_model/lib/bsic.py, with side 1 full duplex too.

# =========================================
# Imports and defines
# =========================================
import numpy as np

from .non_joint_sic_bsic import find_max_generalized_eigenvector
from .non_joint_sic_bsic import project_out_leakage
from .non_joint_sic_bsic import solve_matched_direction

JOINT_ITERATIONS = 20 # rounds of the alternation - it settles well before this

# =========================================
# Joint max sinr
# =========================================

# Every weight is solved against one direction, so both matrices are rank one and a
# single degree of freedom goes on the interference. Each side then changes the other
# side's problem, which is what the loop is for.
def joint_max_sinr_bsic(h00, h01, h10, h11, noise_power):
    """Both links solved together, most wanted power per unit of leakage.

    All four weights are alternated until they agree, starting from uniform.
    """
    weights = create_uniform_start(h00, h11)
    tx_0 = weights["tx_0"]
    rx_0 = weights["rx_0"]
    tx_1 = weights["tx_1"]
    rx_1 = weights["rx_1"]
    for _ in range(JOINT_ITERATIONS):
        tx_0 = solve_max_sinr_direction(h10.T @ rx_1, h00.T @ rx_0, noise_power)
        rx_0 = solve_max_sinr_direction(h01 @ tx_1, h00 @ tx_0, noise_power)
        tx_1 = solve_max_sinr_direction(h01.T @ rx_0, h11.T @ rx_1, noise_power)
        rx_1 = solve_max_sinr_direction(h10 @ tx_0, h11 @ tx_1, noise_power)
    return {"tx_0": tx_0, "rx_0": rx_0, "tx_1": tx_1, "rx_1": rx_1}

def solve_max_sinr_direction(wanted, leakage, noise_power):
    """Most |w^T wanted|^2 per unit of |w^T leakage|^2 and noise floor.

    Both sides are one direction here, so two rank one matrices decide it.
    """
    signal = np.outer(wanted.conj(), wanted)
    interference = np.outer(leakage.conj(), leakage)
    noise = noise_power * np.eye(len(signal))
    return find_max_generalized_eigenvector(signal, interference + noise)

# =========================================
# Joint zero forcing
# =========================================

# With the transmit weight fixed the leakage arrives from one direction only, so the
# null is exact and the noise floor never enters it. Moving the transmit weight moves
# that direction, so the null has to be taken again - the loop again.
def joint_zf_bsic(h00, h01, h10, h11, noise_power):
    """Both links solved together, each side nulling what it leaks on itself.

    One receive element pays for the null, so a side needs at least two.
    """
    weights = create_uniform_start(h00, h11)
    tx_0 = weights["tx_0"]
    tx_1 = weights["tx_1"]
    rx_0 = weights["rx_0"]
    rx_1 = weights["rx_1"]
    for _ in range(JOINT_ITERATIONS):
        rx_0 = project_out_leakage(solve_matched_direction(h01 @ tx_1), h00 @ tx_0)
        rx_1 = project_out_leakage(solve_matched_direction(h10 @ tx_0), h11 @ tx_1)
        tx_0 = solve_matched_direction(h10.T @ rx_1)
        tx_1 = solve_matched_direction(h01.T @ rx_0)
    return {"tx_0": tx_0, "rx_0": rx_0, "tx_1": tx_1, "rx_1": rx_1}

# =========================================
# Shared by both of them
# =========================================

def create_uniform_start(h00, h11):
    """Every element driven and heard equally - where both loops start from."""
    weights = {}
    for side, si in ((0, h00), (1, h11)):
        receivers, elements = si.shape
        weights["tx_%d" % side] = np.ones(elements, dtype=complex) / np.sqrt(elements)
        weights["rx_%d" % side] = np.ones(receivers, dtype=complex) / np.sqrt(receivers)
    return weights

# =========================================
# The algorithms, by the name a notebook asks for
# =========================================

JOINT_BSIC_ALGORITHMS = {"joint_max_sinr": joint_max_sinr_bsic,
                         "joint_zf": joint_zf_bsic}
