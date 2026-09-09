# The helpers more than one executable needs: arming a burst, raw capture to IQ,
# building the transmitted waveform, and the file chores a run ends with. Nothing
# here touches pynq directly, so it imports on a laptop.

# =========================================
# Imports and defines
# =========================================
import json
import os
import time

import numpy as np

# =========================================
# Setting up the converters
# =========================================
def tune_dacs(overlay, dac_nco, dac_zone, n_ch):
    """Set the transmit NCO and zone, no phase shift, and mute every DAC.

    The DACs come up silent so a run transmits only once asked to.
    """
    overlay.d_centre_freq = dac_nco
    overlay.d_nyquist_zone = dac_zone
    overlay.d_phases = [0.0] * n_ch
    overlay.d_gain = [0.0] * n_ch
    overlay.configure_dacs()

def tune_adcs(overlay, adc_nco, adc_zone, n_ch, n_cap):
    """Set the receive NCO and zone, and open all the channels.

    The channel list and data_size must both be set, or the driver quietly hands
    back only the first four channels.
    """
    overlay.active_rf_channels = list(range(n_ch))
    overlay.channels = n_ch * 2
    overlay.data_size = n_cap * overlay.channels
    overlay.centre_freq = adc_nco
    overlay.nyquist_zone = adc_zone
    overlay.phases = [0.0] * n_ch
    overlay.configure_adcs()

# =========================================
# Building the transmitted waveform
# =========================================
def create_tone_samples(n_samples, sr, freq_hz, amp):
    """Single harmonic, complex baseband, scaled to the DAC full scale."""
    t = np.arange(n_samples) / sr
    return amp * np.exp(1j * 2 * np.pi * freq_hz * t)

def create_zc_chirp_samples(n_samples, sr, bw_hz, amp):
    """Zadoff-Chu-style CAZAC chirp: constant amplitude, quadratic phase, and an
    instantaneous frequency sweeping linearly from 0 to bw_hz over the window.
    """
    t = np.arange(n_samples) / sr
    duration = n_samples / sr
    chirp_rate = bw_hz / duration
    phase = np.pi * chirp_rate * t**2
    return amp * np.exp(1j * phase)

def write_tone_to_players(overlay, tone):
    """Interleave I and Q into int16 and copy into every player memory.

    All four tiles get the same waveform, so the gain table alone decides which
    DAC transmits - that is what keeps the transmitted phase identical from burst
    to burst.
    """
    overlay.signal[0::2] = np.int16(tone.real)
    overlay.signal[1::2] = np.int16(tone.imag)
    players = [overlay.dac0_player, overlay.dac1_player,
               overlay.dac2_player, overlay.dac3_player]
    for player in players:
        overlay.dac_data_mem_write(overlay.signal, player)

def snap_tone_to_fft_bin(tone_mhz, adc_sr, n_cap):
    """Return the tone moved onto an exact FFT bin of the capture window.

    Every estimate here is one FFT value read at the tone bin, so a tone between
    bins spreads over its neighbours and the phase read off the peak is not the
    phase of the channel. The nearest bin is always what was meant, so this
    corrects instead of rejecting.
    """
    bin_hz = adc_sr / n_cap
    snapped_mhz = round(tone_mhz * 1e6 / bin_hz) * bin_hz / 1e6
    if snapped_mhz == 0:
        raise ValueError("tone is below half a bin (%g Hz) - it snaps to DC"
                         % bin_hz)
    if snapped_mhz != tone_mhz:
        print("tone %g -> %.6f MHz, nearest bin of the %g Hz grid"
              % (tone_mhz, snapped_mhz, bin_hz))
    return snapped_mhz

def find_tone_bin(tone_mhz, n_samples, adc_sr):
    """FFT bin index of the tone - exact, because the tone was snapped first."""
    return int(round(tone_mhz * 1e6 * n_samples / adc_sr))

# =========================================
# Taking a burst
# =========================================
def capture_aligned(overlay, trig_hold_s):
    """One burst with TX and RX started by the same trigger edge.

    The player enable is dac_enable OR trig_cap, so leaving dac_enable low hands
    the enable to trig_cap: one rising edge restarts the waveform at sample 0 and
    arms the capture on the same clock. The phase of what comes back is then a
    property of the path and not of when the trigger fired.

    get_custom_data_xm655() is avoided - it fires its own trigger and would undo
    the alignment.
    """
    overlay.dacs_off()
    overlay.trig_cap.off()
    overlay.trig_cap.on()
    time.sleep(trig_hold_s)
    overlay.trig_cap.off()
    return overlay._get_selected_xm655_data()

def convert_raw_to_iq(raw, n_ch):
    """Flat interleaved int16 stream -> one complex row per ADC.

    The board hands back all the lanes together - I and Q of ADC 0, then of
    ADC 1 - so the channels are walked one at a time.
    """
    raw = np.asarray(raw, dtype=np.int16).ravel()
    lanes = n_ch * 2
    if raw.size % lanes:
        raise ValueError("%d samples is not divisible by %d lanes" % (raw.size, lanes))
    frames = raw.reshape(-1, lanes)
    iq = np.zeros((n_ch, frames.shape[0]), dtype=np.complex128)
    for channel in range(n_ch):
        i_samples = frames[:, 2 * channel]
        q_samples = frames[:, 2 * channel + 1]
        iq[channel] = i_samples + 1j * q_samples
    return iq

# =========================================
# Writing a run to disk
# =========================================
def clear_dir(folder):
    """Delete an earlier run's files so nothing stale reads as a result of this one.

    Only the files directly in the folder - a subfolder was put there on purpose.
    """
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            os.remove(path)

def save_json_params(folder, params, name = "params.json"):
    """Record the settings the numbers came from.

    Without it the arrays saved beside it are anonymous.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "w") as handle:
        json.dump(params, handle, indent=2)
    return path
