# A stand-in for doaMtsOverlay, for working on the notebooks with no board on the
# desk. It answers only the parts of the driver this project actually calls, and
# behind them sits one fixed channel matrix, built once at import, that every
# capture is pushed through - so a capture comes back with the tone, the beam and
# the leakage a real one would have, and an algorithm can go and estimate it.

# =========================================
# Imports and defines
# =========================================
import numpy as np

N_CH = 16
PATH_PER_TILE = 4
PLAYER_SAMPLES = 131072      # IQ pairs one tile player holds, as on the board
DECIMATION = 4               # DAC_SR / ADC_SR - the ADC path sees every 4th sample
NOISE_COUNTS = 0.01          # receiver noise, in ADC counts
COUPLING = 0.1               # how much of a DAC reaches an ADC, on every path
CHANNEL_SEED = 7             # fixed, so two runs of a notebook are comparable

# =========================================
# The channel matrix
# =========================================
def create_channel_matrix():
    """The one 16 by 16 channel this fake board lives in: constant, complex, symmetric.

    The board has 16 transmitters and 16 receivers and no idea what any of them
    is wired to, so neither does this - every DAC reaches every ADC, and which
    ones form a system is the notebook's business alone. Symmetric because a
    passive array is reciprocal: the path from element i to element j is the path
    from j to i, and an algorithm that estimates it should be able to lean on
    that. Every entry gets its own random phase, since a matrix with equal phases
    would let a uniform beam null the leakage by accident and flatter every
    algorithm. Built once at import, so every capture, every notebook and every
    re-run see the same channel.
    """
    source = np.random.default_rng(CHANNEL_SEED)
    channel = np.zeros((N_CH, N_CH), dtype = complex)
    for row in range(N_CH):
        for column in range(row, N_CH):
            entry = COUPLING * np.exp(1j * source.uniform(0, 2 * np.pi))
            channel[row, column] = entry
            channel[column, row] = entry
    return channel

CHANNEL_MATRIX = create_channel_matrix()

# =========================================
# The fake board
# =========================================
class dummy_zcy216_overlay:
    """The same handle the notebooks get from doaMtsOverlay, without the ZCU216.

    Only the attributes and methods this project uses are here. Anything else
    raises the usual AttributeError, which is on purpose: a notebook that starts
    needing more of the driver should say so loudly instead of quietly running
    against a fake that guessed.
    """
    def __init__(self, bitfile_name = "mts.bit", **kwargs):
        """Come up in the same state the real overlay does: silent DACs, all channels open.

        The bitstream name is accepted and ignored, so the one line a notebook
        swaps is the class name and nothing else.
        """
        self.channel_matrix = CHANNEL_MATRIX
        self.noise_source = np.random.default_rng(CHANNEL_SEED + 1)

        self.dac0_player = np.zeros(2 * PLAYER_SAMPLES, dtype = np.int16)
        self.dac1_player = np.zeros(2 * PLAYER_SAMPLES, dtype = np.int16)
        self.dac2_player = np.zeros(2 * PLAYER_SAMPLES, dtype = np.int16)
        self.dac3_player = np.zeros(2 * PLAYER_SAMPLES, dtype = np.int16)
        self.signal = np.zeros(2 * PLAYER_SAMPLES, dtype = np.int16)

        self.trig_cap = DummyTrigger()

        self.d_centre_freq = [0.0] * PATH_PER_TILE
        self.d_nyquist_zone = [1] * PATH_PER_TILE
        self.d_phases = [0.0] * N_CH
        self.d_gain = [0.0] * N_CH

        self.centre_freq = [0.0] * PATH_PER_TILE
        self.nyquist_zone = [1] * PATH_PER_TILE
        self.phases = [0.0] * N_CH
        self.active_rf_channels = list(range(N_CH))
        self.channels = N_CH * 2
        self.data_size = 16384 * self.channels
        self.da = 2

        self.applied_gain = list(self.d_gain)
        self.applied_phases = list(self.d_phases)

    def configure_dacs(self):
        """Latch the gain and phase tables, the way pushing them to the QMC block would.

        The latch is what makes the fake honest: a capture uses what was last
        configured, so a notebook that sets d_gain and forgets this call sees the
        old beam here exactly as it would on the board.
        """
        self.applied_gain = list(self.d_gain)
        self.applied_phases = list(self.d_phases)

    def configure_adcs(self):
        """Nothing to tune off the board - the receive NCO only moves a tone this fake never mixes."""
        pass

    def dacs_off(self):
        """Nothing to mute - this fake plays a capture out of the players when asked, never between."""
        pass

    def dac_data_mem_write(self, signal, mem):
        """Fill a player memory with the waveform, repeating or trimming it to the exact size."""
        signal = np.asarray(signal, dtype = np.int16).ravel()
        if len(signal) == 0:
            raise ValueError("cannot write an empty signal to DAC memory")
        repeats = len(mem) // len(signal) + 1
        mem[:] = np.tile(signal, repeats)[:len(mem)]

    def _get_selected_xm655_data(self):
        """One capture: read the players, run them through the channel matrix, interleave I and Q.

        The name keeps the leading underscore of the driver method because
        capture_aligned() in lib/common_functions.py calls exactly that.
        """
        n_samples = self.data_size // (len(self.active_rf_channels) * 2)
        transmitted = read_players_as_transmit(self, n_samples)
        weights = create_applied_weights(self.applied_gain, self.applied_phases)
        received = apply_channel(self.channel_matrix, weights, transmitted)
        received = add_receiver_noise(received, self.noise_source)
        return pack_interleaved_iq(received, self.active_rf_channels, self.data_size)

class DummyTrigger:
    """Swallows the trigger edges capture_aligned() fires.

    In software there is nothing to align - the capture is computed from the
    players the moment it is asked for - so the edges only have to be accepted.
    """
    def on(self):
        pass

    def off(self):
        pass

# =========================================
# What a capture goes through
# =========================================
def read_players_as_transmit(overlay, n_samples):
    """What each DAC is playing, decimated to the ADC rate.

    All four DACs of a tile read that tile's player, which is why a notebook can
    only give a different tone to a different tile. Taking every DECIMATION-th
    sample stands in for the interpolation and decimation chain, so a tone lands
    on the bin the notebooks compute against ADC_SR.
    """
    players = [overlay.dac0_player, overlay.dac1_player,
               overlay.dac2_player, overlay.dac3_player]
    stop = 2 * n_samples * DECIMATION
    step = 2 * DECIMATION
    transmitted = np.zeros((N_CH, n_samples), dtype = complex)
    for channel in range(N_CH):
        player = players[channel // PATH_PER_TILE]
        transmitted[channel] = player[0:stop:step] + 1j * player[1:stop:step]
    return transmitted

def create_applied_weights(gain, phases):
    """The gain table and the mixer phase offsets, as one complex weight per DAC."""
    return np.array(gain) * np.exp(1j * np.deg2rad(phases))

def apply_channel(channel, weights, transmitted):
    """Every DAC into every ADC through the channel matrix - the whole physics of this fake.

    This is the line the notebooks are really testing against: what an ADC hears
    is the weighted transmit vector multiplied by the row of the matrix that
    belongs to it.
    """
    received = np.zeros((N_CH, transmitted.shape[1]), dtype = complex)
    for adc in range(N_CH):
        for dac in range(N_CH):
            received[adc] += channel[adc, dac] * weights[dac] * transmitted[dac]
    return received

def add_receiver_noise(received, source):
    """A noise floor, so cancellation has something to stop at instead of running to zero."""
    noise = source.standard_normal(received.shape) + 1j * source.standard_normal(received.shape)
    return received + NOISE_COUNTS * noise

def pack_interleaved_iq(received, active_rf_channels, data_size):
    """Flatten to the int16 lane order the driver hands back: ch0 I, ch0 Q, ch1 I, ...

    Rounded to int16 on purpose - the real capture is quantised, and a fake with
    infinite dynamic range would report cancellation depths no board can reach.
    Clipped and not wrapped, because that is what a saturating ADC does.
    """
    lanes = len(active_rf_channels) * 2
    packed = np.zeros(received.shape[1] * lanes, dtype = np.int16)
    for lane, channel in enumerate(active_rf_channels):
        packed[2 * lane::lanes] = clip_to_adc_range(np.real(received[channel]))
        packed[2 * lane + 1::lanes] = clip_to_adc_range(np.imag(received[channel]))
    return packed[:data_size]

def clip_to_adc_range(samples):
    """Hold a too-loud sample at full scale instead of letting int16 wrap it to nonsense."""
    return np.int16(np.clip(samples, -32768, 32767))
