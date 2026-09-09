import pynq 
from pynq import Overlay, MMIO
import xrfclk
import xrfdc
import numpy as np
import time
import os
import subprocess

from pynq import GPIO
import subprocess as sp
import netifaces as ni
import socket
import struct
import scipy.io
from scipy.signal import chirp
from scipy.signal import sawtooth
from scipy.signal import gausspulse

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
CLOCKWIZARD_LOCK_ADDRESS = 0x0004
CLOCKWIZARD_RESET_ADDRESS = 0x0000
CLOCKWIZARD_RESET_TOKEN = 0x000A
MTS_START_TILE = 0x01
MAX_DAC_TILES = 4
MAX_ADC_TILES = 4
DAC_REF_TILE = 2
ADC_REF_TILE = 2

RFSOC4X2_LMK_FREQ = 500.0
RFSOC4X2_LMX_FREQ = 500.0
RFSOC4X2_DAC_TILES = 0b0101
RFSOC4X2_ADC_TILES = 0b0101
RFSOC4X2_PATH_PER_TILE = 2

ZCU216_LMK_FREQ = 500.0
ZCU216_LMX_FREQ = 2000.0
# ZCU216_LMX_FREQ = 5000.0

ZCU216_DAC_TILES = 0b1111
ZCU216_ADC_TILES = 0b1111
ZCU216_PATH_PER_TILE = 4

DATASTREAM = 1 
# XM655 now has RF channels 0..15 available.
# For now stream/calibrate only the first four RF channels: 0, 1, 2, 3.
ACTIVE_RF_CHANNELS = [0, 1, 2, 3]
IQ_LANES_PER_RF_CHANNEL = 2
CHANNELS = len(ACTIVE_RF_CHANNELS) * IQ_LANES_PER_RF_CHANNEL
DATA_SIZE = (2**14)*CHANNELS
splitSym = "#" # For parsing commands
EVENT_SRC = xrfdc.EVNT_SRC_SYSREF # For MTS

NYQUIST_ZONE = [1, 1, 1, 1]
CENTRE_FREQ = [1000, 1000, 1000, 1000]
PHASES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
A_IQ_PHASES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
D_PHASES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
D_IQ_PHASES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
A_GAIN = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
D_GAIN = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
D_NYQUIST_ZONE = [1, 1, 1, 1]
D_CENTRE_FREQ = [5500, 5500, 5500, 5500]

DACPOWER = [1, 1, 1, 1]
ADC_FS = 250

DA = 2 # Sync DACs with ADCs

class doaMtsOverlay(Overlay):
    """
    The MTS overlay demonstrates the RFSoC multi-tile synchronization capability that enables
    multiple RF DAC and ADC tiles to achieve latency alignment. This capability is key to enabling
    Massive MIMO, phased array RADAR applications and beamforming.
    """
    def __init__(self, bitfile_name='mts.bit', **kwargs):
        """
         This overlay class supports the MTS overlay.  It configures the PL gpio, internal memories,
         PL-DRAM and DMA interfaces.  Additional helper methods are provided to: configure and verify
         MTS, verify the DACRAM and read captured samples from the internal ADC
         memories and the PL-DDR4 memory.  In addition to the bitfile_name, the active ADC and DAC
         tiles must be provided to use in the MTS initialization.
        """
        board = os.getenv('BOARD') 
        # Run lsmod command to get the loaded modules list
        # output = subprocess.check_output(['lsmod'])
        # # Check if "zocl" is present in the output
        # if b'zocl' in output:
        #     # If present, remove the module using rmmod command
        #     rmmod_output = subprocess.run(['rmmod', 'zocl'])
        #     # Check return code
        #     assert rmmod_output.returncode == 0, "Could not restart zocl. Please Shutdown All Kernels and then restart"
        #     # If successful, load the module using modprobe command
        #     modprobe_output = subprocess.run(['modprobe', 'zocl'])
        #     assert modprobe_output.returncode == 0, "Could not restart zocl. It did not restart as expected"
        # else:
        #     modprobe_output = subprocess.run(['modprobe', 'zocl'])
        #     # Check return code
        #     assert modprobe_output.returncode == 0, "Could not restart ZOCL!"

        self.ACTIVE_DAC_TILES = ZCU216_DAC_TILES
        self.ACTIVE_ADC_TILES = ZCU216_ADC_TILES
        self.PATH_PER_TILE = ZCU216_PATH_PER_TILE

        self.active_rf_channels = ACTIVE_RF_CHANNELS.copy()
        self.channels = CHANNELS
        self.data_size = DATA_SIZE
        self.dataStream = DATASTREAM
        self.event_src = EVENT_SRC 
        # ADC parameters
        self.phases = PHASES.copy()
        self.centre_freq = CENTRE_FREQ.copy()
        self.nyquist_zone = NYQUIST_ZONE.copy()
        # DAC parameters
        self.d_phases = D_PHASES.copy()
        self.d_centre_freq = D_CENTRE_FREQ.copy()
        self.d_nyquist_zone = D_NYQUIST_ZONE.copy()   
        self.dacPower = DACPOWER.copy() 
        self.a_gain = A_GAIN.copy()
        self.d_gain = D_GAIN.copy()
        self.a_iq_phase = A_IQ_PHASES.copy()
        self.d_iq_phase = D_IQ_PHASES.copy()
        self.adc_fs = ADC_FS

        self.da = DA        
    
        self.init_rf_clks()
        time.sleep(0.5)
        super().__init__(resolve_binary_path(bitfile_name), **kwargs)  
        
        self.xrfdc = self.usp_rf_data_converter_0       
        self.xrfdc.mts_dac_config.RefTile = DAC_REF_TILE  # DAC tile distributing reference clock
        self.xrfdc.mts_adc_config.RefTile = ADC_REF_TILE  # ADC                
        
        self.adc_tiles = {}
        self.adc = {}
        bitvector = self.ACTIVE_ADC_TILES        
        for n in range(MAX_ADC_TILES):
            if (bitvector & 0x1):
                self.adc_tiles[n] = self.xrfdc.adc_tiles[n]
                z = n * self.PATH_PER_TILE
                for i in range(self.PATH_PER_TILE):
                    self.adc[z + i] = self.adc_tiles[n].blocks[i]
            bitvector = bitvector >> 1

        self.dac_tiles = {}
        self.dac = {}
        bitvector = self.ACTIVE_DAC_TILES
        for n in range(MAX_DAC_TILES):
            if (bitvector & 0x1):
                self.dac_tiles[n] = self.xrfdc.dac_tiles[n]
                z = n * self.PATH_PER_TILE
                for i in range(self.PATH_PER_TILE):
                    self.dac[z + i] = self.dac_tiles[n].blocks[i]
            bitvector = bitvector >> 1

        # map PL GPIO registers
        self.dac_enable =  self.control.dac_gpio.channel1[0]
        self.trig_cap = self.control.adc_gpio_0.channel1[0]
        self.adc_master = self.control.adc_gpio_0.channel1[1]

        self.clk_phase_sync = self.control.clk_phase_sync.channel1[0]
        self.axi_gpio_spi_mux = self.control.axi_gpio_spi_mux.channel1[0]
        # DAC TILE Player Memory - DACs will play this waveform
        self.dac0_player = self.memdict_to_view("hier_play0/axi_bram_ctrl_0")
        self.dac1_player = self.memdict_to_view("hier_play1/axi_bram_ctrl_0")
        self.dac2_player = self.memdict_to_view("hier_play2/axi_bram_ctrl_0")
        self.dac3_player = self.memdict_to_view("hier_play3/axi_bram_ctrl_0")
        self.dac_signal = [
            np.zeros(len(self.dac0_player), dtype=np.int16),
            np.zeros(len(self.dac1_player), dtype=np.int16),
            np.zeros(len(self.dac2_player), dtype=np.int16),
            np.zeros(len(self.dac3_player), dtype=np.int16),
        ]
        self.signal = np.zeros(len(self.dac0_player), dtype=np.int16)
        # self.signal = np.zeros(2 * self.dac0_player.shape[0], dtype=np.int16)
        self.source = 4

        # ADC TILE Capture Memories
        self.adc_capture0 = self.memdict_to_view("hier_cap0/axi_bram_ctrl_0")
        self.adc_capture1 = self.memdict_to_view("hier_cap1/axi_bram_ctrl_0")
        self.adc_capture2 = self.memdict_to_view("hier_cap2/axi_bram_ctrl_0")
        self.adc_capture3 = self.memdict_to_view("hier_cap3/axi_bram_ctrl_0")

        # Reset GPIOs and bring to known state
        self.dac_enable.off()
        self.trig_cap.off() 

        # MTS power up sequence
        time.sleep(1)
        self.init_tile_sync()
        self.verify_clock_tree()
        time.sleep(1)
        # self.sync_tiles()
        self.configure_dacs()
        self.configure_adcs()
        time.sleep(1)
        # self.sync_tiles()
        result = None
        while result is None:
            try:
                # connect
                self.sync_tiles()
                result = True
                print('Success')
            except:
                time.sleep(0.1)
                print('Error')
                pass

    def init_rf_clks(self, lmk_freq=ZCU216_LMK_FREQ, lmx_freq=ZCU216_LMX_FREQ):
        """Initialise the LMK and LMX clocks for the radio hierarchy.

        The radio clocks are required to talk to the RF-DCs and only need
        to be initialised once per session.

        """        
        xrfclk.set_ref_clks(lmk_freq=lmk_freq, lmx_freq=lmx_freq)

    def memdict_to_view(self, ip, dtype='int16'):
        """ Configures access to internal memory via MMIO"""
        baseAddress = self.mem_dict[ip]["phys_addr"]
        mem_range = self.mem_dict[ip]["addr_range"]
        ipmmio = MMIO(baseAddress, mem_range)
        return ipmmio.array[0:ipmmio.length].view(dtype)

    def verify_clock_tree(self):
        """ Verify the PL and PL_SYSREF clocks are active by verifying an MMCM is in the LOCKED state"""
        Xstatus = self.clocktreeMTS.MTSclkwiz.read(CLOCKWIZARD_LOCK_ADDRESS) # reads the LOCK register
        # the ClockWizard AXILite registers are NOT fully mapped: refer to PG065
        if (Xstatus != 1):
            raise Exception("The MTS ClockTree has failed to LOCK. Please verify board clocking configuration")

    def trigger_capture(self):
        """ Trigger for buffer capture"""        
        self.trig_cap.off()
        self.trig_cap.on() # actually triggers adc[A..D] to capture too
        # if self.da == 2:
        #     time.sleep(0.01) # For DAC sync              
        if self.da == 1:
            time.sleep(0.01) # For DAC sync        
        self.trig_cap.off()

    def tile_capture(self, buffer, tile_ram):
        """ Captures ADC samples from three channels and stores to internal memories """
        if not np.issubdtype(buffer.dtype, np.int16):
            raise Exception("buffer not defined or np.int16!")
        self.trigger_capture()
        buffer[0] = np.copy(tile_ram[0:len(buffer[0])])     

    def tiles_capture(self, buffer):
        """ Captures ADC samples from all channels and stores to internal memories """
        if not np.issubdtype(buffer.dtype, np.int16):
            raise Exception("buffer not defined or np.int16!")
        self.trigger_capture()
        buffer[0] = np.copy(self.adc_capture0[0:len(buffer[0])])
        buffer[1] = np.copy(self.adc_capture1[0:len(buffer[1])])
        buffer[2] = np.copy(self.adc_capture2[0:len(buffer[2])])
        buffer[3] = np.copy(self.adc_capture3[0:len(buffer[3])])
        
    def sync_tiles(self, dacTarget=-1, adcTarget=-1):
        """ Configures RFSoC MTS alignment"""
        # Set which RF tiles use MTS and turn MTS off
        if self.ACTIVE_DAC_TILES > 0:
            self.xrfdc.mts_dac_config.Tiles = self.ACTIVE_DAC_TILES # group defined in binary 0b1111
            self.xrfdc.mts_dac_config.SysRef_Enable = 1
            self.xrfdc.mts_dac_config.Target_Latency = dacTarget 
            self.xrfdc.mts_dac()
        else:
            self.xrfdc.mts_dac_config.Tiles = 0x0
            self.xrfdc.mts_dac_config.SysRef_Enable = 0
        if self.ACTIVE_ADC_TILES > 0:
            self.xrfdc.mts_adc_config.Tiles = self.ACTIVE_ADC_TILES
            self.xrfdc.mts_adc_config.SysRef_Enable = 1
            self.xrfdc.mts_adc_config.Target_Latency = adcTarget
            self.xrfdc.mts_adc()
        else:
            self.xrfdc.mts_adc_config.Tiles = 0x0
            self.xrfdc.mts_adc_config.SysRef_Enable = 0

    def init_tile_sync(self):
        """ Resets the MTS alignment engine"""
        self.xrfdc.mts_dac_config.Tiles = 0b0001 # turn only one tile on first
        self.xrfdc.mts_adc_config.Tiles = 0b0001
        self.xrfdc.mts_dac_config.SysRef_Enable = 1
        self.xrfdc.mts_adc_config.SysRef_Enable = 1
        self.xrfdc.mts_dac_config.Target_Latency = -1
        self.xrfdc.mts_adc_config.Target_Latency = -1
        self.xrfdc.mts_dac()
        self.xrfdc.mts_adc()
        # Reset MTS ClockWizard MMCM - refer to PG065
        self.clocktreeMTS.MTSclkwiz.mmio.write_reg(CLOCKWIZARD_RESET_ADDRESS, CLOCKWIZARD_RESET_TOKEN)
        time.sleep(0.1)
        # Reset only user selected DAC tiles
        bitvector = self.ACTIVE_DAC_TILES
        for n in range(MAX_DAC_TILES):
            if (bitvector & 0x1):
                self.xrfdc.dac_tiles[n].Reset()
            bitvector = bitvector >> 1
        # Reset ADC FIFO of only user selected tiles - restarts MTS engine
        for toggleValue in (0, 1):
            bitvector = self.ACTIVE_ADC_TILES
            for n in range(MAX_ADC_TILES):
                if (bitvector & 0x1):
                    self.xrfdc.adc_tiles[n].SetupFIFOBoth(toggleValue)
                bitvector = bitvector >> 1
                
    # def mts_sync(self, dacTarget=-1, adcTarget=-1): #TODO REWORK
    #     """
    #     MTS
    #     """

    #     # self.clocktreeMTS.MTSclkwiz.mmio.write_reg(CLOCKWIZARD_RESET_ADDRESS, CLOCKWIZARD_RESET_TOKEN)
    #     # time.sleep(0.1)
    #     self.xrfdc.mts_dac_config.Tiles = self.ACTIVE_DAC_TILES
    #     self.xrfdc.mts_adc_config.Tiles = self.ACTIVE_ADC_TILES
    #     self.xrfdc.mts_dac_config.SysRef_Enable = 1
    #     self.xrfdc.mts_adc_config.SysRef_Enable = 1
    #     self.xrfdc.mts_dac_config.Target_Latency = dacTarget
    #     self.xrfdc.mts_adc_config.Target_Latency = adcTarget
    #     self.xrfdc.mts_dac()
    #     self.xrfdc.mts_adc()
    #     # self.clocktreeMTS.MTSclkwiz.mmio.write_reg(CLOCKWIZARD_RESET_ADDRESS, CLOCKWIZARD_RESET_TOKEN)
    #     self.verify_clock_tree()
    #     # self.xrfdc.mts_adc_config.SysRef_Enable = 0
    #     time.sleep(0.1)

    #     self.configure_adc_tiles(0)
    #     self.configure_adc_tiles(2)
    #     bitvector = self.ACTIVE_DAC_TILES
    #     for n in range(MAX_DAC_TILES):
    #         if (bitvector & 0x1):
    #             self.dac_tiles[n].Reset()
    #             self.configure_dac_tiles(n)
    #         bitvector = bitvector >> 1
    #     # Reset ADC FIFO of only user selected tiles - restarts MTS engine
    #     bitvector = self.ACTIVE_ADC_TILES
    #     for n in range(MAX_ADC_TILES):
    #         if (bitvector & 0x1):
    #             self.configure_adc_tiles(n)
    #         bitvector = bitvector >> 1
    #     self.configure_dacs()
    #     self.configure_adcs()
    #     # time.sleep(1)
    #     # self.xrfdc.mts_adc_config.SysRef_Enable = 0

    def configure_adc_tiles(self, tile_num, pll_freq=500.00, sample_freq=5000.00):
            """
            Single ADC tile tune
            """
            # self.adc_tiles[tile_num].Reset()
            # self.adc_tiles[tile_num].StartUp()
            # time.sleep(0.5)
            # self.adc_tiles[tile_num].DynamicPLLConfig(1, pll_freq, sample_freq)
            self.adc_tiles[tile_num].SetupFIFOBoth(0)
            self.adc_tiles[tile_num].SetupFIFOBoth(1)
            # self.adc_tiles[tile_num].SetupFIFO(True)

    def configure_adc(self, dev_num, nyquist_zone=NYQUIST_ZONE, centre_freq=CENTRE_FREQ, event_src=EVENT_SRC, mixer_phase=0.0, gain=A_GAIN, iq_phase = A_IQ_PHASES):
            """
            Single ADC block tune
            """         
            self.adc[dev_num].NyquistZone = nyquist_zone
            # self.adc[dev_num].Dither = 1 # Selects if dither is enabled for the selected tile. Dither should be enabled unless the sample rate is under 0.75 times the maximum sampling rate for the RF-ADC.
            # self.adc[dev_num].CalibrationMode = 2 # Mode 1, optimized for the input signal locates from 0.4 * Fs to Fs/2 Mode 2, optimized for the input signal locates from 0 to 0.4 * Fs
            # self.adc[dev_num].CalFreeze = {
            #     'CalFrozen'             : 0, 
            #     'DisableFreezePin'      : 0, 
            #     'FreezeCalibration'     : 0
            #     }
            # self.adc[dev_num].CoarseDelaySettings = {
            #     'CoarseDelay'           : 0, 
            #     'EventSource'           : event_src
            #     }
            self.adc[dev_num].MixerSettings = {
            'Freq'                  : centre_freq,
            'PhaseOffset'           : mixer_phase,
            'EventSource'           : event_src,
            'CoarseMixFreq'         : xrfdc.COARSE_MIX_BYPASS,
            'MixerMode'             : xrfdc.MIXER_MODE_R2C,
            'FineMixerScale'        : xrfdc.MIXER_SCALE_AUTO,
            'MixerType'             : xrfdc.MIXER_TYPE_FINE
        }
            self.adc[dev_num].ResetNCOPhase()

            self.adc[dev_num].QMCSettings = {
                'EnablePhase'           : 1,
                'EnableGain'            : 1,
                'GainCorrectionFactor'  : gain,
                'PhaseCorrectionFactor' : iq_phase,
                'OffsetCorrectionFactor': 0,
                'EventSource'           : event_src
                }
            if not(event_src==xrfdc.EVNT_SRC_SYSREF):
                self.adc[dev_num].UpdateEvent(xrfdc.EVENT_MIXER)
                self.adc[dev_num].UpdateEvent(xrfdc.EVENT_CRSE_DLY)
                self.adc[dev_num].UpdateEvent(xrfdc.EVENT_QMC)
            
    def configure_adcs(self):
            """
            All ADCs blocks tune
            """
            bitvector = self.ACTIVE_ADC_TILES
            for n in range(MAX_ADC_TILES):
                if (bitvector & 0x1):
                    for i in range(self.PATH_PER_TILE):
                        self.configure_adc(n*self.PATH_PER_TILE+i, self.nyquist_zone[n], self.centre_freq[n], self.event_src, self.phases[n*self.PATH_PER_TILE+i], self.a_gain[n*self.PATH_PER_TILE+i], self.a_iq_phase[n*self.PATH_PER_TILE+i])
                bitvector = bitvector >> 1

    def configure_dac_tiles(self, tile_num, pll_freq=500.00, sample_freq=5000.00):
        """
        Single DAC tile tune
        """
        # self.dac_tiles[tile_num].Reset()
        # self.dac_tiles[tile_num].StartUp()
        # time.sleep(0.5)
        # self.dac_tiles[tile_num].DynamicPLLConfig(1, pll_freq, sample_freq)
        self.dac_tiles[tile_num].SetupFIFO(False)
        self.dac_tiles[tile_num].SetupFIFO(True)

    def configure_dac(self, dev_num, nyquist_zone=NYQUIST_ZONE, centre_freq=CENTRE_FREQ, event_src=EVENT_SRC, mixer_phase=0.0, gain=D_GAIN, iq_phase = D_IQ_PHASES):
        """
        Single DAC block tune
        """
        
        self.dac[dev_num].NyquistZone = nyquist_zone
        self.dac[dev_num].SetDACVOP(40500) # Max output current
        # self.dac[dev_num].DataPathMode = 1
        # self.dac[dev_num].IMRPassMode = 0
        # self.dac[dev_num].InvSincFIR = 1

        # self.dac[dev_num].CoarseDelaySettings = {
        #     'CoarseDelay'           : 0, 
        #     'EventSource'           : event_src
        #     }
        self.dac[dev_num].MixerSettings = {
            'Freq'                  : centre_freq,
            'PhaseOffset'           : mixer_phase,
            'EventSource'           : event_src,
            'CoarseMixFreq'         : xrfdc.COARSE_MIX_BYPASS,
            'MixerMode'             : xrfdc.MIXER_MODE_C2R,
            'FineMixerScale'        : xrfdc.MIXER_SCALE_AUTO,
            'MixerType'             : xrfdc.MIXER_TYPE_FINE
        }
        self.dac[dev_num].QMCSettings = {
            'EnablePhase'           : 1,
            'EnableGain'            : 1,
            'GainCorrectionFactor'  : gain,
            'PhaseCorrectionFactor' : iq_phase,
            'OffsetCorrectionFactor': 0,
            'EventSource'           : event_src
            }
        self.dac[dev_num].ResetNCOPhase()

        if not(event_src==xrfdc.EVNT_SRC_SYSREF):
                self.dac[dev_num].UpdateEvent(xrfdc.EVENT_MIXER)
                self.dac[dev_num].UpdateEvent(xrfdc.EVENT_CRSE_DLY)
                self.dac[dev_num].UpdateEvent(xrfdc.EVENT_QMC)            

    def configure_dacs(self):
        """
        All DAC blocks tune
        """
        bitvector = self.ACTIVE_DAC_TILES
        for n in range(MAX_DAC_TILES):
            if (bitvector & 0x1):
                for i in range(self.PATH_PER_TILE):
                    self.configure_dac(n*self.PATH_PER_TILE+i, self.d_nyquist_zone[n], self.d_centre_freq[n], self.event_src, self.d_phases[n*self.PATH_PER_TILE+i], self.d_gain[n*self.PATH_PER_TILE+i], self.d_iq_phase[n*self.PATH_PER_TILE+i])
            bitvector = bitvector >> 1
            
        
    def get_custom_data(self, tile_ram):
        """
        Data from 4 (I + Q) channels, int16
        """
        AlignedCaptureSamples = np.zeros((1,len(tile_ram)),dtype=np.int16)
        self.tile_capture(AlignedCaptureSamples, tile_ram)
        custom_data = AlignedCaptureSamples[0][:self.data_size]
        return custom_data
    
    def get_all_custom_data(self):
        """
        Data from 4 (I + Q) channels x 4 Tiles, int16
        """
        AlignedCaptureSamples = np.zeros((4,len(self.adc_capture0)),dtype=np.int16)
        self.tiles_capture(AlignedCaptureSamples)
        custom_data = AlignedCaptureSamples[:, :self.data_size]
        return custom_data
    
    def handle_client_connection(self, client_socket):
        while True:
            # Receive data from client until terminator is found
            data = b''
            while True:
                chunk = client_socket.recv(1)  # Receive one byte at a time
                if chunk == b'\r' or chunk == b'\n' or chunk == b'':  # ASCII CR or LF character
                    break
                data += chunk

            if not data:
                break

            message = data.decode('utf-8')
            return message
        
    def _parse_command_values(self, var_str):
        """Parse slash-separated command values, preserving decimals when present."""
        values = []
        for item in var_str.split('/'):
            item = item.strip()
            if not item:
                continue
            value = float(item)
            values.append(int(value) if value.is_integer() else value)
        return values

    def handle_commands(self, commandList):
        for command_str in commandList:
            command_str = command_str.strip()
            if not command_str:
                continue
            parts = command_str.split(maxsplit=1)
            command = parts[0]
            var = self._parse_command_values(parts[1]) if len(parts) > 1 else []
            match command:
                case "fc":
                    self.centre_freq = [var[0], var[0], var[0], var[0]]
                    self.nyquist_zone = [var[1], var[1], var[1], var[1]] 
                    self.d_centre_freq = [var[2], var[2], var[2], var[2]]
                    self.d_nyquist_zone = [var[3], var[3], var[3], var[3]]
                    self.configure_dacs()
                    self.configure_adcs()
                    # self.sync_tiles()
                case "cal": # Only for ADCs
                    # self.channels = var[0]
                    self.cal_bw = var[0]
                    # self.calibrate()
                    self.calibrate_xm655()
                case "phase":
                    # Apply four user coefficients to RF channels 0, 1, 2, 3.
                    for ch, val in zip(self.active_rf_channels, var):
                        self.phases[ch] = val/100
                    self.configure_adcs()
                    # self.sync_tiles()
                case "dphase":
                    # Apply four user coefficients to DAC channels 0, 1, 2, 3.
                    for ch, val in zip(self.active_rf_channels, var):
                        self.d_phases[ch] = val/100
                        if ch in self.dac:
                            self.dac[ch].MixerSettings['PhaseOffset'] = self.d_phases[ch]
                    # self.configure_dacs()
                    # self.sync_tiles()
                case "again":
                    for ch, val in zip(self.active_rf_channels, var):
                        self.a_gain[ch] = val/100
                    self.configure_adcs()
                case "dgain":
                    for ch, val in zip(self.active_rf_channels, var):
                        self.d_gain[ch] = val/100
                    self.configure_dacs()
                case "aiqphase":
                    for ch, val in zip(self.active_rf_channels, var):
                        self.a_iq_phase[ch] = val/100
                    self.configure_adcs()
                case "diqphase":
                    for ch, val in zip(self.active_rf_channels, var):
                        self.d_iq_phase[ch] = val/100
                    self.configure_dacs()                    
                case "dataChan":
                    self.data_size = int(var[0])

                case "dataStream":              
                    self.dataStream = int(var[0])
                case "da":
                    self.da = int(var[0])
                    match self.da:
                        case 2: # DACs = ADCs sync
                            self.dacs_off()
                            self.trig_cap.off()
                            self.trig_cap.on() 
                            self.trig_cap.off()
                            # self.dac_enable.on()
                            # self.trig_cap.on() 
                        case 1: # DACs = DACs sync
                            self.dacs_off()
                            self.trig_cap.off()
                            self.trig_cap.on() 
                            # self.dacs_on()
                            self.trig_cap.off()                            
                        case 0: # DACs =/= ADCs without sync
                            self.dacs_off()
                            self.dacs_on()
                            
                case "source":
                    self.source = int(var[0])

                    DAC_SR = var[1]*1E9  # Sample rate of DACs (10GSPS)
                    Fc = var[2]*1E6 # Center frequency of waveform
                    Fe = var[3]*1E6 # Maximum frequency of chirp at end of record
                    DAC_Amplitude = float(var[4]) # 14bit DAC +16383/-16384     
                    X_axis = (1/DAC_SR) * np.arange(0, self.dac0_player.shape[0]//2)
                    self.signal = np.zeros(len(self.dac0_player), dtype=np.int16)
                    match self.source:
                        case 4: # File
                            mat = scipy.io.loadmat('data.mat')
                            sig = mat.get('sigInt16')
                            if sig is None:
                                raise KeyError("data.mat does not contain variable 'sigInt16'")
                            self.signal = np.asarray(sig, dtype=np.int16).ravel()
                            # self.signal[-5000:] = 0
                            # self.signal[:10000] = 0
                            # self.signal[-10000:] = 0                            
                        case 3: # Chirp
                            self.signal[0::2] = np.int16(DAC_Amplitude * chirp(np.arange(0, self.dac0_player.shape[0]/2)/DAC_SR,
                                      f0=Fc, f1=Fe, t1=(self.dac0_player.shape[0]/DAC_SR), 
                                      phi=0.0, method='linear'))
                            self.signal[1::2] = np.int16(DAC_Amplitude * chirp(np.arange(0, self.dac0_player.shape[0]/2)/DAC_SR,
                                      f0=Fc, f1=Fe, t1=(self.dac0_player.shape[0]/DAC_SR), 
                                      phi=90.0, method='linear'))
                        case 2: # SAW
                            # self.signal[0::2] = np.int16(DAC_Amplitude * sawtooth(2 * np.pi * Fc * X_axis))
                            # self.signal[1::2] = np.int16(DAC_Amplitude * sawtooth(2 * np.pi * Fc * X_axis + np.pi/2))
                            self.signal[0::2] = np.int16(DAC_Amplitude * np.sin(2*np.pi*Fc*X_axis))
                            self.signal[1::2] = np.int16(DAC_Amplitude * np.cos(2*np.pi*Fc*X_axis))
                            self.signal[:5000] = 0
                            self.signal[-5000:] = 0
                        case 1: # CW
                            self.signal[0::2] = np.int16(DAC_Amplitude * np.sin(2*np.pi*Fc*X_axis))
                            self.signal[1::2] = np.int16(DAC_Amplitude * np.cos(2*np.pi*Fc*X_axis))
                            
                        case 0: # Zeros
                            self.signal = np.zeros(len(self.dac0_player), dtype=np.int16)
                            
                    self.dac_data_mem_write(self.signal, self.dac0_player)
                    self.dac_data_mem_write(self.signal, self.dac1_player)
                    self.dac_data_mem_write(self.signal, self.dac2_player)
                    self.dac_data_mem_write(self.signal, self.dac3_player)
                case "dac0":    
                    self.dac_signal[0] = var
            
                    self.dac_data_mem_write(self.dac_signal[0], self.dac0_player)
            
                case "dac1":
                    self.dac_signal[1] = var
            
                    self.dac_data_mem_write(self.dac_signal[1], self.dac1_player)
            
                case "dacPow":
                    self.dacPower = var
                    
                case "sync":
                    result = None
                    while result is None:
                        try:
                            # connect
                            self.sync_tiles()
                            result = True
                            print('Success')
                        except:
                            time.sleep(0.1)
                            print('Error')
                            pass
                    
    def dacs_control(self): #TODO REWORK
        if any(self.dacPower[:len(self.active_rf_channels)]):
            self.dac_enable.on()
        else:
            self.dac_enable.off()

    def dacs_off(self):
        self.dac_enable.off()
        
    def dacs_on(self):
        self.dac_enable.on()
        
    def cphase(self, rawData):
        # Calibrates phases using FFT phase estimation
        # Takes the maximum of the abs value of the FFT of the signal
        # and calculates its angle, then calculates the right phase shift
        # with respect to the first rawData vector.

        # INPUTS:
        # rawData: 4 X n raw signal matrix

        # OUTPUTS:
        # fixed_rawData: 4 X n aligned signal matrix
        # weights: 4 X 1 weights vector for phase alignment

        n_channels = rawData.shape[0]
        max_val = np.zeros(n_channels, dtype=np.cdouble)
        for i in range(n_channels):
            rawFt = np.fft.fftshift(np.fft.fft(rawData[i, :]))
            max_idx = np.argmax(np.abs(rawFt))
            max_val[i] = rawFt[max_idx]
        angs = np.angle(max_val)
        cal_angs = angs[0] - angs

        weights = np.exp(1j * cal_angs)

        return weights
    
    def calibrate(self, channels=CHANNELS):
        coefs_i = [0, 0, 0, 0] # Initial phases because of hardware inputs swap
        self.phases = coefs_i
        self.configure_adcs()
        self.sync_tiles()

        N = len(self.adc_capture) // self.channels

        # Data acq from FPGA
        AlignedCaptureSamples = np.zeros((1,len(self.adc_capture)),dtype=np.int16)
        self.internal_capture(AlignedCaptureSamples)

        ShapedCaptureSamples = np.zeros((channels, N), dtype=np.int16)
        for i in range(self.channels):
            ShapedCaptureSamples[i] = AlignedCaptureSamples[0][i::self.channels]

        iqData = np.zeros((4, N), dtype=np.cdouble)
        idx = 0
        for i in range(0, self.channels, 2):
            iqData[idx] = np.conjugate(ShapedCaptureSamples[i] - 1j * ShapedCaptureSamples[i+1])
            idx = idx + 1
        np.angle(self.cphase(iqData), deg=True)
        coefs = np.zeros((4, 1), dtype=np.double)
        coefs = np.angle(self.cphase(iqData), deg=True)

        coefs_new = coefs_i + coefs
        coefs_new = np.clip(coefs_new, -179, 179)
        self.phases = coefs_new
        self.configure_adcs()
        self.sync_tiles()

        # Data acq from FPGA
        AlignedCaptureSamples = np.zeros((1,len(self.adc_capture)),dtype=np.int16)
        self.internal_capture(AlignedCaptureSamples)

        ShapedCaptureSamples = np.zeros((self.channels, N), dtype=np.int16)
        for i in range(self.channels):
            ShapedCaptureSamples[i] = AlignedCaptureSamples[0][i::self.channels]

        iqData = np.zeros((4, N), dtype=np.cdouble)
        idx = 0
        for i in range(0, self.channels, 2):
            iqData[idx] = np.conjugate(ShapedCaptureSamples[i] - 1j * ShapedCaptureSamples[i+1])
            idx = idx + 1

        coefs = np.zeros((4, 1), dtype=np.double)
        coefs = np.angle(self.cphase(iqData), deg=True)
        
    def dac_data_mem_write(self, signal, mem):
        """Align signal length to DAC memory and write int16 samples."""
        signal = np.asarray(signal, dtype=np.int16).ravel()
        N = len(signal)
        mem_size = len(mem)
        if N == 0:
            raise ValueError("Cannot write an empty signal to DAC memory")
        if N > mem_size:  # Truncate signal to fit memory
            signal = signal[:mem_size]
        else:  # Repeat signal to fill memory
            signal = np.tile(signal, mem_size // N + 1)[:mem_size]
        mem[:] = signal
                
    def _xm655_capture_buffers(self):
        """Return the four capture memories, one per ADC tile."""
        return [self.adc_capture0, self.adc_capture1, self.adc_capture2, self.adc_capture3]

    def _get_selected_xm655_data(self):
        """Pack selected XM655 RF channels as interleaved I/Q int16 streams.

        Hardware now exposes RF channels 0..15.  self.active_rf_channels selects which
        RF channels are sent to the client.  With [0, 1, 2, 3], the output layout is:
        ch0 I, ch0 Q, ch1 I, ch1 Q, ch2 I, ch2 Q, ch3 I, ch3 Q.
        """
        if not self.active_rf_channels:
            raise ValueError("No active RF channels selected")

        captures = self._xm655_capture_buffers()
        lanes_per_capture = self.PATH_PER_TILE * IQ_LANES_PER_RF_CHANNEL
        output_lanes = len(self.active_rf_channels) * IQ_LANES_PER_RF_CHANNEL
        samples_per_lane = min(len(capture) for capture in captures) // lanes_per_capture

        final_buffer = np.zeros(samples_per_lane * output_lanes, dtype=np.int16)

        for out_idx, rf_channel in enumerate(self.active_rf_channels):
            capture_idx = rf_channel // self.PATH_PER_TILE
            path_idx = rf_channel % self.PATH_PER_TILE
            input_i_lane = path_idx * IQ_LANES_PER_RF_CHANNEL
            input_q_lane = input_i_lane + 1
            output_i_lane = out_idx * IQ_LANES_PER_RF_CHANNEL
            output_q_lane = output_i_lane + 1

            if rf_channel < 0 or capture_idx >= len(captures):
                raise ValueError(f"RF channel {rf_channel} is outside available XM655 capture memories")

            final_buffer[output_i_lane::output_lanes] = captures[capture_idx][input_i_lane::lanes_per_capture][:samples_per_lane]
            final_buffer[output_q_lane::output_lanes] = captures[capture_idx][input_q_lane::lanes_per_capture][:samples_per_lane]

        return final_buffer[:self.data_size]

    def get_custom_data_xm655(self):
        """Data from selected XM655 RF channels, int16 interleaved I/Q."""
        self.trigger_capture()
        return self._get_selected_xm655_data()

    def get_custom_data_xm655_2(self):
        """Data from selected XM655 RF channels for DAC/ADC sync mode."""
        self.trig_cap.on()
        return self._get_selected_xm655_data()        

    def calibrate_xm655(self):
            
            channels = len(self.active_rf_channels) * IQ_LANES_PER_RF_CHANNEL
            coefs_i = np.zeros(len(self.active_rf_channels)) 
            self.phases = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.configure_adcs()
            # self.sync_tiles()
            
            # Data acq from FPGA
            AlignedCaptureSamples = self.get_custom_data_xm655()
            N = len(AlignedCaptureSamples) // channels
            
            ShapedCaptureSamples = np.zeros((channels, N), dtype=np.int16)
            for i in range(channels):
                ShapedCaptureSamples[i] = AlignedCaptureSamples[i::channels]

            iqData = np.zeros((len(self.active_rf_channels), N), dtype=np.cdouble)
            idx = 0
            for i in range(0, channels, 2):
                iqData[idx] = np.conjugate(ShapedCaptureSamples[i] - 1j * ShapedCaptureSamples[i+1])
                idx = idx + 1
                
            iqDataFilt = self.filtSig(iqData, self.adc_fs, self.cal_bw)
            np.angle(self.cphase(iqDataFilt), deg=True)
            coefs = np.zeros((len(self.active_rf_channels), 1), dtype=np.double)
            coefs = np.angle(self.cphase(iqDataFilt), deg=True)

            coefs_new = coefs_i + coefs
            indices = np.abs(coefs_new) > 179
            coefs_new[indices] = -(coefs_new[indices] - np.clip(coefs_new[indices], -179, 179))
            print(coefs_new)

            # coefs_new = np.clip(coefs_new, -179, 179)
            for ch, coef in zip(self.active_rf_channels, coefs_new):
                self.phases[ch] = coef            
            self.configure_adcs()
            # self.sync_tiles()

            # Data acq from FPGA
            AlignedCaptureSamples = self.get_custom_data_xm655()

            ShapedCaptureSamples = np.zeros((channels, N), dtype=np.int16)
            for i in range(channels):
                ShapedCaptureSamples[i] = AlignedCaptureSamples[i::channels]

            iqData = np.zeros((len(self.active_rf_channels), N), dtype=np.cdouble)            
            idx = 0
            for i in range(0, channels, 2):
                iqData[idx] = np.conjugate(ShapedCaptureSamples[i] - 1j * ShapedCaptureSamples[i+1])
                idx = idx + 1
            iqDataFilt = self.filtSig(iqData, self.adc_fs, self.cal_bw)
            coefs = np.zeros((len(self.active_rf_channels), 1), dtype=np.double)
            coefs = np.angle(self.cphase(iqDataFilt), deg=True)
            print(coefs)
            
    def filtSig(self, sig, fs, bw):
        """Band-limit each channel around DC using FFT along the sample axis."""
        sig = np.asarray(sig)
        n_samples = sig.shape[-1]
        if n_samples == 0:
            return sig

        freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, d=1/fs))
        passband = np.abs(freqs) <= (bw / 2)

        sigFd = np.fft.fftshift(np.fft.fft(sig, axis=-1), axes=-1)
        sigNew = np.zeros_like(sigFd)
        sigNew[..., passband] = sigFd[..., passband]

        sigFilt = np.fft.ifft(np.fft.ifftshift(sigNew, axes=-1), axis=-1)
        return sigFilt
        
 # -------------------------------------------------------------------------------------------------       
def resolve_binary_path(bitfile_name):
    """ this helper function is necessary to locate the bit file during overlay loading"""
    if os.path.isfile(bitfile_name):
        return bitfile_name
    elif os.path.isfile(os.path.join(MODULE_PATH, bitfile_name)):
        return os.path.join(MODULE_PATH, bitfile_name)
    else:
        raise FileNotFoundError(f'Cannot find {bitfile_name}.')
# -------------------------------------------------------------------------------------------------