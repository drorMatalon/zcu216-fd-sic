# ZCU216 full duplex bench

Six notebooks that drive a ZCU216 with an XM655 balun card: transmit a tone out of a
beamforming array, capture it on 16 receivers, measure the channels between two systems,
and cancel a system's own transmission out of its own ears.

```
                       config/parameters.json
                                 |
        +---------------+---------------+---------------------+
        |               |               |                     |
        v               v               v                     v
 capture_and_record  calibration_  channel_estimation    no_channel_sic
   plots only        algo_measur.   output/channels/    output/no_channel_sic/
  (board alive?)   output/captures/ h00 h01 h10 h11      trains itself on
                   (weight sweeps)        |              solo captures
                                          v
                                +---------+---------+
                                |                   |
                                v                   v
                         one_side_fd_sic     two_side_fd_sic
                            output/sic/      output/two_side_sic/
                        results.md + signals  results.md + signals
```

`channel_estimation` must run before `one_side_fd_sic` and `two_side_fd_sic` - both
design their beamformers on the matrices the estimation left in `output/channels/`.
`no_channel_sic` needs no estimate: it probes the board itself. The rest are independent.

## About the ZCU216

The board is a Zynq UltraScale+ RFSoC Gen 3 evaluation kit. The XM655 balun card breaks
the converters out to **16 RF channels**, arranged as **4 tiles of 4** on both the DAC and
the ADC side. Everything in this project is indexed by `overlay.dac[]` / capture index
over 0..15, which is *not* the RF channel label silkscreened on the card - on this board
the tile that reaches the antennas is tile 2, so system 0 transmits on DACs 8..11.

Three things about the hardware shape every parameter in the config:

- **MTS (multi tile synchronization)** is what makes a measured phase mean anything. All
  four tiles are latency aligned at power up, and every burst is started by a single
  trigger edge that arms TX and RX on the same clock - in the PL the player enable is
  `dac_enable OR trig_cap`, so leaving `dac_enable` low hands the enable to `trig_cap`.
  Without it, the phase of a capture would be a property of *when the trigger fired*
  rather than of the path.
- **The baseband rates are fixed by the bitstream.** The DAC interpolates x10 from
  10 GSPS and the C2R mixer eats one factor of two, giving `DAC_SR = 1 GHz`. The ADC
  decimates x10 from 2.5 GSPS, giving `ADC_SR = 250 MHz`. Changing these numbers in the
  config does not change the hardware - it only makes the software lie about it.
- **The receive fold.** The ADC samples at 2.5 GSPS, so a 4900 MHz signal folds down to
  `5000 - 4900 = 100 MHz`. The rule is **`ADC_NCO = 5000 - DAC_NCO`**, and the tone comes
  back at its own baseband frequency. Do not go much above 4900 MHz: the receive mixer
  also produces an image twice the fold away, and if the fold is small that image lands
  in band and ruins the capture.

The bitstream and its metadata live in `lib/mts/` next to the driver that loads them;
PYNQ needs `mts.hwh` beside `mts.bit` under the same basename.

## Config options

`config/parameters.json` holds everything the four executables share. A notebook reads it
and adds only what is specific to its own experiment.

| key | what it is | if it is wrong |
|---|---|---|
| `board.n_ch` | RF channels on the XM655, 16 | the capture is de-interleaved into the wrong number of lanes and every channel is garbage |
| `board.n_tile` | ADC and DAC tiles, 4 | the per-tile validators stop catching a short NCO table |
| `board.dac_sr` | DAC baseband rate, 1e9 Hz | the transmitted tone is not at the frequency you asked for |
| `board.adc_sr` | ADC baseband rate, 250e6 Hz | the FFT bin grid is wrong, so the tone lands between bins and every measured phase is meaningless |
| `rf.dac_nco` | transmit NCO in MHz, one per tile | the tone leaves at the wrong RF frequency |
| `rf.dac_zone` | transmit Nyquist zone, one per tile | the wanted image is attenuated instead of the unwanted one |
| `rf.adc_nco` | receive NCO in MHz, one per tile | must be `5000 - dac_nco`; otherwise the tone does not land in the captured band |
| `rf.adc_zone` | receive Nyquist zone, 2 because the fold is even | the fold is mixed to the wrong side and the tone disappears |
| `capture.n_cap` | samples per channel in one capture | must be a power of two; it sets the FFT bin grid the tone is snapped to |
| `capture.trig_hold_s` | how long `trig_cap` stays high | too short and the DAC stops playing mid capture, leaving the tail silent |
| `signal.tone_0_mhz` | baseband CW tone system 0 transmits, offset down from the tile NCO | snapped onto the nearest FFT bin at run time, so a value between bins is corrected, not rejected |
| `signal.tone_1_mhz` | baseband CW tone system 1 transmits - only the SIC notebooks read it | must land on a different FFT bin than `tone_0_mhz`, or the leakage and the link cannot be told apart |
| `signal.amp` | DAC drive, 16383 for the 14 bit full scale | above 16383 the samples wrap and the tone is no longer a tone |
| `bsic.regularization` | stage 2 lambda of `no_channel_sic`, added to the self interference covariance, in ADC counts squared | too small and the receive weight chases noise in the near singular covariance; too large and it stops nulling and only chases the link |
| `systems.dacs_0` | system 0's transmit elements, `overlay.dac[]` indices | weights go to DACs that are not wired to the antennas, and the capture is only crosstalk - it still looks like a signal, but the phase is noise |
| `systems.adcs_0` | system 0's own receivers - the near field | the self interference matrix `H00` describes the wrong ears |
| `systems.dacs_1` | system 1's transmit elements | `H01` is measured from the wrong node |
| `systems.adcs_1` | system 1's receivers - the far field | `H10` scores the beam at the wrong place |

A DAC or an ADC listed under both systems puts the same measurement into two matrices and
nothing downstream can tell; the notebooks check for that at startup.

## Executable files

Run each notebook top to bottom. Only one kernel at a time may hold the overlay.

| notebook | what it does | writes |
|---|---|---|
| `capture_and_record.ipynb` | transmit a tone or a chirp, capture all 16 ADCs, plot IQ and spectrum | nothing - it is the smoke test |
| `calibration_algo_measurments.ipynb` | sweep a table of beam weights, saving far and near field captures separately, and check trigger repeatability | `output/captures/` |
| `channel_estimation.ipynb` | excite every element of both systems, one-hot or Hadamard, and cut out the four channel matrices `H00 H01 H10 H11` | `output/channels/` |
| `one_side_fd_sic.ipynb` | only system 0 cancels, system 1 just transmits a second tone - depth, link preservation, SIR and far field reach over six aligned captures | `output/sic/` |
| `two_side_fd_sic.ipynb` | both sides cancelling at once, each on its own tone - depth, link preservation and SIR per side, over eight aligned captures | `output/two_side_sic/` |
| `no_channel_sic.ipynb` | two stage BSIC with no channel estimate: probe each side one DAC at a time, transmit down the quietest direction, then train the receive weights on solo captures | `output/no_channel_sic/` |

## Layout

```
config/parameters.json   the settings every executable shares
lib/
  config_parser.py       reads that file
  common_functions.py    arming a burst, raw -> IQ, tone generation, file chores
  mts/                   the board driver, the bitstream and its metadata
  fd/
    non_joint_sic_bsic.py  beamforming cancellation, one side at a time
    joint_sic_bsic.py      beamforming cancellation, both sides together
    dsic.py                digital cancellation
output/                  captures, channels and cancellation runs
```
