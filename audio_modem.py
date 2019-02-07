from transmitter import *
from receiver import *
import analysis

import log
import multiprocessing
import matplotlib.pyplot as plt


if __name__ == '__main__':
    log.info('MAIN')
    multiprocessing.freeze_support()

    # # Todo only one channel can have control of the audio device at once
    # with analysis.AnalysisChannel(channel='ch1') as ac1:
    #     ac1.r.record()
    #     ac1.test_transmission(bit_count=10000, threshold=0.2)
    #     ac1.calculate_error()
    #     ac1.plot_error()
    #     ac1.plot_demodulating_blocks()
    #     plt.show()

    r = Receiver(PamDemodulator('ch1'))
    t = Transmitter(PamModulator('ch1'))

    fig, ax = plt.subplots(nrows=3)

    pulse_shift, window_magnitude, beta = 1024, 5, 0.5
    pulse = r.sig.get_root_raised_cosine(pulse_shift, beta, width=window_magnitude)

    # t.show_modulated_signal(Packet([1,-1,1,0,1,0,1,0]))
    sig = PamModulator('ch1').general_pam_mod([1,-1,1,0,1,0,1,0], 1024, pulse)
    ax[0].plot(sig)
    bits = PamDemodulator('ch1').general_pam_demod(sig, 1024, 8, pulse)
    ax[1].plot(bits)
    plt.show()

    # fig, ax = plt.subplots(nrows=2, sharex='all')
    # ax0 = ax[0]
    # ax1 = ax[1]
    # raw_audio = named_deque()
    # listen_queue = named_deque()
    # filter_data = named_deque()
    # r.record(raw_audio, listen_queue, filter_data)
    # filter_output_queue = named_deque()
    # r.convolve_queue(filter_data, filter_output_queue, r.sig.get_sync_pulse_matched_filter())
    # r.show(raw_audio, (fig, ax0), show=False, interval=500)
    # r.show(filter_output_queue, (fig, ax1), show=False, interval=500)
    # plt.show()


