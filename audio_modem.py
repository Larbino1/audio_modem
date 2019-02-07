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

    t.show_modulated_signal(Packet([1,0,1,0,1,0,1,0]))
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


