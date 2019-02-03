from transmitter import *
from receiver import *
import analysis

import log
import multiprocessing
import matplotlib.pyplot as plt


def initialise(modulation=(AmPamDemodulator(), AmPamModulator())):
    r = Receiver(modulation[0])
    t = Transmitter(modulation[1])
    return r, t


if __name__ == '__main__':
    log.info('MAIN')
    multiprocessing.freeze_support()

    with analysis.AnalysisChannel() as ac:
        ac.test_transmission()
        ac.calculate_error()
        figax = plt.plot()
        ac.plot_error(figax)
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


