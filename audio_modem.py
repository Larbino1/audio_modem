from transmitter import *
from receiver import *

import log
import multiprocessing
import matplotlib.pyplot as plt


def initialise(modulation=(AmPamDemodulator(), AmPamModulator())):
    r = Receiver(modulation[0])
    t = Transmitter(modulation[1])
    return r, t


random.seed(100)
test_packet = Packet([(random.getrandbits(1)) for i in range(1000)])

if __name__ == '__main__':
    log.info('MAIN')
    multiprocessing.freeze_support()

    r, t = initialise()

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

    data_bits_output_queue = named_deque()

    r.record()
    log.special(f'Test_data: = {r.bop.bit_array_to_str(test_packet.unpack())}')
    t.transmit(test_packet)
    r.listen(data_bits_output_queue, threshold=0.5)

    time.sleep(10)

    received_bits = np.concatenate(list(data_bits_output_queue))
    print(r.bop.bit_array_to_str(received_bits))
    print(len(received_bits))

    test_data = test_packet.unpack()

    # TODO plot errors against time for more insight
    errors = 0
    assert len(test_data) == len(received_bits), 'did not receive correct number of bits'
    for b1, b2 in zip(test_data, received_bits):
        if b1 != b2:
            errors+=1
    log.special(f'No of errors: {errors}')
    log.special(f'Percent error: {100*errors/len(test_data):3}%')

    received_bytes = np.packbits(received_bits)
    text = bytes(received_bytes)
    print(text)

    r.stop()
    t.stop()



