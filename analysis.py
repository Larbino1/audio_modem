from transmitter import *
from receiver import *

import log
import multiprocessing
import matplotlib.pyplot as plt


class AnalysisChannel:
    def __init__(self, modulation=(AmPamDemodulator(), AmPamModulator())):
        self.r = Receiver(modulation[0])
        self.t = Transmitter(modulation[1])

        self.sent_data = None
        self.received_data = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.r.stop()
        self.t.stop()

    def test_transmission(self):
        random.seed(100)
        test_packet = Packet([(random.getrandbits(1)) for i in range(100)])
        log.info(f'Sending test transmission - length={len(test_packet.unpack())}')
        self.sent_data = test_packet.unpack()

        self.r.record()
        data_bits_output_queue = named_deque()
        self.r.listen(data_bits_output_queue, threshold=0.5)

        # Transmit and wait till demodulated
        self.t.transmit(test_packet)
        while not self.r.demodulator.demodulated_flag:
            time.sleep(0.1)
        self.r.demodulator.demodulated_flag = False

        self.received_data = np.concatenate(list(data_bits_output_queue))

        received_bytes = np.packbits(self.received_data)
        text = bytes(received_bytes)
        log.special(text)

    def calculate_error(self):
        errors = 0
        N1, N2 = len(self.sent_data), len(self.received_data)
        if N1 != N2:
            log.warning('Did not receive correct number of bits, expected {N1}, received {N2}')
        for b1, b2 in zip(self.sent_data, self.received_data):
            if b1 != b2:
                errors += 1
        log.special(f'No of errors: {errors}')
        log.special(f'Percent error: {100*errors/max(N1, N2):3}%')

    def plot_error(self, figax):
        N1, N2 = len(self.sent_data), len(self.received_data)
        if N1 != N2:
            log.error('Cannot plot errors with arrays of different lengths')
            return None
        errors = np.zeros(N1)
        for b1, b2, i in zip(self.sent_data, self.received_data, range(N1)):
            if b1 != b2:
                errors[i] = 1
        fig, ax = figax
        ax.plot(errors)



