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

    def test_transmission(self, bit_count=100, threshold=0.5):
        random.seed(100)
        test_packet = Packet([(random.getrandbits(1)) for i in range(bit_count)])
        log.info(f'Sending test transmission - length={len(test_packet.unpack())}')
        self.sent_data = test_packet.unpack()

        self.r.record()
        data_bits_output_queue = named_deque()
        self.r.listen(data_bits_output_queue, threshold=threshold)

        # Transmit and wait till demodulated
        self.t.transmit(test_packet)
        while not self.r.demodulator.demodulated_flag:
            time.sleep(0.1)
        self.r.demodulator.demodulated_flag = False
        self.received_data = np.concatenate(list(data_bits_output_queue))

        received_bytes = np.packbits(self.received_data)
        text = bytes(received_bytes)
        log.special(f'Received data in bytes form {text}')

    def calculate_error(self):
        errors = 0
        N1, N2 = len(self.sent_data), len(self.received_data)
        if N1 != N2:
            log.warning(f'Did not receive correct number of bits, expected {N1}, received {N2}')
        for b1, b2 in zip(self.sent_data, self.received_data):
            if b1 != b2:
                errors += 1
        log.special(f'No of errors: {errors}')
        log.special(f'Percent error: {100*errors/max(N1, N2):3}%')

    def plot_error(self, figax=None, title=None):
        N1, N2 = len(self.sent_data), len(self.received_data)
        if N1 != N2:
            log.error('Cannot plot errors with arrays of different lengths')
            return None
        errors = np.zeros(N1)
        for b1, b2, i in zip(self.sent_data, self.received_data, range(N1)):
            if b1 != b2:
                errors[i] = 1
        errors_smooth = scipy_sig.convolve(errors, [0.02]*50, mode='valid')

        if figax:
            fig, ax = figax
        else:
            fig, ax = plt.subplots(nrows=1,)
        ax.plot(errors, color='r')
        ax.plot(errors_smooth, color='b', linestyle='--')
        plt.xlabel('Bit transmitted')
        plt.ylabel('Error')
        plt.title(f'plot_error - {title}')

    def plot_demodulating_blocks(self, figax=None, title=None):
        if not self.r.demodulator.debug_data:
            log.error('Cannot plot decoding blocks, no debug_data exists in r.demodulator')
            return

        # Get/build axis and get data
        demodulator_debug_data = self.r.demodulator.debug_data
        if figax:
            fig, ax = figax
        else:
            N = len(self.r.demodulator.debug_data)
            if N >= 6:
                log.debug('Too much debug data to show all, showing first and last 3 blocks')
                N = 6
                demodulator_debug_data = np.append(demodulator_debug_data[:3], demodulator_debug_data[-3:])
            fig, ax = plt.subplots(nrows=N,)

        # For each axis, plot elements of debug data that are present
        for i, debug_data in enumerate(demodulator_debug_data):
            # Plot audio data for each block
            audio_data = debug_data.get('audio_data', [])
            if len(audio_data):
                ax[i].plot(audio_data)
            else:
                log.warning('No audio data found in demodulator.debug_data')

            bits = debug_data.get('bits', [])
            symbols = debug_data.get('symbols', [])
            pw = debug_data.get('pulse_width')
            pc = debug_data.get('pulse_count')
            if (len(bits) or len(symbols)) and pw and pc:
                if len(bits):
                    symbols = bits
                X = np.linspace(pw//2, pw * (pc - 0.5)//1, len(symbols))
                for x, symbol in zip(X, symbols):
                    ax[i].text(x, 0, symbol)

            xlines = debug_data.get('xlines',[])
            if xlines:
                for line in xlines:
                    ax[i].axhline(line, color='r', linestyle='--')

        plt.suptitle(f'plot_decoding_blocks - {title}')
