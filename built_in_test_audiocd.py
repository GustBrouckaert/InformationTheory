from CD_template.AudioCD import AudioCD

import numpy as np
import math
import matplotlib.pyplot as plt
from CD_template.AudioCD import AudioCD
import wave

def test_scratch_sweep(l_values):
    # --- Load audio once ---
    wave_object = wave.open('Hallelujah.wav','rb')
    Fs = wave_object.getframerate()
    nch = wave_object.getnchannels()
    depth = wave_object.getsampwidth()

    sdata = wave_object.readframes(wave_object.getnframes())
    typ = {1: np.int8, 2: np.int16, 4: np.int32}.get(depth)
    data = np.frombuffer(sdata, dtype=typ) / (2**15)

    ch_1 = data[0::nch]
    ch_2 = data[1::nch]
    audiofile = np.transpose(np.vstack((ch_1, ch_2)))

    # --- Results storage ---
    erasures = []
    failed_interp = []
    undetected = []

    # --- Sweep ---
    for l_scratch in l_values:
        
            print(f"Running for l_scratch = {l_scratch}")

            cd = AudioCD(Fs, 1, 8)
            cd.writeCd(audiofile)

            T_scratch = 600000

            for i in range(math.floor(cd.cd_bits.size / T_scratch)):
                cd.scratchCd(l_scratch, 30000 + i * T_scratch)
           
            
            out, interpolation_flags = cd.readCd()

            # Metrics
            erasures.append(np.sum(interpolation_flags != 0))
            failed_interp.append(np.sum(interpolation_flags == -1))
            undetected.append(
                np.sum(out[interpolation_flags == 0] != 
                       cd.scaled_quantized_padded_original[interpolation_flags == 0])
            )

    print(failed_interp)

    # --- Plot ---
    plt.figure()
    plt.plot(l_values, erasures, label="Erasures")
    plt.plot(l_values, failed_interp, label="Interpolation failed")
    plt.plot(l_values, undetected, label="Undetected errors")
    plt.xlabel("Scratch length (bits)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("CIRC performance vs scratch length")
    plt.grid()
    plt.show()

def test_scratch_location_sweep(scratch_locations):
    import numpy as np
    import wave
    import matplotlib.pyplot as plt

    # --- Load audio once ---
    wave_object = wave.open('Hallelujah.wav','rb')
    Fs = wave_object.getframerate()
    nch = wave_object.getnchannels()
    depth = wave_object.getsampwidth()

    sdata = wave_object.readframes(wave_object.getnframes())
    typ = {1: np.int8, 2: np.int16, 4: np.int32}.get(depth)
    data = np.frombuffer(sdata, dtype=typ) / (2**15)

    ch_1 = data[0::nch]
    ch_2 = data[1::nch]
    audiofile = np.transpose(np.vstack((ch_1, ch_2)))

    # --- Fixed scratch length ---
    l_scratch = 4096

    # --- Results storage ---
    erasures = []
    failed_interp = []
    undetected = []

    # --- Sweep over locations ---
    for loc in scratch_locations:
        print(f"Running for scratch_location = {loc}")

        cd = AudioCD(Fs, 1, 8)
        cd.writeCd(audiofile)

        # SINGLE scratch
        cd.scratchCd(l_scratch, int(loc))

        out, interpolation_flags = cd.readCd()

        # Metrics
        erasures.append(np.sum(interpolation_flags != 0))
        failed_interp.append(np.sum(interpolation_flags == -1))
        undetected.append(
            np.sum(out[interpolation_flags == 0] != 
                   cd.scaled_quantized_padded_original[interpolation_flags == 0])
        )

    # --- Plot ---
    plt.figure()
    plt.plot(scratch_locations, erasures, label="Erasures")
    plt.plot(scratch_locations, failed_interp, label="Interpolation failed")
    plt.plot(scratch_locations, undetected, label="Undetected errors")
    plt.xlabel("Scratch start position (bits)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("CIRC performance vs scratch location (length = 4096)")
    plt.grid()
    plt.show()



def load_audio():
    wave_object = wave.open('Hallelujah.wav','rb')
    Fs = wave_object.getframerate()
    nch = wave_object.getnchannels()
    depth = wave_object.getsampwidth()

    sdata = wave_object.readframes(wave_object.getnframes())
    typ = {1: np.int8, 2: np.int16, 4: np.int32}.get(depth)
    data = np.frombuffer(sdata, dtype=typ) / (2**15)

    ch_1 = data[0::nch]
    ch_2 = data[1::nch]
    audiofile = np.transpose(np.vstack((ch_1, ch_2)))

    return Fs, audiofile

def test_scratches():
    Fs, audiofile = load_audio()

    configs = [0, 1, 2, 3]
    scratch_lengths = [100, 3000, 10000]
    T_scratch = 600000

    results_erasures = np.zeros((len(configs), len(scratch_lengths)))
    results_failed = np.zeros((len(configs), len(scratch_lengths)))

    for ci, config in enumerate(configs):
        for li, l_scratch in enumerate(scratch_lengths):

            print(f"Config {config}, scratch length {l_scratch}")

            cd = AudioCD(Fs, config, 8)
            cd.writeCd(audiofile)

            # apply periodic scratches
            for i in range(math.floor(cd.cd_bits.size / T_scratch)):
                cd.scratchCd(l_scratch, 30000 + i * T_scratch)

            out, flags = cd.readCd()

            erasures = np.sum(flags != 0)
            failed = np.sum(flags == -1)

            total_samples = len(flags)

            results_erasures[ci, li] = erasures
            results_failed[ci, li] = failed

    # --- Plot ---
    #eps = 1e-10
    for ci, config in enumerate(configs):
        plt.plot(scratch_lengths, results_erasures[ci], marker='o', label=f"C{config} erasures")
        plt.plot(scratch_lengths, results_failed[ci], marker='x', linestyle='--', label=f"C{config} failed_interpolation")
    #plt.yscale('log')
    plt.xlabel("Scratch length (bits)")
    plt.ylabel("Count")
    plt.title("Scratch performance comparison")
    plt.legend()
    plt.grid()
    plt.show()


def test_random_errors():
    Fs, audiofile = load_audio()

    configs = [1, 2, 3]  # config 0 excluded
    p_values = np.logspace(-1 - math.log10(2), -3, 10)

    results_erasures = np.zeros((len(configs), len(p_values)))
    results_failed = np.zeros((len(configs), len(p_values)))

    for ci, config in enumerate(configs):
        for pi, p in enumerate(p_values):

            print(f"Config {config}, p={p:.5f}")

            cd = AudioCD(Fs, config, 8)
            cd.writeCd(audiofile)

            cd.bitErrorsCd(p)

            out, flags = cd.readCd()

            erasures = np.sum(flags != 0)
            failed = np.sum(flags == -1)

            total_samples = flags.size

            results_erasures[ci, pi] = erasures / total_samples
            results_failed[ci, pi] = failed / total_samples

    # --- Plot ---
    plt.figure()

    for ci, config in enumerate(configs):
        plt.semilogx(p_values, results_erasures[ci], marker='o', label=f"C{config} erasures")
        plt.semilogx(p_values, results_failed[ci], marker='x', linestyle='--', label=f"C{config} failed")

    plt.xlabel("Bit error probability (p)")
    plt.ylabel("Probability")
    plt.title("Random error performance")
    plt.legend()
    plt.grid()
    plt.show()


# l_values_high_resolution = np.linspace(3000, 5000, 100, dtype=int)
# l_values_low_resolution = np.linspace(5000, 8000, 10, dtype=int)
# scratch_locations = [3000000 + i for i in range(0, 28672, 512)] # Scratch in the middle of the disc, test only 1 scratch
# #test_scratch_sweep(np.concatenate((l_values_high_resolution, l_values_low_resolution))) # Scratch in the middle of the disc, test only 1 scratch
# test_scratch_location_sweep(scratch_locations)
# Run the built-in test
#AudioCD.test()

#test_scratches()
test_random_errors()