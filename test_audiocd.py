"""
Test script for AudioCD CIRC implementation
Helps answer assignment questions about burst error correction, interpolation, and performance comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from CD_template.AudioCD import AudioCD
import math

# ============================================================================
# PART 1: Test with different scratch widths (Question 3 & 5a)
# ============================================================================

def test_scratch_performance():
    """
    Test performance with scratches of different widths (100, 3000, 10000 bits)
    repeated with period 600000 bits (simulating disc rotation)
    """
    print("\n" + "="*80)
    print("TEST 1: SCRATCH ERROR PERFORMANCE")
    print("="*80)
    
    # Create a simple audio signal for testing
    Fs = 44100  # CD audio sample rate
    duration = 0.5  # 0.5 seconds
    t = np.arange(int(Fs * duration)) / Fs
    # Simple sine wave test signal
    audio_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine
    audiofile = np.column_stack((audio_signal, audio_signal))
    
    scratch_widths = [100, 3000, 10000]
    configurations = [0, 1, 2, 3]
    config_names = {0: "No CIRC", 1: "CIRC Standard", 2: "Concatenated RS", 3: "Single 32,24 RS"}
    
    results = {cfg: [] for cfg in configurations}
    
    for scratch_width in scratch_widths:
        print(f"\n--- Scratch width: {scratch_width} bits ---")
        
        for config in configurations:
            cd = AudioCD(Fs, config, 8)
            cd.writeCd(audiofile)
            
            # Add scratches at regular intervals (600000 bits period)
            T_scratch = 600000
            n_scratches = math.floor(cd.cd_bits.size / T_scratch)
            for i in range(n_scratches):
                location = 30000 + i * T_scratch
                cd.scratchCd(scratch_width, location)
            
            # Read and check for errors
            audio_out, interpolation_flags = cd.readCd()
            
            # Calculate statistics
            n_erasures = np.sum(interpolation_flags != 0)
            n_failed_interp = np.sum(interpolation_flags == -1)
            total_samples = interpolation_flags.size
            
            prob_erasure = n_erasures / total_samples
            prob_failed = n_failed_interp / total_samples
            
            print(f"  Config {config} ({config_names[config]}):")
            print(f"    Erasure probability: {prob_erasure:.6f} ({n_erasures}/{total_samples})")
            print(f"    Failed interpolation probability: {prob_failed:.6f} ({n_failed_interp}/{total_samples})")
            
            results[config].append({
                'scratch_width': scratch_width,
                'prob_erasure': prob_erasure,
                'prob_failed': prob_failed,
                'n_erasures': n_erasures,
                'n_failed': n_failed_interp
            })
    
    return results


# ============================================================================
# PART 2: Test with random bit errors (Question 5b)
# ============================================================================

def test_random_bit_errors():
    """
    Test performance with random bit errors at various BER values
    BER range: 0.05 to 0.001 (10 logarithmically spaced values)
    """
    print("\n" + "="*80)
    print("TEST 2: RANDOM BIT ERROR PERFORMANCE")
    print("="*80)
    
    # Create a simple audio signal for testing
    Fs = 44100  # CD audio sample rate
    duration = 0.5  # 0.5 seconds
    t = np.arange(int(Fs * duration)) / Fs
    audio_signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine
    audiofile = np.column_stack((audio_signal, audio_signal))
    
    # Generate BER values: logarithmically spaced from 0.05 to 0.001
    # Use the formula from the assignment
    ber_values = np.logspace(-1 - math.log10(2), -3, 10)
    
    configurations = [1, 2, 3]  # Skip config 0 (no error correction)
    config_names = {1: "CIRC Standard", 2: "Concatenated RS", 3: "Single 32,24 RS"}
    
    results = {cfg: {'ber': [], 'prob_erasure': [], 'prob_failed': []} for cfg in configurations}
    
    for ber in ber_values:
        print(f"\nTesting BER = {ber:.6f}")
        
        for config in configurations:
            cd = AudioCD(Fs, config, 8)
            cd.writeCd(audiofile)
            
            # Add random bit errors
            cd.bitErrorsCd(ber)
            
            # Read and check for errors
            audio_out, interpolation_flags = cd.readCd()
            
            # Calculate statistics
            n_erasures = np.sum(interpolation_flags != 0)
            n_failed_interp = np.sum(interpolation_flags == -1)
            total_samples = interpolation_flags.size
            
            prob_erasure = n_erasures / total_samples
            prob_failed = n_failed_interp / total_samples
            
            results[config]['ber'].append(ber)
            results[config]['prob_erasure'].append(prob_erasure)
            results[config]['prob_failed'].append(prob_failed)
            
            print(f"  Config {config} ({config_names[config]}):")
            print(f"    Erasure probability: {prob_erasure:.6f}")
            print(f"    Failed interpolation probability: {prob_failed:.6f}")
    
    return results


# ============================================================================
# PART 3: Plotting and Visualization
# ============================================================================

def plot_scratch_results(scratch_results):
    """Plot scratch performance results"""
    print("\nGenerating scratch performance plots...")
    
    configurations = [1, 2, 3]
    config_names = {1: "CIRC Standard", 2: "Concatenated RS", 3: "Single 32,24 RS"}
    colors = {1: 'b', 2: 'g', 3: 'r'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Erasure probability
    ax = axes[0]
    for config in configurations:
        scratch_widths = [r['scratch_width'] for r in scratch_results[config]]
        prob_erasures = [r['prob_erasure'] for r in scratch_results[config]]
        ax.plot(scratch_widths, prob_erasures, 'o-', label=config_names[config], color=colors[config], linewidth=2)
    ax.set_xlabel('Scratch Width (bits)', fontsize=12)
    ax.set_ylabel('Erasure Probability', fontsize=12)
    ax.set_title('Erasure Probability vs Scratch Width', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Failed interpolation probability
    ax = axes[1]
    for config in configurations:
        scratch_widths = [r['scratch_width'] for r in scratch_results[config]]
        prob_failed = [r['prob_failed'] for r in scratch_results[config]]
        ax.plot(scratch_widths, prob_failed, 's-', label=config_names[config], color=colors[config], linewidth=2)
    ax.set_xlabel('Scratch Width (bits)', fontsize=12)
    ax.set_ylabel('Failed Interpolation Probability', fontsize=12)
    ax.set_title('Failed Interpolation vs Scratch Width', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scratch_performance.png', dpi=150, bbox_inches='tight')
    print("Saved: scratch_performance.png")
    plt.close()


def plot_ber_results(ber_results):
    """Plot random bit error performance results"""
    print("\nGenerating BER performance plots...")
    
    configurations = [1, 2, 3]
    config_names = {1: "CIRC Standard", 2: "Concatenated RS", 3: "Single 32,24 RS"}
    colors = {1: 'b', 2: 'g', 3: 'r'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Erasure probability vs BER (log-log scale)
    ax = axes[0]
    for config in configurations:
        ber_vals = np.array(ber_results[config]['ber'])
        prob_erasures = np.array(ber_results[config]['prob_erasure'])
        ax.loglog(ber_vals, prob_erasures, 'o-', label=config_names[config], color=colors[config], linewidth=2, markersize=8)
    ax.set_xlabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_ylabel('Erasure Probability', fontsize=12)
    ax.set_title('Erasure Probability vs BER (Log-Log Scale)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Failed interpolation probability vs BER (log-log scale)
    ax = axes[1]
    for config in configurations:
        ber_vals = np.array(ber_results[config]['ber'])
        prob_failed = np.array(ber_results[config]['prob_failed'])
        ax.loglog(ber_vals, prob_failed, 's-', label=config_names[config], color=colors[config], linewidth=2, markersize=8)
    ax.set_xlabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_ylabel('Failed Interpolation Probability', fontsize=12)
    ax.set_title('Failed Interpolation vs BER (Log-Log Scale)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('ber_performance.png', dpi=150, bbox_inches='tight')
    print("Saved: ber_performance.png")
    plt.close()


# ============================================================================
# PART 4: Analysis and Answer Generation
# ============================================================================

def analyze_burst_correction_capability():
    """
    Answer Question 3: Maximum burst duration that can be corrected
    
    Key facts:
    - C2 can correct up to 4 erasures
    - Each frame contains 24 data bytes
    - Delay lines of unequal length create interleaving
    - Scanning velocity: 1.3 m/s
    - Sample rate: 44.1 kHz
    """
    print("\n" + "="*80)
    print("QUESTION 3: MAXIMUM BURST DURATION")
    print("="*80)
    
    print("""
THEORETICAL ANALYSIS:
- C2 (Reed-Solomon code) can correct up to 4 erasures per codeword
- Each C2 codeword contains 24 data bytes + 4 parity bytes = 28 bytes total
- After CIRC encoding, the frame structure is: 28 bytes per C2 frame
- The delay lines of unequal length spread errors across multiple C2 frames
- With proper interleaving, a burst is distributed across different C2 frames

Maximum burst that can be ALWAYS corrected:
- Since C2 can correct 4 erasures, and symbols are interleaved across frames,
  we need to ensure a burst doesn't affect more than 4 symbols in any single C2 codeword
- With the 27-frame unequal delay interleaving, a burst can be spread across
  multiple frames, providing protection
- The effective maximum correctable burst = 4 symbols × 27 frames × 8 bits/symbol
  = 4 × 27 × 8 = 864 bits (approximately)

More realistic assessment:
- The interleaving spreads burst errors, so a continuous burst in the channel
  becomes scattered in different C2 codewords
- With 27 frames of interleaving, a burst ≤ 27 frames can often be corrected
- Since each frame is 28 bytes = 224 bits, a burst of ~27 × 224 = 6,048 bits
  can potentially be recovered

PHYSICAL SCRATCH CALCULATION:
- Scanning velocity: 1.3 m/s
- Bit length on CD: The CD has a linear bit density
- Data rate: 44.1 kHz × 16 bits × 2 channels = 1.4112 Mbit/s
- Bit period: 1 / 1.4112 MHz ≈ 709 ns
- Physical distance per bit: 1.3 m/s × 709 ns ≈ 0.92 µm

Maximum scratch width:
- For ~864 bits burst: ~864 × 0.92 µm ≈ 0.8 mm
- For ~6,048 bits burst: ~6,048 × 0.92 µm ≈ 5.6 mm

This should be compared with simulation results from test_scratch_performance()
    """)


def print_analysis_summary():
    """Print summary of key findings for all questions"""
    print("\n" + "="*80)
    print("SUMMARY FOR ASSIGNMENT QUESTIONS")
    print("="*80)
    
    print("""
QUESTION 1: CIRC COMPONENT FUNCTIONS
--------------------------------------
1. Delay of 2 frames at input:
   - Introduces initial delay
   - Allows data buffering for processing

2. Interleaving sequence:
   - Spreads consecutive data bytes across different frames
   - Converts burst errors into isolated single errors
   - Protects against burst errors

3. C2 encoder:
   - Reed-Solomon (255,251,5) code
   - Adds 4 parity symbols per codeword
   - Corrects random errors
   - Protects against: random errors

4. Delay lines of unequal length:
   - 27 frames of delays
   - Each delay line has different length (0-26 frames)
   - Interleaves C2 codewords across C1 codewords
   - Converts C2 errors into isolated C1 errors
   - Protects against: burst errors

5. C1 encoder:
   - Reed-Solomon (255,251,5) code
   - Adds 4 parity symbols per codeword
   - Corrects errors flagged by C2 decoder
   - Can correct both errors and erasures
   - Protects against: burst errors (that survived C2)

6. Delay of 1 frame + inversions:
   - Delay for synchronization
   - Inversions of parity bytes add error detection capability
   - Prepares data for EFM modulation

ERROR TYPES PROTECTED:
- C1: Primarily protects against BURST ERRORS (via interleaving from unequal delays)
- C2: Primarily protects against RANDOM ERRORS (within each codeword)
- Together: Can correct both random and burst errors


QUESTION 2: Implementation
--------------------------------------
✓ All 10 functions implemented:
  - CIRC_enc_delay_interleave
  - CIRC_enc_C2
  - CIRC_enc_delay_unequal
  - CIRC_enc_C1
  - CIRC_enc_delay_inv
  - CIRC_dec_delay_inv
  - CIRC_dec_C1
  - CIRC_dec_delay_unequal
  - CIRC_dec_C2
  - CIRC_dec_deinterleave_delay


QUESTION 4: LINEAR INTERPOLATION
--------------------------------------
Why linear interpolation works:
- Audio signals have high correlation between adjacent samples
- Errors flagged by CIRC decoder are typically isolated (≤8 samples)
- Linear interpolation between adjacent valid samples is perceptually acceptable
- Maximum of 8 consecutive erasures matches musical signal characteristics

Relation to sample frequency:
- At 44.1 kHz, 8 samples = 181 µs duration
- This is short enough (< 1 ms) that linear interpolation is imperceptible
- Ear cannot perceive audio gaps shorter than ~2-3 ms
- Hence interpolation of up to 8 samples is inaudible


QUESTION 5: PERFORMANCE COMPARISON
--------------------------------------
Run test_scratch_performance() for different scratch widths
Run test_random_bit_errors() for BER analysis

Expected behavior:
- Config 0 (No CIRC): High error rate, unacceptable
- Config 1 (CIRC Standard): Best performance for burst errors
- Config 2 (Concatenated RS): Good performance, simpler
- Config 3 (Single 32,24 RS): Simpler but less effective for bursts


QUESTION 6: EFM MODULATION
--------------------------------------
EFM = Eight-to-Fourteen Modulation
- Converts 8-bit data into 14-bit encoded words
- Ensures sufficient transitions for clock recovery
- Prevents long runs of same bit (aids timing recovery)
- Adds error detection capability through specific bit patterns
- Why: CD players need stable clock signal from pit edges
         EFM guarantees sufficient transitions for reliable clock extraction
    """)


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# AudioCD CIRC IMPLEMENTATION TEST AND ANALYSIS")
    print("#"*80)
    
    # Question 1 & 3 Analysis
    analyze_burst_correction_capability()
    
    # Question 3 & 5a: Test scratch performance
    print("\nRunning scratch performance tests...")
    scratch_results = test_scratch_performance()
    plot_scratch_results(scratch_results)
    
    # Question 5b: Test random bit errors
    print("\nRunning random bit error tests...")
    ber_results = test_random_bit_errors()
    plot_ber_results(ber_results)
    
    # Print summary
    print_analysis_summary()
    
    print("\n" + "#"*80)
    print("# TEST COMPLETE")
    print("#"*80)
    print("\nGenerated files:")
    print("  - scratch_performance.png")
    print("  - ber_performance.png")
    print("\nUse these results to answer assignment questions 3, 4, and 5")
