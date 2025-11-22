# ASK & FSK
# Aim
Write a simple Python program for the modulation and demodulation of ASK and FSK.
# Tools required
CO-LAB
# Program

# Program for ASK 
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
# Butterworth low-pass filter for demodulation
def butter_lowpass_filter(data, cutoff, fs, order=5):
  nyquist = 0.5 * fs
  normal_cutoff = cutoff / nyquist
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return lfilter(b, a, data)
# Parameters
fs = 1000 
f_carrier = 50 
bit_rate = 10 
T = 1 
t = np.linspace(0, T, int(fs * T), endpoint=False)
# Message signal (binary data)
bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
message_signal = np.repeat(bits, bit_duration)
# Carrier signal
carrier = np.sin(2 * np.pi * f_carrier * t)
# ASK Modulation
ask_signal = message_signal * carrier
# ASK Demodulation
demodulated = ask_signal * carrier 
filtered_signal = butter_lowpass_filter(demodulated, f_carrier, fs)
decoded_bits = (filtered_signal[::bit_duration] > 0.25).astype(int)
# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Message Signal (Binary)', color='b')
plt.title('Message Signal')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(t, carrier, label='Carrier Signal', color='g')
plt.title('Carrier Signal')
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(t, ask_signal, label='ASK Modulated Signal', color='r')
plt.title('ASK Modulated Signal')
plt.grid(True)
plt.subplot(4, 1, 4)
plt.step(np.arange(len(decoded_bits)), decoded_bits, label='Decoded Bits', color='r', marker='x')
plt.title('Decoded Bits')
plt.tight_layout()
plt.show()


```
# Program for FSK

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Butterworth low-pass filter for demodulation
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Parameters
fs = 1000         # Sampling frequency
bit_rate = 10     # Bits per second
f1 = 50           # Frequency for bit 1
f2 = 100          # Frequency for bit 0
T = 1             # Total signal duration in seconds
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Message signal (binary data)
bits = np.random.randint(0, 2, bit_rate)
bit_duration = fs // bit_rate
message_signal = np.repeat(bits, bit_duration)

# FSK Modulation
fsk_signal = np.zeros_like(t)
for i, bit in enumerate(bits):
    start = i * bit_duration
    end = (i + 1) * bit_duration
    freq = f1 if bit == 1 else f2
    fsk_signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])

# FSK Demodulation
# Multiply by both carriers and low-pass filter
carrier1 = np.sin(2 * np.pi * f1 * t)
carrier2 = np.sin(2 * np.pi * f2 * t)

demod1 = butter_lowpass_filter(fsk_signal * carrier1, bit_rate, fs)
demod2 = butter_lowpass_filter(fsk_signal * carrier2, bit_rate, fs)

# Decision based on which correlation is stronger
decoded_bits = np.zeros(bit_rate)
for i in range(bit_rate):
    start = i * bit_duration
    end = (i + 1) * bit_duration
    power1 = np.sum(demod1[start:end]**2)
    power2 = np.sum(demod2[start:end]**2)
    decoded_bits[i] = 1 if power1 > power2 else 0
decoded_bits = decoded_bits.astype(int)

# Plotting
plt.figure(figsize=(12, 10))
plt.subplot(5, 1, 1)
plt.step(np.arange(len(bits)), bits, where='post', label='Original Bits', color='b')
plt.title('Original Message Bits')
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t, message_signal, label='Message Signal (Binary)', color='b')
plt.title('Message Signal')
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t, fsk_signal, label='FSK Modulated Signal', color='r')
plt.title('FSK Modulated Signal')
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t, demod1, label='Demodulated at f1 (bit=1)', color='g')
plt.plot(t, demod2, label='Demodulated at f2 (bit=0)', color='orange')
plt.title('Demodulated Signals')
plt.legend()
plt.grid(True)

plt.subplot(5, 1, 5)
plt.step(np.arange(len(decoded_bits)), decoded_bits, where='post', label='Decoded Bits', color='r', marker='x')
plt.title('Decoded Bits')
plt.grid(True)

plt.tight_layout()
plt.show()

```

# Output Waveform

# ASK
<img width="1190" height="790" alt="image" src="https://github.com/user-attachments/assets/0e4aa155-c05c-409b-8446-921422ffa193" />
# FSK
<img width="1190" height="989" alt="image" src="https://github.com/user-attachments/assets/0e9afb4d-40c3-4298-a880-5b45bf1eb8cd" />


# Results

Thus successfully simulated and demoonstrated the ASK and FSK 


