import matplotlib.pyplot as plt
import numpy as np

# Parametry sygnału
amplitude = 1
frequency = 1.0  # w hercach (Hz)
duration = 4.0  # czas trwania w sekundach
sample_step = 0.001  # krok próbkowania w sekundach

# Tworzenie wektora czasu
time_vector = np.arange(0, duration, sample_step)

# Funkcja generująca sygnał prostokątny
def generate_square_wave(amplitude, frequency, time_vector):
    return (amplitude / 2) * np.sign(np.sin(2 * np.pi * frequency * time_vector)) + amplitude / 2

# Funkcja dodająca zakłócenia o rozkładzie jednostajnym
def add_uniform_noise(signal, noise_level):
    return signal + np.random.uniform(-noise_level, noise_level, len(signal))

# Funkcja estymująca sygnał (odszumianie)
def estimate_signal(signal, horizon):
    estimated = np.copy(signal)
    for i in range(horizon, len(signal)):
        estimated[i] = np.mean(signal[i-horizon:i])
    return estimated

# Funkcja obliczająca MSE
def calculate_mse(original, estimated):
    return np.mean((original - estimated) ** 2)

# Sygnał oryginalny
original_signal = generate_square_wave(amplitude, frequency, time_vector)

# Ustawienie poziomów szumów i horyzontów
noise_levels = np.arange(0.2, 1.01, 0.2)
selected_horizons = np.arange(1, 51)

# Wyznaczenie optymalnego horyzontu dla każdego poziomu szumu
optimal_horizons = {}

for noise_level in noise_levels:
    noisy_signal = add_uniform_noise(original_signal, noise_level)
    mse_values = [calculate_mse(original_signal, estimate_signal(noisy_signal, h)) for h in selected_horizons]
    optimal_horizon = selected_horizons[np.argmin(mse_values)]
    optimal_horizons[noise_level] = optimal_horizon

# Funkcja generująca i zapisująca wykresy
def generate_and_save_plots(noise_level, horizons):
    noisy_signal = add_uniform_noise(original_signal, noise_level)
    mse_values = [calculate_mse(original_signal, estimate_signal(noisy_signal, h)) for h in selected_horizons]

    for horizon in horizons:
        denoised_signal = estimate_signal(noisy_signal, horizon)

        # Ustawienie wielkości wykresów dla jednej serii
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Wykres sygnału oryginalnego i zaszumionego
        axs[0].plot(time_vector, original_signal, label='Sygnał oryginalny', color='blue', linewidth=2)
        axs[0].scatter(time_vector, noisy_signal, color='red', s=1, label='Sygnał zaszumiony')
        axs[0].set_title(f'Sygnał oryginalny i zaszumiony (szum = {noise_level})')
        axs[0].set_xlabel('Czas [s]')
        axs[0].set_ylabel('Amplituda')
        axs[0].legend()
        axs[0].grid(True)

        # Wykres sygnału odszumionego przy danym horyzoncie
        axs[1].plot(time_vector, original_signal, label='Sygnał oryginalny', color='blue', linewidth=2)
        axs[1].plot(time_vector, denoised_signal, label=f'Sygnał odszumiony (H={horizon})', color='green', linewidth=1)
        axs[1].set_title(f'Sygnał odszumiony dla H = {horizon}')
        axs[1].set_xlabel('Czas [s]')
        axs[1].set_ylabel('Amplituda')
        axs[1].legend()
        axs[1].grid(True)

        # Wykres MSE od horyzontu z zaznaczonym aktualnym horyzontem
        axs[2].plot(selected_horizons, mse_values, label='MSE od H', color='blue')
        axs[2].scatter(horizon, mse_values[horizon-1], color='red', label=f'H = {horizon}')
        axs[2].set_title(f'MSE dla różnych H (szum = {noise_level})')
        axs[2].set_xlabel('Horyzont pamięci')
        axs[2].set_ylabel('MSE')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        # Zapisanie wykresu do pliku PNG
        file_name = f'noise_{noise_level}_horizons_{horizon}.png'
        fig.savefig(file_name)
        plt.close(fig)

# Generowanie i zapisywanie wykresów dla wybranych horyzontów
for noise_level, optimal_horizon in optimal_horizons.items():
    horizons_to_plot = [optimal_horizon]
    if optimal_horizon - 3 >= 1:
        horizons_to_plot.append(optimal_horizon - 3)
    if optimal_horizon + 3 <= 50:
        horizons_to_plot.append(optimal_horizon + 3)

    generate_and_save_plots(noise_level, horizons_to_plot)
