


# Määrittele havainnoista kurssilla oppimasi perusteella seuraavat asiat ja esitä ne numeroina visualisoinnissasi:
#
# - Askelmäärä laskettuna suodatetusta kiihtyvyysdatasta
#
# - Askelmäärä laskettuna kiihtyvyysdatasta Fourier-analyysin perusteella
#
# - Keskinopeus (GPS-datasta)
#
# - Kuljettu matka (GPS-datasta)
#
# - Askelpituus (lasketun askelmäärän ja matkan perusteella)
#
# Esitä seuraavat kuvaajat
#
# - Suodatettu kiihtyvyysdata, jota käytit askelmäärän määrittelemiseen.
#
# - Analyysiin valitun kiihtyvyysdatan komponentin tehospektritiheys
#
# - Reittisi kartalla

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium
from scipy.signal import butter, filtfilt


df = pd.read_csv('Linear Accelerometer.csv')
gps = pd.read_csv('Location.csv')


acc_time = df['Time (s)']
acc_x = df['X (m/s^2)']
acc_y = df['Y (m/s^2)']
acc_z = df['Z (m/s^2)']

gps_time = gps['Time (s)']
latitudes = gps['Latitude (°)']
longitudes = gps['Longitude (°)']
heights = gps['Height (m)']


time_intervals = np.diff(acc_time)
fs = 1 / np.mean(time_intervals)


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

cutoff = 3.0
fs = 1 / np.mean(time_intervals)

acc_filtered_scipy = butter_lowpass_filter(acc_x, cutoff, fs)
acc_filtered = butter_lowpass_filter(acc_x, cutoff, fs)

jaksot = 0
for i in range(len(acc_filtered_scipy) - 1):
    if acc_filtered_scipy[i] / acc_filtered_scipy[i + 1] < 0:
        jaksot += 1

step_count_scipy_filtered = np.floor(jaksot / 2)

df = pd.read_csv('Linear Accelerometer.csv')

f = df['X (m/s^2)']
t = df['Time (s)']
N = len(df)
dt = np.max(t)/len(t)

fourier = np.fft.fft(f,N)
psd = fourier *np.conj(fourier)/N
freq = np.fft.fftfreq(N,dt)
L = np.arange(1,int(N/2))
plt.plot(freq[L],psd[L].real)
plt.ylabel('Teho')
plt.xlabel('Taajuus')
plt.grid()
plt.axis([0,10,0,500])

walking_range = (freq >= 1) & (freq <= 3)
psd_walking = psd[walking_range]

dominant_freq_fourier = freq[walking_range][np.argmax(psd_walking)]

total_time = t.iloc[-1] - t.iloc[0]
step_count_fourier = (dominant_freq_fourier * total_time)



def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


distances = [haversine(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1])
             for i in range(len(latitudes) - 1)]
total_distance = sum(distances)
average_speed = total_distance / (gps_time.iloc[-1] - gps_time.iloc[0])
average_speed_kmh = average_speed * 3.6


step_length = total_distance / step_count_scipy_filtered




st.subheader('Filteröity Kiihtyvyys data (x-komponentti)')
plt.figure()
plt.plot(acc_time, acc_filtered)
plt.xlabel('Aika (s)')
plt.ylabel('Kiihtyvyys (m/s^2)')
st.pyplot(plt)


st.subheader('Kiihtyvyysdatan X tehospektri')
plt.figure()
plt.plot(freq[L], psd[L].real)
plt.xlabel('Taajuus (Hz)')
plt.ylabel('Teho')
st.pyplot(plt)


st.write(f"Askelmäärä suodetettusta datasta: {step_count_scipy_filtered}")
st.write(f"Askelmäärä Fourier analyysilla: {int(step_count_fourier)}")
st.write(f"Keskinopeus: {average_speed_kmh:.2f} km/h")
st.write(f"Matka: {total_distance:.2f} meters")
st.write(f"Askelpituus: {step_length:.2f} meters")

# kartta
start_lat = gps['Latitude (°)'].mean()
start_long = gps['Longitude (°)'].mean()

map = folium.Map(location = [start_lat,start_long], zoom_start = 14)

folium.PolyLine(gps[['Latitude (°)', 'Longitude (°)']], color ='blue', weight = 3.5, opacity = 1).add_to(map)

st_map = st_folium(map, width=900, height=650)



