import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import folium
from streamlit_folium import st_folium

# Ladataan kiihtyvyys- ja GPS-data
path_acc = "Linear_Acceleration.csv"
path_gps = "Location.csv"
df_acc = pd.read_csv(path_acc)
df_gps = pd.read_csv(path_gps)

# Haversine-kaava kahden pisteen välisen etäisyyden laskemiseen
# Maapallon säde kilometreinä
R = 6371.0  # km

# Muunnetaan asteet radiaaneiksi
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Leveysasteiden ja pituusasteiden erot
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversinen kaava
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Etäisyys kilometreinä
    distance = R * c
    return distance

# Tarkistetaan, että tiedosto ladattu
if path_gps is not None:
    df = pd.read_csv(path_gps)
    
    # Lasketaan peräkkäisten pisteiden väliset etäisyydet
    total_distance = 0  # Kuljettu matka
    
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, ['Latitude (°)', 'Longitude (°)']]
        lat2, lon2 = df.loc[i, ['Latitude (°)', 'Longitude (°)']]
        
        # Lasketaan etäisyys
        distance = haversine(lat1, lon1, lat2, lon2)
        
        # Summataan etäisyys kuljettuun matkaan
        total_distance += distance

   
# Keskinopeus laskenta
# Lasketaan kokonaisaika
    time_start = df.loc[0, 'Time (s)']
    time_end = df.loc[len(df) - 1, 'Time (s)']
    total_time = time_end - time_start  # sekunteina

    # Muutetaan aika tunneiksi km/h
    total_time_hours = total_time / 3600.0

    # Lasketaan keskinopeus (km/h)
    average_speed = total_distance / total_time_hours



# Lasketaan taajuus
# Lasketaan aikavälit peräkkäisten aikaleimojen välillä
df_acc = pd.read_csv("Linear Acceleration.csv")

# Lasketaan aikavälit kahden peräkkäisen aikaleiman välillä
time_diffs = np.diff(df_acc['Time (s)'])

# Lasketaan keskimääräinen aikaväli
mean_time_diff = np.mean(time_diffs)

# Näytteenottotaajuus (Hz) on aikavälien käänteisluku
fs = 1 / mean_time_diff

# Tulostetaan näytteenottotaajuus
print(f"Arvioitu näytteenottotaajuus: {fs:.2f} Hz")

# Suodatetun kiihtyvyysdatan x -komponentti
# Butterworth-kaistanpäästösuodattimen määrittely ja suodatus
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Suodatusparametrit
lowcut = 0.5  # Alaraja taajuudelle (Hz)
highcut = 3.0  # Yläraja taajuudelle (Hz)
fs = 198.79  # Phyphox-data on näytteenotettu Hz taajuudella (laskettu edellä)

# Valitaan x-komponentti suodatukseen
ax_filtered = bandpass_filter(df_acc['Linear Acceleration x (m/s^2)'], lowcut, highcut, fs)

# Piirretään suodatettu x-komponentti
#st.write("## Suodatettu x-komponentti")
suodatettu_df = pd.DataFrame({'Aika (s)': df_acc['Time (s)'], 'Suodatettu x': ax_filtered})

#st.line_chart(suodatettu_df.set_index('Aika (s)'))

# Askeleiden määrän laskeminen nollakohtien ylityksistä
zero_crossings = np.where(np.diff(np.sign(ax_filtered)))[0]
num_steps = len(zero_crossings) // 2


# Tehospektri laskenta
# Tarkistetaan, että Data sisältää tarvittavat sarakkeet
if 'Linear Acceleration x (m/s^2)' in df_acc.columns and 'Time (s)' in df_acc.columns:
    # Otetaan signaali ja aika
    f = df_acc['Linear Acceleration x (m/s^2)']  # Signaali
    t = df_acc['Time (s)']  # Aika
    N = len(df_acc)  # Havaintojen määrä
    dt = t.iloc[1] - t.iloc[0]  # Oletetaan tasavälinen näytteistys

    # Fourier-muunnos
    fourier = np.fft.fft(f, N)  # Fourier-muunnos
    psd = fourier * np.conj(fourier) / N  # Tehospektri
    freq = np.fft.fftfreq(N, dt)  # Taajuudet

    # Rajataan pois nollataajuus ja negatiiviset taajuudet
    L = freq > 0

    # Muodostetaan DataFrame taajuuksista ja tehoista (positiiviset taajuudet)
    chart_data = pd.DataFrame(np.transpose(np.array([freq[L], psd[L].real])), columns=["freq", "psd"])

    # Etsitään askeltempon alueet (oletusarvoisesti 1-3 Hz)
    step_freq_range = (1, 3)  # askeltempon taajuusalue (Hz)
    step_indices = np.where((freq[L] >= step_freq_range[0]) & (freq[L] <= step_freq_range[1]))

    # Lasketaan askelten määrä
    # Askelmäärä laskemalla amplitudien summan tai kynnysarvojen mukaan
    step_count = np.sum(psd[L][step_indices] > 0.05)  # Kynnysarvo amplitudille

    # Kuljettu matka (km) -> muunnos senttimetreiksi
    total_distance_cm = total_distance * 100000  # km to cm

    # Lasketaan askelpituus (cm)
    step_length_cm = total_distance_cm / np.maximum(step_count, 1)  # Varmistetaan, että jakaja ei ole nolla


st.title('Liike- ja reittianalyysi')

# Näytetään laskelmat sivun yläreunassa
st.write(f"Arvioitu askelten määrä suodatetusta datasta: {num_steps} askelta")
st.write(f"Askelmäärä Fourier-analyysin perusteella: {step_count} askelta")
st.write(f"Keskinopeus: {average_speed:.2f} km/h")
st.write(f"Kuljettu matka: {total_distance:.2f} km")
st.write(f"Askelpituus: {step_length_cm:.0f} cm")

# Piirretään suodatettu kiihtyvyysdata
st.write("## Suodatettuna x-komponentti")
st.line_chart(suodatettu_df, x='Aika (s)', y='Suodatettu x')

# Piirretään tehospektri Streamlitin line_chart-funktiolla
st.write("## Tehospektri")
st.line_chart(chart_data.set_index('freq'))



# Reitti kartalla GPS-datan perusteella
st.write("## Karttakuva")

# Lasketaan keskipiste reitille kartan keskittämistä varten
start_lat = df_gps['Latitude (°)'].mean()
start_long = df_gps['Longitude (°)'].mean()

# Luodaan kartta Foliumilla, asetetaan keskikohta ja zoom-taso
map = folium.Map(location=[start_lat, start_long], zoom_start=14)

# Piirretään reitti kartalle PolyLine-funktiolla
folium.PolyLine(df_gps[['Latitude (°)', 'Longitude (°)']].values, color='blue', weight=3.5, opacity=1).add_to(map)

# Näytetään kartta Streamlitin avulla
st_map = st_folium(map, width=900, height=650)

