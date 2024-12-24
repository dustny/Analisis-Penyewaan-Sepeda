import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="Dashboard Analisis Peminjaman Sepeda", layout="wide")

# Title
st.title("Dashboard Analisis Peminjaman Sepeda")

# Cache data to prevent reload unless necessary
@st.cache_data(show_spinner=True)
def load_data():
    """
    Fungsi untuk memuat dataset dan melakukan preprocessing awal.
    - Mengonversi kolom 'dteday' menjadi tipe datetime.
    """
    hour_df = pd.read_csv('main_data.csv')
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])  # Konversi ke datetime
    return hour_df

# Load data
hour_df = load_data()

# Sidebar filters
st.sidebar.header("Filter")
selected_hour = st.sidebar.slider("Pilih Jam", 0, 23, (0, 23))
selected_weather = st.sidebar.multiselect(
    "Pilih Kondisi Cuaca", 
    options=[1, 2, 3, 4],
    default=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Cerah / Sedikit Berawan",
        2: "Berkabut / Berawan",
        3: "Hujan Ringan / Salju Ringan",
        4: "Hujan Deras / Badai"
    }.get(x, "Unknown")
)

# Filter data based on selections
filtered_data = hour_df[
    (hour_df['hr'].between(selected_hour[0], selected_hour[1])) &
    (hour_df['weathersit'].isin(selected_weather))
]

# KPI
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Rentals", int(filtered_data['cnt'].sum()))
with col2:
    st.metric("Casual Users (Avg)", round(filtered_data['casual'].mean(), 2))
with col3:
    st.metric("Registered Users (Avg)", round(filtered_data['registered'].mean(), 2))

# Average Usage by Hour
def plot_hourly_usage():
    hourly_usage = filtered_data.groupby('hr')[['casual', 'registered']].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(hourly_usage['hr'], hourly_usage['casual'], label='Casual Users (Avg)', marker='o')
    plt.plot(hourly_usage['hr'], hourly_usage['registered'], label='Registered Users (Avg)', marker='o')
    plt.title("Rata-rata Penggunaan Sepeda per Jam")
    plt.xlabel("Jam")
    plt.ylabel("Rata-rata Pengguna")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(alpha=0.3)
    st.pyplot(plt)

# Rentals by Weather Condition
def plot_weather_usage():
    weather_analysis = filtered_data.groupby('weathersit')['cnt'].mean().reset_index()
    weather_conditions = {
        1: "Cerah / Sedikit Berawan",
        2: "Berkabut / Berawan",
        3: "Hujan Ringan / Salju Ringan",
        4: "Hujan Deras / Badai"
    }
    weather_analysis['Weather'] = weather_analysis['weathersit'].map(weather_conditions)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=weather_analysis, x='Weather', y='cnt', palette='coolwarm')
    plt.title("Rata-rata Peminjaman Berdasarkan Kondisi Cuaca")
    plt.xlabel("Kondisi Cuaca")
    plt.ylabel("Rata-rata Peminjaman")
    plt.xticks(rotation=15)
    st.pyplot(plt)

# Call the functions to render the plots
plot_hourly_usage()
plot_weather_usage()

# Customer Segmentation (RFM Analysis)
st.subheader("Segmentasi Pelanggan Berdasarkan RFM")

# Filter data untuk menghitung RFM berdasarkan jam dan cuaca yang dipilih
filtered_rfm_data = filtered_data.groupby('dteday').agg(
    Recency=('dteday', lambda x: (pd.to_datetime(filtered_data['dteday']).max() - x.max()).days),
    Frequency=('cnt', 'count'),
    Monetary=('cnt', 'sum')
).reset_index()

# RFM Scoring
try:
    filtered_rfm_data['R_Score'] = pd.qcut(filtered_rfm_data['Recency'], 4, labels=[4, 3, 2, 1])
except ValueError as e:
    print(f"Error in Recency scoring: {e}")
    
try:
    filtered_rfm_data['F_Score'] = pd.qcut(filtered_rfm_data['Frequency'], 4, labels=[1, 2, 3, 4], duplicates='drop')
except ValueError as e:
    print(f"Error in Frequency scoring: {e}")
    filtered_rfm_data['F_Score'] = pd.cut(filtered_rfm_data['Frequency'], bins=4, labels=[1, 2, 3, 4])

try:
    filtered_rfm_data['M_Score'] = pd.qcut(filtered_rfm_data['Monetary'], 4, labels=[1, 2, 3, 4], duplicates='drop')
except ValueError as e:
    print(f"Error in Monetary scoring: {e}")
    filtered_rfm_data['M_Score'] = pd.cut(filtered_rfm_data['Monetary'], bins=4, labels=[1, 2, 3, 4])

# Combine RFM scores into a single score
filtered_rfm_data['RFM_Score'] = filtered_rfm_data['R_Score'].astype(int) + \
                                  filtered_rfm_data['F_Score'].astype(int) + \
                                  filtered_rfm_data['M_Score'].astype(int)

# Define customer segments
def segment_customer(row):
    if row['RFM_Score'] >= 9:
        return 'VIP'
    elif row['RFM_Score'] >= 5:
        return 'Regular'
    else:
        return 'Churn Risk'

filtered_rfm_data['Segment'] = filtered_rfm_data.apply(segment_customer, axis=1)

# Visualisasi Segmentasi
plt.figure(figsize=(8, 5))
sns.countplot(data=filtered_rfm_data, x='Segment', palette='coolwarm', order=['VIP', 'Regular', 'Churn Risk'], hue='Segment')
plt.legend([], [], frameon=False)  
plt.title('Segmentasi Pelanggan Berdasarkan RFM')
plt.xlabel('Segment')
plt.ylabel('Jumlah Pelanggan')
st.pyplot(plt)

# Display Segment Summary Statistics
segment_summary = filtered_rfm_data.groupby('Segment').agg(
    Recency=('Recency', 'mean'),
    Frequency=('Frequency', 'mean'),
    Monetary=('Monetary', 'mean'),
    RFM_Score=('RFM_Score', 'mean')
).reset_index()

st.write("### Segment Summary Statistics")
st.dataframe(segment_summary)

# Insight Generation
segment_counts = filtered_rfm_data['Segment'].value_counts()
st.write("### Segment Distribution")
st.write(segment_counts)