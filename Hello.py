import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("day.csv")
    return df

# Main function
def main():
    st.title("Bike Sharing Clustering & Visualitation Dashboard ")

    # Load data
    df = load_data()

    # Pilih fitur yang ingin digunakan untuk clustering
    selected_features = ['temp', 'hum', 'windspeed']

    # Memilih jumlah cluster
    num_clusters = st.slider("Jumlah Cluster:", 2, 10)

    # Membuat objek KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)

    # Melakukan clustering
    clusters = kmeans.fit_predict(df[selected_features])

    # Menambahkan kolom hasil clustering ke dataframe
    df['cluster'] = clusters

    # Menampilkan data yang telah di-cluster
    st.subheader("Data setelah di-cluster:")
    st.write(df)

    # Menampilkan scatter plot berdasarkan dua fitur yang dipilih
    x_axis = st.selectbox("Pilih Fitur untuk Sumbu X:", selected_features)
    y_axis = st.selectbox("Pilih Fitur untuk Sumbu Y:", selected_features, index=1)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='cluster', palette='viridis', legend='full')
    plt.title(f"Clustering berdasarkan {x_axis} vs {y_axis}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    st.pyplot()
    
    
    st.title("Data Visualitation")
    #Pertanyaan 1
    st.subheader("Jumlah peminjaman sepeda berdasarkan bulan")
    monthly_rentals = df.groupby('mnth')['cnt'].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='mnth', y='cnt', data=monthly_rentals, palette='Blues_r')
    plt.title('Jumlah Peminjaman Sepeda Berdasarkan Bulan')
    plt.xlabel('Bulan')
    plt.ylabel('Jumlah Peminjaman')
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    #Pertanyaan 2
    st.subheader("Perbandingan jumlah peminjaman sepeda pada Working Day dan Other Day")
    weekday_counts = df[df['workingday'] == 1]['cnt'].sum()
    workingday_counts = df[df['workingday'] == 0]['cnt'].sum()

    plt.figure(figsize=(8, 8))
    plt.pie([weekday_counts, workingday_counts], labels=['Working Day', 'Other Day'], autopct='%1.1f%%', colors=['#A0E9FF', '#CDF5FD'])
    plt.title('Perbandingan Jumlah Peminjaman Sepeda pada Working Day dan Other Day')
    st.pyplot()

    #Pertanyaan 3 
    st.subheader("Jumlah peminjaman sepeda berdasarkan musim")
    season_rentals = df.groupby('season')['cnt'].sum().reset_index()
    custom_colors = ['#D5F0C1', '#FFCF81', '#FFBE98', '#AEE2FF']

    plt.figure(figsize=(8, 6))
    sns.barplot(x='season', y='cnt', data=season_rentals, palette=custom_colors)
    plt.title('Jumlah Peminjaman Sepeda Berdasarkan Musim')
    plt.xlabel('Musim')
    plt.ylabel('Jumlah Peminjaman')
    plt.xticks([0, 1, 2, 3], ['Spring', 'Summer', 'Fall', 'Winter'])
    st.pyplot()

     #Pertanyaan 4
    st.subheader("Jumlah peminjaman sepeda berdasarkan cuaca dengan donut chart")
    weather_rentals = df.groupby('weathersit')['cnt'].sum().reset_index()

    plt.figure(figsize=(8, 8))
    plt.pie(weather_rentals['cnt'], labels=weather_rentals['weathersit'], autopct='%1.1f%%', colors=['#FFBE98', '#F7DED0', '#FEECE2'], wedgeprops=dict(width=0.4))
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Jumlah Peminjaman Sepeda Berdasarkan Cuaca')
    plt.axis('equal')
    st.pyplot()

    # Pertanyaan 5
    st.subheader('Pengaruh Peminjaman Sepeda Berdasarkan Kecepatan Angin')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='windspeed', y='cnt', data=df, color='orange')
    plt.title('Peminjaman Sepeda Berdasarkan Kecepatan Angin')
    plt.xlabel('Kecepatan Angin')
    plt.ylabel('Jumlah Peminjaman')
    st.pyplot()

if __name__ == "__main__":
    main()
