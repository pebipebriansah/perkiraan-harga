import os
import pandas as pd
from io import BytesIO
from sklearn.linear_model import LinearRegression
from azure.storage.blob import BlobServiceClient
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def load_data_from_blob():
    print("📦 Menghubungkan ke Azure Blob...")
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = os.getenv("AZURE_STORAGE_CONTAINER")
    blob_name = os.getenv("AZURE_STORAGE_BLOB")

    if not all([conn_str, container, blob_name]):
        raise Exception("❌Variabel .env tidak lengkap!")

    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
        stream = blob_client.download_blob()
        data = stream.readall()
        df = pd.read_excel(BytesIO(data))
        print("✅ Data berhasil dimuat.")
        return df
    except Exception as e:
        raise Exception(f"❌ Gagal memuat data: {e}")

def train_model(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace(r"[()/]", "", regex=True)  # hapus tanda ( ) dan /
    
    expected_cols = ['Bulan', 'Harga_Cabai_Rpkg', 'Curah_Hujan_mm']

    if not all(col in df.columns for col in expected_cols):
        raise Exception(f"❌ Kolom data tidak lengkap. Kolom ditemukan: {df.columns.tolist()}")

    X = df[['Bulan', 'Curah_Hujan_mm']]
    y = df['Harga_Cabai_Rpkg']

    model = LinearRegression().fit(X, y)
    return model

def predict(model, bulan, curah_hujan):
    import pandas as pd
    X_pred = pd.DataFrame({'Bulan': [bulan], 'Curah_Hujan_mm': [curah_hujan]})
    return model.predict(X_pred)[0]

def predict_next_month(model, current_bulan, curah_hujan):
    next_bulan = current_bulan + 1 if current_bulan < 12 else 1
    predicted_harga = predict(model, next_bulan, curah_hujan)
    return next_bulan, predicted_harga

if __name__ == "__main__":
    try:
        df = load_data_from_blob()
        model = train_model(df)

        current_bulan = 5  # Contoh bulan saat ini, misal Mei
        curah_hujan = 120.0  # Contoh curah hujan bulan ini (mm)

        harga_saat_ini = predict(model, current_bulan, curah_hujan)
        bulan_depan, harga_depan = predict_next_month(model, current_bulan, curah_hujan)

        print(f"Prediksi harga cabai untuk bulan {current_bulan} dengan curah hujan {curah_hujan} mm adalah Rp{harga_saat_ini:,.2f}")
        print(f"Prediksi harga cabai untuk bulan {bulan_depan} (bulan depan) dengan curah hujan {curah_hujan} mm adalah Rp{harga_depan:,.2f}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
