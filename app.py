import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

st.set_page_config(
    page_title="Dashboard Prediksi & Forecast Saham Makanan Olahan",
    layout="wide"
)

np.random.seed(42)
tf.random.set_seed(42)

def compute_technical_indicators(df):
    """
    Menghitung 7 fitur teknikal stasioner (X1 hingga X7) berdasarkan data OHLCV.
    """
    df['X1_Close_Return'] = df['Close'].pct_change()
    df['X2_Volume_Return'] = df['Volume'].pct_change()
    
    sma_20 = df['Close'].rolling(window=20).mean()
    df['X3_SMA_Ratio'] = df['Close'] / sma_20
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['X4_MACD_Hist'] = macd_line - signal_line
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['X5_RSI'] = 100 - (100 / (1 + rs))
    
    std_20 = df['Close'].rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    df['X6_Bollinger_B'] = (df['Close'] - lower_band) / (upper_band - lower_band)
    
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['X7_Stochastic_K'] = ((df['Close'] - low_14) / (high_14 - low_14)) * 100
    
    df = df.bfill().fillna(0)
    df['Y_Target_Return'] = df['X1_Close_Return']
    return df

def load_and_preprocess_source(source):
    """
    Membaca dan membersihkan CSV saham serta menambahkan fitur indikator teknikal.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            return pd.DataFrame()
        with open(source, 'r', encoding='utf-8-sig', errors='replace') as file:
            raw_lines = [line.strip() for line in file.readlines() if line.strip()]
    else:
        raw_bytes = source.getvalue().decode('utf-8-sig', errors='replace')
        raw_lines = [line.strip() for line in raw_bytes.splitlines() if line.strip()]

    if not raw_lines:
        return pd.DataFrame()

    data_rows = []
    for line in raw_lines:
        if ';' in line:
            row = line.split(';')
        elif '\t' in line:
            row = line.split('\t')
        else:
            try:
                row = next(csv.reader([line], skipinitialspace=True))
            except Exception:
                row = line.split(',')
                
        row = [str(x).strip() for x in row]
        data_rows.append(row)
        
    df = pd.DataFrame(data_rows)
    
    header_idx = -1
    for i, row in enumerate(data_rows):
        if any(x and ('date' in str(x).lower() or 'tanggal' in str(x).lower()) for x in row):
            header_idx = i
            break
            
    if header_idx != -1:
        header = [str(x).replace('"', '').replace('*', '').strip().capitalize() for x in data_rows[header_idx]]
        header = [h if h else f"Col_{j}" for j, h in enumerate(header)]
        num_cols = df.shape[1]
        if len(header) < num_cols:
            header.extend([f"Col_{j}" for j in range(len(header), num_cols)])
        elif len(header) > num_cols:
            header = header[:num_cols]
        df.columns = header
        df = df.iloc[header_idx+1:].reset_index(drop=True)
    else:
        default_headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj close', 'Volume']
        num_cols = df.shape[1]
        df.columns = default_headers[:num_cols] + [f"Col_{j}" for j in range(len(default_headers), num_cols)]

    date_col = next((c for c in df.columns if 'date' in str(c).lower() or 'tanggal' in str(c).lower()), None)
    if date_col:
        df = df.dropna(subset=[date_col])
        id_to_en = {'jan': 'Jan', 'peb': 'Feb', 'mar': 'Mar', 'apr': 'Apr', 'mei': 'May', 'jun': 'Jun',
                    'jul': 'Jul', 'agt': 'Aug', 'agu': 'Aug', 'sep': 'Sep', 'okt': 'Oct', 'nov': 'Nov', 'des': 'Dec'}
        
        def clean_date(d_str):
            if not isinstance(d_str, str): return d_str
            d_str_clean = ' '.join(d_str.replace(',', '').strip().split())
            d_lower = d_str_clean.lower()
            for k, v in id_to_en.items():
                if k in d_lower: return d_lower.replace(k, v)
            return d_str_clean
            
        df[date_col] = df[date_col].apply(clean_date)
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in target_cols:
        actual_col = next((c for c in df.columns if col.lower() in str(c).lower()), None)
        if actual_col:
            df[actual_col] = df[actual_col].astype(str).str.replace(',', '', regex=False).str.replace('-', '', regex=False).str.strip()
            df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce')
            if actual_col != col:
                df.rename(columns={actual_col: col}, inplace=True)
                
    df = df.dropna(subset=['Close'])
    if len(df) < 20:
        return pd.DataFrame()

    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return compute_technical_indicators(df)

def create_sliding_window(X_data, y_data, close_prices, lookback=5):
    X_seq, y_seq, prev_close = [], [], []
    for i in range(lookback, len(X_data)):
        X_seq.append(X_data[i-lookback:i])
        y_seq.append(y_data[i])
        prev_close.append(close_prices[i-1])
    return np.array(X_seq), np.array(y_seq), np.array(prev_close)

def prepare_dataset(df, lookback=5, train_ratio=0.8):
    feature_cols = ['X1_Close_Return', 'X2_Volume_Return', 'X3_SMA_Ratio', 
                    'X4_MACD_Hist', 'X5_RSI', 'X6_Bollinger_B', 'X7_Stochastic_K']
    
    X_raw = df[feature_cols].values
    y_raw = df['Y_Target_Return'].values
    close_raw = df['Close'].values
    
    train_size = int(len(df) * train_ratio)
    
    X_train_raw, X_test_raw = X_raw[:train_size], X_raw[train_size:]
    y_train_raw, y_test_raw = y_raw[:train_size], y_raw[train_size:]
    close_train_raw, close_test_raw = close_raw[:train_size], close_raw[train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    X_train, y_train, _ = create_sliding_window(X_train_scaled, y_train_raw, close_train_raw, lookback)
    X_test, y_test, prev_close_test = create_sliding_window(X_test_scaled, y_test_raw, close_test_raw, lookback)
    
    actual_close_test = close_test_raw[lookback:]
    return X_train, y_train, X_test, y_test, prev_close_test, actual_close_test, scaler, feature_cols

def forecast_future_recursive(rfr_model, lstm_model, df, scaler, feature_cols, lookback=5, forecast_steps=30):
    df_temp_rfr = df.copy()
    df_temp_lstm = df.copy()
    
    forecast_rfr_prices = []
    forecast_lstm_prices = []
    
    for _ in range(forecast_steps):
        df_temp_rfr = compute_technical_indicators(df_temp_rfr)
        X_curr = scaler.transform(df_temp_rfr[feature_cols].values[-lookback:])
        X_curr_rfr = X_curr.reshape(1, -1)
        
        pred_ret = rfr_model.predict(X_curr_rfr)[0]
        last_price = df_temp_rfr['Close'].iloc[-1]
        next_price = last_price * (1 + pred_ret)
        forecast_rfr_prices.append(next_price)
        
        last_row = df_temp_rfr.iloc[-1].copy()
        last_row['Close'] = next_price
        last_row['High'] = max(last_row['High'], next_price)
        last_row['Low'] = min(last_row['Low'], next_price)
        df_temp_rfr = pd.concat([df_temp_rfr, pd.DataFrame([last_row])], ignore_index=True)

    for _ in range(forecast_steps):
        df_temp_lstm = compute_technical_indicators(df_temp_lstm)
        X_curr = scaler.transform(df_temp_lstm[feature_cols].values[-lookback:])
        X_curr_lstm = X_curr.reshape(1, lookback, len(feature_cols))
        
        pred_ret = lstm_model.predict(X_curr_lstm, verbose=0).flatten()[0]
        last_price = df_temp_lstm['Close'].iloc[-1]
        next_price = last_price * (1 + pred_ret)
        forecast_lstm_prices.append(next_price)
        
        last_row = df_temp_lstm.iloc[-1].copy()
        last_row['Close'] = next_price
        last_row['High'] = max(last_row['High'], next_price)
        last_row['Low'] = min(last_row['Low'], next_price)
        df_temp_lstm = pd.concat([df_temp_lstm, pd.DataFrame([last_row])], ignore_index=True)

    return forecast_rfr_prices, forecast_lstm_prices

st.title("Dashboard Prediksi dan Analisis")
st.subheader("Sektor Konsumen Primer Industri Makanan Olahan (BEI)")


st.sidebar.header("Control Panel")

data_option = st.sidebar.selectbox("Pilih Data Saham:", ["ICBP", "INDF", "MYOR", "CMRY", "ULTJ", "Unggah File Custom (CSV)"])
target_file_source = None
ticker_name = ""

if data_option == "Unggah File Custom (CSV)":
    uploaded_file = st.sidebar.file_uploader("Unggah File Data Saham (Format CSV)", type=["csv"])
    if uploaded_file is not None:
        target_file_source = uploaded_file
        ticker_name = os.path.splitext(uploaded_file.name)[0].upper()
else:
    file_path = os.path.join("data", f"{data_option}.csv")
    if os.path.exists(file_path):
        target_file_source = file_path
        ticker_name = data_option

model_choice = st.sidebar.selectbox("Pilih Algoritma Model:", ["Bandingkan Kedua Model (RFR vs LSTM)", "Random Forest Regressor (RFR)", "Long Short Term Memory (LSTM)"])
lookback = st.sidebar.slider("Panjang Sliding Window (Hari):", min_value=3, max_value=15, value=5)
forecast_days = st.sidebar.slider("Jumlah Hari Proyeksi Masa Depan:", min_value=1, max_value=30, value=30)

if target_file_source is not None:
    try:
        df = load_and_preprocess_source(target_file_source)
        if df.empty or len(df) < 20:
            st.error(f"Data valid {ticker_name} kurang dari 20 baris.")
        else:
            st.success(f"Berhasil memuat data {ticker_name} ({len(df)} baris data valid).")
            
            with st.spinner("Melatih model dan memproses proyeksi masa depan..."):
                X_train, y_train, X_test, y_test, prev_close_test, actual_close_test, scaler, feature_cols = prepare_dataset(df, lookback=lookback)
                
                X_train_rfr = X_train.reshape(X_train.shape[0], -1)
                X_test_rfr = X_test.reshape(X_test.shape[0], -1)
                
                rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rfr_model.fit(X_train_rfr, y_train)
                
                lstm_model = Sequential([
                    LSTM(units=32, input_shape=(lookback, 7), return_sequences=False),
                    Dropout(0.1),
                    Dense(units=1)
                ])
                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                lstm_model.fit(X_train, y_train, epochs=40, batch_size=16, verbose=0, shuffle=False)
                
                rfr_pred_returns = rfr_model.predict(X_test_rfr)
                lstm_pred_returns = lstm_model.predict(X_test, verbose=0).flatten()
                
                rfr_pred_close = prev_close_test * (1 + rfr_pred_returns)
                lstm_pred_close = prev_close_test * (1 + lstm_pred_returns)
                
                mae_ret_rfr = mean_absolute_error(y_test, rfr_pred_returns)
                rmse_ret_rfr = np.sqrt(mean_squared_error(y_test, rfr_pred_returns))
                mae_rp_rfr = mean_absolute_error(actual_close_test, rfr_pred_close)
                rmse_rp_rfr = np.sqrt(mean_squared_error(actual_close_test, rfr_pred_close))
                
                mae_ret_lstm = mean_absolute_error(y_test, lstm_pred_returns)
                rmse_ret_lstm = np.sqrt(mean_squared_error(y_test, lstm_pred_returns))
                mae_rp_lstm = mean_absolute_error(actual_close_test, lstm_pred_close)
                rmse_rp_lstm = np.sqrt(mean_squared_error(actual_close_test, lstm_pred_close))
                
                forecast_rfr, forecast_lstm = forecast_future_recursive(rfr_model, lstm_model, df, scaler, feature_cols, lookback=lookback, forecast_steps=forecast_days)
                last_known_close = df['Close'].iloc[-1]

            st.markdown("---")
            st.subheader("Evaluasi Akurasi Model (Subset Data Uji)")
            st.write("Pengukuran tingkat kesalahan komparatif antara harga penutupan aktual dan hasil prediksi pada data uji.")
            
            eval_data = []
            if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Random Forest Regressor (RFR)"]:
                eval_data.append({
                    "Algoritma": "Random Forest Regressor (RFR)",
                    "MAE Return": f"{mae_ret_rfr:.5f}",
                    "RMSE Return": f"{rmse_ret_rfr:.5f}",
                    "MAE (Rupiah)": f"Rp {mae_rp_rfr:,.2f}",
                    "RMSE (Rupiah)": f"Rp {rmse_rp_rfr:,.2f}"
                })
            if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Long Short Term Memory (LSTM)"]:
                eval_data.append({
                    "Algoritma": "Long Short Term Memory (LSTM)",
                    "MAE Return": f"{mae_ret_lstm:.5f}",
                    "RMSE Return": f"{rmse_ret_lstm:.5f}",
                    "MAE (Rupiah)": f"Rp {mae_rp_lstm:,.2f}",
                    "RMSE (Rupiah)": f"Rp {rmse_rp_lstm:,.2f}"
                })
                
            st.table(pd.DataFrame(eval_data))

            st.markdown("---")
            st.subheader(f"Proyeksi Masa Depan ({forecast_days} Hari Bursa Ke Depan)")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Harga Penutupan Terakhir", f"Rp {last_known_close:,.2f}")
            
            if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Random Forest Regressor (RFR)"]:
                rfr_change = ((forecast_rfr[-1] - last_known_close) / last_known_close) * 100
                col2.metric(f"Estimasi +{forecast_days} Hari (RFR)", f"Rp {forecast_rfr[-1]:,.2f}", f"{rfr_change:.2f}%")
            
            if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Long Short Term Memory (LSTM)"]:
                lstm_change = ((forecast_lstm[-1] - last_known_close) / last_known_close) * 100
                col3.metric(f"Estimasi +{forecast_days} Hari (LSTM)", f"Rp {forecast_lstm[-1]:,.2f}", f"{lstm_change:.2f}%")

            st.markdown("---")
            st.subheader("Grafik Proyeksi Tren Masa Depan")
            
            fig, ax = plt.subplots(figsize=(10, 4.5))
            hist_subset = df['Close'].tail(60).values
            hist_x = list(range(len(hist_subset)))
            future_x = list(range(len(hist_subset) - 1, len(hist_subset) + forecast_days))
            
            ax.plot(hist_x, hist_subset, label='Harga Historis (60 Hari Terakhir)', color='#1f77b4', linewidth=1.5)
            
            if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Random Forest Regressor (RFR)"]:
                ax.plot(future_x, [hist_subset[-1]] + forecast_rfr, label=f'Proyeksi RFR (+{forecast_days} Hari)', color='#2ca02c', linestyle='--', linewidth=1.8)
                
            if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Long Short Term Memory (LSTM)"]:
                ax.plot(future_x, [hist_subset[-1]] + forecast_lstm, label=f'Proyeksi LSTM (+{forecast_days} Hari)', color='#d62728', linestyle=':', linewidth=1.8)
                
            ax.set_title(f'Tren Proyeksi Masa Depan - {ticker_name}', fontsize=12)
            ax.set_xlabel('Hari Bursa', fontsize=10)
            ax.set_ylabel('Harga Penutupan (IDR)', fontsize=10)
            ax.legend(loc='best')
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)

            st.markdown("---")
            st.subheader(f"Tabel Rincian Estimasi ({forecast_days} Hari Ke Depan)")
            
            forecast_data = []
            for d in range(forecast_days):
                row_dict = {"Hari Ke-+": f"Hari +{d+1}"}
                if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Random Forest Regressor (RFR)"]:
                    row_dict["Proyeksi RFR (Rp)"] = f"Rp {forecast_rfr[d]:,.2f}"
                if model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Long Short Term Memory (LSTM)"]:
                    row_dict["Proyeksi LSTM (Rp)"] = f"Rp {forecast_lstm[d]:,.2f}"
                forecast_data.append(row_dict)
                
            st.dataframe(pd.DataFrame(forecast_data), use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengolah file: {str(e)}")
else:
    st.info("Silakan pilih emiten saham dari dropdown di sebelah kiri untuk memulai analisis.")