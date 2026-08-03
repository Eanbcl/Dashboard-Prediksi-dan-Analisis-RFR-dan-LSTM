import os
import csv
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

def load_and_preprocess_csv(filepath):
    """
    Membaca dan membersihkan CSV saham serta menambahkan fitur indikator teknikal.
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} tidak ditemukan.")
        return pd.DataFrame()
        
    with open(filepath, 'r', encoding='utf-8-sig', errors='replace') as file:
        raw_lines = [line.strip() for line in file.readlines() if line.strip()]
        
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

def build_rfr_model(n_estimators=100, random_state=42):
    return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

def build_lstm_model(time_steps=5, num_features=7, units=32):
    model = Sequential([
        LSTM(units=units, input_shape=(time_steps, num_features), return_sequences=False),
        Dropout(0.1),
        Dense(units=1) 
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

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

def plot_and_save_future_forecast(ticker, historical_prices, forecast_rfr, forecast_lstm, output_dir="Model/plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    hist_len = len(historical_prices)
    hist_x = list(range(hist_len))
    future_x = list(range(hist_len - 1, hist_len + len(forecast_rfr)))
    
    plt.plot(hist_x, historical_prices, label='Harga Historis Aktual', color='blue', linewidth=1.5)
    plt.plot(future_x, [historical_prices[-1]] + forecast_rfr, label='Proyeksi RFR (30 Hari Masa Depan)', color='green', linestyle='--', linewidth=1.8)
    plt.plot(future_x, [historical_prices[-1]] + forecast_lstm, label='Proyeksi LSTM (30 Hari Masa Depan)', color='red', linestyle=':', linewidth=1.8)
    
    plt.title(f'Proyeksi Tren Harga Saham Masa Depan (30 Hari Bursa) - {ticker}', fontsize=12)
    plt.xlabel('Hari Bursa', fontsize=10)
    plt.ylabel('Harga Penutupan (IDR)', fontsize=10)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_path = os.path.join(output_dir, f'{ticker}_30day_forecast.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    emitens = ["ICBP", "INDF", "MYOR", "CMRY", "ULTJ"]
    data_dir = "data"
    output_dir = "Model"
    saved_models_dir = os.path.join(output_dir, "saved_models")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(saved_models_dir, exist_ok=True)
    
    results_summary = []
    future_projections_all = []
    
    print("Memulai proses pemodelan saham, penyimpanan 10 model, dan proyeksi 30 hari masa depan...")
    
    for ticker in emitens:
        filepath = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(filepath):
            print(f"File {filepath} tidak ditemukan. Dilewati.")
            continue
            
        print(f"\nProcessing: {ticker}")
        df = load_and_preprocess_csv(filepath)
        
        if len(df) < 20: 
            print(f"Data valid {ticker} terlalu sedikit.")
            continue
            
        lookback = 5
        X_train, y_train, X_test, y_test, prev_close_test, actual_close_test, scaler, feature_cols = prepare_dataset(df, lookback=lookback)
        
        X_train_rfr = X_train.reshape(X_train.shape[0], -1)
        X_test_rfr = X_test.reshape(X_test.shape[0], -1)
        
        # Training Models
        rfr_model = build_rfr_model()
        rfr_model.fit(X_train_rfr, y_train)
        
        lstm_model = build_lstm_model(time_steps=lookback, num_features=7, units=32)
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, shuffle=False)
        
        # Menyimpan 10 File Model Terlatih (5 RFR + 5 LSTM)
        rfr_save_path = os.path.join(saved_models_dir, f"rfr_model_{ticker}.joblib")
        lstm_save_path = os.path.join(saved_models_dir, f"lstm_model_{ticker}.keras")
        scaler_save_path = os.path.join(saved_models_dir, f"scaler_{ticker}.joblib")
        
        joblib.dump(rfr_model, rfr_save_path)
        lstm_model.save(lstm_save_path)
        joblib.dump(scaler, scaler_save_path)
        
        print(f"  [SAVED] Model RFR -> {rfr_save_path}")
        print(f"  [SAVED] Model LSTM -> {lstm_save_path}")
        
        # Test Set Evaluation
        rfr_pred_returns = rfr_model.predict(X_test_rfr)
        lstm_pred_returns = lstm_model.predict(X_test, verbose=0).flatten()
        
        rfr_pred_close = prev_close_test * (1 + rfr_pred_returns)
        lstm_pred_close = prev_close_test * (1 + lstm_pred_returns)
        
        mae_ret_rfr, rmse_ret_rfr = calculate_metrics(y_test, rfr_pred_returns)
        mae_rp_rfr, rmse_rp_rfr = calculate_metrics(actual_close_test, rfr_pred_close)
        
        mae_ret_lstm, rmse_ret_lstm = calculate_metrics(y_test, lstm_pred_returns)
        mae_rp_lstm, rmse_rp_lstm = calculate_metrics(actual_close_test, lstm_pred_close)
        
        # Proyeksi 30 Hari Ke Depan
        forecast_rfr, forecast_lstm = forecast_future_recursive(rfr_model, lstm_model, df, scaler, feature_cols, lookback=lookback, forecast_steps=30)
        
        last_known_close = df['Close'].iloc[-1]
        
        results_summary.append({
            "Emiten": ticker,
            "Harga Terakhir (Rp)": round(last_known_close, 2),
            "Prediksi +1 Hari RFR": round(forecast_rfr[0], 2),
            "Prediksi +1 Hari LSTM": round(forecast_lstm[0], 2),
            "Prediksi +30 Hari RFR": round(forecast_rfr[-1], 2),
            "Prediksi +30 Hari LSTM": round(forecast_lstm[-1], 2),
            "RFR MAE(Ret)": round(mae_ret_rfr, 5),
            "RFR RMSE(Ret)": round(rmse_ret_rfr, 5),
            "RFR MAE(Rp)": round(mae_rp_rfr, 2),
            "RFR RMSE(Rp)": round(rmse_rp_rfr, 2),
            "LSTM MAE(Ret)": round(mae_ret_lstm, 5),
            "LSTM RMSE(Ret)": round(rmse_ret_lstm, 5),
            "LSTM MAE(Rp)": round(mae_rp_lstm, 2),
            "LSTM RMSE(Rp)": round(rmse_rp_lstm, 2)
        })
        
        for step_i in range(30):
            future_projections_all.append({
                "Emiten": ticker,
                "Hari Ke-+": step_i + 1,
                "Proyeksi RFR (Rp)": round(forecast_rfr[step_i], 2),
                "Proyeksi LSTM (Rp)": round(forecast_lstm[step_i], 2)
            })
            
        plot_and_save_future_forecast(ticker, df['Close'].tail(100).values, forecast_rfr, forecast_lstm, output_dir=os.path.join(output_dir, "plots"))
        print(f"Selesai {ticker} -> Harga Terakhir: Rp {last_known_close:,.2f} | Forecast +30 Hari RFR: Rp {forecast_rfr[-1]:,.2f} | LSTM: Rp {forecast_lstm[-1]:,.2f}")

    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(os.path.join(output_dir, "ringkasan_evaluasi.csv"), index=False)
        
        projections_df = pd.DataFrame(future_projections_all)
        projections_df.to_csv(os.path.join(output_dir, "proyeksi_30_hari.csv"), index=False)
        print(f"\n[OK] Seluruh 10 model, hasil evaluasi, dan proyeksi 30 hari berhasil disimpan di folder: {saved_models_dir}/")

if __name__ == "__main__":
    main()