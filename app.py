import os
import csv
import io
import joblib
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
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

st.set_page_config(
    page_title="Dashboard Prediksi & Analisis Saham",
    layout="wide",
    initial_sidebar_state="expanded"
)

np.random.seed(42)
tf.random.set_seed(42)

def compute_technical_indicators(df):
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
    if isinstance(source, str):
        if not os.path.exists(source): return pd.DataFrame()
        with open(source, 'r', encoding='utf-8-sig', errors='replace') as file:
            raw_lines = [line.strip() for line in file.readlines() if line.strip()]
    else:
        raw_bytes = source.getvalue().decode('utf-8-sig', errors='replace')
        raw_lines = [line.strip() for line in raw_bytes.splitlines() if line.strip()]

    if not raw_lines: return pd.DataFrame()
    data_rows = []
    for line in raw_lines:
        if ';' in line: row = line.split(';')
        elif '\t' in line: row = line.split('\t')
        else:
            try: row = next(csv.reader([line], skipinitialspace=True))
            except Exception: row = line.split(',')
        data_rows.append([str(x).strip() for x in row])
        
    df = pd.DataFrame(data_rows)
    header_idx = next((i for i, row in enumerate(data_rows) if any(x and ('date' in str(x).lower() or 'tanggal' in str(x).lower()) for x in row)), -1)
            
    if header_idx != -1:
        header = [str(x).replace('"', '').replace('*', '').strip().capitalize() for x in data_rows[header_idx]]
        header = [h if h else f"Col_{j}" for j, h in enumerate(header)]
        num_cols = df.shape[1]
        if len(header) < num_cols: header.extend([f"Col_{j}" for j in range(len(header), num_cols)])
        elif len(header) > num_cols: header = header[:num_cols]
        df.columns = header
        df = df.iloc[header_idx+1:].reset_index(drop=True)
    else:
        default_headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj close', 'Volume']
        df.columns = default_headers[:df.shape[1]] + [f"Col_{j}" for j in range(len(default_headers), df.shape[1])]

    date_col = next((c for c in df.columns if 'date' in str(c).lower() or 'tanggal' in str(c).lower()), None)
    if date_col:
        id_to_en = {'jan': 'Jan', 'peb': 'Feb', 'mar': 'Mar', 'apr': 'Apr', 'mei': 'May', 'jun': 'Jun', 'jul': 'Jul', 'agt': 'Aug', 'agu': 'Aug', 'sep': 'Sep', 'okt': 'Oct', 'nov': 'Nov', 'des': 'Dec'}
        def clean_date(d_str):
            if not isinstance(d_str, str): return d_str
            d_lower = ' '.join(d_str.replace(',', '').strip().split()).lower()
            for k, v in id_to_en.items():
                if k in d_lower: return d_lower.replace(k, v)
            return d_lower
        df[date_col] = df[date_col].apply(clean_date)
        df[date_col] = pd.to_datetime(df[date_col], format='mixed', errors='coerce')
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in target_cols:
        actual_col = next((c for c in df.columns if col.lower() in str(c).lower()), None)
        if actual_col:
            df[actual_col] = pd.to_numeric(df[actual_col].astype(str).str.replace(',', '', regex=False).str.replace('-', '', regex=False).str.strip(), errors='coerce')
            if actual_col != col: df.rename(columns={actual_col: col}, inplace=True)
                
    df = df.dropna(subset=['Close'])
    if len(df) < 20: return pd.DataFrame()
    for col in target_cols:
        if col in df.columns: df[col] = df[col].astype(float)
    return compute_technical_indicators(df)

def get_latest_valid_number(row_series):
    for val in row_series.values[1:]:
        v_str = str(val).strip().replace('%', '').replace(',', '')
        if v_str.lower() not in ['n/a', 'nan', 'none', '']:
            try:
                return float(v_str)
            except ValueError:
                continue
    return 0.0

def parse_fundamental(source):
    fund_data = {'PER': 15.0, 'ROE': 0.10, 'ROA': 0.05, 'Interest Coverage': 5.0}
    try:
        if isinstance(source, str):
            if not os.path.exists(source): return fund_data
            df = pd.read_csv(source, sep=';', header=None, engine='python')
        else:
            df = pd.read_csv(source, sep=';', header=None, engine='python')
            
        for i in range(len(df)):
            indicator_name = str(df.iloc[i, 0]).lower()
            
            if 'pe ratio' in indicator_name or 'per' in indicator_name:
                fund_data['PER'] = get_latest_valid_number(df.iloc[i])
            elif 'return on equity' in indicator_name or 'roe' in indicator_name:
                val = get_latest_valid_number(df.iloc[i])
                fund_data['ROE'] = val / 100 if val > 1 else val
            elif 'return on assets' in indicator_name or 'roa' in indicator_name:
                val = get_latest_valid_number(df.iloc[i])
                fund_data['ROA'] = val / 100 if val > 1 else val
            elif 'interest coverage' in indicator_name:
                fund_data['Interest Coverage'] = get_latest_valid_number(df.iloc[i])
                
        return fund_data
    except Exception:
        return fund_data

def create_sliding_window(X_data, y_data, close_prices, lookback=5):
    X_seq, y_seq, prev_close = [], [], []
    for i in range(lookback, len(X_data)):
        X_seq.append(X_data[i-lookback:i])
        y_seq.append(y_data[i])
        prev_close.append(close_prices[i-1])
    return np.array(X_seq), np.array(y_seq), np.array(prev_close)

def forecast_future_recursive(rfr_model, lstm_model, df, scaler, feature_cols, lookback=5, forecast_steps=30):
    df_temp_rfr = df.copy()
    df_temp_lstm = df.copy()
    forecast_rfr_prices, forecast_lstm_prices = [], []
    
    if rfr_model is not None:
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
            df_temp_rfr = pd.concat([df_temp_rfr, pd.DataFrame([last_row])], ignore_index=True)

    if lstm_model is not None:
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
            df_temp_lstm = pd.concat([df_temp_lstm, pd.DataFrame([last_row])], ignore_index=True)

    return forecast_rfr_prices, forecast_lstm_prices

def calculate_fundamental_score(fund_data):
    score = 0
    per = fund_data.get('PER', 15)
    roe = fund_data.get('ROE', 0.10)
    ic = fund_data.get('Interest Coverage', 5)
    
    if 0 < per <= 15: score += 40
    elif 15 < per <= 25: score += 20
    else: score += 10
    
    if roe > 0.15: score += 40
    elif 0.05 <= roe <= 0.15: score += 20
    else: score += 10
        
    if ic > 6: score += 20
    elif 2 <= ic <= 6: score += 10
    else: score += 5
    return score

def calculate_technical_score(mae_pct, trend_pct):
    score = 0
    if mae_pct <= 0.02: score += 50
    elif 0.02 < mae_pct <= 0.05: score += 25
    else: score += 10
    
    if trend_pct > 0.02: score += 50
    elif -0.02 <= trend_pct <= 0.02: score += 25
    else: score += 10
    return score

def get_label(score):
    if score >= 70: return "Bagus"
    elif score >= 45: return "Wajar"
    else: return "Jelek"

st.title("Dashboard Prediksi dan Analisis Saham")
st.markdown("Sektor Konsumen Primer Industri Makanan Olahan (BEI)")

with st.expander("📊 Tabel Rekapitulasi Hasil Prediksi & Evaluasi Skripsi (Asli)", expanded=False):
    st.markdown("Tabel di bawah ini menampilkan kompilasi hasil evaluasi metrik dan prediksi yang didapatkan dari perhitungan penelitian (Model Random Forest dan LSTM).")
    
    skripsi_csv = """Emiten,Harga Terakhir (Rp),Prediksi +1 Hari RFR,Prediksi +1 Hari LSTM,Prediksi +30 Hari RFR,Prediksi +30 Hari LSTM,RFR MAE(Ret),RFR RMSE(Ret),RFR MAE(Rp),RFR RMSE(Rp),LSTM MAE(Ret),LSTM RMSE(Ret),LSTM MAE(Rp),LSTM RMSE(Rp)
ICBP,7125.0,7138.46,7124.11,6976.15,6082.03,0.01448,0.0193,115.18,152.92,0.01439,0.01948,113.86,152.82
INDF,6600.0,6619.1,6724.59,6114.04,7371.88,0.01298,0.01687,92.52,118.63,0.01373,0.0176,98.38,124.94
MYOR,1815.0,1805.8,1818.75,1753.93,1736.0,0.01734,0.02262,36.08,47.04,0.01748,0.02235,36.52,46.95
CMRY,4790.0,4762.32,4759.18,4726.2,6687.2,0.0189,0.02599,96.34,133.47,0.01902,0.02583,96.1,130.27
ULTJ,1500.0,1492.97,1505.97,1491.24,1680.81,0.01414,0.02053,20.92,30.73,0.01386,0.02033,20.45,30.29"""
    
    df_skripsi = pd.read_csv(io.StringIO(skripsi_csv))
    
    st.dataframe(
        df_skripsi.style.format({
            "Harga Terakhir (Rp)": "Rp {:,.0f}",
            "Prediksi +1 Hari RFR": "Rp {:,.2f}",
            "Prediksi +1 Hari LSTM": "Rp {:,.2f}",
            "Prediksi +30 Hari RFR": "Rp {:,.2f}",
            "Prediksi +30 Hari LSTM": "Rp {:,.2f}",
            "RFR MAE(Ret)": "{:.5f}",
            "RFR RMSE(Ret)": "{:.5f}",
            "RFR MAE(Rp)": "Rp {:,.2f}",
            "RFR RMSE(Rp)": "Rp {:,.2f}",
            "LSTM MAE(Ret)": "{:.5f}",
            "LSTM RMSE(Ret)": "{:.5f}",
            "LSTM MAE(Rp)": "Rp {:,.2f}",
            "LSTM RMSE(Rp)": "Rp {:,.2f}",
        }),
        use_container_width=True
    )

with st.sidebar:
    st.header("Control Panel")
    
    data_source = st.radio("Pilih Sumber Data:", ["Database Skripsi (ICBP, INDF, dkk)", "Unggah File Custom (CSV)"])
    
    emiten_configs = []
    
    if data_source == "Database Skripsi (ICBP, INDF, dkk)":
        selected_emitens = st.multiselect("Pilih Emiten untuk Dianalisis/Bandingkan:", 
                                          ["Bandingkan 5 Emiten Skripsi", "ICBP", "INDF", "MYOR", "CMRY", "ULTJ"], default=["ICBP"])
        
        if "Bandingkan 5 Emiten Skripsi" in selected_emitens:
            target_list = ["ICBP", "INDF", "MYOR", "CMRY", "ULTJ"]
        else:
            target_list = selected_emitens
            
        for ticker in target_list:
            emiten_configs.append({
                "name": ticker,
                "tech_source": os.path.join("data", f"{ticker}.csv"),
                "fund_source": os.path.join("data fundamental", f"Fundamental_{ticker}.csv"),
                "is_custom": False
            })
    else:
        st.info("Pilih jumlah emiten yang ingin diunggah dan dibandingkan.")
        num_custom = st.number_input("Jumlah Emiten Custom:", min_value=1, max_value=5, value=1)
        for i in range(int(num_custom)):
            st.markdown(f"**Data Emiten {i+1}**")
            t_file = st.file_uploader(f"Historis CSV (Emiten {i+1})", type=["csv"], key=f"t_{i}")
            f_file = st.file_uploader(f"Fundamental CSV (Emiten {i+1})", type=["csv"], key=f"f_{i}")
            if t_file and f_file:
                emiten_configs.append({
                    "name": t_file.name.split('.')[0].upper(),
                    "tech_source": t_file,
                    "fund_source": f_file,
                    "is_custom": True
                })

    st.markdown("---")
    model_choice = st.selectbox("Pilih Algoritma Model:", ["Bandingkan Kedua Model (RFR vs LSTM)", "Random Forest Regressor (RFR)", "Long Short Term Memory (LSTM)"])
    lookback = st.slider("Panjang Sliding Window (Hari):", min_value=3, max_value=15, value=5)
    forecast_days = st.slider("Horizon Proyeksi (Hari):", min_value=1, max_value=30, value=30)
    
    run_button = st.button("Jalankan Analisis")

@st.cache_data(show_spinner=False)
def process_emiten(config, model_choice, lookback, forecast_days):
    try:
        df = load_and_preprocess_source(config["tech_source"])
        fund_data = parse_fundamental(config["fund_source"])
        if df.empty: return None
        
        feature_cols = ['X1_Close_Return', 'X2_Volume_Return', 'X3_SMA_Ratio', 'X4_MACD_Hist', 'X5_RSI', 'X6_Bollinger_B', 'X7_Stochastic_K']
        X_raw = df[feature_cols].values
        y_raw = df['Y_Target_Return'].values
        close_raw = df['Close'].values
        train_size = int(len(df) * 0.8)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_raw[:train_size])
        X_test_scaled = scaler.transform(X_raw[train_size:])
        
        X_train, y_train, _ = create_sliding_window(X_train_scaled, y_raw[:train_size], close_raw[:train_size], lookback)
        X_test, y_test, prev_close_test = create_sliding_window(X_test_scaled, y_raw[train_size:], close_raw[train_size:], lookback)
        actual_close_test = close_raw[train_size+lookback:]
        
        X_train_rfr = X_train.reshape(X_train.shape[0], -1)
        X_test_rfr = X_test.reshape(X_test.shape[0], -1)
        
        rfr_model, lstm_model = None, None
        train_rfr = model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Random Forest Regressor (RFR)"]
        train_lstm = model_choice in ["Bandingkan Kedua Model (RFR vs LSTM)", "Long Short Term Memory (LSTM)"]
        
        if not config["is_custom"] and lookback == 5:
            rfr_path = os.path.join("Model", "saved_models", f"rfr_model_{config['name']}.joblib")
            lstm_path = os.path.join("Model", "saved_models", f"lstm_model_{config['name']}.keras")
            if train_rfr and os.path.exists(rfr_path):
                rfr_model = joblib.load(rfr_path)
                train_rfr = False
            if train_lstm and os.path.exists(lstm_path):
                lstm_model = load_model(lstm_path)
                train_lstm = False
                
        if train_rfr:
            rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rfr_model.fit(X_train_rfr, y_train)
        if train_lstm:
            lstm_model = Sequential([LSTM(32, input_shape=(lookback, 7)), Dropout(0.1), Dense(1)])
            lstm_model.compile(optimizer=Adam(0.001), loss='mse')
            lstm_model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, shuffle=False)
            
        rfr_metrics, lstm_metrics = {}, {}
        
        if rfr_model is not None:
            rfr_pred = rfr_model.predict(X_test_rfr)
            pred_close = prev_close_test * (1 + rfr_pred)
            rfr_metrics['mae_rp'] = mean_absolute_error(actual_close_test, pred_close)
            rfr_metrics['mae_pct'] = rfr_metrics['mae_rp'] / np.mean(actual_close_test)
            rfr_metrics['rmse_rp'] = np.sqrt(mean_squared_error(actual_close_test, pred_close))
            
        if lstm_model is not None:
            lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
            pred_close = prev_close_test * (1 + lstm_pred)
            lstm_metrics['mae_rp'] = mean_absolute_error(actual_close_test, pred_close)
            lstm_metrics['mae_pct'] = lstm_metrics['mae_rp'] / np.mean(actual_close_test)
            lstm_metrics['rmse_rp'] = np.sqrt(mean_squared_error(actual_close_test, pred_close))
            
        forecast_rfr, forecast_lstm = forecast_future_recursive(rfr_model, lstm_model, df, scaler, feature_cols, lookback, forecast_days)
        last_price = df['Close'].iloc[-1]
        
        best_mae_pct = lstm_metrics.get('mae_pct', rfr_metrics.get('mae_pct', 0))
        best_forecast = forecast_lstm if forecast_lstm else forecast_rfr
        trend_pct = (best_forecast[-1] - last_price) / last_price if best_forecast else 0
        
        tech_score = calculate_technical_score(best_mae_pct, trend_pct)
        fund_score = calculate_fundamental_score(fund_data)
        
        return {
            "name": config["name"], "df": df, "fund": fund_data, "last_price": last_price,
            "forecast_rfr": forecast_rfr, "forecast_lstm": forecast_lstm,
            "rfr_metrics": rfr_metrics, "lstm_metrics": lstm_metrics,
            "tech_score": tech_score, "fund_score": fund_score, "trend_pct": trend_pct
        }
    except Exception:
        return None

if run_button:
    if len(emiten_configs) == 0:
        st.warning("Silakan pilih emiten dari database atau unggah file custom terlebih dahulu.")
    else:
        results = []
        with st.spinner("Memproses data analisis..."):
            for config in emiten_configs:
                res = process_emiten(config, model_choice, lookback, forecast_days)
                if res: results.append(res)
                
        if not results:
            st.error("Gagal memproses data. Pastikan format CSV sesuai.")
        else:
            st.subheader(f"Grafik Proyeksi Tren ({forecast_days} Hari Kedepan)")
            fig, ax = plt.subplots(figsize=(12, 5))
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            for idx, res in enumerate(results):
                c = colors[idx % len(colors)]
                hist_subset = res['df']['Close'].tail(60).values
                hist_x = np.arange(len(hist_subset))
                future_x = np.arange(len(hist_subset) - 1, len(hist_subset) + forecast_days)
                base_price = hist_subset[0]
                
                ax.plot(hist_x, (hist_subset/base_price - 1)*100, label=f"Historis {res['name']}", color=c, linewidth=2)
                
                if res['forecast_rfr']:
                    rfr_full = np.array([hist_subset[-1]] + res['forecast_rfr'])
                    ax.plot(future_x, (rfr_full/base_price - 1)*100, label=f"Proyeksi RFR {res['name']}", color=c, linestyle='--', linewidth=1.5)
                    
                if res['forecast_lstm']:
                    lstm_full = np.array([hist_subset[-1]] + res['forecast_lstm'])
                    ax.plot(future_x, (lstm_full/base_price - 1)*100, label=f"Proyeksi LSTM {res['name']}", color=c, linestyle=':', linewidth=1.5)
                    
            ax.set_ylabel("Pergerakan Harga (%)", fontsize=10)
            ax.set_xlabel(f"Hari Bursa (60 Hari Terakhir + {forecast_days} Hari Kedepan)", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            st.pyplot(fig)
            
            st.markdown("---")
            st.subheader("Rincian Metrik & Penilaian")
            
            for res in results:
                st.markdown(f"### Emiten: {res['name']}")
                col_skor, col_metrik = st.columns([1, 2])
                
                with col_skor:
                    t_lbl = get_label(res['tech_score'])
                    f_lbl = get_label(res['fund_score'])
                    st.metric("Skor Teknikal", f"{res['tech_score']}/100", f"Status: {t_lbl}")
                    st.metric("Skor Fundamental", f"{res['fund_score']}/100", f"Status: {f_lbl}")
                    
                with col_metrik:
                    eval_data = []
                    if res['rfr_metrics']:
                        eval_data.append({
                            "Algoritma": "RFR",
                            "MAE (Rp)": f"Rp {res['rfr_metrics']['mae_rp']:,.2f}",
                            "RMSE (Rp)": f"Rp {res['rfr_metrics']['rmse_rp']:,.2f}",
                            f"Prediksi +{forecast_days}H": f"Rp {res['forecast_rfr'][-1]:,.0f}"
                        })
                    if res['lstm_metrics']:
                        eval_data.append({
                            "Algoritma": "LSTM",
                            "MAE (Rp)": f"Rp {res['lstm_metrics']['mae_rp']:,.2f}",
                            "RMSE (Rp)": f"Rp {res['lstm_metrics']['rmse_rp']:,.2f}",
                            f"Prediksi +{forecast_days}H": f"Rp {res['forecast_lstm'][-1]:,.0f}"
                        })
                    st.write("**Evaluasi Akurasi Prediksi:**")
                    st.table(pd.DataFrame(eval_data))
                    
                    fund_metrics = {
                        "PER": f"{res['fund'].get('PER', 0):.2f}x",
                        "ROE": f"{res['fund'].get('ROE', 0)*100:.1f}%",
                        "ROA": f"{res['fund'].get('ROA', 0)*100:.1f}%",
                        "Int. Coverage": f"{res['fund'].get('Interest Coverage', 0):.2f}"
                    }
                    st.write("**Indikator Fundamental:**")
                    st.json(fund_metrics, expanded=False)
                st.divider()

            st.subheader("Kesimpulan Analisis")
            results.sort(key=lambda x: x['tech_score'] + x['fund_score'], reverse=True)
            
            if len(results) > 1:
                best_emiten = results[0]
                conclusion = f"Berdasarkan analisis komparatif bobot skor teknikal (presisi model & tren) dan skor fundamental (valuasi & profitabilitas), **{best_emiten['name']}** menempati peringkat tertinggi.\n\n"
                txt_conclusion = "KESIMPULAN ANALISIS KOMPARATIF\n" + "="*50 + "\n\n"
                
                for i, res in enumerate(results):
                    t_lbl = get_label(res['tech_score'])
                    f_lbl = get_label(res['fund_score'])
                    conclusion += f"* **Peringkat {i+1} - {res['name']}**: Sinyal Teknikal **{t_lbl}** (Skor {res['tech_score']}), Fundamental **{f_lbl}** (Skor {res['fund_score']}).\n"
                    
                    txt_conclusion += f"Peringkat {i+1}. Emiten: {res['name']}\n"
                    txt_conclusion += f"   - Sinyal Teknikal  : {t_lbl} (Skor: {res['tech_score']})\n"
                    txt_conclusion += f"   - Nilai Fundamental: {f_lbl} (Skor: {res['fund_score']} | PER: {res['fund'].get('PER', 0):.2f}x | ROE: {res['fund'].get('ROE', 0)*100:.1f}% | ROA: {res['fund'].get('ROA', 0)*100:.1f}% | Int. Coverage: {res['fund'].get('Interest Coverage', 0):.2f})\n\n"
            else:
                res = results[0]
                t_lbl = get_label(res['tech_score'])
                f_lbl = get_label(res['fund_score'])
                conclusion = f"Berdasarkan analisis performa, emiten **{res['name']}** mencatatkan Sinyal Teknikal **{t_lbl}** (Skor {res['tech_score']}) dengan estimasi tren {res['trend_pct']*100:.1f}%, dan Nilai Fundamental **{f_lbl}** (Skor {res['fund_score']} dengan PER {res['fund'].get('PER', 0):.2f}x).\n\n"
                
                txt_conclusion = "KESIMPULAN ANALISIS PREDIKTIF & FUNDAMENTAL\n" + "="*50 + "\n\n"
                txt_conclusion += f"Emiten: {res['name']}\n"
                txt_conclusion += f"   - Sinyal Teknikal  : {t_lbl} (Skor: {res['tech_score']})\n"
                txt_conclusion += f"   - Nilai Fundamental: {f_lbl} (Skor: {res['fund_score']} | PER: {res['fund'].get('PER', 0):.2f}x | ROE: {res['fund'].get('ROE', 0)*100:.1f}% | ROA: {res['fund'].get('ROA', 0)*100:.1f}% | Int. Coverage: {res['fund'].get('Interest Coverage', 0):.2f})\n\n"

            st.info(conclusion)
            
            st.download_button(
                label="📥 Unduh Hasil Kesimpulan (TXT)",
                data=txt_conclusion,
                file_name="Kesimpulan.txt",
                mime="text/plain",
                use_container_width=True
            )
            
else:
    st.info("Pilih parameter dari panel di sebelah kiri dan klik 'Jalankan Analisis'.")