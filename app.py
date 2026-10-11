import os
import csv
import traceback
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

# ==========================================
# KONFIGURASI
# ==========================================
st.set_page_config(
    page_title="Dashboard Prediksi & Analisis Saham",
    layout="wide",
    initial_sidebar_state="expanded"
)

np.random.seed(42)
tf.random.set_seed(42)

# Jumlah epoch LSTM saat model dilatih ulang di dashboard (sama dengan main.py)
LSTM_EPOCHS = 50

# ==========================================
# FUNGSI TEKNIKAL & DATA
# ==========================================
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

# ==========================================
# DATA FUNDAMENTAL (BASIS TTM)
# ==========================================
# Format file: baris pertama berisi label kuartal (kolom pertama = kuartal terbaru),
# lalu baris indikator (EPS, PE Ratio, Return on Equity, Return on Assets, Interest Coverage).
# PER TTM            = harga akhir kuartal terbaru / jumlah EPS 4 kuartal terakhir
#                      (harga akhir kuartal = PE Ratio kuartal terbaru x EPS kuartal terbaru)
# ROE TTM, ROA TTM   = jumlah 4 kuartal terakhir (pendekatan nilai setahun)
# Interest Coverage  = rata-rata 4 kuartal terakhir
TTM_QUARTERS = 4

def _read_fund_rows(source):
    if isinstance(source, str):
        if not os.path.exists(source): return []
        with open(source, 'r', encoding='utf-8-sig', errors='replace') as f:
            text = f.read()
    else:
        text = source.getvalue().decode('utf-8-sig', errors='replace')
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines: return []
    first = lines[0]
    if first.count('\t') >= first.count(';') and first.count('\t') > 0:
        return [[c.strip() for c in ln.split('\t')] for ln in lines]
    if ';' in first:
        return [[c.strip() for c in ln.split(';')] for ln in lines]
    return [[c.strip() for c in row] for row in csv.reader(lines)]

def _to_float(s):
    s = str(s).strip().replace('%', '').replace(',', '')
    if s.lower() in ('', 'n/a', 'nan', 'none', '-'):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _latest_quarters(cells, n=TTM_QUARTERS):
    """Ambil n kuartal terbaru dari satu baris. Semuanya harus berupa angka, kalau tidak hasilnya None."""
    vals = [_to_float(c) for c in cells[1:1 + n]]
    if len(vals) < n or any(v is None for v in vals):
        return None
    return vals

def parse_fundamental(source):
    fund = {'PER': None, 'ROE': None, 'ROA': None, 'Interest Coverage': None, 'Periode': None}
    try:
        rows = _read_fund_rows(source)
        if not rows:
            return fund

        header = rows[0]
        labels = [h for h in header[1:1 + TTM_QUARTERS] if h]
        if len(labels) == TTM_QUARTERS:
            fund['Periode'] = f"{labels[-1]} - {labels[0]}"

        eps = pe = None
        for cells in rows[1:]:
            name = cells[0].strip().lower()
            if name.startswith('eps'):
                eps = _latest_quarters(cells)
            elif name.startswith('pe ratio'):
                pe = _latest_quarters(cells)
            elif name.startswith('return on equity'):
                q = _latest_quarters(cells)
                if q: fund['ROE'] = sum(q) / 100
            elif name.startswith('return on assets'):
                q = _latest_quarters(cells)
                if q: fund['ROA'] = sum(q) / 100
            elif name.startswith('interest coverage'):
                q = _latest_quarters(cells)
                if q: fund['Interest Coverage'] = sum(q) / len(q)

        if eps and pe:
            eps_ttm = sum(eps)
            harga_akhir_kuartal = pe[0] * eps[0]
            if eps_ttm > 0 and harga_akhir_kuartal > 0:
                fund['PER'] = harga_akhir_kuartal / eps_ttm
        return fund
    except Exception:
        traceback.print_exc()
        return fund

def fmt_num(value, decimals=2, suffix=''):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:,.{decimals}f}{suffix}"

def fmt_pct_ratio(value, decimals=2):
    """value berupa rasio (0.1415 -> 14.15%)."""
    if value is None:
        return "n/a"
    return f"{value * 100:,.{decimals}f}%"

def fmt_ic(value):
    if value is None:
        return "n/a"
    return "> 100x" if value > 100 else f"{value:,.2f}"

# ==========================================
# WINDOWING & PROYEKSI
# ==========================================
def create_sliding_window(X_data, y_data, close_prices, lookback=5):
    X_seq, y_seq, prev_close = [], [], []
    for i in range(lookback, len(X_data)):
        X_seq.append(X_data[i-lookback:i])
        y_seq.append(y_data[i])
        prev_close.append(close_prices[i-1])
    return np.array(X_seq), np.array(y_seq), np.array(prev_close)

def _extend_row(df_temp, next_price):
    last_row = df_temp.iloc[-1].copy()
    last_row['Close'] = next_price
    # Sama dengan main.py: High dan Low ikut melebar agar indikator stokastik tetap valid
    last_row['High'] = max(last_row['High'], next_price)
    last_row['Low'] = min(last_row['Low'], next_price)
    return pd.concat([df_temp, pd.DataFrame([last_row])], ignore_index=True)

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
            df_temp_rfr = _extend_row(df_temp_rfr, next_price)

    if lstm_model is not None:
        for _ in range(forecast_steps):
            df_temp_lstm = compute_technical_indicators(df_temp_lstm)
            X_curr = scaler.transform(df_temp_lstm[feature_cols].values[-lookback:])
            X_curr_lstm = X_curr.reshape(1, lookback, len(feature_cols))
            pred_ret = lstm_model.predict(X_curr_lstm, verbose=0).flatten()[0]
            last_price = df_temp_lstm['Close'].iloc[-1]
            next_price = last_price * (1 + pred_ret)
            forecast_lstm_prices.append(next_price)
            df_temp_lstm = _extend_row(df_temp_lstm, next_price)

    return forecast_rfr_prices, forecast_lstm_prices

# ==========================================
# RINGKASAN (KESIMPULAN) DESKRIPTIF
# ==========================================
# Tidak ada skor, ambang, atau label penilaian. Kesimpulan hanya merangkum angka yang sudah ada:
# proyeksi tiap model dibandingkan harga terakhir, tingkat kesalahan (MAE return) hasil evaluasi,
# dan rasio fundamental TTM sebagai pembanding deskriptif.
def describe_forecast(label, forecast_price, last_price):
    forecast_price = float(forecast_price)
    last_price = float(last_price)
    diff = forecast_price - last_price
    diff_pct = diff / last_price
    if diff > 0:
        posisi = "di atas harga terakhir"
    elif diff < 0:
        posisi = "di bawah harga terakhir"
    else:
        posisi = "sama dengan harga terakhir"
    return {
        "label": label,
        "price": forecast_price,
        "diff": diff,
        "diff_pct": diff_pct,
        "posisi": posisi,
        "sign": int(diff > 0) - int(diff < 0)
    }

def build_summary(res, forecast_days):
    last_price = res['last_price']
    items = []
    if res['forecast_rfr']:
        items.append(describe_forecast("RFR", res['forecast_rfr'][-1], last_price))
    if res['forecast_lstm']:
        items.append(describe_forecast("LSTM", res['forecast_lstm'][-1], last_price))

    if len(items) == 2:
        kesepakatan = "searah" if items[0]['sign'] == items[1]['sign'] else "berbeda arah"
    else:
        kesepakatan = "-"

    mae_parts = []
    if res['rfr_metrics']:
        mae_parts.append(f"RFR {res['rfr_metrics']['mae_ret'] * 100:.2f}%")
    if res['lstm_metrics']:
        mae_parts.append(f"LSTM {res['lstm_metrics']['mae_ret'] * 100:.2f}%")

    fund = res['fund']
    text = f"{res['name']}: harga terakhir Rp {last_price:,.0f} ({res['last_date']}). "
    for it in items:
        text += (f"Proyeksi {it['label']} {forecast_days} hari ke depan Rp {it['price']:,.0f} "
                 f"({it['diff_pct'] * 100:+.2f}%, {it['posisi']}). ")
    if kesepakatan == "searah":
        text += "Kedua model memberi proyeksi yang searah terhadap harga terakhir. "
    elif kesepakatan == "berbeda arah":
        text += "Kedua model memberi proyeksi yang berbeda arah terhadap harga terakhir. "
    if mae_parts:
        text += f"MAE (return) hasil evaluasi data uji: {', '.join(mae_parts)}. "
    periode = f" ({fund['Periode']})" if fund.get('Periode') else ""
    text += (f"Sebagai pembanding deskriptif, rasio fundamental TTM{periode}: "
             f"PER {fmt_num(fund['PER'], 2, 'x')}, ROE {fmt_pct_ratio(fund['ROE'])}, "
             f"ROA {fmt_pct_ratio(fund['ROA'])}, interest coverage {fmt_ic(fund['Interest Coverage'])}.")

    row = {
        "Emiten": res['name'],
        "Harga Terakhir": f"Rp {last_price:,.0f}",
        "Tanggal Data Akhir": res['last_date'],
    }
    for it in items:
        row[f"Proyeksi {it['label']} (+{forecast_days}H)"] = f"Rp {it['price']:,.0f} ({it['diff_pct'] * 100:+.2f}%)"
    row["Kedua Model"] = kesepakatan
    row["MAE Return"] = ", ".join(mae_parts) if mae_parts else "-"
    row["PER TTM"] = fmt_num(fund['PER'], 2, 'x')
    return text, row

DISCLAIMER = ("Catatan: hasil ini merupakan proyeksi model dari data historis dan rasio fundamental sebagai pembanding "
              "deskriptif. Hasil ini bukan rekomendasi investasi dan bukan jaminan arah harga.")

# ==========================================
# MAIN APP & UI
# ==========================================
st.title("Dashboard Prediksi dan Analisis Saham")
st.markdown("Sektor Konsumen Primer Industri Makanan Olahan (BEI)")

# --- FILTER DATA SKRIPSI ---
with st.sidebar:
    st.header("Control Panel")
    data_source = st.radio("Pilih Sumber Data:", ["Dataset Skripsi", "Unggah File Custom (CSV)"])
    emiten_configs = []

    if data_source == "Dataset Skripsi":
        target_list = st.multiselect(
            "Pilih Emiten untuk Dianalisis/Bandingkan:",
            ["ICBP", "INDF", "MYOR", "CMRY", "ULTJ"],
            default=["ICBP"]
        )

        for ticker in target_list:
            emiten_configs.append({
                "name": ticker,
                "tech_source": os.path.join("data", f"{ticker}.csv"),
                "fund_source": os.path.join("data fundamental", f"Fundamental_{ticker}.csv"),
                "is_custom": False
            })
    else:
        st.info("Pilih jumlah emiten yang ingin diunggah dan dibandingkan.")

        # TEMPLATE DOWNLOADS
        tech_template = "Date,Open,High,Low,Close,Volume\n2024-01-01,1000,1050,990,1020,50000\n2024-01-02,1020,1060,1010,1050,60000\n"
        fund_template = (
            "Dalam IDR;Q2 2026;Q1 2026;Q4 2025;Q3 2025\n"
            "EPS (Quarter);100.00;90.00;110.00;95.00\n"
            "PE Ratio (Quarter);15.00;16.00;14.00;15.50\n"
            "Return on Assets (Quarter);2.00%;1.90%;2.10%;1.80%\n"
            "Return on Equity (Quarter);4.00%;3.80%;4.20%;3.60%\n"
            "Interest Coverage (Quarter);5.00;4.50;5.50;4.80\n"
        )

        st.download_button("Unduh Template Teknikal (CSV)", data=tech_template, file_name="template_teknikal.csv", mime="text/csv", help="Format riwayat harga saham harian.")
        st.download_button("Unduh Template Fundamental (CSV)", data=fund_template, file_name="template_fundamental.csv", mime="text/csv", help="Format data fundamental per kuartal (pemisah titik koma, kolom pertama = kuartal terbaru, minimal 4 kuartal).")

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

# --- TABEL EVALUASI SKRIPSI (HANYA MUNCUL DI MODE DATASET) ---
if data_source == "Dataset Skripsi":
    with st.expander("Tabel Evaluasi Skripsi"):
        try:
            eval_path = os.path.join("Model", "ringkasan_evaluasi.csv")
            eval_df = pd.read_csv(eval_path)
            format_dict = {
                'Harga Terakhir (Rp)': 'Rp {:,.2f}',
                'Prediksi +1 Hari RFR': 'Rp {:,.2f}',
                'Prediksi +1 Hari LSTM': 'Rp {:,.2f}',
                'Prediksi +30 Hari RFR': 'Rp {:,.2f}',
                'Prediksi +30 Hari LSTM': 'Rp {:,.2f}',
                'RFR MAE(Ret)': '{:.5f}',
                'RFR RMSE(Ret)': '{:.5f}',
                'RFR MAE(Rp)': 'Rp {:,.2f}',
                'RFR RMSE(Rp)': 'Rp {:,.2f}',
                'LSTM MAE(Ret)': '{:.5f}',
                'LSTM RMSE(Ret)': '{:.5f}',
                'LSTM MAE(Rp)': 'Rp {:,.2f}',
                'LSTM RMSE(Rp)': 'Rp {:,.2f}',
            }
            st.dataframe(eval_df.style.format(format_dict))
        except Exception:
            st.error("File ringkasan_evaluasi.csv tidak ditemukan di direktori Model.")

# --- FUNGSI PROSES EMITEN ---
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
            lstm_model.fit(X_train, y_train, epochs=LSTM_EPOCHS, batch_size=16, verbose=0, shuffle=False)

        rfr_metrics, lstm_metrics = {}, {}

        if rfr_model is not None:
            rfr_pred = rfr_model.predict(X_test_rfr)
            pred_close = prev_close_test * (1 + rfr_pred)
            rfr_metrics['mae_ret'] = mean_absolute_error(y_test, rfr_pred)
            rfr_metrics['rmse_ret'] = np.sqrt(mean_squared_error(y_test, rfr_pred))
            rfr_metrics['mae_rp'] = mean_absolute_error(actual_close_test, pred_close)
            rfr_metrics['rmse_rp'] = np.sqrt(mean_squared_error(actual_close_test, pred_close))

        if lstm_model is not None:
            lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
            pred_close = prev_close_test * (1 + lstm_pred)
            lstm_metrics['mae_ret'] = mean_absolute_error(y_test, lstm_pred)
            lstm_metrics['rmse_ret'] = np.sqrt(mean_squared_error(y_test, lstm_pred))
            lstm_metrics['mae_rp'] = mean_absolute_error(actual_close_test, pred_close)
            lstm_metrics['rmse_rp'] = np.sqrt(mean_squared_error(actual_close_test, pred_close))

        forecast_rfr, forecast_lstm = forecast_future_recursive(rfr_model, lstm_model, df, scaler, feature_cols, lookback, forecast_days)
        last_price = df['Close'].iloc[-1]

        date_col = next((c for c in df.columns if 'date' in str(c).lower() or 'tanggal' in str(c).lower()), None)
        last_date = df[date_col].iloc[-1].strftime('%d %b %Y') if date_col else "-"

        return {
            "name": config["name"], "df": df, "fund": fund_data, "last_price": last_price, "last_date": last_date,
            "forecast_rfr": forecast_rfr, "forecast_lstm": forecast_lstm,
            "rfr_metrics": rfr_metrics, "lstm_metrics": lstm_metrics
        }
    except Exception:
        traceback.print_exc()
        return None

# --- EKSEKUSI PROSES MAIN ---
if run_button:
    if len(emiten_configs) == 0:
        st.warning("Silakan pilih emiten dari panel di sebelah kiri dan klik 'Jalankan Analisis'.")
    else:
        results = []
        with st.spinner("Memproses data komparasi..."):
            for config in emiten_configs:
                res = process_emiten(config, model_choice, lookback, forecast_days)
                if res: results.append(res)

        if not results:
            st.error("Gagal memproses data. Pastikan format CSV sesuai.")
        else:
            # 1. GRAFIK KOMPARASI GABUNGAN
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

            # 2. RINGKASAN PERBANDINGAN FUNDAMENTAL (TTM)
            st.subheader("Ringkasan Perbandingan Fundamental (TTM)")
            fund_summary = []
            for res in results:
                f = res['fund']
                fund_summary.append({
                    "Emiten": res['name'],
                    "PER TTM": fmt_num(f['PER'], 2, 'x'),
                    "ROE TTM": fmt_pct_ratio(f['ROE']),
                    "ROA TTM": fmt_pct_ratio(f['ROA']),
                    "Interest Coverage": fmt_ic(f['Interest Coverage']),
                    "Periode TTM": f['Periode'] or "-"
                })
            st.table(pd.DataFrame(fund_summary))
            st.caption("TTM = empat kuartal terakhir pada data fundamental. PER = harga akhir kuartal terbaru dibagi jumlah EPS empat kuartal; "
                       "ROE dan ROA = jumlah empat kuartal (pendekatan nilai setahun); interest coverage = rata-rata empat kuartal. "
                       "n/a berarti data tidak lengkap untuk dihitung.")

            st.markdown("**Grafik Visualisasi Indikator Fundamental**")
            nan = float('nan')
            fund_viz_df = pd.DataFrame({
                'Emiten': [r['name'] for r in results],
                'PER TTM (x)': [r['fund']['PER'] if r['fund']['PER'] is not None else nan for r in results],
                'ROE TTM (%)': [r['fund']['ROE'] * 100 if r['fund']['ROE'] is not None else nan for r in results],
                'ROA TTM (%)': [r['fund']['ROA'] * 100 if r['fund']['ROA'] is not None else nan for r in results],
                'Interest Coverage (x, maks. 100)': [min(r['fund']['Interest Coverage'], 100) if r['fund']['Interest Coverage'] is not None else nan for r in results]
            }).set_index('Emiten')

            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.write("Valuasi: Price to Earning Ratio (PER) TTM")
                st.bar_chart(fund_viz_df['PER TTM (x)'])
                st.write("Kemampuan Membayar Bunga: Interest Coverage (nilai di atas 100 ditampilkan 100)")
                st.bar_chart(fund_viz_df['Interest Coverage (x, maks. 100)'])
            with col_viz2:
                st.write("Profitabilitas: Return on Equity (ROE) TTM")
                st.bar_chart(fund_viz_df['ROE TTM (%)'])
                st.write("Profitabilitas: Return on Assets (ROA) TTM")
                st.bar_chart(fund_viz_df['ROA TTM (%)'])

            # FITUR VIEWER DATA MENTAH (HANYA MUNCUL DI MODE DATASET)
            if data_source == "Dataset Skripsi":
                with st.expander("Lihat Data Mentah & Histori Fundamental"):
                    for res in results:
                        try:
                            fund_path = os.path.join("data fundamental", f"Fundamental_{res['name']}.csv")
                            rows = _read_fund_rows(fund_path)
                            if rows:
                                st.markdown(f"**Data Fundamental {res['name']}**")
                                width = max(len(r) for r in rows)
                                padded = [r + [''] * (width - len(r)) for r in rows]
                                st.dataframe(pd.DataFrame(padded[1:], columns=padded[0]))
                        except Exception:
                            pass

            st.markdown("---")

            # 3. METRIK EVALUASI
            st.subheader("Rincian Metrik Prediksi & Evaluasi")
            for res in results:
                st.markdown(f"### Emiten: {res['name']}")
                col_info, col_metrik = st.columns([1, 2])

                with col_info:
                    st.metric("Harga Terakhir", f"Rp {res['last_price']:,.0f}")
                    st.caption(f"Data harga berakhir pada {res['last_date']}")

                with col_metrik:
                    eval_data = []
                    if res['rfr_metrics']:
                        eval_data.append({
                            "Algoritma": "RFR",
                            "MAE (Return)": f"{res['rfr_metrics'].get('mae_ret', 0):.5f}",
                            "RMSE (Return)": f"{res['rfr_metrics'].get('rmse_ret', 0):.5f}",
                            "MAE (Rp)": f"Rp {res['rfr_metrics']['mae_rp']:,.2f}",
                            "RMSE (Rp)": f"Rp {res['rfr_metrics']['rmse_rp']:,.2f}",
                            f"Prediksi +{forecast_days}H": f"Rp {res['forecast_rfr'][-1]:,.0f}"
                        })
                    if res['lstm_metrics']:
                        eval_data.append({
                            "Algoritma": "LSTM",
                            "MAE (Return)": f"{res['lstm_metrics'].get('mae_ret', 0):.5f}",
                            "RMSE (Return)": f"{res['lstm_metrics'].get('rmse_ret', 0):.5f}",
                            "MAE (Rp)": f"Rp {res['lstm_metrics']['mae_rp']:,.2f}",
                            "RMSE (Rp)": f"Rp {res['lstm_metrics']['rmse_rp']:,.2f}",
                            f"Prediksi +{forecast_days}H": f"Rp {res['forecast_lstm'][-1]:,.0f}"
                        })
                    st.write("**Evaluasi Akurasi Prediksi:**")
                    st.table(pd.DataFrame(eval_data))
                st.divider()

            # 4. KESIMPULAN (RINGKASAN DESKRIPTIF)
            st.subheader("Kesimpulan Analisis")
            summary_rows = []
            txt_conclusion = "RINGKASAN HASIL PROYEKSI DAN PEMBANDING FUNDAMENTAL\n" + "=" * 50 + "\n\n"
            for res in results:
                text, row = build_summary(res, forecast_days)
                summary_rows.append(row)
                st.markdown(text)
                txt_conclusion += text + "\n\n"

            if len(summary_rows) > 1:
                st.markdown("**Tabel ringkasan:**")
                st.table(pd.DataFrame(summary_rows).fillna("-"))

            st.info(DISCLAIMER)
            txt_conclusion += DISCLAIMER + "\n"
            st.download_button(
                label="Unduh Hasil Kesimpulan (TXT)",
                data=txt_conclusion,
                file_name="Kesimpulan.txt",
                mime="text/plain",
                use_container_width=True
            )
else:
    st.info("Pilih parameter dari panel di sebelah kiri dan klik 'Jalankan Analisis'.")