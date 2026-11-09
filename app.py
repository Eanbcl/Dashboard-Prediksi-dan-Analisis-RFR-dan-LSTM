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
    page_title="Predikta",
    layout="wide",
    initial_sidebar_state="expanded"
)

np.random.seed(42)
tf.random.set_seed(42)

# Jumlah epoch LSTM saat model dilatih ulang di dashboard (sama dengan main.py)
LSTM_EPOCHS = 50

# Folder dan file hasil main_revisi.py serta periode data yang disamakan (1.074 hari bursa per emiten)
MODEL_DIR = "Model_revisi"
EVAL_FILE = "ringkasan_evaluasi_main_revisi.csv"
START_DATE = "2021-12-06"
END_DATE = "2026-06-02"

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

def load_and_preprocess_source(source, period=None):
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
        if period:
            df = df[(df[date_col] >= period[0]) & (df[date_col] <= period[1])].reset_index(drop=True)

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

def fund_display_values(fund):
    """Nilai fundamental dalam satuan tampilan: PER (x), ROE (%), ROA (%), Interest Coverage (x)."""
    return {
        'PER': fund['PER'],
        'ROE': fund['ROE'] * 100 if fund['ROE'] is not None else None,
        'ROA': fund['ROA'] * 100 if fund['ROA'] is not None else None,
        'IC': fund['Interest Coverage'],
    }

INDICATORS = [
    ('PER', 'PER', 'x',
     "Price to Earning Ratio (PER) TTM: harga saham dibagi jumlah laba per saham (EPS) empat kuartal terakhir. "
     "Makin tinggi nilainya, makin besar harga yang dibayar pasar untuk setiap Rp 1 laba."),
    ('ROE', 'ROE', '%',
     "Return on Equity (ROE) TTM: laba bersih terhadap ekuitas pemegang saham, dijumlahkan dari empat kuartal."),
    ('ROA', 'ROA', '%',
     "Return on Assets (ROA) TTM: laba bersih terhadap total aset, dijumlahkan dari empat kuartal."),
    ('IC', 'Interest Coverage', 'x',
     "Interest Coverage: berapa kali laba operasi dapat menutup beban bunga, dirata-ratakan dari empat kuartal. "
     "Nilai di atas 100 kali ditampilkan sebagai > 100x."),
]

def render_indicator(key, label, unit, desc, funds, highlight):
    st.write(desc)
    vals = {name: fund_display_values(f)[key] for name, f in funds.items()}
    valid = {k: v for k, v in vals.items() if v is not None}
    if not valid:
        st.info("Data fundamental belum lengkap untuk dibandingkan.")
        return
    use_avg = key != 'IC'
    avg = sum(valid.values()) / len(valid) if use_avg else None
    order = sorted(valid, key=lambda k: valid[k], reverse=True)
    cap = 100.0 if key == 'IC' else None

    fig, ax = plt.subplots(figsize=(8, 3.6))
    heights = [min(valid[n], cap) if cap else valid[n] for n in order]
    colors = ['#d62728' if n in highlight else '#9bb7d4' for n in order]
    bars = ax.bar(order, heights, color=colors, width=0.6)
    top = max(heights) if heights else 1
    for bar, n in zip(bars, order):
        v = valid[n]
        txt = "> 100x" if (cap and v > cap) else f"{v:,.2f}{unit}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02, txt,
                ha='center', va='bottom', fontsize=9)
    if avg is not None:
        ax.axhline(avg, color='#444444', linestyle='--', linewidth=1)
        ax.text(0.99, avg, f"rata-rata {avg:,.2f}{unit} ", va='bottom', ha='right', fontsize=8, color='#444444',
                transform=ax.get_yaxis_transform())
    ax.set_ylabel(f"{label} ({unit})", fontsize=9)
    ax.set_ylim(min(0, min(heights)), top * 1.18)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)

    kalimat = f"Urutan dari tertinggi: {', '.join(order)}."
    if avg is not None:
        kalimat += f" Rata-rata {len(valid)} emiten: {avg:,.2f}{unit}."
        for n in order:
            if n in highlight:
                posisi = "di atas" if valid[n] > avg else ("di bawah" if valid[n] < avg else "sama dengan")
                kalimat += f" {n} {valid[n]:,.2f}{unit}, {posisi} rata-rata."
    st.caption(kalimat)

    rows = []
    for rank, n in enumerate(order, start=1):
        v = valid[n]
        row = {
            "Peringkat": rank,
            "Emiten": n + (" (dipilih)" if n in highlight else ""),
            f"{label} ({unit})": "lebih dari 100x" if (cap and v > cap) else f"{v:,.2f}",
        }
        if avg is not None:
            row["Selisih dari rata-rata"] = f"{v - avg:+,.2f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

def render_fundamental_section(funds, highlight):
    st.subheader("Perbandingan Fundamental Antar Emiten (TTM)")
    if len(funds) < 2:
        st.caption("Tambahkan lebih dari satu emiten untuk melihat perbandingan.")
    st.caption("Batang merah = emiten yang dipilih. TTM = empat kuartal terakhir (Q3 2025 sampai Q2 2026). "
               "Rasio ini hanya pembanding deskriptif, bukan masukan model.")
    tabs = st.tabs([i[1] for i in INDICATORS])
    for tab, (key, label, unit, desc) in zip(tabs, INDICATORS):
        with tab:
            render_indicator(key, label, unit, desc, funds, highlight)

@st.cache_data(show_spinner=False)
def load_all_fundamentals():
    funds = {}
    for t in ["ICBP", "INDF", "MYOR", "CMRY", "ULTJ"]:
        funds[t] = parse_fundamental(os.path.join("data fundamental", f"Fundamental_{t}.csv"))
    return funds

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
        text += "Kedua model memproyeksikan arah yang sama dari harga terakhir. "
    elif kesepakatan == "berbeda arah":
        text += "Arah proyeksi kedua model berbeda dari harga terakhir. "
    if mae_parts:
        text += f"MAE (return) pada data uji: {', '.join(mae_parts)}. "
    periode = f" ({fund['Periode']})" if fund.get('Periode') else ""
    text += (f"Sebagai pembanding, rasio fundamental TTM{periode}: "
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

DISCLAIMER = ("Catatan: ini proyeksi model dari data historis, dengan rasio fundamental sebagai pembanding. "
              "Bukan saran investasi dan bukan jaminan harga akan bergerak ke arah itu.")

# --- FUNGSI PROSES EMITEN ---
@st.cache_data(show_spinner=False)
def process_emiten(config, model_choice, lookback, forecast_days):
    try:
        period = None if config["is_custom"] else (START_DATE, END_DATE)
        df = load_and_preprocess_source(config["tech_source"], period)
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
            rfr_path = os.path.join(MODEL_DIR, "saved_models", f"rfr_model_{config['name']}.joblib")
            lstm_path = os.path.join(MODEL_DIR, "saved_models", f"lstm_model_{config['name']}.keras")
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
        test_rfr, test_lstm = None, None

        if rfr_model is not None:
            rfr_pred = rfr_model.predict(X_test_rfr)
            pred_close = prev_close_test * (1 + rfr_pred)
            rfr_metrics['mae_ret'] = mean_absolute_error(y_test, rfr_pred)
            rfr_metrics['rmse_ret'] = np.sqrt(mean_squared_error(y_test, rfr_pred))
            rfr_metrics['mae_rp'] = mean_absolute_error(actual_close_test, pred_close)
            rfr_metrics['rmse_rp'] = np.sqrt(mean_squared_error(actual_close_test, pred_close))
            test_rfr = pred_close

        if lstm_model is not None:
            lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
            pred_close = prev_close_test * (1 + lstm_pred)
            lstm_metrics['mae_ret'] = mean_absolute_error(y_test, lstm_pred)
            lstm_metrics['rmse_ret'] = np.sqrt(mean_squared_error(y_test, lstm_pred))
            lstm_metrics['mae_rp'] = mean_absolute_error(actual_close_test, pred_close)
            lstm_metrics['rmse_rp'] = np.sqrt(mean_squared_error(actual_close_test, pred_close))
            test_lstm = pred_close

        forecast_rfr, forecast_lstm = forecast_future_recursive(rfr_model, lstm_model, df, scaler, feature_cols, lookback, forecast_days)
        last_price = df['Close'].iloc[-1]

        date_col = next((c for c in df.columns if 'date' in str(c).lower() or 'tanggal' in str(c).lower()), None)
        last_date = df[date_col].iloc[-1].strftime('%d %b %Y') if date_col else "-"

        return {
            "name": config["name"], "df": df, "fund": fund_data, "fund_source": config["fund_source"], "last_price": last_price, "last_date": last_date,
            "forecast_rfr": forecast_rfr, "forecast_lstm": forecast_lstm,
            "rfr_metrics": rfr_metrics, "lstm_metrics": lstm_metrics,
            "test_dates": df[date_col].iloc[train_size + lookback:].values if date_col else None,
            "test_actual": actual_close_test, "test_rfr": test_rfr, "test_lstm": test_lstm
        }
    except Exception:
        traceback.print_exc()
        return None

def render_results(results, forecast_days):
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

    st.markdown("---")
    funds_cmp = {r['name']: r['fund'] for r in results}
    highlight = list(funds_cmp.keys())
    render_fundamental_section(funds_cmp, highlight)

    # FITUR VIEWER DATA MENTAH (HANYA MUNCUL DI MODE DATASET)
    if False:
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


# ==========================================
# TAMPILAN (PREDIKTA)
# ==========================================
TICKERS = ["ICBP", "INDF", "MYOR", "CMRY", "ULTJ"]
PASTEL = ["#c9f2e3", "#d9c9fb", "#fdeab8", "#d4f5b8", "#fbcbe6"]
LINE_COLORS = {"Aktual": "#334155", "RFR": "#2563eb", "LSTM": "#f97316"}
DATASET_MODEL = "Bandingkan Kedua Model (RFR vs LSTM)"
DATASET_LOOKBACK = 5
DATASET_HORIZON = 30

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 1rem;}
.pcard {border-radius: 16px; padding: 14px 16px; color: #1f2937; height: 100%;}
.pcard .tk {font-weight: 700; font-size: 1.05rem;}
.pcard .dt {font-size: 0.72rem; opacity: 0.7;}
.pcard .px {font-size: 1.35rem; font-weight: 700; margin: 2px 0;}
.pcard .pj {font-size: 0.78rem; margin-top: 4px;}
.pstat {background: rgba(128,128,128,0.08); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;}
.pstat .l {font-size: 0.72rem; opacity: 0.7;}
.pstat .v {font-size: 1.05rem; font-weight: 600;}
.pbrand {font-size: 1.5rem; font-weight: 800; letter-spacing: 0.5px; margin-top: -3.2rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 2px;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {width: 100%; padding: 9px 14px; border-radius: 10px; cursor: pointer;}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child:not(:has([data-testid="stMarkdownContainer"])) {display: none;}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child:not([data-testid="stMarkdownContainer"]):not(:has(p)) {display: none;}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {color: inherit; margin: 0;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background: rgba(128,128,128,0.14);}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-selected="true"], section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background: rgba(128,128,128,0.24); font-weight: 600;}
.psub {font-size: 0.75rem; opacity: 0.65; margin-bottom: 28px;}
</style>
""", unsafe_allow_html=True)


def sparkline_svg(values, w=150, h=36, color="#1f2937"):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    rng = (hi - lo) or 1.0
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 2 - (v - lo) / rng * (h - 4)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{pts}"/></svg>')


def pct_change(forecast, last_price):
    if not forecast:
        return None
    return (forecast[-1] / last_price - 1) * 100


def fmt_pct(v):
    return "-" if v is None else f"{v:+.2f}%".replace(".", ",")


def rp(v):
    return "Rp " + f"{v:,.0f}".replace(",", ".")


@st.cache_data(show_spinner=False)
def load_dataset_results():
    out = []
    for t in TICKERS:
        cfg = {"name": t, "tech_source": os.path.join("data", f"{t}.csv"),
               "fund_source": os.path.join("data fundamental", f"Fundamental_{t}.csv"), "is_custom": False}
        r = process_emiten(cfg, DATASET_MODEL, DATASET_LOOKBACK, DATASET_HORIZON)
        if r:
            out.append(r)
    return out


def render_cards(results):
    cols = st.columns(len(results))
    for i, (col, r) in enumerate(zip(cols, results)):
        spark = sparkline_svg(r["df"]["Close"].tail(60).values)
        pr, pl = pct_change(r["forecast_rfr"], r["last_price"]), pct_change(r["forecast_lstm"], r["last_price"])
        html = (f'<div class="pcard" style="background:{PASTEL[i % len(PASTEL)]}">'
                f'<div class="tk">{r["name"]}</div><div class="dt">{r["last_date"]}</div>'
                f'<div class="px">{rp(r["last_price"])}</div>{spark}'
                f'<div class="pj">+30 hari<br>RFR {fmt_pct(pr)} | LSTM {fmt_pct(pl)}</div></div>')
        col.markdown(html, unsafe_allow_html=True)


def stat_box(label, value):
    return f'<div class="pstat"><div class="l">{label}</div><div class="v">{value}</div></div>'


def plot_test_prediction(r):
    if r.get("test_dates") is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.6))
    d = pd.to_datetime(r["test_dates"])
    ax.plot(d, r["test_actual"], label="Aktual", color=LINE_COLORS["Aktual"], linewidth=1.6)
    if r["test_rfr"] is not None:
        ax.plot(d, r["test_rfr"], label="RFR", color=LINE_COLORS["RFR"], linewidth=1.1)
    if r["test_lstm"] is not None:
        ax.plot(d, r["test_lstm"], label="LSTM", color=LINE_COLORS["LSTM"], linewidth=1.1)
    ax.set_ylabel("Harga penutupan (Rp)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def metric_row(r):
    a, b = r["rfr_metrics"], r["lstm_metrics"]
    c = st.columns(4)
    c[0].metric("MAE return RFR", f"{a['mae_ret']:.5f}".replace(".", ",") if a else "-")
    c[1].metric("MAE return LSTM", f"{b['mae_ret']:.5f}".replace(".", ",") if b else "-")
    c[2].metric("MAE Rp RFR", f"{a['mae_rp']:.2f}".replace(".", ",") if a else "-")
    c[3].metric("MAE Rp LSTM", f"{b['mae_rp']:.2f}".replace(".", ",") if b else "-")


def eval_table(results):
    rows = []
    for r in results:
        for nm, m in (("RFR", r["rfr_metrics"]), ("LSTM", r["lstm_metrics"])):
            if m:
                rows.append({"Emiten": r["name"], "Model": nm, "MAE return": m["mae_ret"], "RMSE return": m["rmse_ret"],
                             "MAE (Rp)": m["mae_rp"], "RMSE (Rp)": m["rmse_rp"]})
    return pd.DataFrame(rows)


def page_dashboard(results):
    render_cards(results)
    names = [r["name"] for r in results]
    sel = st.selectbox("Emiten", names, label_visibility="collapsed")
    r = next(x for x in results if x["name"] == sel)
    df = r["df"]
    left, right = st.columns([3, 1.2])
    with left:
        with st.container(border=True):
            st.markdown(f"**{sel}: harga aktual vs prediksi pada data uji**")
            fig = plot_test_prediction(r)
            if fig is not None:
                st.pyplot(fig, clear_figure=True)
            metric_row(r)
    with right:
        with st.container(border=True):
            st.markdown("**Statistik data (Tabel 4.1)**")
            ret = df["Close"].pct_change().dropna()
            st.markdown(
                stat_box("Jumlah observasi", f"{len(df):,}".replace(",", "."))
                + stat_box("Harga Close min | maks", f"{rp(df['Close'].min())} | {rp(df['Close'].max())}")
                + stat_box("Harga Close rata-rata", rp(df["Close"].mean()))
                + stat_box("Simpangan baku return", f"{ret.std():.5f}".replace(".", ",")),
                unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Ringkasan lima emiten**")
        rows = []
        for x in results:
            fv = fund_display_values(x["fund"])
            rows.append({"Emiten": x["name"], "Harga terakhir": rp(x["last_price"]),
                         "+30H RFR": fmt_pct(pct_change(x["forecast_rfr"], x["last_price"])),
                         "+30H LSTM": fmt_pct(pct_change(x["forecast_lstm"], x["last_price"])),
                         "MAE return RFR": round(x["rfr_metrics"]["mae_ret"], 5),
                         "MAE return LSTM": round(x["lstm_metrics"]["mae_ret"], 5),
                         "PER TTM": fmt_num(fv.get("PER"))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(DISCLAIMER)


def page_proyeksi(results):
    c1, c2 = st.columns([3, 1])
    names = [r["name"] for r in results]
    pick = c1.multiselect("Emiten", names, default=names)
    days = c2.slider("Horizon (hari)", 1, DATASET_HORIZON, DATASET_HORIZON)
    chosen = [r for r in results if r["name"] in pick]
    if not chosen:
        st.info("Pilih minimal satu emiten.")
        return
    with st.container(border=True):
        st.markdown("**Perubahan harga proyeksi terhadap harga terakhir (%)**")
        fig, ax = plt.subplots(figsize=(10, 3.8))
        for i, r in enumerate(chosen):
            c = PASTEL[names.index(r["name"]) % 5]
            col = ["#0f766e", "#6d28d9", "#b45309", "#4d7c0f", "#be185d"][names.index(r["name"]) % 5]
            x = np.arange(1, days + 1)
            if r["forecast_rfr"]:
                ax.plot(x, (np.array(r["forecast_rfr"][:days]) / r["last_price"] - 1) * 100, color=col, label=f"{r['name']} RFR")
            if r["forecast_lstm"]:
                ax.plot(x, (np.array(r["forecast_lstm"][:days]) / r["last_price"] - 1) * 100, color=col, linestyle="--", label=f"{r['name']} LSTM")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xticks(range(1, days + 1))
        ax.set_xlim(0.5, days + 0.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Hari ke-")
        ax.set_ylabel("%")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with st.container(border=True):
        st.markdown(f"**Ringkasan pada hari ke-{days}**")
        rows = []
        for r in chosen:
            for nm, f in (("RFR", r["forecast_rfr"]), ("LSTM", r["forecast_lstm"])):
                if f:
                    rows.append({"Emiten": r["name"], "Model": nm, "Harga terakhir": rp(r["last_price"]),
                                 f"Harga hari ke-{days}": rp(f[days - 1]),
                                 "Perubahan": fmt_pct((f[days - 1] / r["last_price"] - 1) * 100)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown("**Angka proyeksi per hari**")
        satuan = h2.selectbox("Satuan", ["Harga (Rp)", "Perubahan (%)"], label_visibility="collapsed")
        data = {}
        for r in chosen:
            for nm, f in (("RFR", r["forecast_rfr"]), ("LSTM", r["forecast_lstm"])):
                if f:
                    v = np.array(f[:days], dtype=float)
                    data[f"{r['name']} {nm}"] = v if satuan == "Harga (Rp)" else (v / r["last_price"] - 1) * 100
        tab = pd.DataFrame(data, index=pd.Index(range(1, days + 1), name="Hari ke-")).reset_index()
        fmt = "%.0f" if satuan == "Harga (Rp)" else "%+.2f"
        st.dataframe(tab, hide_index=True, width="stretch", height=min(35 * (days + 1) + 3, 420),
                     column_config={c: st.column_config.NumberColumn(format=fmt) for c in tab.columns if c != "Hari ke-"})
    st.caption(DISCLAIMER)


def page_evaluasi(results):
    et = eval_table(results)
    l, r_ = st.columns([1.2, 1])
    with l:
        with st.container(border=True):
            st.markdown("**Metrik evaluasi pada data uji (Tabel 4.2)**")
            st.dataframe(et, hide_index=True, width="stretch",
                         column_config={"MAE return": st.column_config.NumberColumn(format="%.5f"),
                                        "RMSE return": st.column_config.NumberColumn(format="%.5f"),
                                        "MAE (Rp)": st.column_config.NumberColumn(format="%.2f"),
                                        "RMSE (Rp)": st.column_config.NumberColumn(format="%.2f")})
    with r_:
        with st.container(border=True):
            mt = {"MAE return": "mae_ret", "RMSE return": "rmse_ret", "MAE (Rp)": "mae_rp", "RMSE (Rp)": "rmse_rp"}
            pick = st.selectbox("Metrik", list(mt.keys()), label_visibility="collapsed")
            k = mt[pick]
            fig, ax = plt.subplots(figsize=(5, 3.2))
            x = np.arange(len(results))
            ax.bar(x - 0.2, [q["rfr_metrics"][k] for q in results], 0.4, label="RFR", color=LINE_COLORS["RFR"])
            ax.bar(x + 0.2, [q["lstm_metrics"][k] for q in results], 0.4, label="LSTM", color=LINE_COLORS["LSTM"])
            ax.set_xticks(x)
            ax.set_xticklabels([q["name"] for q in results])
            ax.set_title(f"{pick} per emiten", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
    sel = st.selectbox("Lihat prediksi data uji untuk", [q["name"] for q in results])
    q = next(x for x in results if x["name"] == sel)
    with st.container(border=True):
        fig = plot_test_prediction(q)
        if fig is not None:
            st.pyplot(fig, clear_figure=True)
    try:
        with st.expander("Berkas ringkasan evaluasi hasil main_revisi.py"):
            st.dataframe(pd.read_csv(os.path.join(MODEL_DIR, EVAL_FILE)), hide_index=True)
    except Exception:
        pass


def page_fundamental(results):
    names = [r["name"] for r in results]
    hl = st.multiselect("Pilihan emiten", names, default=names)
    with st.container(border=True):
        render_fundamental_section(load_all_fundamentals(), hl)


def page_deskriptif(results):
    with st.container(border=True):
        st.markdown("**Statistik deskriptif data penelitian (Tabel 4.1)**")
        rows = []
        for r in results:
            df = r["df"]
            ret = df["Close"].pct_change().dropna()
            rows.append({"Emiten": r["name"], "Observasi": len(df), "Close min (Rp)": df["Close"].min(),
                         "Close maks (Rp)": df["Close"].max(), "Close rata-rata (Rp)": df["Close"].mean(),
                         "Simpangan baku return": ret.std()})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Close min (Rp)": st.column_config.NumberColumn(format="%.0f"),
                                    "Close maks (Rp)": st.column_config.NumberColumn(format="%.0f"),
                                    "Close rata-rata (Rp)": st.column_config.NumberColumn(format="%.2f"),
                                    "Simpangan baku return": st.column_config.NumberColumn(format="%.5f")})
        st.caption(f"Periode {START_DATE} sampai {END_DATE}.")
    with st.container(border=True):
        st.markdown("**Pergerakan harga Close (indeks, awal periode = 100)**")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        cols = ["#0f766e", "#6d28d9", "#b45309", "#4d7c0f", "#be185d"]
        for i, r in enumerate(results):
            c = r["df"]["Close"].values
            ax.plot(c / c[0] * 100, color=cols[i % 5], label=r["name"], linewidth=1.2)
        ax.set_xlabel("Hari bursa ke-")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    with st.expander("Catatan data dan model"):
        st.markdown("Tujuh fitur teknikal, target return harian berikutnya, pembagian 80:20 berurutan, *sliding window* 5 hari. "
                    "*Random Forest Regressor* (100 pohon) dan LSTM (32 unit, *dropout* 0,1, 50 epoch). "
                    "PER, ROE, dan ROA dihitung TTM (empat kuartal terakhir).")
    st.caption(DISCLAIMER)


def page_unggah():
    st.markdown("Gunakan data sendiri: unggah riwayat harga dan data fundamental tiap emiten, lalu jalankan analisis.")
    tech_template = "Date,Open,High,Low,Close,Volume\n2024-01-01,1000,1050,990,1020,50000\n2024-01-02,1020,1060,1010,1050,60000\n"
    fund_template = (
        "Dalam IDR;Q2 2026;Q1 2026;Q4 2025;Q3 2025\n"
        "EPS (Quarter);100.00;90.00;110.00;95.00\n"
        "PE Ratio (Quarter);15.00;16.00;14.00;15.50\n"
        "Return on Assets (Quarter);2.00%;1.90%;2.10%;1.80%\n"
        "Return on Equity (Quarter);4.00%;3.80%;4.20%;3.60%\n"
        "Interest Coverage (Quarter);5.00;4.50;5.50;4.80\n"
    )
    d1, d2 = st.columns(2)
    d1.download_button("Unduh Template Teknikal (CSV)", data=tech_template, file_name="template_teknikal.csv", mime="text/csv")
    d2.download_button("Unduh Template Fundamental (CSV)", data=fund_template, file_name="template_fundamental.csv", mime="text/csv",
                       help="Pemisah titik koma, kolom pertama = kuartal terbaru, minimal 4 kuartal.")
    c1, c2, c3, c4 = st.columns(4)
    num = c1.number_input("Jumlah emiten", 1, 5, 1)
    model_choice = c2.selectbox("Model", [DATASET_MODEL, "Random Forest Regressor (RFR)", "Long Short Term Memory (LSTM)"])
    lookback = c3.slider("Sliding window (hari)", 3, 15, 5)
    horizon = c4.slider("Horizon (hari)", 1, 30, 30)
    configs = []
    for i in range(int(num)):
        u1, u2 = st.columns(2)
        t_file = u1.file_uploader(f"Historis CSV (Emiten {i+1})", type=["csv"], key=f"t_{i}")
        f_file = u2.file_uploader(f"Fundamental CSV (Emiten {i+1})", type=["csv"], key=f"f_{i}")
        if t_file and f_file:
            configs.append({"name": t_file.name.split('.')[0].upper(), "tech_source": t_file, "fund_source": f_file, "is_custom": True})
    if st.button("Jalankan Analisis"):
        if not configs:
            st.warning("Unggah berkas historis dan fundamental terlebih dahulu.")
        else:
            res = []
            with st.spinner("Memproses data..."):
                for cfg in configs:
                    x = process_emiten(cfg, model_choice, lookback, horizon)
                    if x:
                        res.append(x)
            if not res:
                st.error("Gagal memproses data. Pastikan format CSV sesuai.")
            else:
                render_results(res, horizon)


MENU = ["Dashboard", "Proyeksi", "Evaluasi Model", "Fundamental", "Analisis Deskriptif", "Unggah Data Sendiri"]
with st.sidebar:
    st.markdown('<div class="pbrand">Predikta</div><div class="psub">Analisis harga saham makanan olahan</div>', unsafe_allow_html=True)
    page = st.radio("Menu", MENU, label_visibility="collapsed")

if page == "Unggah Data Sendiri":
    page_unggah()
else:
    with st.spinner("Memuat hasil model..."):
        _results = load_dataset_results()
    if not _results:
        st.error("Data tidak ditemukan. Pastikan folder data, data fundamental, dan Model_revisi ada di samping app.py.")
    elif page == "Dashboard":
        page_dashboard(_results)
    elif page == "Proyeksi":
        page_proyeksi(_results)
    elif page == "Analisis Deskriptif":
        page_deskriptif(_results)
    elif page == "Evaluasi Model":
        page_evaluasi(_results)
    else:
        page_fundamental(_results)