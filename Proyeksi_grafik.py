import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import main_revisi as m

MODEL_DIR = "Model_revisi"
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
DIR_30 = os.path.join(PLOTS_DIR, "proyeksi_30_hari")
DIR_UJI = os.path.join(PLOTS_DIR, "perbandingan_data_uji")
DIR_LATIH_UJI = os.path.join(PLOTS_DIR, "perbandingan_data_latih_dan_uji")
EMITENS = ["ICBP", "INDF", "MYOR", "CMRY", "ULTJ"]
LOOKBACK = 5
TRAIN_RATIO = 0.8

def _style(title, xlabel):
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel("Harga Penutupan (IDR)", fontsize=10)
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=30)

def plot_uji(ticker, dates, actual, p_rfr, p_lstm):
    os.makedirs(DIR_UJI, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Harga Aktual", color="blue", linewidth=1.5)
    plt.plot(dates, p_rfr, label="Prediksi RFR", color="green", linestyle="--", linewidth=1.5)
    plt.plot(dates, p_lstm, label="Prediksi LSTM", color="red", linestyle=":", linewidth=1.8)
    _style(f"Grafik Prediksi Harga Saham Harian - {ticker}", "Tanggal (Data Uji)")
    path = os.path.join(DIR_UJI, f"{ticker}_data_uji.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def plot_latih_uji(ticker, d_tr, a_tr, r_tr, l_tr, d_te, a_te, r_te, l_te):
    os.makedirs(DIR_LATIH_UJI, exist_ok=True)
    plt.figure(figsize=(14, 6))
    dates = np.concatenate([d_tr, d_te])
    plt.plot(dates, np.concatenate([a_tr, a_te]), label="Harga Aktual", color="blue", linewidth=1.3)
    plt.plot(dates, np.concatenate([r_tr, r_te]), label="Prediksi RFR", color="green", linestyle="--", linewidth=1.1)
    plt.plot(dates, np.concatenate([l_tr, l_te]), label="Prediksi LSTM", color="red", linestyle=":", linewidth=1.5)
    plt.axvline(d_te[0], color="black", linestyle="-.", linewidth=1.2, label="Batas data latih | data uji")
    _style(f"Perbandingan Prediksi Model dengan Harga Aktual (Data Latih dan Data Uji) - {ticker}", "Tanggal")
    path = os.path.join(DIR_LATIH_UJI, f"{ticker}_data_latih_dan_uji.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path

def main():
    for ticker in EMITENS:
        df = m.load_and_preprocess_csv(os.path.join("data", f"{ticker}.csv"))
        (X_train, y_train, X_test, y_test, prev_close_test,
         actual_close_test, scaler, feature_cols) = m.prepare_dataset(df, lookback=LOOKBACK, train_ratio=TRAIN_RATIO)

        rfr = joblib.load(os.path.join(MODEL_DIR, "saved_models", f"rfr_model_{ticker}.joblib"))
        lstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "saved_models", f"lstm_model_{ticker}.keras"))

        close = df["Close"].values
        dates = df["Date"].values
        n_tr = int(len(df) * TRAIN_RATIO)

        prev_tr = close[LOOKBACK - 1:n_tr - 1]
        act_tr = close[LOOKBACK:n_tr]
        d_tr = dates[LOOKBACK:n_tr]
        r_tr = prev_tr * (1 + rfr.predict(X_train.reshape(X_train.shape[0], -1)))
        l_tr = prev_tr * (1 + lstm.predict(X_train, verbose=0).flatten())

        d_te = dates[n_tr + LOOKBACK:]
        r_te = prev_close_test * (1 + rfr.predict(X_test.reshape(X_test.shape[0], -1)))
        l_te = prev_close_test * (1 + lstm.predict(X_test, verbose=0).flatten())

        p1 = plot_uji(ticker, d_te, actual_close_test, r_te, l_te)
        p2 = plot_latih_uji(ticker, d_tr, act_tr, r_tr, l_tr, d_te, actual_close_test, r_te, l_te)

        f_rfr, f_lstm = m.forecast_future_recursive(rfr, lstm, df, scaler, feature_cols,
                                                    lookback=LOOKBACK, forecast_steps=30)
        m.plot_and_save_future_forecast(ticker, close[-100:], f_rfr, f_lstm, output_dir=DIR_30)

        mae_r = np.mean(np.abs(actual_close_test - r_te))
        mae_l = np.mean(np.abs(actual_close_test - l_te))
        print(f"{ticker}: uji={len(d_te)} titik, latih={len(d_tr)} titik | MAE Rp RFR {mae_r:.2f}, LSTM {mae_l:.2f} "
              f"| +30 hari RFR {f_rfr[-1]:.2f}, LSTM {f_lstm[-1]:.2f}")
    print("\nSelesai. Cek folder:", PLOTS_DIR)

if __name__ == "__main__":
    main()