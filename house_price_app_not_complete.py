import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta


# === Hàm GM(1,1) ===
def gm11_forecast(data_series, forecast_steps):
    data = data_series.values
    n = len(data)
    x0 = data.copy()
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2
    B = np.column_stack((-z1, np.ones(n - 1)))
    Y = x0[1:]
    params = np.linalg.inv(B.T @ B) @ B.T @ Y
    a, b = params
    fitted = np.zeros(n)
    fitted[0] = x0[0]
    for i in range(1, n):
        fitted[i] = (x0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * i)
    forecast = np.zeros(forecast_steps)
    for k in range(forecast_steps):
        forecast[k] = (x0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (n + k))
    return fitted, forecast


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# === Load dữ liệu ===
try:
    df = pd.read_csv("HousePricingHCM_v2.csv", parse_dates=["Date"])
    df = df.set_index('Date').to_period('D')
except:
    messagebox.showerror("Lỗi",
                         "Không tìm thấy file HousePricingHCM_v2.csv\nHãy đặt file này cùng thư mục với chương trình!")
    exit()

districts = df.columns.tolist()

# === Tạo cửa sổ GUI ===
root = tk.Tk()
root.title("Dự báo Giá Nhà TP.HCM - Phiên bản GUI")
root.geometry("1100x800")
root.configure(bg="#f0f0f0")

title = tk.Label(root, text="DỰ BÁO GIÁ NHÀ TP.HCM", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#2c3e50")
title.pack(pady=10)

frame_top = tk.Frame(root, bg="#f0f0f0")
frame_top.pack(pady=10)

tk.Label(frame_top, text="Chọn Quận:", font=("Helvetica", 12), bg="#f0f0f0").pack(side=tk.LEFT, padx=10)
combo = ttk.Combobox(frame_top, values=districts, width=20, font=("Helvetica", 12), state="readonly")
combo.pack(side=tk.LEFT, padx=10)
combo.set("Quận 1")

btn = tk.Button(frame_top, text="XEM KẾT QUẢ", font=("Helvetica", 12, "bold"), bg="#3498db", fg="white", width=15)
btn.pack(side=tk.LEFT, padx=10)

# Khu vực biểu đồ
frame_chart = tk.Frame(root)
frame_chart.pack(pady=10, fill=tk.BOTH, expand=True)

# Khu vực kết quả văn bản
text = tk.Text(root, height=18, font=("Courier", 11), bg="white", relief="solid", borderwidth=1)
text.pack(pady=10, padx=20, fill=tk.BOTH)


def show_result():
    district = combo.get()
    if not district:
        return
    text.delete(1.0, tk.END)
    y = df[district].dropna()
    if len(y) == 0:
        messagebox.showinfo("Thông báo", "Quận này không có dữ liệu!")
        return

    text.insert(tk.END, f"ĐANG XỬ LÝ QUẬN: {district}\n")
    text.insert(tk.END, "=" * 70 + "\n\n")

    # Biểu đồ
    for widget in frame_chart.winfo_children():
        widget.destroy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y.index.to_timestamp(), y.values, label=district, color="#3498db", linewidth=2)
    ax.set_title(f"Giá Nhà {district} (2017 - 2022)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Giá (Triệu VND/m²)")
    ax.set_xlabel("Năm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, frame_chart)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Prophet data
    df_prophet = pd.DataFrame({'ds': y.index.to_timestamp(), 'y': y.values})

    # Holt-Winters
    model_hw = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=365)
    fit_hw = model_hw.fit(optimized=True)
    y_fitted_hw = fit_hw.fittedvalues.values

    # Prophet
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode='additive', changepoint_prior_scale=0.05)
    m.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    m.fit(df_prophet)
    forecast_prophet = m.predict(df_prophet)
    y_fitted_prophet = forecast_prophet['yhat'].values

    # GM(1,1)
    fitted_gm, forecast_gm = gm11_forecast(y, 365)

    # In-sample
    y_actual = y.values
    mae_hw = mean_absolute_error(y_actual, y_fitted_hw)
    rmse_hw = np.sqrt(mean_squared_error(y_actual, y_fitted_hw))
    mape_hw = mean_absolute_percentage_error(y_actual, y_fitted_hw)

    mae_prophet = mean_absolute_error(y_actual, y_fitted_prophet)
    rmse_prophet = np.sqrt(mean_squared_error(y_actual, y_fitted_prophet))
    mape_prophet = mean_absolute_percentage_error(y_actual, y_fitted_prophet)

    mae_gm = mean_absolute_error(y_actual, fitted_gm)
    rmse_gm = np.sqrt(mean_squared_error(y_actual, fitted_gm))
    mape_gm = mean_absolute_percentage_error(y_actual, fitted_gm)

    text.insert(tk.END, "ĐÁNH GIÁ HIỆU SUẤT TRÊN TOÀN BỘ DỮ LIỆU LỊCH SỬ (In-sample)\n")
    text.insert(tk.END, "=" * 60 + "\n")
    text.insert(tk.END,
                f"Holt-Winters Exponential Smoothing:\n   MAE: {mae_hw:.2f} | RMSE: {rmse_hw:.2f} | MAPE: {mape_hw:.2f}%\n\n")
    text.insert(tk.END,
                f"Facebook Prophet:\n   MAE: {mae_prophet:.2f} | RMSE: {rmse_prophet:.2f} | MAPE: {mape_prophet:.2f}%\n\n")
    text.insert(tk.END, f"Grey Model GM(1,1):\n   MAE: {mae_gm:.2f} | RMSE: {rmse_gm:.2f} | MAPE: {mape_gm:.2f}%\n\n")

    # Dự báo 10 ngày
    last_date = y.index[-1].to_timestamp()
    start_date = last_date + timedelta(days=1)
    text.insert(tk.END,
                f"DỰ BÁO GIÁ NHÀ {district} BẰNG GM(1,1) - 10 NGÀY ĐẦU TIÊN (từ {start_date.strftime('%d/%m/%Y')}):\n")
    text.insert(tk.END, "-" * 50 + "\n")
    for i in range(10):
        date_str = (start_date + timedelta(days=i)).strftime('%d/%m/%Y')
        price = round(forecast_gm[i], 2)
        text.insert(tk.END, f"{date_str}    {price} triệu/m²\n")
    text.insert(tk.END, "\n")

    # Overfitting
    train_ratio = 0.8
    train_size = int(len(y) * train_ratio)
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Holt-Winters train/test
    model_hw_train = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=365)
    fit_hw_train = model_hw_train.fit(optimized=True)
    mae_train_hw = mean_absolute_error(y_train.values, fit_hw_train.fittedvalues.values)
    forecast_test_hw = fit_hw_train.forecast(len(y_test)).values
    mae_test_hw = mean_absolute_error(y_test.values, forecast_test_hw)

    # Prophet train/test
    df_train = df_prophet.iloc[:train_size].copy()
    m_train = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                      seasonality_mode='additive', changepoint_prior_scale=0.05)
    m_train.add_seasonality(name='yearly', period=365.25, fourier_order=10)
    m_train.fit(df_train)
    forecast_train_prophet = m_train.predict(df_train)
    mae_train_prophet = mean_absolute_error(y_train.values, forecast_train_prophet['yhat'].values)
    future_test = m_train.make_future_dataframe(periods=len(y_test), freq='D')
    forecast_full = m_train.predict(future_test)
    mae_test_prophet = mean_absolute_error(y_test.values, forecast_full['yhat'].values[train_size:])

    # GM train/test
    fitted_train_gm, forecast_test_gm = gm11_forecast(y_train, len(y_test))
    mae_train_gm = mean_absolute_error(y_train.values, fitted_train_gm)
    mae_test_gm = mean_absolute_error(y_test.values, forecast_test_gm)

    text.insert(tk.END, "KIỂM TRA OVERFITTING (Train 80% đầu - Test 20% cuối)\n")
    text.insert(tk.END, "=" * 70 + "\n")
    text.insert(tk.END,
                f"Holt-Winters:  Train MAE: {mae_train_hw:.2f} | Test MAE: {mae_test_hw:.2f} | Diff: {mae_test_hw - mae_train_hw:.2f}\n")
    text.insert(tk.END,
                f"Prophet:       Train MAE: {mae_train_prophet:.2f} | Test MAE: {mae_test_prophet:.2f} | Diff: {mae_test_prophet - mae_train_prophet:.2f}\n")
    text.insert(tk.END,
                f"GM(1,1):       Train MAE: {mae_train_gm:.2f} | Test MAE: {mae_test_gm:.2f} | Diff: {mae_test_gm - mae_train_gm:.2f}\n")
    text.insert(tk.END, "\nHoàn thành! Bạn có thể chọn quận khác và nhấn nút lại.")


btn.config(command=show_result)

# Khởi động
combo.set("District 1")
show_result()  # Tự động hiển thị Quận 1 khi mở
root.mainloop()