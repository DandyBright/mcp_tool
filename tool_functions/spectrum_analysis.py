import json
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

def compute_peak_spectrum_json_only(data_path: str, config_path: str = None) -> str:
    """
    频谱分析与峰值提取 (FFT + Peak Detection + Visualization)。
    计算频谱，提取峰值，并可选生成频谱图文件。
    
    Args:
        data_path (str): 数据文件路径 (.csv/.xlsx)。
        config_path (str): [可选] JSON 配置文件路径。
        
    Returns:
        str: JSON 格式结果，包含数据和图片路径。
    """
    results = {
        "status": "failed",
        "message": "",
        "params": {},
        "spectrum_data": {},
        "detected_peaks": [],
        "visualization": {"image_path": None}
    }

    try:
        # 1. 加载数据
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='gbk')
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的数据格式")

        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty:
            raise ValueError("未找到数值型数据列")
        
        signal = df_numeric.iloc[:, 0].values
        signal = signal - np.mean(signal) # 去直流

        # 2. 加载配置
        config = {}
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception:
                pass

        fs = config.get('sampling_rate', 10000.0)
        top_k = config.get('top_k_peaks', 10)
        min_distance = config.get('min_peak_distance', 5)
        
        # 可视化配置
        enable_plot = config.get('enable_plot', False) # 默认不画图，需显式开启
        plot_save_path = config.get('plot_save_path', './spectrum_plot.png')

        # 3. FFT 计算
        N = len(signal)
        yf = rfft(signal)
        xf = rfftfreq(N, 1 / fs)
        amplitude = np.abs(yf) * 2 / N
        
        # 4. 峰值检测
        threshold_height = np.max(amplitude) * 0.05
        peaks, properties = find_peaks(
            amplitude, 
            height=threshold_height, 
            distance=min_distance
        )
        
        peak_freqs = xf[peaks]
        peak_amps = amplitude[peaks]
        
        # 排序取 Top K
        sorted_indices = np.argsort(peak_amps)[::-1]
        top_indices = sorted_indices[:top_k]
        final_peak_freqs = peak_freqs[top_indices]
        final_peak_amps = peak_amps[top_indices]

        # 5. 可视化绘制 (Matplotlib) 
        image_abs_path = None
        if enable_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(xf, amplitude, label='Spectrum', color='blue', linewidth=1)
            
            # 标记峰值
            plt.plot(final_peak_freqs, final_peak_amps, "x", color='red', label='Peaks')
            
            plt.title("Frequency Domain Spectrum")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(plot_save_path)), exist_ok=True)
            plt.savefig(plot_save_path, dpi=100)
            plt.close() # 关闭画布，释放内存
            
            image_abs_path = os.path.abspath(plot_save_path)

        # 6. 构造结果
        limit_points = 2000
        step = max(1, len(xf) // limit_points)
        
        results["status"] = "success"
        results["message"] = "分析完成" + ("，图片已生成" if image_abs_path else "")
        
        results["params"] = {"sampling_rate": fs, "signal_length": N}
        
        results["spectrum_data"] = {
            "frequency": np.round(xf[::step], 2).tolist(),
            "amplitude": np.round(amplitude[::step], 4).tolist()
        }
        
        detected_peaks_list = []
        for f, a in zip(final_peak_freqs, final_peak_amps):
            detected_peaks_list.append({"frequency_hz": round(float(f), 2), "amplitude": round(float(a), 4)})
        
        detected_peaks_list.sort(key=lambda x: x["frequency_hz"])
        results["detected_peaks"] = detected_peaks_list
        
        results["visualization"] = {
            "generated": enable_plot,
            "image_path": image_abs_path
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=False, indent=2)