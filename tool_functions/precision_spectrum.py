import json
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann, hamming, blackman

def compute_precision_spectrum_json_only(data_path: str, config_path: str = None) -> str:
    """
    高精度频谱分析 (FFT + Windowing + Spectrum Correction)。
    
    包含：
    1. 窗函数处理 (抑制泄漏)
    2. 幅值恢复 (Coherent Gain Compensation)
    3. 频率/幅值校正 (基于比值校正法，消除栅栏效应)
    
    Args:
        data_path (str): 数据文件路径 (.csv/.xlsx)。
        config_path (str): [可选] 配置路径。
        
    Returns:
        str: JSON 格式结果，包含校正前后的频率、幅值对比。
    """
    results = {
        "status": "failed",
        "message": "",
        "params": {},
        "raw_peak": {},
        "corrected_peak": {},
        "spectrum_preview": {} 
    }

    try:
        # -------------------------------------------------------
        # 1. 加载数据
        # -------------------------------------------------------
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
        
        # 提取信号并去直流
        signal = df_numeric.iloc[:, 0].values
        signal = signal - np.mean(signal)
        N = len(signal)

        # -------------------------------------------------------
        # 2. 加载配置
        # -------------------------------------------------------
        config = {}
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception:
                pass

        fs = config.get('sampling_rate', 10000.0)
        window_type = config.get('window_type', 'hann') # 默认汉宁窗

        # -------------------------------------------------------
        # 3. 加窗处理 (Windowing)
        # -------------------------------------------------------
        # 获取窗函数数组
        if window_type == 'hann':
            window = hann(N)
        elif window_type == 'hamming':
            window = hamming(N)
        elif window_type == 'blackman':
            window = blackman(N)
        else:
            window = np.ones(N) # 矩形窗 (无窗)

        # 计算 "恢复系数" (Coherent Gain)
        # 加窗会削弱总能量，必须在FFT后进行补偿，否则幅值会偏小
        # 对于 Hanning 窗，coherent_gain = 0.5，意味着幅值被削减了一半
        coherent_gain = np.sum(window) / N
        
        # 信号加窗
        signal_windowed = signal * window

        # -------------------------------------------------------
        # 4. FFT 计算
        # -------------------------------------------------------
        yf = rfft(signal_windowed)
        xf = rfftfreq(N, 1 / fs)
        
        # 计算幅值谱
        # 公式: |FFT| * 2 / N / 恢复系数
        amplitude = np.abs(yf) * 2 / N / coherent_gain
        
        # -------------------------------------------------------
        # 5. 寻找最大峰值 (作为校正演示)
        # -------------------------------------------------------
        # 实际场景中可能需要校正 Top N 个峰，这里演示最显著的主峰校正
        # 忽略直流分量附近 (索引 0-2)
        search_start_idx = 3 
        if N > 10:
            max_idx = np.argmax(amplitude[search_start_idx:]) + search_start_idx
        else:
            max_idx = np.argmax(amplitude)

        y_max = amplitude[max_idx]   # 原始峰值幅值
        x_max = xf[max_idx]          # 原始峰值频率
        
        # -------------------------------------------------------
        # 6. 谱校正 (Spectrum Correction) - Hanning 窗比值校正法
        # -------------------------------------------------------
        # 只有加了 Hanning 窗，比值校正公式才最准确
        # 如果峰值恰好在两个频率点之间，直接取 max_idx 会有误差
        
        corrected_freq = x_max
        corrected_amp = y_max
        correction_info = "无校正 (非 Hanning 窗或边界点)"

        if window_type == 'hann' and 1 < max_idx < len(amplitude) - 1:
            # 获取左右相邻点的幅值
            y_prev = amplitude[max_idx - 1]
            y_next = amplitude[max_idx + 1]
            
            # 判断主峰偏向哪一边
            if y_next > y_prev:
                # 峰值偏右
                v = y_next / y_max
                delta = (2 * v) / (1 + v)  # 频率偏移量 (-1, 1)
                direction = 1
            else:
                # 峰值偏左
                v = y_prev / y_max
                delta = -(2 * v) / (1 + v) # 注意负号
                direction = -1
            
            # 频率分辨率
            df_res = fs / N
            
            # 1. 校正频率
            corrected_freq = x_max + (delta * direction * df_res) # 这里简化逻辑，delta本身带符号更通用
            # 更正通用的 delta 计算 (适用于 Hanning 的代数推导):
            # 设 y0 为最大值，y1 为次大值(邻居)。
            # delta = (2*y1) / (y0 + y1)  --> 这种写法通常用于 ratio = y1/y0
            # 让我们使用标准的比值公式:
            
            # 重新严谨计算:
            if y_next > y_prev:
                alpha = 1 # 向右偏
                ratio = y_next / y_max
            else:
                alpha = -1 # 向左偏
                ratio = y_prev / y_max
            
            # Hanning 窗的校正公式
            delta = (2 * ratio) / (1 + ratio)  # delta 范围 [0, 1]
            
            final_delta = alpha * delta
            corrected_freq = xf[max_idx] + final_delta * df_res
            
            # 2. 校正幅值
            # Hanning 窗的幅值校正系数公式:
            # Amp_true = Amp_read * (1/sinc(delta)) * ... 
            # 常用近似公式或精确公式: A = A_read * (2*pi*delta * (1-delta^2)) / sin(pi*delta)
            # 注意: 当 delta 趋近 0 时，上述公式分母为0。需要处理。
            
            if abs(final_delta) < 1e-4:
                factor = 1.0
            else:
                factor = (np.pi * final_delta * (1 - final_delta**2)) / np.sin(np.pi * final_delta)
                # 这里的 factor 通常还需要乘以前面的 coherent gain 的反向操作?
                # 不，前面已经除以 coherent_gain (0.5) 归一化了。
                # 这里的 factor 是 "Picket Fence Effect" 的补偿系数。
                # 对于 Hanning，最大栅栏误差只有 15% 左右 (Factor max ≈ 1.15)
                # 修正公式引用自《机械故障诊断信号处理技术》
                
                # 另一种常用的简易校正: 
                # A_corr = (y0 + y1) * (2.356 / pi)? 不，太偏门。
                
                # 使用通用校正因子 (Hanning Correction Factor):
                # factor = 1.0 / (np.sinc(final_delta) * (1 - final_delta**2)) # numpy sinc is sin(pi*x)/(pi*x)
                # 这是最标准的。
                factor = 1.0 / (np.sinc(final_delta) * (1 - final_delta**2))

            corrected_amp = y_max * factor
            correction_info = f"Hanning 比值校正 (Offset: {round(final_delta, 4)})"

        # -------------------------------------------------------
        # 7. 构造结果
        # -------------------------------------------------------
        limit_points = 1000
        step = max(1, len(xf) // limit_points)

        results["status"] = "success"
        results["message"] = f"分析完成 ({correction_info})"
        
        results["params"] = {
            "sampling_rate": fs,
            "window_type": window_type,
            "freq_resolution": round(fs/N, 4),
            "N": N
        }
        
        results["raw_peak"] = {
            "frequency": round(float(x_max), 4),
            "amplitude": round(float(y_max), 4),
            "index": int(max_idx)
        }
        
        results["corrected_peak"] = {
            "frequency": round(float(corrected_freq), 4),
            "amplitude": round(float(corrected_amp), 4),
            "amplitude_increase_ratio": round(float(corrected_amp / y_max - 1) * 100, 2) # 提升了百分之多少
        }
        
        results["spectrum_preview"] = {
            "frequency": np.round(xf[::step], 2).tolist(),
            "amplitude": np.round(amplitude[::step], 4).tolist()
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)

    return json.dumps(results, ensure_ascii=False, indent=2)