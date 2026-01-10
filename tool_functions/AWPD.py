import json
import pandas as pd
import numpy as np
import os
import pywt
import matplotlib.pyplot as plt

def denoise_awpd_json_only(data_path: str, config_path: str = None) -> str:
    """
    AWPD (Adaptive Wavelet Packet Denoising) 自适应小波包去噪。
    
    原理：
    利用小波包变换将信号分解到精细频带，根据噪声水平自适应计算阈值，
    对小波系数进行软/硬阈值处理，最后重构信号。
    
    Args:
        data_path (str): 数据文件路径 (.csv/.xlsx)。
        config_path (str): [可选] 配置路径。
        
    Returns:
        str: JSON 格式结果，包含去噪指标 (SNR, RMSE) 和可视化路径。
    """
    results = {
        "status": "failed",
        "message": "",
        "params": {},
        "metrics": {},
        "data_preview": {},
        "visualization": {"image_path": None}
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
        
        # 提取信号
        raw_signal = df_numeric.iloc[:, 0].values.astype(float)
        
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

        # 小波基函数：'db8', 'sym5', 'coif3' 等
        wavelet_name = config.get('wavelet', 'db8') 
        # 分解层数：通常 3-5 层
        level = config.get('level', 3)
        # 阈值模式：'soft' (软阈值, 更平滑) 或 'hard' (硬阈值, 保留特征)
        mode = config.get('threshold_mode', 'soft')
        # 可视化
        enable_plot = config.get('enable_plot', False)
        plot_save_path = config.get('plot_save_path', './awpd_result.png')

        # -------------------------------------------------------
        # 3. 小波包分解 (Decomposition)
        # -------------------------------------------------------
        # 创建小波包对象
        wp = pywt.WaveletPacket(data=raw_signal, wavelet=wavelet_name, mode='symmetric', maxlevel=level)
        
        # 获取最底层的节点 (叶子节点)
        # 这些节点代表了信号在不同频段的精细分量
        nodes = [node.path for node in wp.get_level(level, 'natural')]

        # -------------------------------------------------------
        # 4. 自适应阈值计算与去噪 (Thresholding)
        # -------------------------------------------------------
        # 策略：使用全局通用阈值 (Universal Threshold) 或基于噪声估计的阈值
        # 这里实现一种鲁棒的噪声估计方法：MAD (Median Absolute Deviation)
        
        # 1. 估计噪声标准差 sigma
        # 通常利用最高频层的细节系数来估计噪声 (因为高频主要由噪声主导)
        # 获取第一层的高频系数 (Details)
        d1_coeffs = pywt.wavedec(raw_signal, wavelet_name, level=1)[-1]
        sigma = np.median(np.abs(d1_coeffs)) / 0.6745
        
        # 2. 计算通用阈值 (VisuShrink)
        # threshold = sigma * sqrt(2 * log(N))
        threshold = sigma * np.sqrt(2 * np.log(len(raw_signal)))
        
        # 3. 对每个节点的系数应用阈值
        for node_path in nodes:
            # 获取节点系数
            coeffs = wp[node_path].data
            
            # 应用 pywt 自带的阈值函数
            # substitute=0 表示低于阈值的置为0
            new_coeffs = pywt.threshold(coeffs, threshold, mode=mode, substitute=0)
            
            # 更新小波包节点数据
            wp[node_path].data = new_coeffs

        # -------------------------------------------------------
        # 5. 信号重构 (Reconstruction)
        # -------------------------------------------------------
        denoised_signal = wp.reconstruct(update=True)
        
        # 修正重构后的长度 (可能会因卷积导致边缘多了几个点)
        if len(denoised_signal) > len(raw_signal):
            denoised_signal = denoised_signal[:len(raw_signal)]
            
        # -------------------------------------------------------
        # 6. 计算指标 (Metrics)
        # -------------------------------------------------------
        # 计算残差 (Residual) = 也就是被去除的 "噪声"
        noise_residual = raw_signal - denoised_signal
        
        # RMSE
        rmse = np.sqrt(np.mean(noise_residual**2))
        
        # SNR (信噪比) - 估计值
        # 假设 denoise_signal 是纯净信号 P_signal，noise_residual 是 P_noise
        p_signal = np.sum(denoised_signal**2)
        p_noise = np.sum(noise_residual**2)
        if p_noise == 0:
            snr = 100 # 避免除零
        else:
            snr = 10 * np.log10(p_signal / p_noise)

        # -------------------------------------------------------
        # 7. 可视化
        # -------------------------------------------------------
        image_abs_path = None
        if enable_plot:
            plt.figure(figsize=(12, 6))
            
            plot_len = min(len(raw_signal), 2000)
            t = np.arange(plot_len)
            
            plt.plot(t, raw_signal[:plot_len], color='lightgray', label='Noisy Signal', alpha=0.8)
            plt.plot(t, denoised_signal[:plot_len], color='blue', label=f'AWPD ({mode}, L={level})', linewidth=1)
            
            plt.title(f"Wavelet Packet Denoising (Thresh={round(threshold,3)})")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            os.makedirs(os.path.dirname(os.path.abspath(plot_save_path)), exist_ok=True)
            plt.savefig(plot_save_path, dpi=100)
            plt.close()
            image_abs_path = os.path.abspath(plot_save_path)

        # -------------------------------------------------------
        # 8. 构造结果
        # -------------------------------------------------------
        limit_points = 1000
        step = max(1, len(raw_signal) // limit_points)
        
        results["status"] = "success"
        results["message"] = f"AWPD 去噪完成 (Threshold={round(threshold, 4)})"
        
        results["params"] = {
            "wavelet": wavelet_name,
            "level": level,
            "threshold_mode": mode,
            "calculated_threshold": round(float(threshold), 4)
        }
        
        results["metrics"] = {
            "rmse": round(float(rmse), 4),
            "estimated_snr": round(float(snr), 2)
        }
        
        results["data_preview"] = {
            "raw": np.round(raw_signal[::step], 4).tolist(),
            "denoised": np.round(denoised_signal[::step], 4).tolist()
        }
        
        results["visualization"] = {
            "generated": enable_plot,
            "image_path": image_abs_path
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=False, indent=2)