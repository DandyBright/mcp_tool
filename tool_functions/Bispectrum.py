import json
import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

def compute_diagonal_bispectrum_json_only(data_path: str, config_path: str = None) -> str:
    """
    1.5维谱 (双谱对角切片) 计算模块。
    用于检测信号中的非线性相位耦合 (Quadratic Phase Coupling, QPC)。
    常用于早期裂纹检测、严重的机械松动诊断。
    
    Args:
        data_path (str): 数据文件路径 (.csv/.xlsx)。
        config_path (str): [可选] 配置路径。
        
    Returns:
        str: JSON 格式结果，包含双谱对角线的频率和幅值。
    """
    results = {
        "status": "failed",
        "message": "",
        "params": {},
        "bispectrum_data": {},
        "top_nonlinear_features": []
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
        
        # 提取信号并去直流 (DC Offset 对高阶谱影响很大，必须去除)
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
        
        # -------------------------------------------------------
        # 3. 计算 FFT
        # -------------------------------------------------------
        # 使用 rfft 获取正半轴频谱
        yf = rfft(signal)
        xf = rfftfreq(N, 1 / fs)
        
        # -------------------------------------------------------
        # 4. 计算双谱对角切片 (Diagonal Bispectrum)
        # -------------------------------------------------------
        # 定义: B(f) = X(f) * X(f) * conj(X(2f))
        # 这表示频率 f 的二次谐波是否与 2f 发生相位耦合
        
        # 这里的索引处理需要非常小心：
        # 如果 yf 的长度是 M，那么索引 k 代表频率 f_k
        # 我们需要索引 2k 代表频率 2*f_k
        # 因此，k 的范围只能遍历到 M/2，否则 2k 会越界
        
        limit_idx = len(yf) // 2
        
        # 矢量化计算 (比 for 循环快 100 倍)
        # component 1: X(f) -> yf[:limit_idx]
        # component 2: X(f) -> yf[:limit_idx]
        # component 3: X*(2f) -> np.conj(yf[0 : 2*limit_idx : 2]) 
        # 注意: 切片 step=2 即可取到 0, 2, 4, ... 对应 2f
        
        X_f = yf[:limit_idx]
        X_2f_conj = np.conj(yf[0 : 2*limit_idx : 2])
        
        # 修正长度匹配 (切片可能因奇偶数差1位)
        min_len = min(len(X_f), len(X_2f_conj))
        X_f = X_f[:min_len]
        X_2f_conj = X_2f_conj[:min_len]
        freq_axis = xf[:min_len]
        
        # 计算双谱 (复数结果)
        B_diag_complex = X_f * X_f * X_2f_conj
        
        # 取模 (Magnitude) 得到 1.5维幅值谱
        # 归一化处理：通常除以 N^3 或其他因子，这里为了展示趋势直接取模
        B_diag_amp = np.abs(B_diag_complex)

        # -------------------------------------------------------
        # 5. 提取特征 (Top QPC Peaks)
        # -------------------------------------------------------
        # 找出非线性耦合最强的频率点
        top_k = 5
        sorted_indices = np.argsort(B_diag_amp)[::-1][:top_k]
        
        nonlinear_features = []
        for idx in sorted_indices:
            # 过滤掉 0Hz 附近的干扰
            if freq