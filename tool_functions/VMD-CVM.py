import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from vmdpy import VMD  # 核心依赖

def train_vmd_cvm_json_only(data_path: str, config_path: str = None) -> str:
    """
    VMD-CVM: 变分模态分解 + K值自动寻优。
    
    通过迭代不同的模态数 K (从 min_k 到 max_k)，计算“中心频率差异”或“重构误差”，
    自动选择最佳的 K 值，并返回该 K 值下的分解结果 (IMFs)。
    
    Args:
        data_path (str): 数据文件路径 (.csv/.xlsx)，单列时间序列数据。
        config_path (str): [可选] 配置文件路径，包含 VMD 参数范围。
        
    Returns:
        str: JSON 格式结果，包含最佳 K 值、各模态中心频率及可视化路径。
    """
    results = {
        "status": "failed",
        "message": "",
        "best_params": {},
        "optimization_history": [],
        "decomposition_result": {},
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
        
        # 提取信号 (假设第一列)
        signal = df_numeric.iloc[:, 0].values.astype(float)
        # VMD 对边界效应敏感，通常不需要去均值，但在分解前做归一化是好习惯
        # 这里保持原始幅值以便物理分析，但在 VMD 内部计算时需注意
        
        N = len(signal)

        # -------------------------------------------------------
        # 2. 加载配置 & 参数初始化
        # -------------------------------------------------------
        config = {}
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except Exception:
                pass

        # VMD 固定参数 (通常不需要变动)
        alpha = config.get('alpha', 2000)       # 带宽限制 (越大约平滑)
        tau = 0                                 # 噪声容忍度 (0 表示无噪声容忍，强去噪)
        DC = 0                                  # 是否含有直流分量 (0:无)
        init = 1                                # 初始化方式 (1:均匀分布)
        tol = 1e-7                              # 收敛准则
        
        # 寻优范围
        min_k = config.get('min_k', 2)
        max_k = config.get('max_k', 8)          # 通常轴承故障分解到 8 层足够
        
        # 可视化配置
        enable_plot = config.get('enable_plot', False)
        plot_save_path = config.get('plot_save_path', './vmd_result.png')

        # -------------------------------------------------------
        # 3. K 值寻优循环 (Cross-Validation / Optimization Loop)
        # -------------------------------------------------------
        # 策略：观察中心频率 (Center Frequencies)。
        # 如果 K 设定过大，VMD 会强制分裂出一个极窄的频带，
        # 导致两个模态的中心频率非常接近 (Mode Mixing / Over-decomposition)。
        # 判据：当两个模态的中心频率距离小于阈值时，认为 K 过大。
        
        best_k = min_k
        best_u = None
        best_omega = None
        history = []
        
        # 频率距离阈值 (例如：归一化频率距离小于 0.05 则认为重复)
        min_freq_dist_threshold = 0.02 
        
        stop_searching = False
        
        for k in range(min_k, max_k + 1):
            if stop_searching:
                break
                
            # 执行 VMD
            # u: 分解出的 IMF 矩阵 [K, N]
            # u_hat: 频谱
            # omega: 中心频率演变 [iterations, K]
            u, u_hat, omega = VMD(signal, alpha, tau, k, DC, init, tol)
            
            # 获取最后一次迭代的中心频率
            final_centers = sorted(omega[-1])
            
            # 计算相邻中心频率的最小距离
            min_dist = 1.0 # 初始化最大值
            if k > 1:
                diffs = np.diff(final_centers)
                min_dist = np.min(diffs)
            
            # 记录历史
            history.append({
                "k": k,
                "min_center_freq_distance": round(float(min_dist), 5),
                "center_freqs": [round(float(f), 4) for f in final_centers]
            })
            
            # 判定逻辑
            if k > min_k and min_dist < min_freq_dist_threshold:
                # 发现过分解现象，回退到上一个 K 为最佳
                best_k = k - 1
                stop_searching = True
                # 注意：此时 best_u 还是上这一轮的(k)，我们需要的是 k-1 的结果
                # 但为了代码简单，我们通常记录上一轮的 best
            else:
                # 如果没有过分解，暂定当前 K 为最佳，并保存结果
                best_k = k
                best_u = u
                best_omega = final_centers

        # -------------------------------------------------------
        # 4. 整理最佳结果
        # -------------------------------------------------------
        # 如果循环跑完都没触发阈值，就用 max_k
        # 此时 best_u 已经是 max_k 的结果了
        
        # 计算每个 IMF 的能量占比 (Energy Ratio)
        # 能量 = sum(x^2)
        imf_energies = np.sum(best_u**2, axis=1)
        total_energy = np.sum(imf_energies)
        energy_ratios = imf_energies / total_energy
        
        # -------------------------------------------------------
        # 5. 可视化 (绘制最佳 K 的 IMFs)
        # -------------------------------------------------------
        image_abs_path = None
        if enable_plot and best_u is not None:
            # 动态调整画布高度
            fig_h = max(6, best_k * 1.5)
            plt.figure(figsize=(10, fig_h))
            
            # 绘制原始信号
            plt.subplot(best_k + 1, 1, 1)
            plt.plot(signal, color='black', linewidth=0.8)
            plt.title(f"Original Signal (Best K={best_k})")
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # 绘制各个 IMF
            for i in range(best_k):
                plt.subplot(best_k + 1, 1, i + 2)
                plt.plot(best_u[i], color='blue', linewidth=0.8)
                plt.ylabel(f"IMF {i+1}")
                plt.grid(True, linestyle='--', alpha=0.5)
                # 在右侧标注能量占比
                plt.text(0.01, 0.9, f"Energy: {energy_ratios[i]*100:.1f}%", 
                         transform=plt.gca().transAxes, fontsize=9, backgroundcolor='white')

            plt.tight_layout()
            
            os.makedirs(os.path.dirname(os.path.abspath(plot_save_path)), exist_ok=True)
            plt.savefig(plot_save_path, dpi=100)
            plt.close()
            image_abs_path = os.path.abspath(plot_save_path)

        # -------------------------------------------------------
        # 6. 构造返回 JSON
        # -------------------------------------------------------
        # IMF 数据量大，进行降采样
        limit_points = 1000
        step = max(1, N // limit_points)
        
        imf_preview = {}
        for i in range(best_k):
            imf_preview[f"IMF_{i+1}"] = np.round(best_u[i][::step], 4).tolist()

        results["status"] = "success"
        results["message"] = f"VMD 优化完成，最佳模态数 K={best_k}"
        
        results["best_params"] = {
            "best_k": int(best_k),
            "alpha": alpha,
            "tau": tau
        }
        
        results["optimization_history"] = history
        
        results["decomposition_result"] = {
            "imf_count": int(best_k),
            "center_frequencies_normalized": [round(float(f), 4) for f in best_omega],
            "energy_ratios": [round(float(e), 4) for e in energy_ratios],
            "imf_data_preview": imf_preview,
            "note": f"数据已降采样 (step={step})"
        }
        
        results["visualization"] = {
            "generated": enable_plot,
            "image_path": image_abs_path
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        import traceback
        # results["debug"] = traceback.format_exc()

    return json.dumps(results, ensure_ascii=False, indent=2)