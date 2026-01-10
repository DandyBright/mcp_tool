import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

def denoise_tvd_json_only(data_path: str, config_path: str = None) -> str:
    """
    TVD (Total Variation Denoising) 总变差去噪模块。
    
    优势: 在去除高斯白噪声的同时，保留信号的边缘和突变特征 (Edge-Preserving)。
    场景: 适用于轴承故障冲击信号、方波信号、或任何包含阶跃变化的传感器数据。
    
    Args:
        data_path (str): 数据文件路径 (.csv/.xlsx)。
        config_path (str): [可选] 配置路径，包含正则化权重 weight。
        
    Returns:
        str: JSON 格式结果，包含去噪后的信号预览、信噪比改善指标及可视化路径。
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
        
        # 提取原始信号
        raw_signal = df_numeric.iloc[:, 0].values.astype(float)
        # 暂时去除均值以方便处理，最后可以加回来，或者TVD本身对均值不敏感
        # 但为了保持信号原始物理意义，这里不做强制去直流，除非为了可视化

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

        # 关键参数: weight (正则化参数 lambda)
        # weight 越小 -> 保留越多的原信号细节 (去噪弱)
        # weight 越大 -> 信号越平滑，阶梯感越强 (去噪强)
        # 经验值通常在 0.1 到 0.5 之间 (取决于信号幅值归一化情况)
        weight = config.get('weight', 0.1)
        
        # 迭代次数 (Chambolle 算法是迭代求解的)
        n_iter = config.get('n_iter_max', 100)
        
        # 可视化配置
        enable_plot = config.get('enable_plot', False)
        plot_save_path = config.get('plot_save_path', './tvd_result.png')

        # -------------------------------------------------------
        # 3. 执行 TVD 算法 (Core)
        # -------------------------------------------------------
        # 使用 skimage 的 chambolle 算法求解最小化问题:
        # min ||u - f||^2 + weight * TV(u)
        
        denoised_signal = denoise_tv_chambolle(
            raw_signal, 
            weight=weight, 
            max_num_iter=n_iter
        )

        # -------------------------------------------------------
        # 4. 计算去噪指标
        # -------------------------------------------------------
        # 1. RMSE (均方根误差): 衡量去噪信号偏离原信号的程度 (即去除了多少"噪音")
        noise_removed = raw_signal - denoised_signal
        rmse = np.sqrt(np.mean(noise_removed**2))
        
        # 2. TVReduction (总变差缩减率): 衡量平滑程度
        # Total Variation = sum(|x[i+1] - x[i]|)
        tv_raw = np.sum(np.abs(np.diff(raw_signal)))
        tv_denoised = np.sum(np.abs(np.diff(denoised_signal)))
        tv_reduction_ratio = (tv_raw - tv_denoised) / (tv_raw + 1e-8)

        # -------------------------------------------------------
        # 5. 可视化 (可选)
        # -------------------------------------------------------
        image_abs_path = None
        if enable_plot:
            plt.figure(figsize=(12, 6))
            
            # 为了看清细节，通常只画一部分，或者画全图
            # 这里画前 1000 个点或全图
            plot_len = min(len(raw_signal), 2000)
            t_axis = np.arange(plot_len)
            
            plt.plot(t_axis, raw_signal[:plot_len], color='lightgray', label='Raw (Noisy)', alpha=0.7)
            plt.plot(t_axis, denoised_signal[:plot_len], color='red', label=f'TVD (w={weight})', linewidth=1.5)
            
            plt.title("Total Variation Denoising Result")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            os.makedirs(os.path.dirname(os.path.abspath(plot_save_path)), exist_ok=True)
            plt.savefig(plot_save_path, dpi=100)
            plt.close()
            
            image_abs_path = os.path.abspath(plot_save_path)

        # -------------------------------------------------------
        # 6. 构造结果
        # -------------------------------------------------------
        # 降采样预览数据
        limit_points = 1000
        step = max(1, len(raw_signal) // limit_points)
        
        results["status"] = "success"
        results["message"] = f"TVD 去噪完成 (Weight={weight})"
        
        results["params"] = {
            "weight": weight,
            "n_iter_max": n_iter,
            "signal_length": len(raw_signal)
        }
        
        results["metrics"] = {
            "rmse_diff": round(float(rmse), 4),
            "tv_reduction": f"{round(tv_reduction_ratio * 100, 2)}%"
        }
        
        results["data_preview"] = {
            "raw_sample": np.round(raw_signal[::step], 4).tolist(),
            "denoised_sample": np.round(denoised_signal[::step], 4).tolist(),
            "step": step
        }
        
        results["visualization"] = {
            "generated": enable_plot,
            "image_path": image_abs_path
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)

    return json.dumps(results, ensure_ascii=False, indent=2)