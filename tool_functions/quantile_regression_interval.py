import json
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def quantile_regression_interval(data_path: str, config_path: str, label_path: str) -> str:
    """
    分位数回归（预测区间）算法模块。
    使用分位数回归构建预测区间，返回不同分位数的预测结果和区间评估指标。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含分位数回归参数和区间设置。
        label_path (str): 标签文件路径 (.txt)，每行一个连续值标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含分位数预测、区间评估和可视化图像。
    """
    
    # 初始化结果字典
    results = {
        "status": "failed",  # 执行状态
        "message": "",  # 执行消息
        "quantile_params": {},  # 分位数回归参数
        "prediction_intervals": {},  # 预测区间结果
        "interval_metrics": {},  # 区间评估指标
        "quantile_models": {},  # 各分位数模型信息
        "visualization": {},  # 可视化数据
        "uncertainty_analysis": {}  # 不确定性分析
    }

    try:
        # 1. 加载特征数据
        if data_path.endswith('.csv'):
            try:
                df = pd.read_csv(data_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding='gbk')
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的数据格式，请提供 .csv 或 .xlsx/.xls 文件")
            
        X = df.values  # 特征矩阵
        feature_names = df.columns.tolist()  # 特征名称
        n_samples, n_features = X.shape  # 样本数和特征数
        
        # 2. 加载标签数据（连续值）
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [float(line.strip()) for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            with open(label_path, 'r', encoding='gbk') as f:
                labels = [float(line.strip()) for line in f.readlines() if line.strip()]
        except ValueError as e:
            raise ValueError(f"标签必须是数值类型: {e}")
        
        y = np.array(labels)  # 转换为 numpy 数组
        
        # 3. 加载配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
        
        # 提取配置参数
        quantile_params = config.get('quantile_params', {})  # 分位数回归参数
        interval_params = config.get('interval_params', {})  # 预测区间参数
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})  # 训练参数
        visualization_params = config.get('visualization_params', {})  # 可视化参数
        
        # 分位数参数
        quantiles = quantile_params.get('quantiles', [0.05, 0.5, 0.95])  # 预测的分位数
        alpha = quantile_params.get('alpha', 0.05)  # 正则化参数
        solver = quantile_params.get('solver', 'highs-ds')  # 求解器
        fit_intercept = quantile_params.get('fit_intercept', True)  # 是否拟合截距项
        
        # 区间参数
        confidence_level = interval_params.get('confidence_level', 0.90)  # 置信水平
        interval_type = interval_params.get('interval_type', 'symmetric')  # 区间类型: 'symmetric' or 'asymmetric'
        
        # 4. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=train_params.get('test_size', 0.2),
            random_state=train_params.get('random_state', 42)
        )
        
        # 5. 特征标准化
        scaler_X = StandardScaler()  # 特征标准化
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()  # 目标变量标准化
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_original = y_test.copy()  # 保存原始y_test用于评估
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 6. 训练分位数回归模型（每个分位数一个模型）
        quantile_models = {}  # 存储每个分位数的模型
        quantile_predictions = {}  # 存储每个分位数的预测结果
        
        for q in quantiles:
            # 创建分位数回归模型
            qr = QuantileRegressor(
                quantile=q,
                alpha=alpha,
                solver=solver,
                fit_intercept=fit_intercept
            )
            
            # 训练模型
            qr.fit(X_train_scaled, y_train_scaled)
            
            # 在测试集上进行预测
            y_pred_scaled = qr.predict(X_test_scaled)
            
            # 将预测值转换回原始尺度
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # 保存模型和预测结果
            quantile_models[q] = qr
            quantile_predictions[q] = y_pred
        
        # 7. 构建预测区间
        # 根据配置的分位数确定区间上下界
        quantiles_sorted = sorted(quantiles)
        
        # 找到中位数分位数（作为点预测）
        median_idx = np.argmin(np.abs(np.array(quantiles_sorted) - 0.5))
        median_quantile = quantiles_sorted[median_idx]
        y_pred_median = quantile_predictions[median_quantile]
        
        # 确定区间上下界的分位数
        if interval_type == 'symmetric':
            # 对称区间：例如，90%置信区间使用0.05和0.95分位数
            lower_quantile = (1 - confidence_level) / 2
            upper_quantile = 1 - lower_quantile
            
            # 找到最接近的分位数
            lower_idx = np.argmin(np.abs(np.array(quantiles_sorted) - lower_quantile))
            upper_idx = np.argmin(np.abs(np.array(quantiles_sorted) - upper_quantile))
            
            lower_bound = quantile_predictions[quantiles_sorted[lower_idx]]
            upper_bound = quantile_predictions[quantiles_sorted[upper_idx]]
        else:
            # 非对称区间：使用配置的最小和最大分位数
            lower_bound = quantile_predictions[min(quantiles_sorted)]
            upper_bound = quantile_predictions[max(quantiles_sorted)]
        
        # 确保区间有效性（下限 ≤ 上限）
        for i in range(len(lower_bound)):
            if lower_bound[i] > upper_bound[i]:
                # 交换上下界
                lower_bound[i], upper_bound[i] = upper_bound[i], lower_bound[i]
        
        # 8. 计算区间评估指标
        # 区间覆盖率（实际值落在区间内的比例）
        in_interval_mask = (y_test_original >= lower_bound) & (y_test_original <= upper_bound)
        interval_coverage = np.mean(in_interval_mask)
        
        # 区间宽度统计
        interval_widths = upper_bound - lower_bound
        mean_width = np.mean(interval_widths)
        median_width = np.median(interval_widths)
        width_std = np.std(interval_widths)
        
        # 区间不对称性
        interval_asymmetry = np.mean(upper_bound - y_pred_median) / np.mean(y_pred_median - lower_bound)
        
        # 9. 计算分位数损失
        def quantile_loss(y_true, y_pred, q):
            """计算分位数损失"""
            errors = y_true - y_pred
            return np.mean(np.maximum(q * errors, (q - 1) * errors))
        
        quantile_losses = {}
        for q in quantiles:
            quantile_losses[f"quantile_{q}"] = quantile_loss(y_test_original, quantile_predictions[q], q)
        
        # 10. 计算点预测性能指标（使用中位数分位数）
        mse = mean_squared_error(y_test_original, y_pred_median)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差
        mae = mean_absolute_error(y_test_original, y_pred_median)  # 平均绝对误差
        r2 = r2_score(y_test_original, y_pred_median)  # R²分数
        
        # 11. 不确定性分析
        # 计算预测的标准差（使用多个分位数的预测）
        if len(quantiles) >= 2:
            # 使用所有分位数的预测计算不确定性
            all_predictions = np.array(list(quantile_predictions.values()))
            prediction_std = np.std(all_predictions, axis=0)  # 每个样本的预测标准差
            mean_prediction_std = np.mean(prediction_std)  # 平均不确定性
            
            # 计算不确定性分解
            uncertainty_components = {
                "aleatoric": mean_prediction_std**2,  # 偶然不确定性
                "epistemic": width_std**2,  # 认知不确定性（区间宽度的方差）
                "total": mean_prediction_std**2 + width_std**2  # 总不确定性
            }
        else:
            uncertainty_components = {}
            mean_prediction_std = 0
        
        # 12. 生成可视化图像
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 预测区间可视化（测试集样本排序后）
        ax1 = axes[0, 0]
        # 对测试样本按实际值排序
        sort_idx = np.argsort(y_test_original)
        y_test_sorted = y_test_original[sort_idx]
        lower_sorted = lower_bound[sort_idx]
        upper_sorted = upper_bound[sort_idx]
        median_sorted = y_pred_median[sort_idx]
        
        # 绘制区间
        x_range = np.arange(len(y_test_sorted))
        ax1.fill_between(x_range, lower_sorted, upper_sorted, alpha=0.3, color='skyblue', label=f'{confidence_level*100:.0f}% Prediction Interval')
        
        # 绘制实际值和预测值
        ax1.plot(x_range, y_test_sorted, 'bo', alpha=0.6, markersize=4, label='Actual Values')
        ax1.plot(x_range, median_sorted, 'r-', linewidth=1.5, label='Median Prediction')
        
        ax1.set_xlabel('Test Samples (Sorted by Actual Value)')
        ax1.set_ylabel('Target Value')
        ax1.set_title('Prediction Intervals on Test Set')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 分位数回归系数对比
        ax2 = axes[0, 1]
        # 提取不同分位数的系数
        quantile_coefs = []
        quantile_labels = []
        
        for q in quantiles_sorted:
            model = quantile_models[q]
            coefs = model.coef_
            quantile_coefs.append(coefs)
            quantile_labels.append(f'q={q}')
        
        # 选择前10个最重要的特征（按中位数系数的绝对值排序）
        median_coef_idx = quantiles_sorted.index(median_quantile)
        median_coefs = quantile_coefs[median_coef_idx]
        
        if n_features <= 10:
            # 如果特征数小于等于10，显示所有特征
            top_indices = range(n_features)
        else:
            # 否则选择系数绝对值最大的10个特征
            top_indices = np.argsort(np.abs(median_coefs))[-10:][::-1]
        
        # 绘制每个特征在不同分位数下的系数
        for i, idx in enumerate(top_indices):
            feature_coefs = [coefs[idx] for coefs in quantile_coefs]
            ax2.plot(quantiles_sorted, feature_coefs, '-o', label=feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}')
        
        ax2.set_xlabel('Quantile')
        ax2.set_ylabel('Coefficient Value')
        ax2.set_title('Quantile Regression Coefficients')
        ax2.legend(loc='best', fontsize='small')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 实际值 vs 预测值（带区间）
        ax3 = axes[1, 0]
        # 计算每个样本的区间宽度（用于点的大小）
        point_sizes = 20 + 50 * (interval_widths / np.max(interval_widths))
        
        scatter = ax3.scatter(y_test_original, y_pred_median, c=interval_widths, 
                             s=point_sizes, alpha=0.6, cmap='viridis', edgecolors='w', linewidth=0.5)
        
        # 添加误差线（显示区间）
        for i in range(min(30, len(y_test_original))):  # 最多显示30个误差线
            ax3.plot([y_test_original[i], y_test_original[i]], 
                    [lower_bound[i], upper_bound[i]], 'gray', alpha=0.5, linewidth=0.5)
        
        # 绘制理想预测线（y=x）
        min_val = min(y_test_original.min(), y_pred_median.min())
        max_val = max(y_test_original.max(), y_pred_median.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal Prediction')
        
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values (Median)')
        ax3.set_title(f'Actual vs Predicted with Intervals (R²={r2:.4f})')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax3, label='Interval Width')
        
        # 子图4: 区间覆盖率分析
        ax4 = axes[1, 1]
        # 计算分箱后的区间覆盖率
        n_bins = min(10, len(y_test_original) // 5)
        if n_bins >= 3:
            # 按实际值分箱
            bins = np.linspace(y_test_original.min(), y_test_original.max(), n_bins+1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            coverage_by_bin = []
            mean_width_by_bin = []
            
            for i in range(n_bins):
                bin_mask = (y_test_original >= bins[i]) & (y_test_original < bins[i+1])
                if np.sum(bin_mask) > 0:
                    bin_coverage = np.mean(in_interval_mask[bin_mask])
                    bin_width = np.mean(interval_widths[bin_mask])
                    coverage_by_bin.append(bin_coverage)
                    mean_width_by_bin.append(bin_width)
                else:
                    coverage_by_bin.append(0)
                    mean_width_by_bin.append(0)
            
            # 绘制覆盖率
            ax4.plot(bin_centers, coverage_by_bin, 'bo-', linewidth=2, markersize=8, label='Coverage Rate')
            ax4.axhline(confidence_level, color='r', linestyle='--', alpha=0.7, label=f'Target ({confidence_level*100:.0f}%)')
            
            ax4_twin = ax4.twinx()
            ax4_twin.plot(bin_centers, mean_width_by_bin, 'g^-', linewidth=2, markersize=8, label='Mean Interval Width')
            
            ax4.set_xlabel('Actual Value (Binned)')
            ax4.set_ylabel('Coverage Rate', color='blue')
            ax4_twin.set_ylabel('Interval Width', color='green')
            ax4.set_title('Coverage Rate and Interval Width by Value Range')
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 将图像转换为base64字符串
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # 13. 提取特征重要性（基于中位数模型的系数绝对值）
        median_coef_abs = np.abs(median_coefs)
        top_indices = np.argsort(median_coef_abs)[-10:][::-1]  # 取前10个最重要的特征
        top_features = {
            feature_names[i]: {
                "coefficient": float(median_coefs[i]),
                "abs_coefficient": float(median_coef_abs[i])
            } for i in top_indices
        }
        
        # 14. 构建最终结果
        results["status"] = "success"
        results["message"] = "分位数回归预测区间分析完成"
        results["quantile_params"] = {
            "quantiles": quantiles,
            "confidence_level": confidence_level,
            "interval_type": interval_type,
            "alpha": alpha,
            "solver": solver
        }
        results["prediction_intervals"] = {
            "lower_bound": lower_bound.tolist(),
            "upper_bound": upper_bound.tolist(),
            "median_prediction": y_pred_median.tolist(),
            "test_actual": y_test_original.tolist()
        }
        results["interval_metrics"] = {
            "coverage_rate": round(float(interval_coverage), 4),
            "interval_width": {
                "mean": round(float(mean_width), 4),
                "median": round(float(median_width), 4),
                "std": round(float(width_std), 4),
                "min": round(float(np.min(interval_widths)), 4),
                "max": round(float(np.max(interval_widths)), 4)
            },
            "asymmetry_ratio": round(float(interval_asymmetry), 4),
            "quantile_losses": quantile_losses
        }
        results["performance_metrics"] = {
            "point_prediction": {
                "mse": round(float(mse), 6),
                "rmse": round(float(rmse), 6),
                "mae": round(float(mae), 6),
                "r2_score": round(float(r2), 4)
            }
        }
        results["uncertainty_analysis"] = {
            "prediction_std_mean": round(float(mean_prediction_std), 4),
            "uncertainty_components": uncertainty_components,
            "uncertainty_correlation": round(float(np.corrcoef(y_test_original, interval_widths)[0, 1]), 4)
        }
        results["top_features"] = top_features
        results["visualization"] = {
            "plot_image": f"data:image/png;base64,{img_base64}",  # base64编码的图像
            "plot_size": [1400, 1000]  # 图像尺寸（像素）
        }
        results["data_info"] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "y_statistics": {
                "mean": round(float(np.mean(y)), 4),
                "std": round(float(np.std(y)), 4),
                "min": round(float(np.min(y)), 4),
                "max": round(float(np.max(y)), 4)
            }
        }

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=True, indent=2)