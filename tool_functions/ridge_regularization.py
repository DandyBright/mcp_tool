import json
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from io import BytesIO
import base64


def ridge_regularization_path(data_path: str, config_path: str, label_path: str) -> str:
    """
    岭回归正则化路径可视化模块。
    计算岭回归在不同正则化强度下的系数路径，返回路径数据和可视化图像。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含岭回归参数和可视化设置。
        label_path (str): 标签文件路径 (.txt)，每行一个连续值标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含正则化路径数据、最佳模型和可视化图像。
    """
    
    # 初始化结果字典
    results = {
        "status": "failed",  # 执行状态
        "message": "",  # 执行消息
        "ridge_params": {},  # 岭回归参数
        "path_data": {},  # 正则化路径数据
        "best_model": {},  # 最佳模型信息
        "performance_metrics": {},  # 模型性能指标
        "visualization": {},  # 可视化数据
        "coefficient_statistics": {}  # 系数统计信息
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
        ridge_params = config.get('ridge_params', {})  # 岭回归参数
        path_params = config.get('path_params', {})  # 正则化路径参数
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})  # 训练参数
        visualization_params = config.get('visualization_params', {})  # 可视化参数
        
        # 正则化路径参数
        alpha_min = path_params.get('alpha_min', 1e-10)  # 最小正则化参数（对数刻度）
        alpha_max = path_params.get('alpha_max', 1e10)  # 最大正则化参数（对数刻度）
        n_alphas = path_params.get('n_alphas', 100)  # 正则化参数数量
        cv_folds = path_params.get('cv_folds', 5)  # 交叉验证折数
        scoring = path_params.get('scoring', 'neg_mean_squared_error')  # 评分指标
        
        # 岭回归参数
        fit_intercept = ridge_params.get('fit_intercept', True)  # 是否拟合截距项
        normalize = ridge_params.get('normalize', False)  # 是否标准化
        solver = ridge_params.get('solver', 'auto')  # 求解器
        
        # 4. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=train_params.get('test_size', 0.2),
            random_state=train_params.get('random_state', 42)
        )
        
        # 5. 特征标准化（对岭回归非常重要）
        scaler_X = StandardScaler()  # 特征标准化
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        scaler_y = StandardScaler()  # 目标变量标准化
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_original = y_test.copy()  # 保存原始y_test用于评估
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 6. 生成正则化参数网格（对数均匀分布）
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alphas)  # 对数均匀分布
        
        # 7. 计算正则化路径（系数随alpha变化的路径）
        coefs_path = []  # 存储每个alpha对应的系数
        intercepts_path = []  # 存储每个alpha对应的截距
        cv_scores = []  # 存储每个alpha的交叉验证分数
        
        for alpha in alphas:
            # 创建岭回归模型
            ridge = Ridge(
                alpha=alpha,
                fit_intercept=fit_intercept,
                normalize=normalize,
                solver=solver,
                random_state=42
            )
            
            # 拟合模型
            ridge.fit(X_train_scaled, y_train_scaled)
            
            # 保存系数和截距
            coefs_path.append(ridge.coef_.copy())
            if fit_intercept:
                intercepts_path.append(ridge.intercept_)
            else:
                intercepts_path.append(0)
            
            # 计算交叉验证分数
            cv_score = cross_val_score(
                ridge, 
                X_train_scaled, 
                y_train_scaled,
                scoring=scoring,
                cv=cv_folds,
                n_jobs=-1
            )
            cv_scores.append(np.mean(cv_score))
        
        # 转换为numpy数组
        coefs_path = np.array(coefs_path)  # 形状: (n_alphas, n_features)
        intercepts_path = np.array(intercepts_path)  # 形状: (n_alphas,)
        cv_scores = np.array(cv_scores)  # 形状: (n_alphas,)
        
        # 8. 寻找最佳alpha（交叉验证分数最高）
        best_idx = np.argmax(cv_scores)  # 负MSE时分数越高越好
        best_alpha = alphas[best_idx]
        best_cv_score = cv_scores[best_idx]
        
        # 使用最佳alpha训练最终模型
        best_ridge = Ridge(
            alpha=best_alpha,
            fit_intercept=fit_intercept,
            normalize=normalize,
            solver=solver,
            random_state=42
        )
        best_ridge.fit(X_train_scaled, y_train_scaled)
        
        # 预测
        y_pred_scaled = best_ridge.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 9. 计算模型性能指标
        mse = mean_squared_error(y_test_original, y_pred)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差
        mae = mean_absolute_error(y_test_original, y_pred)  # 平均绝对误差
        r2 = r2_score(y_test_original, y_pred)  # R²分数
        
        # 10. 计算系数统计信息
        final_coefs = best_ridge.coef_
        coef_stats = {
            "nonzero_count": int(np.sum(np.abs(final_coefs) > 1e-10)),  # 非零系数数量
            "max_abs_coef": float(np.max(np.abs(final_coefs))),  # 最大绝对值系数
            "min_abs_coef": float(np.min(np.abs(final_coefs[final_coefs != 0])) 
                                 if np.any(final_coefs != 0) else 0),  # 最小非零系数绝对值
            "mean_abs_coef": float(np.mean(np.abs(final_coefs))),  # 系数绝对值均值
            "std_coef": float(np.std(final_coefs))  # 系数标准差
        }
        
        # 获取最重要的特征（按系数绝对值排序）
        coef_abs = np.abs(final_coefs)
        top_indices = np.argsort(coef_abs)[-10:][::-1]  # 取前10个最重要的特征
        top_features = {
            feature_names[i]: {
                "coefficient": float(final_coefs[i]),
                "abs_coefficient": float(coef_abs[i])
            } for i in top_indices
        }
        
        # 11. 生成可视化图像
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 系数路径（所有特征）
        ax1 = axes[0, 0]
        for i in range(min(20, n_features)):  # 最多显示20个特征
            ax1.plot(alphas, coefs_path[:, i], label=feature_names[i] if i < len(feature_names) else f"Feature_{i}")
        ax1.axvline(best_alpha, color='r', linestyle='--', alpha=0.7, label=f'Best α={best_alpha:.2e}')
        ax1.set_xscale('log')  # x轴使用对数刻度
        ax1.set_xlabel('Regularization Strength (α)')
        ax1.set_ylabel('Coefficient Value')
        ax1.set_title('Ridge Coefficient Paths')
        ax1.legend(loc='best', fontsize='small')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 交叉验证分数路径
        ax2 = axes[0, 1]
        ax2.plot(alphas, cv_scores, 'b-', linewidth=2)
        ax2.axvline(best_alpha, color='r', linestyle='--', alpha=0.7, label=f'Best α={best_alpha:.2e}')
        ax2.axhline(best_cv_score, color='g', linestyle=':', alpha=0.7, label=f'Best CV Score={best_cv_score:.4f}')
        ax2.set_xscale('log')
        ax2.set_xlabel('Regularization Strength (α)')
        ax2.set_ylabel('Cross-Validation Score')
        ax2.set_title('CV Score vs Regularization Strength')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 最终模型系数条形图（前20个最重要的特征）
        ax3 = axes[1, 0]
        if len(top_features) > 0:
            sorted_features = sorted(top_features.items(), key=lambda x: x[1]["abs_coefficient"], reverse=True)
            feature_labels = [item[0] for item in sorted_features]
            coef_values = [item[1]["coefficient"] for item in sorted_features]
            
            y_pos = np.arange(len(feature_labels))
            colors = ['red' if val < 0 else 'blue' for val in coef_values]
            ax3.barh(y_pos, coef_values, color=colors)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(feature_labels, fontsize=9)
            ax3.set_xlabel('Coefficient Value')
            ax3.set_title('Top Feature Coefficients (Final Model)')
            ax3.axvline(0, color='black', linewidth=0.5)
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 子图4: 预测结果散点图
        ax4 = axes[1, 1]
        ax4.scatter(y_test_original, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
        
        # 绘制理想预测线（y=x）
        min_val = min(y_test_original.min(), y_pred.min())
        max_val = max(y_test_original.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Ideal Prediction')
        
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title(f'Actual vs Predicted (R²={r2:.4f})')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 将图像转换为base64字符串
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # 12. 提取详细路径数据（为了减少数据量，选择部分alpha点）
        # 选择大约20个点用于前端显示
        display_indices = np.linspace(0, len(alphas)-1, min(20, len(alphas))).astype(int)
        display_alphas = alphas[display_indices].tolist()
        display_coefs = coefs_path[display_indices, :].tolist()
        display_cv_scores = cv_scores[display_indices].tolist()
        
        # 为每个特征创建路径数据
        feature_paths = {}
        for i, feature_name in enumerate(feature_names):
            feature_paths[feature_name] = {
                "alphas": display_alphas,
                "coefficients": [float(coef) for coef in coefs_path[:, i][display_indices]]
            }
        
        # 13. 构建最终结果
        results["status"] = "success"
        results["message"] = "岭回归正则化路径分析完成"
        results["ridge_params"] = {
            "best_alpha": float(best_alpha),
            "alpha_range": [float(alpha_min), float(alpha_max)],
            "n_alphas": n_alphas,
            "fit_intercept": fit_intercept,
            "solver": solver
        }
        results["path_data"] = {
            "alphas": display_alphas,  # 用于显示的alpha值
            "all_alphas": alphas.tolist(),  # 所有alpha值（用于高级分析）
            "cv_scores": display_cv_scores,  # 交叉验证分数
            "feature_paths": feature_paths,  # 每个特征的系数路径
            "intercepts": intercepts_path.tolist()  # 截距路径
        }
        results["best_model"] = {
            "alpha": float(best_alpha),
            "intercept": float(best_ridge.intercept_ if fit_intercept else 0),
            "coefficients": {
                feature_names[i]: float(final_coefs[i]) for i in range(len(feature_names))
            }
        }
        results["performance_metrics"] = {
            "test_set": {
                "mse": round(float(mse), 6),
                "rmse": round(float(rmse), 6),
                "mae": round(float(mae), 6),
                "r2_score": round(float(r2), 4)
            },
            "cross_validation": {
                "best_score": round(float(best_cv_score), 6),
                "mean_cv_score": round(float(np.mean(cv_scores)), 6)
            }
        }
        results["coefficient_statistics"] = coef_stats
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