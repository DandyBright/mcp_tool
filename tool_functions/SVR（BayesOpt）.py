import json
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import warnings
warnings.filterwarnings('ignore')


def svr_bayesopt_tuning(data_path: str, config_path: str, label_path: str) -> str:
    """
    SVR 超参数贝叶斯优化调优模块。
    使用贝叶斯优化自动搜索 SVR 最佳超参数，返回优化结果和模型性能。
    
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含贝叶斯优化和 SVR 参数设置。
        label_path (str): 标签文件路径 (.txt)，每行一个连续值标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含优化过程、最佳参数、模型性能等。
    """
    
    # 初始化结果字典
    results = {
        "status": "failed",  # 执行状态
        "message": "",  # 执行消息
        "bayesopt_params": {},  # 贝叶斯优化参数
        "best_params": {},  # 最佳超参数
        "optimization_history": [],  # 优化历史记录
        "model_performance": {},  # 模型性能指标
        "feature_importance": {},  # 特征重要性（基于SVR系数）
        "cross_validation": {}  # 交叉验证结果
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
        bayesopt_params = config.get('bayesopt_params', {})  # 贝叶斯优化参数
        svr_params = config.get('svr_params', {})  # SVR 模型参数
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})  # 训练参数
        feature_scaling = config.get('feature_scaling', True)  # 是否进行特征缩放
        
        # 贝叶斯优化参数
        n_iter = bayesopt_params.get('n_iter', 50)  # 优化迭代次数
        init_points = bayesopt_params.get('init_points', 10)  # 初始探索点数量
        kappa = bayesopt_params.get('kappa', 2.576)  # 探索-利用平衡参数（UCB公式中的κ）
        random_state = bayesopt_params.get('random_state', 42)  # 随机种子
        
        # SVR 超参数范围
        param_bounds = svr_params.get('param_bounds', {
            'C': (0.1, 100),  # 正则化参数
            'gamma': (0.001, 1),  # RBF核的gamma参数
            'epsilon': (0.01, 0.5)  # epsilon-insensitive损失函数的epsilon
        })
        
        # 4. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=train_params.get('test_size', 0.2),
            random_state=train_params.get('random_state', 42)
        )
        
        # 5. 特征缩放（对SVR非常重要）
        if feature_scaling:
            scaler_X = StandardScaler()  # 特征标准化
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            
            scaler_y = StandardScaler()  # 目标变量标准化
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_original = y_test.copy()  # 保存原始y_test用于评估
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
            y_train_scaled = y_train
            y_test_scaled = y_test
            scaler_y = None
        
        # 6. 定义贝叶斯优化的目标函数
        def svr_cv_score(C, gamma, epsilon):
            """
            SVR交叉验证得分函数，作为贝叶斯优化的目标函数
            
            Args:
                C: 正则化参数
                gamma: RBF核参数
                epsilon: epsilon-insensitive损失函数参数
                
            Returns:
                float: 负均方误差（优化器会最大化这个值，即最小化MSE）
            """
            # 创建SVR模型
            model = SVR(
                C=float(C),
                gamma=float(gamma),
                epsilon=float(epsilon),
                kernel='rbf',  # 使用RBF核
                shrinking=True,  # 启用收缩启发式
                cache_size=500,  # 缓存大小（MB）
                max_iter=-1  # 无限制迭代
            )
            
            # 计算交叉验证得分（负均方误差）
            cv_scores = cross_val_score(
                model, 
                X_train_scaled, 
                y_train_scaled,
                scoring='neg_mean_squared_error',  # 负MSE
                cv=5,  # 5折交叉验证
                n_jobs=1  # 单进程运行
            )
            
            return np.mean(cv_scores)  # 返回平均得分
        
        # 7. 创建贝叶斯优化器
        optimizer = BayesianOptimization(
            f=svr_cv_score,  # 目标函数
            pbounds=param_bounds,  # 参数边界
            random_state=random_state,  # 随机种子
            verbose=0  # 控制输出详细程度
        )
        
        # 8. 执行贝叶斯优化
        optimizer.maximize(
            init_points=init_points,  # 初始探索点
            n_iter=n_iter,  # 迭代次数
            kappa=kappa  # 探索-利用平衡参数
        )
        
        # 9. 获取最佳参数并训练最终模型
        best_params = optimizer.max['params']  # 最佳参数组合
        
        # 创建最佳SVR模型
        best_svr = SVR(
            C=float(best_params['C']),
            gamma=float(best_params['gamma']),
            epsilon=float(best_params['epsilon']),
            kernel='rbf'
        )
        
        # 在整个训练集上训练模型
        best_svr.fit(X_train_scaled, y_train_scaled)
        
        # 10. 模型预测
        y_pred_scaled = best_svr.predict(X_test_scaled)  # 预测（缩放后）
        
        # 如果进行了特征缩放，需要将预测值转换回原始尺度
        if feature_scaling:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_for_eval = y_test_original
        else:
            y_pred = y_pred_scaled
            y_test_for_eval = y_test_scaled
        
        # 11. 计算模型性能指标
        mse = mean_squared_error(y_test_for_eval, y_pred)  # 均方误差
        rmse = np.sqrt(mse)  # 均方根误差
        mae = mean_absolute_error(y_test_for_eval, y_pred)  # 平均绝对误差
        r2 = r2_score(y_test_for_eval, y_pred)  # R²分数
        evs = explained_variance_score(y_test_for_eval, y_pred)  # 解释方差分数
        
        # 计算预测误差统计
        errors = y_test_for_eval - y_pred
        error_stats = {
            "mean_error": float(np.mean(errors)),  # 平均误差
            "std_error": float(np.std(errors)),  # 误差标准差
            "max_error": float(np.max(np.abs(errors))),  # 最大绝对误差
            "median_abs_error": float(np.median(np.abs(errors)))  # 中位数绝对误差
        }
        
        # 12. 提取特征重要性（基于SVR支持向量的权重）
        # 对于RBF核，可以通过计算每个特征的平均权重贡献来估计重要性
        n_features = X_train_scaled.shape[1]
        if best_svr.kernel == 'rbf' and best_svr.support_vectors_.shape[0] > 0:
            # 使用支持向量和对应的对偶系数计算特征重要性
            support_vectors = best_svr.support_vectors_
            dual_coef = best_svr.dual_coef_
            
            # 计算每个特征的重要性（基于支持向量的加权平均）
            feature_importance = np.zeros(n_features)
            for i in range(support_vectors.shape[0]):
                feature_importance += np.abs(support_vectors[i] * dual_coef[0, i])
            
            # 归一化到[0, 1]范围
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
            
            # 创建特征重要性字典
            feature_imp_dict = dict(zip(feature_names, feature_importance.tolist()))
            # 排序并取Top 10
            sorted_features = dict(sorted(feature_imp_dict.items(), 
                                         key=lambda item: item[1], 
                                         reverse=True)[:10])
        else:
            sorted_features = {}
        
        # 13. 收集优化历史记录
        optimization_history = []
        for i, res in enumerate(optimizer.res):
            record = {
                "iteration": i + 1,
                "target": float(res['target']),
                "params": {k: float(v) for k, v in res['params'].items()}
            }
            optimization_history.append(record)
        
        # 14. 交叉验证性能评估
        cv_scores = cross_val_score(
            best_svr,
            X_train_scaled,
            y_train_scaled,
            scoring='neg_mean_squared_error',
            cv=5
        )
        
        # 15. 构建最终结果
        results["status"] = "success"
        results["message"] = "SVR 贝叶斯优化调优完成"
        results["bayesopt_params"] = {
            "n_iter": n_iter,
            "init_points": init_points,
            "kappa": kappa,
            "random_state": random_state,
            "total_evaluations": len(optimizer.res)
        }
        results["best_params"] = {k: float(v) for k, v in best_params.items()}
        results["optimization_history"] = optimization_history
        results["model_performance"] = {
            "test_metrics": {
                "mse": round(float(mse), 6),
                "rmse": round(float(rmse), 6),
                "mae": round(float(mae), 6),
                "r2_score": round(float(r2), 4),
                "explained_variance_score": round(float(evs), 4)
            },
            "error_statistics": error_stats,
            "best_target_score": round(float(optimizer.max['target']), 6)
        }
        results["feature_importance"] = sorted_features
        results["cross_validation"] = {
            "cv_scores_mean": round(float(np.mean(cv_scores)), 6),
            "cv_scores_std": round(float(np.std(cv_scores)), 6),
            "cv_scores_all": [float(score) for score in cv_scores]
        }
        results["data_info"] = {
            "n_samples": len(y),
            "n_features": n_features,
            "feature_scaling": feature_scaling,
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