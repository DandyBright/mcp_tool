import os
import json
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
import shap  
import platform
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 设置后端与字体
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def train_predict_explain_xgboost(data_path: str, config_path: str, label_path: str) -> str:
    """
    XGBoost 分类 + SHAP 解释工具。
    除了常规指标，还会生成 SHAP 摘要图来解释模型决策。
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 XGBoost 模型参数。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含模型指标、特征重要性和预测摘要。    
    """
    results = {
        "status": "failed",
        "message": "",
        "metrics": {},
        "output_files": []
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
            
        X = df.values
        feature_names = df.columns.tolist()
        
        # 2. 加载标签
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            with open(label_path, 'r', encoding='gbk') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
                
        le = LabelEncoder()
        y = le.fit_transform(labels)
        
        # 3. 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                config = yaml.safe_load(f)
            
        model_params = config.get('model_params', {})
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})
        
        # 4. 训练
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_params['test_size'], random_state=train_params['random_state']
        )
        
        # 自动识别二分类/多分类
        if len(le.classes_) > 2:
            model_params['objective'] = 'multi:softmax'
            model_params['num_class'] = len(le.classes_)
        else:
            model_params['objective'] = 'binary:logistic'

        model = xgb.XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # 5. 预测与评估
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # --- 路径准备 ---
        dir_name = os.path.dirname(data_path)
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        
        # ==========================================
        # 6. SHAP 解释计算 (核心新增部分)
        # ==========================================
        # TreeExplainer 专门用于树模型 (XGBoost/LightGBM/RandomForest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # 如果是二分类，shap_values 是一个数组
        # 如果是多分类，shap_values 是一个列表（每个类别一个数组），我们通常取类别 0 或需要特定处理
        # 这里为了通用性，如果是多分类，我们只画第一个类别的解释，或者聚合展示
        
        plt.figure(figsize=(10, 8))
        
        # 处理多分类 SHAP 值的格式兼容性
        shap_data_to_plot = shap_values
        if isinstance(shap_values, list):
            # 多分类：默认展示第1个类别的特征影响，或者使用 summary_plot 的多类模式
            # shap.summary_plot 支持列表输入
            pass 
            
        # 绘制 SHAP 蜂群图 (Summary Plot)
        # show=False 允许我们后续保存
        shap.summary_plot(shap_data_to_plot, X_test, feature_names=feature_names, show=False)
        
        shap_img_path = os.path.join(dir_name, f"{base_name}_shap_XGBoost.png")
        plt.savefig(shap_img_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        # 尝试自动打开 SHAP 图
        try:
            if platform.system() == "Windows": os.startfile(shap_img_path)
            elif platform.system() == "Darwin": subprocess.call(["open", shap_img_path])
            else: subprocess.call(["xdg-open", shap_img_path])
        except: pass
        
        results["output_files"].append(shap_img_path)

        # 7. 构造返回结果
        results["status"] = "success"
        results["message"] = f"训练完成。SHAP 解释图已保存至: {shap_img_path}"
        results["metrics"] = {"accuracy": round(float(accuracy), 4)}
        
    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=False, indent=2)
