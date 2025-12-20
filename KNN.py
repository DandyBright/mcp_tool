import json
import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report
from fastmcp import FastMCP

app = FastMCP("KNN_GridSearch_JSON_Only")

@app.tool()
def train_knn_grid_json_only(data_path: str, config_path: str, label_path: str) -> str:
    """
    KNN 分类 + GridSearch 网格搜索。
    返回 JSON 格式的训练结果和特征重要性数据。
    Args:
        data_path (str): 数据文件路径 (.csv 或 .xlsx/xls)，包含特征矩阵。
        config_path (str): YAML 配置文件路径，包含 KNN 模型参数。
        label_path (str): 标签文件路径 (.txt)，每行一个分类标签，需与数据行数对应。
        
    Returns:
        str: JSON 格式的执行结果，包含模型指标、特征重要性(基于排列重要性)和预测摘要。    
    """
    results = {
        "status": "failed",
        "message": "",
        "best_params": {},
        "metrics": {},
        "feature_importance": {}
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
            
        param_grid = config.get('param_grid', {})
        grid_settings = config.get('grid_search_params', {})
        train_params = config.get('train_params', {'test_size': 0.2, 'random_state': 42})
        
        # 4. 数据划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=train_params.get('test_size', 0.2), 
            random_state=train_params.get('random_state', 42)
        )
        
        # 5. 执行 GridSearch (安全模式)
        base_knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            estimator=base_knn,
            param_grid=param_grid,
            cv=grid_settings.get('cv', 5),
            scoring=grid_settings.get('scoring', 'accuracy'),
            n_jobs=1,       
            verbose=0       
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        
        # 6. 预测与评估
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # 7. 计算特征重要性
        perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
        importances = perm_importance.importances_mean
        
        feature_imp_dict = dict(zip(feature_names, [float(i) for i in importances]))
        feature_imp_dict = {k: max(0.0, v) for k, v in feature_imp_dict.items()}
        # 排序并取 Top 10
        sorted_features = dict(sorted(feature_imp_dict.items(), key=lambda item: item[1], reverse=True)[:10])
        
        # 8. 构造结果 (纯数据)
        results["status"] = "success"
        results["message"] = "GridSearch 完成"
        results["best_params"] = best_params
        results["metrics"] = {
            "best_cv_accuracy": round(float(best_cv_score), 4),
            "test_accuracy": round(float(accuracy), 4),
            "classification_report": report
        }
        results["feature_importance"] = sorted_features

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)
        
    return json.dumps(results, ensure_ascii=True, indent=2)

if __name__ == "__main__":
    app.run(transport="stdio")