import json
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def random_forest_model(csv_path: str,target_column: str,task_type: str,test_size: float,n_estimators: int) -> str:
    try:
        data = pd.read_csv(csv_path)
        target_col = target_column

        if target_col not in data.columns:
            return {"error": f"目标列 '{target_col}' 不存在于数据集中"}

        # 数据预处理
        for col in data.select_dtypes(include=['object']).columns:
            data[col].fillna(data[col].mode()[0], inplace=True) if data[col].dtype == 'object' else data[col].fillna(data[col].mean(), inplace=True)
            data[col] = LabelEncoder().fit_transform(data[col])

        X = data.drop(columns=[target_col])
        y = data[target_col]

        if task_type == "classification":
            y = LabelEncoder().fit_transform(y)
            model = RandomForestClassifier(random_state=42,n_estimators=n_estimators)
        elif task_type == "regression":
            model = RandomForestRegressor(random_state=42,n_estimators=n_estimators)
        else:
            return {"error": "任务类型必须是 classification 或 regression"}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = round(time.time() - start_time, 4)

        y_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, model.predict(X_train)) if task_type == "classification" else r2_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred) if task_type == "classification" else r2_score(y_test, y_pred)

        feature_importances = dict(zip(X.columns, model.feature_importances_))

        result = {
            "feature_importance_dict": feature_importances,
            "training_time": training_time,
            "n_estimators": n_estimators,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4)
        }

        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"error": str(e)}