import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import shap

RANDOM_STATE = 42

def preprocess_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features = X.select_dtypes(include=['int64','float64']).columns  # 数据按照数字和其他进行分列
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[ # 数字量进行张量化？
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return X, y, preprocessor

def train_and_evaluate(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    }

    results = {}
    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_pred_proba)
        results[name] = auc

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig("output/roc_curves.png")
    plt.close()

    print("\n\nModel Performance (AUC):")
    for k,v in results.items():
        print(f"{k}: {v:.3f}")

    # Example SHAP for Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    # rf.fit(X_train, y_train)
    # explainer = shap.TreeExplainer(rf)
    # shap_values = explainer.shap_values(X_train)
    # shap.summary_plot(shap_values[1], X_train, show=False)
    # plt.savefig("shap_summary.png")
    # plt.close()

    # 假设你有：X_train, y_train, preprocessor (ColumnTransformer), rf (RandomForestClassifier)
    # 或者你是用 Pipeline clf = Pipeline([('preprocessor', preprocessor), ('classifier', rf)])

    # 1. 如果你单独使用 preprocessor + model（推荐）
    X_train_trans = preprocessor.fit_transform(X_train)   # 训练时用 fit_transform，评估/解释时用 transform
    # 如果你在外面已经 fit 过 preprocessor，请用 transform 而不是 fit_transform

    # 确保模型是用变换后的数据训练的
    rf.fit(X_train_trans, y_train)

    # 2. 获取变换后特征名（兼容 OneHotEncoder）
    feature_names = []
    try:
        # ColumnTransformer 中每个 transformer 可能是 ('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)
        # 下面尝试自动拼接各部分的列名（sklearn >=1.0 推荐用 get_feature_names_out）
        for name, trans, cols in preprocessor.transformers_:
            if name == 'remainder' and trans == 'drop':
                continue
            if hasattr(trans, 'named_steps') and 'encoder' in trans.named_steps:
                # 假设 categorical transformer 最后一步是 OneHotEncoder 且命名为 'encoder'
                enc = trans.named_steps['encoder']
                names = list(enc.get_feature_names_out(cols))
                feature_names.extend(names)
            elif hasattr(trans, 'get_feature_names_out'):
                # numeric scaler 等，直接返回原列
                try:
                    names = list(trans.get_feature_names_out(cols))
                except Exception:
                    names = list(cols)
                feature_names.extend(names)
            else:
                # fallback: 原列名
                feature_names.extend(list(cols))
    except Exception:
        # 最保守的做法：如果上面失败，尝试从变换后数组的列数生成占位名
        n_cols = X_train_trans.shape[1]
        feature_names = [f"f{i}" for i in range(n_cols)]

    # 3. 用 TreeExplainer 解释变换后的数据
    import shap
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_train_trans)

    # 4. 兼容 shap_values 的不同返回格式
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # 二分类常见：shap_values[1] 对正类
        sv = shap_values[1]
    else:
        sv = shap_values

    # 5. 将变换后的数据包装成 DataFrame（便于 summary_plot 使用 feature_names）
    import pandas as pd
    X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names)

    # 6. 绘图
    shap.summary_plot(sv, X_train_trans_df, show=False)
    plt.savefig("output/shap_summary.png")
    plt.close()

    return results

def main():
    print("Hello from auto-modeling!")


if __name__ == "__main__":
    # Demo with sample data
    df = pd.read_csv("demo/sample_data.csv")                    # 数据导入
    X, y, preprocessor = preprocess_data(df, target="outcome")  # 预处理数据
    results = train_and_evaluate(X, y, preprocessor)            # 训练和评估
