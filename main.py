# main_optimized.py --- 临床预测模型自动化全流程脚本 ---
# 作者：[你的小红书ID]
# 适用场景：临床二分类预测、科研论文基线模型构建、医学大数据挖掘

import os  # 文件操作
import json  # JSON配置保存
import time  # 计时
import datetime  # 时间戳
import pandas as pd  # 数据处理神器
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 基础绘图
import seaborn as sns  # 使得图表更美观（论文级配色）
import sklearn  # 机器学习核心库

# --- 核心组件导入 ---
from sklearn.model_selection import train_test_split  # 拆分训练集和测试集
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 标准化与独热编码
from sklearn.impute import SimpleImputer  # 缺失值插补（临床数据必备！）
from sklearn.compose import ColumnTransformer  # 列转换器
from sklearn.pipeline import Pipeline  # 构建自动化流水线
from sklearn.linear_model import LogisticRegression  # 逻辑回归（经典基线）
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from xgboost import XGBClassifier  # XGBoost（竞赛大杀器）

# --- 评估指标导入 ---
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report, 
    confusion_matrix, accuracy_score, recall_score, precision_score
)
from sklearn.calibration import calibration_curve  # 校准曲线（高分文章必备）

import shap  # 模型解释性工具（打开黑箱）

# --- 全局配置 ---
# 开启 pandas 输出模式（Sklearn 1.2+ 新特性），彻底告别复杂的列名提取函数！
sklearn.set_config(transform_output="pandas") 

RANDOM_STATE = 42  # 随机种子，锁定结果以便复现
OUTPUT_DIR = "output"  # 结果输出文件夹
DEMO_CSV = "demo/sample_data.csv"  # 数据路径

# 设置绘图风格，更符合科研审美
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_pipeline(X):
    """
    构建预处理流水线：自动识别数值与类别特征，自动插补缺失值
    """
    # 1. 自动识别列类型
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    print(f"[数据分析] 纳入数值变量 {len(numeric_features)} 个，类别变量 {len(categorical_features)} 个。")

    # 2. 定义数值型处理：中位数插补缺失值 -> 标准化
    # 临床数据常有偏态分布，Median插补比Mean更稳健
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 3. 定义类别型处理：众数插补缺失值 -> 独热编码(OneHot)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False 配合 pandas输出
    ])

    # 4. 组合处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ],
        verbose_feature_names_out=False # 保持列名简洁
    )
    
    return preprocessor

def plot_confusion_matrix_custom(y_true, y_pred, title, path):
    """绘制美观的混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_calibration_curve_custom(y_true, y_prob, title, path):
    """绘制校准曲线（Calibration Curve），这是临床模型高分文章的标配"""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=title)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve: {title}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def run_modeling(data_path, target_col):
    """主建模函数（修复ROC绘图覆盖问题版）"""
    ensure_dir(OUTPUT_DIR)
    start_time = time.time()
    
    # ... (前序数据读取与处理代码保持不变) ...
    # 1. 读取数据
    if not os.path.exists(data_path):
        print(f"[错误] 找不到文件：{data_path}")
        return
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = preprocess_pipeline(X)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight='balanced', random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    }
    
    results = {}
    
    # --- 修改点 1：使用面向对象方法创建 ROC 专用画布 ---
    # 显式创建一个 figure 和 axes 对象，确保后续画图都锁定在这个 ax_roc 上
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8)) 
    
    print("\n[INFO] 开始训练模型流水线...")
    
    for name, model in models.items():
        print(f" -> 正在训练: {name} ...")
        
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(X_train, y_train)
        
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred)
        spec = recall_score(y_test, y_pred, pos_label=0)
        
        results[name] = {
            "AUC": auc, "Accuracy": acc, "Sensitivity": sens, "Specificity": spec
        }
        
        # --- 修改点 2：指定在 ax_roc 对象上画图 ---
        # 这样无论中间怎么 open/close 其他窗口，都不会影响这张图
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
        
        # 绘制混淆矩阵（这会创建并关闭它自己的临时画布，不会影响 ax_roc）
        plot_confusion_matrix_custom(y_test, y_pred, name, os.path.join(OUTPUT_DIR, f"cm_{name}.png"))
        
        if name == "LogisticRegression":
            plot_calibration_curve_custom(y_test, y_prob, name, os.path.join(OUTPUT_DIR, f"calib_{name}.png"))

        if name == "RandomForest":
            print(f" -> 生成 {name} 的 SHAP 解释图...")
            try:
                X_train_trans = clf.named_steps['preprocessor'].transform(X_train)
                explainer = shap.TreeExplainer(clf.named_steps['classifier'])
                shap_values = explainer.shap_values(X_train_trans)
                shap_val_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
                
                plt.figure() # SHAP 需要显式开启新画布
                shap.summary_plot(shap_val_to_plot, X_train_trans, show=False)
                plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_rf.png"), bbox_inches='tight', dpi=300)
                plt.close() # 关闭 SHAP 画布
            except Exception as e:
                print(f"[WARN] SHAP 生成失败: {e}")

    # --- 修改点 3：设置 ax_roc 的属性并保存 fig_roc ---
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2) # 对角线
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.set_title('ROC Curves Comparison')
    ax_roc.legend(loc="lower right")
    
    # 保存特定的 figure 对象
    fig_roc.savefig(os.path.join(OUTPUT_DIR, "roc_curves_all.png"), dpi=300)
    plt.close(fig_roc) # 最后关闭这个 ROC 专用画布
    
    # ... (生成 Markdown 报告代码保持不变) ...
    report_path = os.path.join(OUTPUT_DIR, "Clinical_Modeling_Report.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"# 临床预测模型实验报告\n\n")
        f.write(f"**生成时间:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. 模型性能对比 (测试集)\n\n")
        f.write("| 模型 | AUC | Accuracy | Sensitivity | Specificity |\n")
        f.write("|---|---|---|---|---|\n")
        for name, metrics in results.items():
            f.write(f"| {name} | {metrics['AUC']:.3f} | {metrics['Accuracy']:.3f} | {metrics['Sensitivity']:.3f} | {metrics['Specificity']:.3f} |\n")
        
        f.write("\n## 2. 关键图表\n\n")
        f.write("### (1) ROC 曲线\n![ROC](roc_curves_all.png)\n\n")
        f.write("### (2) SHAP 变量重要性 (Random Forest)\n![SHAP](shap_summary_rf.png)\n\n")
        f.write("### (3) 混淆矩阵示例 (Logistic Regression)\n![CM](cm_LogisticRegression.png)\n\n")
    
    print(f"\n[Success] 建模完成！耗时 {time.time()-start_time:.1f} 秒。")
    print(f"请查看文件夹 '{OUTPUT_DIR}' 获取完整报告和高清图表。")

if __name__ == "__main__":
    # 请确保目录下有 sample_data.csv，且目标列名为 outcome
    # 你可以自己造一个简单的csv来测试
    run_modeling(DEMO_CSV, target_col="outcome")