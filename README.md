# Iridium Clinical ML v0.1

## 简介
这是一个用于 **二分类临床预测模型建模** 的 Python 脚本。  
目标是帮助科研人员快速完成数据预处理、模型训练、性能评估与解释。

## 功能
- 数据预处理（缺失值填充、标准化、类别编码）
- 模型训练与对比（Logistic Regression, Random Forest, XGBoost）
- 性能评估（ROC, AUC, 混淆矩阵）
- 可视化（ROC曲线、特征重要性、SHAP解释）
- 自动报告（保存图表与性能结果）

## 安装
```bash
git clone https://github.com/nekorectifier/iridium-clinical-ml.git
cd iridium-clinical-ml
pip install -r requirements.txt
```

# 使用示例

```bash
python main.py
```

# 输出

1. roc_curves.png：模型 ROC 曲线

2. shap_summary.png：SHAP 全局解释图

3. 控制台打印模型性能对比

# 下一步计划

- 支持生存分析（Cox 回归）

- 增加校准曲线与 DCA

- 自动生成 Markdown/PDF 报告