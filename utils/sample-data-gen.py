# generate_demo_data.py
# 运行： python generate_demo_data.py
import numpy as np
import pandas as pd

np.random.seed(424)
n = 6000

# 基本人口学
age = np.random.randint(18, 90, size=n)
sex = np.random.choice(['0','1'], size=n, p=[0.52,0.48])
bmi = np.round(np.random.normal(loc=24, scale=3.5, size=n), 1)

# 生命体征
heart_rate = np.clip(np.random.normal(78, 10, size=n).astype(int), 50, 140)
systolic_bp = np.clip(np.random.normal(130, 15, size=n).astype(int), 80, 220)
diastolic_bp = np.clip(np.random.normal(78, 10, size=n).astype(int), 40, 140)

# 实验室
serum_calcium = np.round(np.random.normal(2.2, 0.15, size=n), 2)  # mmol/L
wbc = np.round(np.random.normal(7.5, 3.0, size=n), 1)  # 10^9/L
wbc = np.clip(wbc, 1.0, 30.0)

# 合并症（示例：银屑病）
psoriasis = np.random.binomial(1, 0.09, size=n)  # ~9% 有银屑病

# 结局（示例：脓毒症），用简单规则模拟：高WBC或低钙或随机因素增加风险
risk_score = (wbc > 10).astype(int) + (serum_calcium < 2.1).astype(int) + np.random.binomial(1, 0.05, size=n)
outcome = (risk_score >= 1).astype(int)  # 若任一风险因子存在则标记为1（示例规则）

# 构建 DataFrame
df = pd.DataFrame({
    'id': np.arange(1, n+1),
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'heart_rate': heart_rate,
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'serum_calcium_mmol_per_L': serum_calcium,
    'wbc_10e9_per_L': wbc,
    'psoriasis': psoriasis,
    'outcome': outcome
})

# 保存 CSV
out_path = "sample_data.csv"
df.to_csv(out_path, index=False)
print(f"Saved demo CSV to {out_path} (n={n})")