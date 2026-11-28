import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. データの読み込み
df = pd.read_csv('/home/ubuntu/upload/data.csv')

# 外部データ（政府統計等）の定義
external_stats = {
    'industry_attrition_rate': 10.2,  # 厚労省：雇用動向調査（情報通信業）
    'industry_avg_monthly_income': 38.5, # 厚労省：賃金構造基本統計調査（情報通信業）
    'industry_avg_overtime': 15.0, # 厚労省：毎月勤労統計調査（情報通信業）
    'industry_remote_rate': 52.0, # 総務省：通信利用動向調査
    'industry_avg_holidays': 118.0 # 厚労省：就労条件総合調査（情報通信業）
}

# ディレクトリ作成
os.makedirs('/home/ubuntu/project/plots_v2', exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

# 2. S社の現状数値算出
s_attrition_rate = df['Attrition'].value_counts(normalize=True)['Yes'] * 100
s_avg_income = df['MonthlyIncome'].mean() / 1000 # 単位：千円 -> 万円に近似（実際はドルベースだが、比較のためスケール調整）
# OverTimeは数値（残業時間）のようなので、0より大きい場合を残業ありとする
s_overtime_rate = (df['OverTime'] > 0).mean() * 100

# 3. 10の観点に基づく詳細分析と図表生成

# [図1] 離職率の市場比較
plt.figure(figsize=(10, 6))
labels = ['IT業界平均', 'S社']
values = [external_stats['industry_attrition_rate'], s_attrition_rate]
sns.barplot(x=labels, y=values, palette=['#CED4DA', '#197278'])
plt.title('図1: 離職率の市場比較（%）')
plt.ylabel('離職率 (%)')
for i, v in enumerate(values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')
plt.savefig('/home/ubuntu/project/plots_v2/fig1_market_attrition.png')
plt.close()

# [図2] 給与水準の市場比較
plt.figure(figsize=(10, 6))
# MonthlyIncomeを万円に換算（仮に10ドル=1000円、100ドル=1万円とする）
s_income_man = df['MonthlyIncome'].mean() / 100
labels = ['IT業界平均', 'S社']
values = [external_stats['industry_avg_monthly_income'], s_income_man]
sns.barplot(x=labels, y=values, palette=['#CED4DA', '#197278'])
plt.title('図2: 平均月収の市場比較（万円）')
plt.ylabel('月収 (万円)')
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.1f}万円", ha='center')
plt.savefig('/home/ubuntu/project/plots_v2/fig2_market_income.png')
plt.close()

# [図3] 残業の有無と離職率の詳細
plt.figure(figsize=(10, 6))
# OverTimeをバイナリ化して分析
df['HasOverTime'] = df['OverTime'].apply(lambda x: 'Yes' if x > 0 else 'No')
ot_attrition = df.groupby('HasOverTime')['Attrition'].value_counts(normalize=True).unstack() * 100
ot_attrition['Yes'].plot(kind='bar', color='#197278')
plt.title('図3: 残業の有無別離職率')
plt.ylabel('離職率 (%)')
plt.xticks(rotation=0)
plt.savefig('/home/ubuntu/project/plots_v2/fig3_overtime_detail.png')
plt.close()

# [図4] 職種別離職率と生産性
plt.figure(figsize=(12, 6))
job_stats = df.groupby('JobRole').agg({'Attrition': lambda x: (x=='Yes').mean()*100, 'PerformanceRating': 'mean'}).reset_index()
sns.scatterplot(data=job_stats, x='Attrition', y='PerformanceRating', hue='JobRole', s=200)
plt.title('図4: 職種別の離職率と生産性（PerformanceRating）')
plt.xlabel('離職率 (%)')
plt.ylabel('平均生産性スコア')
plt.savefig('/home/ubuntu/project/plots_v2/fig4_jobrole_matrix.png')
plt.close()

# [図5] 年齢層別の離職理由推測（月収と満足度）
plt.figure(figsize=(10, 6))
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=['20代', '30代', '40代', '50代以上'])
age_satisfaction = df.groupby('AgeGroup')['EnvironmentSatisfaction'].mean()
age_satisfaction.plot(kind='line', marker='o', color='#197278')
plt.title('図5: 年齢層別の職場環境満足度')
plt.ylabel('平均満足度')
plt.savefig('/home/ubuntu/project/plots_v2/fig5_age_satisfaction.png')
plt.close()

# [図6] 勤続年数と昇進の停滞
plt.figure(figsize=(10, 6))
sns.boxplot(x='Attrition', y='YearsSinceLastPromotion', data=df, palette=['#197278', '#CED4DA'])
plt.title('図6: 離職・在職別の前回昇進からの経過年数')
plt.savefig('/home/ubuntu/project/plots_v2/fig6_promotion_gap.png')
plt.close()

# [図7] ワークライフバランスと生産性
plt.figure(figsize=(10, 6))
sns.pointplot(x='WorkLifeBalance', y='PerformanceRating', data=df, color='#197278')
plt.title('図7: ワークライフバランス評価と生産性の関係')
plt.savefig('/home/ubuntu/project/plots_v2/fig7_wlb_performance.png')
plt.close()

# [図8] 採用チャネル別のROI（定着率×生産性）
df['StayRate'] = df['Attrition'].apply(lambda x: 1 if x=='No' else 0)
channel_roi = df.groupby('HowToEmploy').agg({'StayRate': 'mean', 'PerformanceRating': 'mean'})
channel_roi['ROI_Score'] = channel_roi['StayRate'] * channel_roi['PerformanceRating']
channel_roi['ROI_Score'].sort_values().plot(kind='barh', color='#197278')
plt.title('図8: 採用チャネル別のROIスコア（定着率×生産性）')
plt.savefig('/home/ubuntu/project/plots_v2/fig8_channel_roi.png')
plt.close()

# [図9] ストレスレベルと離職の相関
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='StressRating', hue='Attrition', fill=True, palette=['#197278', '#CED4DA'])
plt.title('図9: ストレスレベルの分布（離職別）')
plt.savefig('/home/ubuntu/project/plots_v2/fig9_stress_density.png')
plt.close()

# [図10] 特徴量重要度（機械学習による要因特定）
le = LabelEncoder()
df_ml = df.copy()
for col in df_ml.select_dtypes(include=['object']).columns:
    df_ml[col] = le.fit_transform(df_ml[col])

X = df_ml.drop(['Attrition', 'AgeGroup'], axis=1)
y = df_ml['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 8))
importances.plot(kind='barh', color='#197278')
plt.title('図10: 離職決定要因の重要度ランキング')
plt.savefig('/home/ubuntu/project/plots_v2/fig10_importance.png')
plt.close()

# 分析データの保存
with open('/home/ubuntu/project/analysis_results.txt', 'w') as f:
    f.write(f"S社離職率: {s_attrition_rate:.2f}%\n")
    f.write(f"業界平均離職率: {external_stats['industry_attrition_rate']}%\n")
    f.write(f"S社平均月収（換算）: {s_income_man:.2f}万円\n")
    f.write(f"業界平均月収: {external_stats['industry_avg_monthly_income']}万円\n")
    f.write(f"残業あり層の離職率: {ot_attrition.loc['Yes', 'Yes']:.2f}%\n")
    if 'No' in ot_attrition.index:
        f.write(f"残業なし層の離職率: {ot_attrition.loc['No', 'Yes']:.2f}%\n")
    else:
        f.write("残業なし層のデータがありません\n")
    f.write(f"重要度トップ3: {importances.index[0]}, {importances.index[1]}, {importances.index[2]}\n")

print("Analysis completed and plots generated.")
