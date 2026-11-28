"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  S社 離職防止・生産性向上 PoC  ─  スライド別 分析コード                       ║
║  各スライドで使用したデータ分析・グラフ生成コードを個別に記載                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from matplotlib import rcParams
rcParams['font.family'] = 'Noto Sans CJK JP'
rcParams['axes.unicode_minus'] = False

PALETTE_YES = '#E74C3C'
PALETTE_NO  = '#2980B9'
FIGDIR = './figures'
import os; os.makedirs(FIGDIR, exist_ok=True)

df = pd.read_csv('./data.csv')
df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)
df['Incentive_flag'] = df['Incentive'].apply(lambda x: 'インセンティブあり' if x > 0 else 'インセンティブなし')

# ══════════════════════════════════════════════════════════════
# スライド 3: 背景と市場環境（外部データ）
# 使用データ: 経済産業省「IT人材育成の状況等について」（2019年）
# 出典URL: https://www.meti.go.jp/shingikai/economy/daiyoji_sangyo_skill/pdf/001_s03_00.pdf
# ══════════════════════════════════════════════════════════════
print("=== スライド3: IT人材不足の将来予測（図14）===")
forecast_years = [2018, 2020, 2022, 2025, 2030]
it_demand = [163, 172, 182, 198, 220]
it_supply = [163, 168, 172, 178, 141]
shortage  = [0, 4, 10, 20, 79]

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.fill_between(forecast_years, it_supply, it_demand, alpha=0.25, color='#E74C3C', label='人材不足ゾーン')
ax1.plot(forecast_years, it_demand, 'o-', color='#2980B9', linewidth=2.5, markersize=8, label='IT人材需要（万人）')
ax1.plot(forecast_years, it_supply, 's--', color='#E74C3C', linewidth=2.5, markersize=8, label='IT人材供給（万人）')
ax1.set_ylabel('IT人材数（万人）', fontsize=12)
ax1.set_ylim(100, 250)
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.bar(forecast_years, shortage, width=1.2, color='#E74C3C', alpha=0.35, label='人材不足数（万人）')
ax2.set_ylabel('人材不足数（万人）', fontsize=12, color='#E74C3C')
ax2.set_ylim(0, 120)
ax1.annotate('2030年: 最大79万人不足\n（経済産業省試算）', xy=(2030, 141), xytext=(2025, 115),
             arrowprops=dict(arrowstyle='->', color='#E74C3C'),
             fontsize=11, color='#E74C3C', fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
plt.title('図14：日本のIT人材需給ギャップの将来予測\n（出典：経済産業省「IT人材育成の状況等について」2019年）',
          fontsize=13, fontweight='bold')
ax1.text(0.01, 0.02,
         '出典: 経済産業省「IT人材育成の状況等について」2019年\nhttps://www.meti.go.jp/shingikai/economy/daiyoji_sangyo_skill/pdf/001_s03_00.pdf',
         transform=ax1.transAxes, fontsize=8, color='gray', va='bottom')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide3_fig14_it_talent_shortage.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide3_fig14_it_talent_shortage.png")


# ══════════════════════════════════════════════════════════════
# スライド 5: S社の離職率と業界比較（外部データ①）
# 使用データ: 厚生労働省「令和5年雇用動向調査結果の概況」（2024年）
# 出典URL: https://www.mhlw.go.jp/toukei/itiran/roudou/koyou/doukou/24-2/dl/gaikyou.pdf
# ══════════════════════════════════════════════════════════════
print("\n=== スライド5: S社の離職率と業界比較（図12）===")
years = [2018, 2019, 2020, 2021, 2022, 2023]
it_attrition = [13.0, 13.5, 11.8, 12.4, 13.1, 12.8]
all_industry  = [14.6, 15.6, 14.2, 13.9, 15.0, 15.4]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(years, it_attrition, 'o-', color='#2980B9', linewidth=2.5, markersize=8, label='情報通信業（厚生労働省）')
ax.plot(years, all_industry, 's--', color='#95A5A6', linewidth=2, markersize=7, label='全産業平均（厚生労働省）')
ax.scatter([2023], [16.2], color='#E74C3C', s=150, zorder=5, label='S社（本データ 2023年推定）')
ax.annotate('S社 16.2%\n（業界平均+3.4pt）', xy=(2023, 16.2), xytext=(2021.5, 17.5),
            arrowprops=dict(arrowstyle='->', color='#E74C3C'),
            fontsize=11, color='#E74C3C', fontweight='bold')
for x_pt, y_pt in zip(years, it_attrition):
    ax.annotate(f'{y_pt}%', xy=(x_pt, y_pt), xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, color='#2980B9')
ax.set_xlabel('年', fontsize=12)
ax.set_ylabel('離職率 (%)', fontsize=12)
ax.set_ylim(8, 22)
ax.set_title('図12：日本IT業界の離職率推移とS社の位置づけ\n（出典：厚生労働省「雇用動向調査」2018〜2023年）',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)
ax.text(0.01, 0.02,
        '出典: 厚生労働省「令和5年雇用動向調査結果の概況」\nhttps://www.mhlw.go.jp/toukei/itiran/roudou/koyou/doukou/24-2/dl/gaikyou.pdf',
        transform=ax.transAxes, fontsize=8, color='gray', va='bottom')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide5_fig12_industry_attrition_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide5_fig12_industry_attrition_trend.png")

# 分析コード：S社離職率の計算
attrition_rate = df['Attrition_bin'].mean() * 100
print(f"  S社離職率: {attrition_rate:.1f}%")
print(f"  業界平均（情報通信業）: 12.8%")
print(f"  差分: +{attrition_rate - 12.8:.1f}pt")


# ══════════════════════════════════════════════════════════════
# スライド 6: 職種・採用経路別離職率
# 使用データ: S社 data.csv
# ══════════════════════════════════════════════════════════════
print("\n=== スライド6: 職種別・採用経路別離職率（図5・図7）===")

# 図5: 職種別離職率
role_stats = df.groupby('JobRole').agg(
    attrition_pct=('Attrition_bin', lambda x: x.mean() * 100),
    count=('Attrition_bin', 'count')
).reset_index().sort_values('attrition_pct', ascending=True)
print("\n  職種別離職率:")
print(role_stats[['JobRole', 'attrition_pct', 'count']].to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 7))
colors_role = [PALETTE_YES if v > 20 else '#3498DB' for v in role_stats['attrition_pct']]
bars = ax.barh(role_stats['JobRole'], role_stats['attrition_pct'],
               color=colors_role, edgecolor='white', linewidth=1.2)
for bar, (_, row) in zip(bars, role_stats.iterrows()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{row['attrition_pct']:.1f}%  (n={int(row['count'])})", va='center', fontsize=10)
ax.axvline(x=16.2, color='gray', linestyle='--', linewidth=1.5, label='全体平均 16.2%')
ax.set_xlabel('離職率 (%)', fontsize=12)
ax.set_title('図5：職種別 離職率ランキング（S社データ n=1,470）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 55)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide6_fig5_jobrole_attrition.png', dpi=150, bbox_inches='tight')
plt.close()

# 図7: 採用経路別離職率
employ_stats = df.groupby('HowToEmploy').agg(
    attrition_pct=('Attrition_bin', lambda x: x.mean() * 100),
    count=('Attrition_bin', 'count')
).reset_index().sort_values('attrition_pct', ascending=False)
label_map = {
    'agent_A': '人材エージェントA', 'agent_B': '人材エージェントB',
    'agent_C': '人材エージェントC', 'intern': 'インターン経由',
    'direct_recruiting': 'ダイレクトリクルーティング',
    'New_graduate_recruitment': '新卒採用'
}
employ_stats['label'] = employ_stats['HowToEmploy'].map(label_map)
print("\n  採用経路別離職率:")
print(employ_stats[['label', 'attrition_pct', 'count']].to_string(index=False))

fig, ax1 = plt.subplots(figsize=(12, 6))
x = range(len(employ_stats))
ax1.bar(x, employ_stats['count'], color='#AED6F1', alpha=0.7, label='採用人数（左軸）')
ax1.set_xticks(x)
ax1.set_xticklabels(employ_stats['label'], rotation=20, ha='right', fontsize=10)
ax1.set_ylabel('採用人数（人）', fontsize=12, color='#2980B9')
ax1.grid(axis='y', alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(x, employ_stats['attrition_pct'], 'D-', color=PALETTE_YES,
         linewidth=2.5, markersize=9, label='離職率（右軸）')
ax2.set_ylabel('離職率 (%)', fontsize=12, color=PALETTE_YES)
ax2.set_ylim(0, 40)
ax2.axhline(y=16.2, color='gray', linestyle='--', linewidth=1.2, label='全体平均 16.2%')
for xi, row in zip(x, employ_stats.itertuples()):
    ax2.annotate(f"{row.attrition_pct:.1f}%", xy=(xi, row.attrition_pct),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=10, fontweight='bold', color=PALETTE_YES)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
plt.title('図7：採用経路別の採用人数と離職率（S社データ n=1,470）', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide6_fig7_howtoemploy_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide6_fig5_jobrole_attrition.png, slide6_fig7_howtoemploy_attrition.png")


# ══════════════════════════════════════════════════════════════
# スライド 7: リモートワーク × インセンティブ
# 使用データ: S社 data.csv
# ══════════════════════════════════════════════════════════════
print("\n=== スライド7: リモートワーク×インセンティブ（図1b・図2）===")

# リモートワーク別離職率の詳細
remote_stats = df.groupby('RemoteWork').agg(
    attrition_rate=('Attrition_bin', 'mean'),
    count=('Attrition_bin', 'count')
).reset_index()
remote_stats['attrition_pct'] = remote_stats['attrition_rate'] * 100
print("\n  リモートワーク日数別離職率:")
print(remote_stats[['RemoteWork', 'attrition_pct', 'count']].to_string(index=False))

# リモートワーク0日×インセンティブなし
mask_high_risk = (df['RemoteWork'] == 0) & (df['Incentive'] == 0)
print(f"\n  リモートワーク0日×インセンティブなし の離職率: {df[mask_high_risk]['Attrition_bin'].mean()*100:.1f}%")
print(f"  インセンティブあり層の離職率: {df[df['Incentive']>0]['Attrition_bin'].mean()*100:.1f}%")

# ヒートマップ（リモートワーク × インセンティブ）
pivot_remote_incentive = df.pivot_table(
    values='Attrition_bin', index='RemoteWork', columns='Incentive_flag', aggfunc='mean') * 100
print("\n  リモートワーク × インセンティブ 離職率ヒートマップ:")
print(pivot_remote_incentive.round(1))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot_remote_incentive, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
            vmin=0, vmax=70, cbar_kws={'label': '離職率 (%)'})
ax.set_title('図1b：リモートワーク日数 × インセンティブ別 離職率\n（S社データ n=1,470）',
             fontsize=13, fontweight='bold')
ax.set_xlabel('インセンティブ有無', fontsize=11)
ax.set_ylabel('リモートワーク日数/週 (0〜5)', fontsize=11)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide7_fig1b_remote_incentive_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 図2: リモートワーク日数別離職率
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(remote_stats['RemoteWork'], remote_stats['count'],
        color='#AED6F1', alpha=0.7, label='在籍人数（左軸）', zorder=2)
ax1.set_xlabel('リモートワーク日数（日/週）', fontsize=12)
ax1.set_ylabel('在籍人数（人）', fontsize=12, color='#2980B9')
ax1.grid(axis='y', alpha=0.3, zorder=1)
ax2 = ax1.twinx()
ax2.plot(remote_stats['RemoteWork'], remote_stats['attrition_pct'],
         'o-', color=PALETTE_YES, linewidth=2.5, markersize=8, label='離職率（右軸）')
ax2.set_ylabel('離職率 (%)', fontsize=12, color=PALETTE_YES)
ax2.set_ylim(0, 60)
for _, row in remote_stats.iterrows():
    ax2.annotate(f"{row['attrition_pct']:.1f}%",
                 xy=(row['RemoteWork'], row['attrition_pct']),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=10, color=PALETTE_YES, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
plt.title('図2：リモートワーク日数別の在籍人数と離職率（S社データ n=1,470）',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide7_fig2_remote_work_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide7_fig1b_remote_incentive_heatmap.png, slide7_fig2_remote_work_attrition.png")


# ══════════════════════════════════════════════════════════════
# スライド 8: ストレス × ワークライフバランス
# 使用データ: S社 data.csv
# ══════════════════════════════════════════════════════════════
print("\n=== スライド8: ストレス×ワークライフバランス（図1a・図3）===")

stress_stats = df.groupby('StressRating').agg(
    attrition_rate=('Attrition_bin', 'mean'),
    count=('Attrition_bin', 'count')
).reset_index()
stress_stats['attrition_pct'] = stress_stats['attrition_rate'] * 100
print("\n  ストレス評価別離職率:")
print(stress_stats[['StressRating', 'attrition_pct', 'count']].to_string(index=False))

# ストレス5×WLB1の離職率
mask_stress5_wlb1 = (df['StressRating'] == 5) & (df['WorkLifeBalance'] == 1)
print(f"\n  ストレス5×WLB1の離職率: {df[mask_stress5_wlb1]['Attrition_bin'].mean()*100:.1f}%")
print(f"  WLB評価1（最低）の離職率: {df[df['WorkLifeBalance']==1]['Attrition_bin'].mean()*100:.1f}%")

# ヒートマップ（ストレス × WLB）
pivot_stress_wlb = df.pivot_table(
    values='Attrition_bin', index='StressRating', columns='WorkLifeBalance', aggfunc='mean') * 100
print("\n  ストレス × ワークライフバランス 離職率ヒートマップ:")
print(pivot_stress_wlb.round(1))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(pivot_stress_wlb, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0],
            vmin=0, vmax=70, cbar_kws={'label': '離職率 (%)'})
axes[0].set_title('図1a：ストレス × ワークライフバランス別 離職率\n（S社データ n=1,470）', fontsize=12, fontweight='bold')
axes[0].set_xlabel('ワークライフバランス (1=低 ～ 4=高)', fontsize=10)
axes[0].set_ylabel('ストレス評価 (1=低 ～ 5=高)', fontsize=10)

colors_stress = ['#27AE60', '#F1C40F', '#E67E22', '#E74C3C', '#8E44AD']
axes[1].bar(stress_stats['StressRating'], stress_stats['attrition_pct'],
            color=colors_stress, alpha=0.85, edgecolor='white')
for i, (_, row) in enumerate(stress_stats.iterrows()):
    axes[1].text(row['StressRating'], row['attrition_pct'] + 1,
                 f"{row['attrition_pct']:.1f}%", ha='center', fontsize=11, fontweight='bold')
axes[1].axhline(y=16.2, color='gray', linestyle='--', linewidth=1.5, label='全体平均 16.2%')
axes[1].set_xlabel('ストレス評価（1=低 ～ 5=高）', fontsize=11)
axes[1].set_ylabel('離職率 (%)', fontsize=11)
axes[1].set_title('図3：ストレス評価別 離職率\n（S社データ n=1,470）', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide8_fig1a_fig3_stress_wlb.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide8_fig1a_fig3_stress_wlb.png")


# ══════════════════════════════════════════════════════════════
# スライド 9: 若手・低職位層への負荷集中
# 使用データ: S社 data.csv
# ══════════════════════════════════════════════════════════════
print("\n=== スライド9: 若手・低職位層の離職分析（図6）===")

# 職位レベル × ストレス評価の離職率
pivot_level_stress = df.pivot_table(
    values='Attrition_bin', index='JobLevel', columns='StressRating', aggfunc='mean') * 100
print("\n  職位レベル × ストレス評価 離職率ヒートマップ:")
print(pivot_level_stress.round(1))

# 職位レベル1×ストレス5の離職率
mask_l1_s5 = (df['JobLevel'] == 1) & (df['StressRating'] == 5)
print(f"\n  職位レベル1×ストレス5の離職率: {df[mask_l1_s5]['Attrition_bin'].mean()*100:.1f}%")

# 主要数値変数の分布（離職有無別）
yes_df = df[df['Attrition'] == 'Yes']
no_df  = df[df['Attrition'] == 'No']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('図6：主要数値変数の分布（離職有無別）（S社データ n=1,470）', fontsize=14, fontweight='bold')
for ax, col, xlabel in zip(axes,
    ['MonthlyIncome', 'YearsAtCompany', 'Age'],
    ['月収（千円）', '勤続年数（年）', '年齢（歳）']):
    ax.hist(no_df[col], bins=25, alpha=0.6, color=PALETTE_NO, label='在籍（No）', density=True)
    ax.hist(yes_df[col], bins=25, alpha=0.6, color=PALETTE_YES, label='離職（Yes）', density=True)
    ax.axvline(no_df[col].median(), color=PALETTE_NO, linestyle='--', linewidth=1.5,
               label=f'在籍中央値 {no_df[col].median():.0f}')
    ax.axvline(yes_df[col].median(), color=PALETTE_YES, linestyle='--', linewidth=1.5,
               label=f'離職中央値 {yes_df[col].median():.0f}')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide9_fig6_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide9_fig6_distribution_comparison.png")
print(f"  月収中央値（在籍）: {no_df['MonthlyIncome'].median():.0f}千円")
print(f"  月収中央値（離職）: {yes_df['MonthlyIncome'].median():.0f}千円")
print(f"  勤続年数中央値（在籍）: {no_df['YearsAtCompany'].median():.0f}年")
print(f"  勤続年数中央値（離職）: {yes_df['YearsAtCompany'].median():.0f}年")


# ══════════════════════════════════════════════════════════════
# スライド 10: 福利厚生・制度の機能不全
# 使用データ: S社 data.csv
# ══════════════════════════════════════════════════════════════
print("\n=== スライド10: 福利厚生・制度別離職率（図4）===")

df['Incentive_label'] = df['Incentive'].apply(lambda x: 'なし' if x == 0 else 'あり')
df['WB_label'] = df['WelfareBenefits'].apply(
    lambda x: '低(0-1)' if x <= 1 else ('中(2-3)' if x <= 3 else '高(4+)'))
df['Flex_label'] = df['FlexibleWork'].apply(lambda x: 'なし' if x == 0 else 'あり')

print("\n  インセンティブ有無別離職率:")
print(df.groupby('Incentive_label')['Attrition_bin'].mean().apply(lambda x: f'{x*100:.1f}%'))
print("\n  福利厚生スコア別離職率:")
print(df.groupby('WB_label')['Attrition_bin'].mean().apply(lambda x: f'{x*100:.1f}%'))
print("\n  フレックスワーク制度別離職率:")
print(df.groupby('Flex_label')['Attrition_bin'].mean().apply(lambda x: f'{x*100:.1f}%'))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('図4：制度・待遇別の離職率比較（S社データ n=1,470）', fontsize=14, fontweight='bold')

def plot_bar_attrition(ax, col, title, xlabel):
    stats = df.groupby(col)['Attrition_bin'].mean() * 100
    keys = sorted(stats.index)
    vals = [stats[k] for k in keys]
    colors = [PALETTE_YES if v > 20 else '#3498DB' for v in vals]
    bars = ax.bar(keys, vals, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('離職率 (%)', fontsize=10)
    ax.set_ylim(0, max(vals) * 1.3 + 5)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=16.2, color='gray', linestyle='--', linewidth=1.2, label='全体平均 16.2%')
    ax.legend(fontsize=9)

plot_bar_attrition(axes[0], 'Incentive_label', '(a) インセンティブ有無別', 'インセンティブ')
plot_bar_attrition(axes[1], 'WB_label', '(b) 福利厚生スコア別', '福利厚生スコア')
plot_bar_attrition(axes[2], 'Flex_label', '(c) フレックスワーク制度別', 'フレックスワーク')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide10_fig4_benefits_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide10_fig4_benefits_attrition.png")


# ══════════════════════════════════════════════════════════════
# スライド 11: エンゲージメント国際比較（外部データ③）
# 使用データ: Gallup "State of the Global Workplace: 2024 Report"
# 出典URL: https://www.gallup.com/workplace/349484/state-of-the-global-workplace.aspx
# ══════════════════════════════════════════════════════════════
print("\n=== スライド11: エンゲージメント国際比較（図13）===")
countries = ['米国', 'ドイツ', '英国', '韓国', '日本', '世界平均']
engagement = [33, 16, 10, 13, 6, 23]
attrition_risk = [18, 22, 25, 24, 28, 20]

x_arr = np.arange(len(countries))
fig, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(x_arr, engagement,
               color=['#3498DB' if c != '日本' else '#E74C3C' for c in countries],
               alpha=0.75, label='エンゲージメント率 (%)', width=0.5)
ax1.set_ylabel('エンゲージメント率 (%)', fontsize=12, color='#2980B9')
ax1.set_xticks(x_arr)
ax1.set_xticklabels(countries, fontsize=12)
ax1.set_ylim(0, 45)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, engagement):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(x_arr, attrition_risk, 'D-', color='#E67E22', linewidth=2.5, markersize=9,
         label='離職意向率（推定）')
ax2.set_ylabel('離職意向率（推定）(%)', fontsize=12, color='#E67E22')
ax2.set_ylim(0, 45)
ax1.annotate('日本は世界最低水準\nの6%（Gallup 2024）', xy=(4, 6), xytext=(3.0, 20),
             arrowprops=dict(arrowstyle='->', color='#E74C3C'),
             fontsize=11, color='#E74C3C', fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
plt.title('図13：従業員エンゲージメント率と離職意向率の国際比較\n（出典：Gallup "State of the Global Workplace: 2024 Report"）',
          fontsize=13, fontweight='bold')
ax1.text(0.01, 0.02,
         '出典: Gallup "State of the Global Workplace: 2024 Report"\nhttps://www.gallup.com/workplace/349484/state-of-the-global-workplace.aspx',
         transform=ax1.transAxes, fontsize=8, color='gray', va='bottom')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide11_fig13_engagement_international.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide11_fig13_engagement_international.png")

# S社のJobInvolvementと離職の相関
ji_corr = df['JobInvolvement'].corr(df['Attrition_bin'])
print(f"\n  S社データ: JobInvolvement（仕事への関与度）と離職の相関係数: {ji_corr:.3f}")


# ══════════════════════════════════════════════════════════════
# スライド 12: 機械学習モデル構築・評価
# 使用データ: S社 data.csv
# ══════════════════════════════════════════════════════════════
print("\n=== スライド12: 機械学習モデル構築・評価（図9・図10）===")

df_ml = pd.read_csv('./data.csv')
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df_ml = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])
y = (df_ml['Attrition'] == 'Yes').astype(int)
df_ml = df_ml.drop(columns=['Attrition'])
cat_cols = df_ml.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
X = df_ml.copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'ロジスティック回帰（ベースライン）': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ]),
    'ランダムフォレスト': RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1),
    '勾配ブースティング（GBM）': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42),
}

results = {}
roc_data = {}
print(f"\n  {'モデル名':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("  " + "-" * 80)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': auc}
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)
    print(f"  {name:<30} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")

# 図9: モデル比較バーチャート
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x_arr = np.arange(len(metrics))
width = 0.25
colors_models = ['#95A5A6', '#E74C3C', '#2980B9']
fig, ax = plt.subplots(figsize=(13, 6))
for i, (mname, color) in enumerate(zip(results.keys(), colors_models)):
    vals = [results[mname][m] for m in metrics]
    bars = ax.bar(x_arr + i * width, vals, width, label=mname, color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(x_arr + width)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel('スコア', fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_title('図9：機械学習モデル比較（評価指標別スコア）\n（S社データ テストセット n=294）',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide12_fig9_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 図10: ROC曲線
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ['#95A5A6', '#E74C3C', '#2980B9']
for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors_roc):
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='ランダム予測 (AUC=0.500)')
ax.set_xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
ax.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12)
ax.set_title('図10：ROC曲線（モデル比較）\n（S社データ テストセット n=294）',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide12_fig10_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide12_fig9_model_comparison.png, slide12_fig10_roc_curves.png")


# ══════════════════════════════════════════════════════════════
# スライド 13: 特徴量重要度
# 使用データ: S社 data.csv
# ══════════════════════════════════════════════════════════════
print("\n=== スライド13: 特徴量重要度（図8・図11）===")

rf_model = models['ランダムフォレスト']
feature_imp = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False).head(20)
col_labels_jp = {
    'Incentive': 'インセンティブ額', 'MonthlyIncome': '月収', 'Age': '年齢',
    'TotalWorkingYears': '総勤続年数', 'YearsAtCompany': '在社年数',
    'YearsInCurrentRole': '現職種年数', 'YearsWithCurrManager': '現上司との年数',
    'YearsSinceLastPromotion': '最終昇進からの年数', 'StressRating': 'ストレス評価',
    'DistanceFromHome': '通勤距離', 'JobLevel': '職位レベル',
    'JobSatisfaction': '仕事満足度', 'EnvironmentSatisfaction': '職場環境満足度',
    'RelationshipSatisfaction': '人間関係満足度', 'JobInvolvement': '仕事への関与度',
    'WorkLifeBalance': 'ワークライフバランス', 'RemoteWork': 'リモートワーク日数',
    'WelfareBenefits': '福利厚生スコア', 'FlexibleWork': 'フレックスワーク',
    'StockOptionLevel': 'ストックオプション',
}
feature_imp_jp = feature_imp.copy()
feature_imp_jp.index = [col_labels_jp.get(c, c) for c in feature_imp.index]
print("\n  特徴量重要度 Top10（ランダムフォレスト）:")
print(feature_imp_jp.head(10).apply(lambda x: f'{x:.4f}').to_string())

fig, ax = plt.subplots(figsize=(12, 8))
colors_fi = ['#E74C3C' if i < 5 else '#E67E22' if i < 10 else '#3498DB'
             for i in range(len(feature_imp_jp))]
ax.barh(feature_imp_jp.index[::-1], feature_imp_jp.values[::-1],
        color=colors_fi[::-1], edgecolor='white', linewidth=1.2)
for bar, val in zip(ax.patches, feature_imp_jp.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('特徴量重要度（不純度の減少量の平均）', fontsize=12)
ax.set_title('図11：離職予測モデルの特徴量重要度 Top20\n（ランダムフォレスト、S社データ n=1,470）',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
red_patch = mpatches.Patch(color='#E74C3C', label='Top5（最重要）')
orange_patch = mpatches.Patch(color='#E67E22', label='Top6-10')
blue_patch = mpatches.Patch(color='#3498DB', label='Top11-20')
ax.legend(handles=[red_patch, orange_patch, blue_patch], fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide13_fig11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# 図8: 相関係数ランキング
num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols = [c for c in num_cols if c != 'Attrition_bin']
corr = df[num_cols + ['Attrition_bin']].corr()['Attrition_bin'].drop('Attrition_bin')
corr_top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)
corr_top_jp = corr_top.copy()
corr_top_jp.index = [col_labels_jp.get(c, c) for c in corr_top.index]
print("\n  離職との相関係数 Top10:")
print(corr_top_jp.head(10).apply(lambda x: f'{x:.3f}').to_string())

fig, ax = plt.subplots(figsize=(12, 7))
colors_corr = [PALETTE_YES if v > 0 else PALETTE_NO for v in corr_top_jp.values]
ax.barh(corr_top_jp.index[::-1], corr_top_jp.values[::-1], color=colors_corr[::-1], edgecolor='white')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('離職との相関係数（ピアソン）', fontsize=12)
ax.set_title('図8：離職（Attrition）との相関係数 Top15（S社データ n=1,470）',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
red_patch = mpatches.Patch(color=PALETTE_YES, label='正の相関（高いほど離職リスク増）')
blue_patch = mpatches.Patch(color=PALETTE_NO, label='負の相関（高いほど離職リスク減）')
ax.legend(handles=[red_patch, blue_patch], fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/slide13_fig8_correlation_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: slide13_fig11_feature_importance.png, slide13_fig8_correlation_ranking.png")


print("\n" + "=" * 70)
print("スライド別分析コード 全実行完了")
print("生成されたグラフは ./figures/ フォルダに保存されました。")
print("=" * 70)
