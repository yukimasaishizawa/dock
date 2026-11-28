#以下は2026年GCI Summerにおける最終課題のために制作した。©︎Yukimasa Ishizawa
#このコード類は全てVS code上で正しく動作するように作られているためgoogle colabなどでは適宜ファイルパスやライブライのインポートなどを修正しなければならない可能性がある。
#全体の見やすさ改善（スペースや改行など）に関しては一部自身のMacBook上でローカルでAIを動かし、使用した。これにはGoogle製Gemma4を使用した。


#必要なライブラリのインポート
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report)
from sklearn.pipeline import Pipeline
from matplotlib import rcParams

#Macに備え付きであるフォントを使用する。
plt.rcParams['font.family'] = 'Hiragino Sans' 
# マイナス記号の文字化けを防ぐ設定を行います。
plt.rcParams['axes.unicode_minus'] = False

#スライドデータ作成：共通定義] 離職（赤）と在籍（青）のカラーパレットを定義し、スライド挿入用のグラフ画像を保存するフォルダを自動作成します。
PALETTE_YES = '#E74C3C'
PALETTE_NO  = '#2980B9'
FIGDIR = './figures'
os.makedirs(FIGDIR, exist_ok=True)



#SECTION1:データ読み込みと基本確認
print("\n[SECTION 1] データ読み込みと基本確認")

#提供されたS社のデータを読み込む
df = pd.read_csv('data.csv')

#データの形状、列一覧、欠損値、基本統計量、スライド冒頭の「全体離職率」を算出
print(f"データ形状: {df.shape[0]}行 × {df.shape[1]}列")
print(f"\n列名一覧:\n{list(df.columns)}")
print(f"\n目的変数 Attrition の分布:")
print(df['Attrition'].value_counts())
overall_avg_pct = (df['Attrition'] == 'Yes').mean() * 100
print(f"\n離職率: {overall_avg_pct:.1f}%")

print(f"\n欠損値の確認:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\n基本統計量:\n{df.describe().round(2)}")
#相関分析や機械学習では今のままのデータでは使えないため、離職（Attrition）のYes/Noを1/0の数値データに変換した列をdfに追加する（get dummieとやっていることは同じ）
df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)


#SECTION2:探索的データ分析（EDA）
print("\n[SECTION 2] 探索的データ分析（EDA）")

#図1(出典：経済産業省)
#経済産業省の予測データを基に、今後のIT需要増加と供給減少による人材不足の予測データを定義し、前提条件とする。今後の分析ではS社の現状把握が大切になると考えられるため、業界平均などの比較値を事前に算出してグラフ化しておく。
forecast_years = [2018, 2020, 2022, 2025, 2030]
it_demand = [163, 172, 182, 198, 220]
it_supply = [163, 168, 172, 178, 141]
shortage  = [0, 4, 10, 20, 79]

#人材不足ゾーンを視覚的に強調し、需要と供給が噛み合わなくなることを視覚的にわかりやすくしたかったため、需要が噛み合っていない部分に薄く色付けした。
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
ax1.annotate('2030年: 最大79万人不足\n（経済産業省試算）', xy=(2030, 141), xytext=(2025, 115), arrowprops=dict(arrowstyle='->', color='#E74C3C'), fontsize=11, color='#E74C3C', fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
plt.title('図1：日本のIT人材需給ギャップの将来予測\n（出典：経済産業省「IT人材育成の状況等について」2019年）', fontsize=13, fontweight='bold', pad=15)
ax1.text(0.01, 0.02, '出典: 経済産業省「IT人材育成の状況等について」2019年\nhttps://www.meti.go.jp/shingikai/economy/daiyoji_sangyo_skill/pdf/001_s03_00.pdf', transform=ax1.transAxes, fontsize=8, color='gray', va='bottom')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig1_it_talent_shortage.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図1 保存完了")


#図2（出典：厚生労働省）
#厚生労働省による情報通信業および全産業の離職率推移データを可視化する。このすぐ後にS社のデータとの比較を行うため、大切な指標となると考えられる。
years = [2018, 2019, 2020, 2021, 2022, 2023]
it_attrition = [13.0, 13.5, 11.8, 12.4, 13.1, 12.8]   # 情報通信業
all_industry  = [14.6, 15.6, 14.2, 13.9, 15.0, 15.4]   # 全産業

#業界平均の推移線にS社の現在地を重ね、離職率が業界平均を上回ることをまとめた図を制作する。ただしS社のデータが何時頃のものかわからないため、リモートワーク導入率や従業員の年齢と勤続年数をある程度参考にし、2023年度のものであると仮定してずにプロットしてある。本来水平線で表しても良かったが、最新のデータとの比較の方が説得力が増しそうだと考えたため今回は一点に絞ってプロットしてある。
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(years, it_attrition, 'o-', color='#2980B9', linewidth=2.5, markersize=8, label='情報通信業（厚生労働省）')
ax.plot(years, all_industry, 's--', color='#95A5A6', linewidth=2, markersize=7, label='全産業平均（厚生労働省）')
ax.scatter([2023], [overall_avg_pct], color='#E74C3C', s=150, zorder=5, label='S社（本データ 2023年推定）')
ax.annotate(f'S社 {overall_avg_pct:.1f}%\n（業界平均+{overall_avg_pct - it_attrition[-1]:.1f}pt）',xy=(2023, overall_avg_pct), xytext=(2021.5, 17.5),arrowprops=dict(arrowstyle='->', color='#E74C3C'), fontsize=11, color='#E74C3C', fontweight='bold')
for x_pt, y_pt in zip(years, it_attrition):
    ax.annotate(f'{y_pt}%', xy=(x_pt, y_pt), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9, color='#2980B9')
ax.set_xlabel('年', fontsize=12)
ax.set_ylabel('離職率 (%)', fontsize=12)
ax.set_ylim(8, 22)
ax.set_title('図2：日本IT業界の離職率推移とS社の位置づけ\n（出典：厚生労働省「雇用動向調査」2018〜2023年）', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(alpha=0.3)
ax.text(0.01, 0.02, '出典: 厚生労働省「令和5年雇用動向調査結果の概況」\nhttps://www.mhlw.go.jp/toukei/itiran/roudou/koyou/doukou/24-2/dl/gaikyou.pdf', transform=ax.transAxes, fontsize=8, color='gray', va='bottom')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig2_industry_attrition_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図2 保存完了")


#図3（出典：Gallup 2024）
#Gallup調査に基づく主要な国別の従業員エンゲージメント率と離職意向率のデータを可視化する。
countries = ['米国', 'ドイツ', '英国', '韓国', '日本', '世界平均']
engagement = [33, 16, 10, 13, 6, 23]
attrition_risk = [18, 22, 25, 24, 28, 20]
x_arr = np.arange(len(countries))

#日本のエンゲージメント率が世界最低水準（6%）であることを赤色で強調した国際比較のグラフを制作する。これは大きな課題であり、S社の現状にも関連することであると考えられるため色使いを工夫し日本の弱点を強調する。
fig, ax1 = plt.subplots(figsize=(12, 6))
bars = ax1.bar(x_arr, engagement, color=['#3498DB' if c != '日本' else '#E74C3C' for c in countries], alpha=0.75, label='エンゲージメント率 (%)', width=0.5)
ax1.set_ylabel('エンゲージメント率 (%)', fontsize=12, color='#2980B9')
ax1.set_xticks(x_arr)
ax1.set_xticklabels(countries, fontsize=12)
ax1.set_ylim(0, 45)
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, engagement):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(x_arr, attrition_risk, 'D-', color='#E67E22', linewidth=2.5, markersize=9, label='離職意向率（推定）')
ax2.set_ylabel('離職意向率（推定）(%)', fontsize=12, color='#E67E22')
ax2.set_ylim(0, 45)
ax1.annotate('日本は世界最低水準\nの6%（Gallup 2024）', xy=(4, 6), xytext=(3.0, 20), arrowprops=dict(arrowstyle='->', color='#E74C3C'), fontsize=11, color='#E74C3C', fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
plt.title('図3：従業員エンゲージメント率と離職意向率の国際比較\n（出典：Gallup "State of the Global Workplace: 2024 Report"）', fontsize=13, fontweight='bold', pad=15)
ax1.text(0.01, 0.02, '出典: Gallup "State of the Global Workplace: 2024 Report"\nhttps://www.gallup.com/workplace/349484/state-of-the-global-workplace.aspx', transform=ax1.transAxes, fontsize=8, color='gray', va='bottom')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig3_engagement_international.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図3 保存完了")


#図4
#採用経路ごとの採用人数と離職率を算出し、採用方法による従業員の離職率の偏りに関して可視化する。これは今後S社がどの採用方法を強化し、どの採用方法を取りやめるかの根拠となる分析であると考える。
fig, ax1 = plt.subplots(figsize=(12, 6))
employ_stats = df.groupby('HowToEmploy').agg(attrition_pct=('Attrition_bin', lambda x: x.mean() * 100), count=('Attrition_bin', 'count')).reset_index().sort_values('attrition_pct', ascending=False)
label_map = {'agent_A': '人材エージェントA', 'agent_B': '人材エージェントB', 'agent_C': '人材エージェントC', 'intern': 'インターン経由', 'direct_recruiting': 'ダイレクトリクルーティング', 'New_graduate_recruitment': '新卒採用'}
employ_stats['label'] = employ_stats['HowToEmploy'].map(label_map)
x = range(len(employ_stats))
ax1.bar(x, employ_stats['count'], color='#AED6F1', alpha=0.7, label='採用人数（左軸）')
ax1.set_xticks(x)
ax1.set_xticklabels(employ_stats['label'], rotation=20, ha='right', fontsize=10)
ax1.set_ylabel('採用人数（人）', fontsize=12, color='#2980B9')
ax1.grid(axis='y', alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(x, employ_stats['attrition_pct'], 'D-', color=PALETTE_YES, linewidth=2.5, markersize=9, label='離職率（右軸）')
ax2.set_ylabel('離職率 (%)', fontsize=12, color=PALETTE_YES)
ax2.set_ylim(0, 40)
ax2.axhline(y=overall_avg_pct, color='gray', linestyle='--', linewidth=1.2, label=f'全体平均 {overall_avg_pct:.1f}%')
for xi, row in zip(x, employ_stats.itertuples()):
    ax2.annotate(f"{row.attrition_pct:.1f}%", xy=(xi, row.attrition_pct), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', color=PALETTE_YES)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
plt.title('図4：採用経路別の採用人数と離職率（S社データ n=1,470）', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig4_howtoemploy_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図4 保存完了")


#図5
#職種別の離職率を算出し、20%を超える高リスク職種を赤色で強調した横棒グラフを制作する。どの職種に問題があるかをはっきりさせるために不可欠であると考える。
fig, ax = plt.subplots(figsize=(12, 7))
role_stats = df.groupby('JobRole').agg(attrition_pct=('Attrition_bin', lambda x: x.mean() * 100), count=('Attrition_bin', 'count')).reset_index().sort_values('attrition_pct', ascending=True)
colors_role = [PALETTE_YES if v > 20 else '#3498DB' for v in role_stats['attrition_pct']]
bars = ax.barh(role_stats['JobRole'], role_stats['attrition_pct'], color=colors_role, edgecolor='white', linewidth=1.2)
for bar, (_, row) in zip(bars, role_stats.iterrows()):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f"{row['attrition_pct']:.1f}%  (n={int(row['count'])})", va='center', fontsize=10)
ax.axvline(x=overall_avg_pct, color='gray', linestyle='--', linewidth=1.5, label=f'全体平均 {overall_avg_pct:.1f}%')
ax.set_xlabel('離職率 (%)', fontsize=12)
ax.set_title('図5：職種別 離職率ランキング（S社データ n=1,470）', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.set_xlim(0, 55)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig5_jobrole_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図5 保存完了")


#図6
#月間残業時間の長さと離職率の関係を見るために可視化する。
df['Attrition_Flag'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
it_benchmark_attrition_rate = 10.2  # 業界ベンチマーク値
benchmarks = {'it_attrition_rate': it_benchmark_attrition_rate}

#残業時間が長くなるほど離職率が高まる相関傾向を視覚化した棒グラフを作成する。このグラフにより、長時間労働を強いられている従業員がいることがわかり、残業時間と離職率の関係が一目でわかる。したがってS社に過労働（残業）が離職を促していることを強くアピールできると考える。
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='OverTime', y='Attrition_Flag', palette='magma')
plt.axhline(y=benchmarks['it_attrition_rate']/100, color='red', linestyle='--', label='業界平均(JISA調査参考)')
for ind, label in enumerate(ax.get_xticklabels()):
    if ind % 5 == 0:
        label.set_visible(True)
    else:
        label.set_visible(False)
plt.title('図6: 残業時間と離職率の関係', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('残業時間（月間）')
plt.ylabel('離職率')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig6_overtime_attrition.png')
plt.close()
print("  → 図6 保存完了")


#図7
#週のリモートワーク日数ごとの在籍人数と離職率を算出し、リモート制度の効果を示すグラフを制作する。このデータはのちに提案としてリモートワークを推奨する根拠となる。
fig, ax1 = plt.subplots(figsize=(12, 6))
remote_stats = df.groupby('RemoteWork').agg(attrition_rate=('Attrition_bin', 'mean'), count=('Attrition_bin', 'count')).reset_index()
remote_stats['attrition_pct'] = remote_stats['attrition_rate'] * 100
ax1.bar(remote_stats['RemoteWork'], remote_stats['count'], color='#AED6F1', alpha=0.7, label='在籍人数（左軸）', zorder=2)
ax1.set_xlabel('リモートワーク日数（日/週）', fontsize=12)
ax1.set_ylabel('在籍人数（人）', fontsize=12, color='#2980B9')
ax1.grid(axis='y', alpha=0.3, zorder=1)
ax2 = ax1.twinx()
ax2.plot(remote_stats['RemoteWork'], remote_stats['attrition_pct'], 'o-', color=PALETTE_YES, linewidth=2.5, markersize=8, label='離職率（右軸）')
ax2.set_ylabel('離職率 (%)', fontsize=12, color=PALETTE_YES)
ax2.set_ylim(0, 60)
for _, row in remote_stats.iterrows():
    ax2.annotate(f"{row['attrition_pct']:.1f}%",xy=(row['RemoteWork'], row['attrition_pct']),xytext=(0, 10), textcoords='offset points',ha='center', fontsize=10, color=PALETTE_YES, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
plt.title('図7：リモートワーク日数別の在籍人数と離職率（S社データ n=1,470）', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig7_remote_work_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図7 保存完了")


#図8
#2つのリスク因子の掛け合わせ（ストレス×WLB、リモート×インセンティブ、職位×ストレス）が離職率に与える影響を可視化するヒートマップを制作する。一度前列におけるヒートマップを制作したが、必要ない部分が多く含まれたことと、文字が小さくなり見ずらかったことを理由に特徴的な相関のみを抜き出して図にまとめることにした。
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('図8：主要リスク因子と離職率の関係（S社データ n=1,470）', fontsize=15, fontweight='bold', y=1.02)
pivot1 = df.pivot_table(values='Attrition_bin', index='StressRating', columns='WorkLifeBalance', aggfunc='mean') * 100
sns.heatmap(pivot1, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0], vmin=0, vmax=70, cbar_kws={'label': '離職率 (%)'})
axes[0].set_title('(a) ストレス × ワークライフバランス', fontsize=12)
axes[0].set_xlabel('ワークライフバランス (1=低 ～ 4=高)', fontsize=10)
axes[0].set_ylabel('ストレス評価 (1=低 ～ 5=高)', fontsize=10)
df['Incentive_flag'] = df['Incentive'].apply(lambda x: 'インセンティブあり' if x > 0 else 'インセンティブなし')
pivot2 = df.pivot_table(values='Attrition_bin', index='RemoteWork', columns='Incentive_flag', aggfunc='mean') * 100
sns.heatmap(pivot2, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], vmin=0, vmax=70, cbar_kws={'label': '離職率 (%)'})
axes[1].set_title('(b) リモートワーク日数 × インセンティブ', fontsize=12)
axes[1].set_xlabel('インセンティブ有無', fontsize=10)
axes[1].set_ylabel('リモートワーク日数/週 (0〜5)', fontsize=10)
pivot3 = df.pivot_table(values='Attrition_bin', index='JobLevel', columns='StressRating', aggfunc='mean') * 100
sns.heatmap(pivot3, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[2], vmin=0, vmax=70, cbar_kws={'label': '離職率 (%)'})
axes[2].set_title('(c) 職位レベル × ストレス評価', fontsize=12)
axes[2].set_xlabel('ストレス評価 (1=低 ～ 5=高)', fontsize=10)
axes[2].set_ylabel('職位レベル (1=低 ～ 5=高)', fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig8_heatmap_risk_factors.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図8 保存完了")


#図9
#従業員の主観的なストレス評価（5段階）別の人数と離職率を算出し、高ストレス層における離職の急増を示すグラフを制作する。
fig, ax1 = plt.subplots(figsize=(10, 6))
stress_stats = df.groupby('StressRating').agg(attrition_rate=('Attrition_bin', 'mean'), count=('Attrition_bin', 'count')).reset_index()
stress_stats['attrition_pct'] = stress_stats['attrition_rate'] * 100
colors_stress = ['#27AE60', '#F1C40F', '#E67E22', '#E74C3C', '#8E44AD']
ax1.bar(stress_stats['StressRating'], stress_stats['count'], color=colors_stress, alpha=0.75, label='在籍人数（左軸）', zorder=2)
ax1.set_xlabel('ストレス評価（1=低 ～ 5=高）', fontsize=12)
ax1.set_ylabel('在籍人数（人）', fontsize=12)
ax1.grid(axis='y', alpha=0.3, zorder=1)
ax2 = ax1.twinx()
ax2.plot(stress_stats['StressRating'], stress_stats['attrition_pct'], 's-', color='#2C3E50', linewidth=2.5, markersize=9, label='離職率（right軸）')
ax2.set_ylabel('離職率 (%)', fontsize=12, color='#2C3E50')
ax2.set_ylim(0, 90)
for _, row in stress_stats.iterrows():
    ax2.annotate(f"{row['attrition_pct']:.1f}%",xy=(row['StressRating'], row['attrition_pct']),xytext=(0, 10), textcoords='offset points',ha='center', fontsize=11, fontweight='bold', color='#2C3E50')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
plt.title('図9：ストレス評価別の在籍人数と離職率（S社データ n=1,470）', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig9_stress_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図9 保存完了")


#図10
#月収・勤続年数・年齢について、在籍者（青）と離職者（赤）の確率密度および中央値の差を比較するヒストグラムを制作する。割合と分布を見比べたいため棒部分を薄く塗りつぶしている。
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('図10：主要数値変数の分布（離職有無別）（S社データ n=1,470）', fontsize=14, fontweight='bold')
yes_df = df[df['Attrition'] == 'Yes']
no_df  = df[df['Attrition'] == 'No']
for ax, col, xlabel in zip(axes, ['MonthlyIncome', 'YearsAtCompany', 'Age'], ['月収（百円）', '勤続年数（年）', '年齢（歳）']):
    ax.hist(no_df[col], bins=25, alpha=0.6, color=PALETTE_NO, label='在籍（No）', density=True)
    ax.hist(yes_df[col], bins=25, alpha=0.6, color=PALETTE_YES, label='離職（Yes）', density=True)
    ax.axvline(no_df[col].median(), color=PALETTE_NO, linestyle='--', linewidth=1.5, label=f'在籍中央値 {no_df[col].median():.0f}')
    ax.axvline(yes_df[col].median(), color=PALETTE_YES, linestyle='--', linewidth=1.5, label=f'離職中央値 {yes_df[col].median():.0f}')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig10_distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図10 保存完了")


#図11
#インセンティブ、福利厚生、フレックス制度が離職抑制に良い影響を与えているのではないかと考え、実際に可視化して確認する。
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('図11：制度・待遇別の離職率比較（S社データ n=1,470）', fontsize=14, fontweight='bold')

def plot_bar_attrition(ax, col, labels_map, title, xlabel):
    stats = df.groupby(col)['Attrition_bin'].mean() * 100
    keys = sorted(stats.index)
    vals = [stats[k] for k in keys]
    xlabels = [labels_map.get(k, str(k)) for k in keys]
    colors = [PALETTE_YES if v > 20 else '#3498DB' for v in vals]
    bars = ax.bar(xlabels, vals, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('離職率 (%)', fontsize=10)
    ax.set_ylim(0, max(vals) * 1.3 + 5)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=overall_avg_pct, color='gray', linestyle='--', linewidth=1.2, label=f'全体平均 {overall_avg_pct:.1f}%')
    ax.legend(fontsize=9)

df['Incentive_label'] = df['Incentive'].apply(lambda x: 'なし' if x == 0 else 'あり')
df['WB_label'] = df['WelfareBenefits'].apply(lambda x: '低(0-1)' if x <= 1 else ('中(2-3)' if x <= 3 else '高(4+)'))
df['Flex_label'] = df['FlexibleWork'].apply(lambda x: 'なし' if x == 0 else 'あり')
plot_bar_attrition(axes[0], 'Incentive_label', {}, '(a) インセンティブ有無別', 'インセンティブ')
plot_bar_attrition(axes[1], 'WB_label', {}, '(b) 福利厚生スコア別', '福利厚生スコア')
plot_bar_attrition(axes[2], 'Flex_label', {}, '(c) フレックスワーク制度別', 'フレックスワーク')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig11_benefits_attrition.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図11 保存完了")


#SECTION3:データ前処理と特徴量エンジニアリング
#機械学習による離職予測を行うため、識別不能な不要列の削除をするなどデータの前処理を行う。
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

#不均衡データに配慮した層化抽出（stratify）を行い、データを訓練用8割・テスト用2割に分割する。全部で1500件程度しかデータがないため、本来ならtest_size=0.3などで行いたいが、trainで過学習を防ぎたいためtrainの比重を重くした。GCIの複数の講義でもtest_size=0.2が用いられていたため参考にした。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"訓練データ（学習用）: {X_train.shape[0]}件, テストデータ（検証用）: {X_test.shape[0]}件")



#SECTION4:機械学習モデルの構築・評価
print("\n[SECTION 4] 機械学習モデルの構築・評価")
print("-" * 50)
#予測精度を比較検証するため、ロジスティック回帰、ランダムフォレスト、勾配ブースティング（GBM）の3つのモデルを構築する。
models = {
    'ロジスティック回帰（ベースライン）': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ]),
    'ランダムフォレスト': RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    '勾配ブースティング（GBM）': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
}
results = {}
roc_data = {}
cv_results = {} 
print("\n【モデル評価結果（テストセットでのホールドアウト評価）】")
print(f"{'モデル名':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("-" * 82)

#各モデルを訓練し、各種性能指標（正解率やAUCなど）などを測定するCVS(cross_val_score)を実行します。
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
    print(f"{name:<30} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {auc:>10.4f}")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = cv_scores

print("\n【モデル交差検証（5-Fold Cross-Validation）結果 - 評価指標: ROC-AUC】")
print(f"{'モデル名':<30} {'各フォールドのスコア':<40} {'平均スコア':>10}")
print("-" * 85)
for name, scores in cv_results.items():
    scores_str = ", ".join([f"{s:.4f}" for s in scores])
    print(f"{name:<30} [{scores_str}] {scores.mean():>10.4f}")



#図12
#各予測モデルの5つの指標を並べ、最もパフォーマンスが優れるモデルを決定・提示するグラフを制作する。
fig, ax = plt.subplots(figsize=(13, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
model_names_short = list(results.keys())
x_arr = np.arange(len(metrics))
width = 0.25
colors_models = ['#95A5A6', '#E74C3C', '#2980B9']
for i, (mname, color) in enumerate(zip(model_names_short, colors_models)):
    vals = [results[mname][m] for m in metrics]
    bars = ax.bar(x_arr + i * width, vals, width, label=mname, color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(x_arr + width)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylabel('スコア', fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_title('図12：機械学習モデル比較（評価指標別スコア）\n（S社データ テストセット n=294）', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.8, color='gray', linestyle=':', linewidth=1)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig12_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図12 保存完了")


#図13
#横軸に偽陽性率、縦軸に真陽性率をとるROC曲線をプロットし、モデルの識別能力（AUC）を可視化する。なお、偽陽性率とは間違っていることをあっていると判定してしまう確率、真陽性率とはあっていることをあっていると正しく言い当てられる確率を表す。AUCはGCIコンペティションで採用されており、そこで算出方法に関して学んだ。モデルの評価に最適だと感じたため今回採用した。
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ['#95A5A6', '#E74C3C', '#2980B9']
for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors_roc):
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='ランダム予測 (AUC=0.500)')
ax.set_xlabel('偽陽性率 (False Positive Rate)', fontsize=12)
ax.set_ylabel('真陽性率 (True Positive Rate)', fontsize=12)
ax.set_title('図13：ROC曲線（モデル比較）\n（S社データ テストセット n=294）', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig13_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図13 保存完了")


#図14
#最も予測精度が高かったランダムフォレストから特徴量重要度を抽出し、各カラムを日本語に直した上で可視化する。なお、各カラムの翻訳にはAIを使用した。
rf_model = models['ランダムフォレスト']
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
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
    'StockOptionLevel': 'ストックオプション', 'NumCompaniesWorked': '転職回数',
    'TrainingTimesLastYear': '年間研修回数', 'PerformanceRating': '業績評価',
    'PerformanceIndex': 'パフォーマンス指数', 'MonthlyAchievement': '月次達成度',
    'BusinessTravel': '出張頻度', 'Department': '部門', 'EducationField': '専攻分野',
    'Gender': '性別', 'MaritalStatus': '婚姻状況', 'JobRole': '職種',
    'HowToEmploy': '採用経路', 'Education': '学歴', 'OverTime': '残業時間',
    'ExtendedLeave': '長期休暇取得',
}
feature_imp_jp = feature_imp.copy()
feature_imp_jp.index = [col_labels_jp.get(c, c) for c in feature_imp.index]

#人事戦略において最優先でアプローチすべき上位要因を視覚化し、ランキング化する。
fig, ax = plt.subplots(figsize=(12, 8))
colors_fi = ['#E74C3C' if i < 5 else '#E67E22' if i < 10 else '#3498DB' for i in range(len(feature_imp_jp))]
ax.barh(feature_imp_jp.index[::-1], feature_imp_jp.values[::-1], color=colors_fi[::-1], edgecolor='white', linewidth=1.2)
for bar, val in zip(ax.patches, feature_imp_jp.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('特徴量重要度（不純度の減少量の平均）', fontsize=12)
ax.set_title('図14：離職予測モデルの特徴量重要度 Top20\n（ランダムフォレスト、S社データ n=1,470）', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
red_patch = mpatches.Patch(color='#E74C3C', label='Top5（最重要）')
orange_patch = mpatches.Patch(color='#E67E22', label='Top6-10')
blue_patch = mpatches.Patch(color='#3498DB', label='Top11-20')
ax.legend(handles=[red_patch, orange_patch, blue_patch], fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig14_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図14 保存完了")


#図15
#各数値変数と離職とのピアソン相関係数を算出し、正負の影響度上位15項目を整理するグラフを制作する。
fig, ax = plt.subplots(figsize=(12, 7))
num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols = [c for c in num_cols if c not in ['Attrition_bin', 'Attrition_Flag']]
corr = df[num_cols + ['Attrition_bin']].corr()['Attrition_bin'].drop('Attrition_bin')
corr_top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)
col_labels = {
    'Incentive': 'インセンティブ額', 'JobLevel': '職位レベル', 'StressRating': 'ストレス評価',
    'TotalWorkingYears': '総勤続年数', 'YearsWithCurrManager': '現上司との年数',
    'Age': '年齢', 'YearsInCurrentRole': '現職種年数', 'ExtendedLeave': '長期休暇取得',
    'YearsAtCompany': '在社年数', 'RemoteWork': 'リモートワーク日数',
    'WelfareBenefits': '福利厚生スコア', 'JobInvolvement': '仕事への関与度',
    'MonthlyIncome': '月収', 'FlexibleWork': 'フレックスワーク',
    'StockOptionLevel': 'ストックオプション',
}
labels = [col_labels.get(c, c) for c in corr_top.index]
colors_corr = [PALETTE_YES if v > 0 else PALETTE_NO for v in corr_top.values]
ax.barh(labels[::-1], corr_top.values[::-1], color=colors_corr[::-1], edgecolor='white', linewidth=1.2)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('離職との相関係数（ピアソン）', fontsize=12)
ax.set_title('図15：離職（Attrition）との相関係数 Top15（S社データ n=1,470）', fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
red_patch = mpatches.Patch(color=PALETTE_YES, label='正の相関（高いほど離職リスク増）')
blue_patch = mpatches.Patch(color=PALETTE_NO, label='負の相関（高いほど離職リスク減）')
ax.legend(handles=[red_patch, blue_patch], fontsize=10)
for bar, val in zip(ax.patches, corr_top.values[::-1]):
    ax.text(val + (0.003 if val >= 0 else -0.003), bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig15_correlation_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → 図15 保存完了")


#SECTION5:結果の出力
print("\n[SECTION 5] 分析結果")
#スライドのまとめや根拠として使用したいため、分析結果から得られた各重要指標をまとめて出力する。
print("\n【EDA主要発見事項】")
print(f"  ・全体離職率: {df['Attrition_bin'].mean()*100:.1f}% (n=1,470)")
print(f"  ・リモートワーク週0日の離職率: {df[df['RemoteWork']==0]['Attrition_bin'].mean()*100:.1f}%")
print(f"  ・リモートワーク週5日の離職率: {df[df['RemoteWork']==5]['Attrition_bin'].mean()*100:.1f}%")
print(f"  ・ストレス評価5の離職率: {df[df['StressRating']==5]['Attrition_bin'].mean()*100:.1f}%")
print(f"  ・インセンティブなし層の離職率: {df[df['Incentive']==0]['Attrition_bin'].mean()*100:.1f}%")

print("\n【機械学習モデル評価結果（ホールドアウト＆交差検証）】")
for name, scores in results.items():
    print(f"  [{name}]")
    for metric, score in scores.items():
        print(f"    {metric}: {score:.4f}")
    print(f"    5-Fold CV ROC-AUC (Mean): {cv_results[name].mean():.4f}")

print("\n【特徴量重要度 Top5（ランダムフォレスト）】")
for feat, imp in feature_imp_jp.head(5).items():
    print(f"  {feat}: {imp:.4f}")

print("\n全分析完了")