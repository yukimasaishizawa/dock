import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import os

# 日本語フォントの設定
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(font_path):
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
else:
    plt.rcParams['font.family'] = 'sans-serif'

# データの読み込み
df = pd.read_csv('/home/ubuntu/project/video_ad_market_data.csv')

# 線形回帰モデルの構築 (市場全体)
X = df[['year']]
y = df['market_size_billion_yen']
model = LinearRegression()
model.fit(X, y)

# 2030年の予測
future_year = np.array([[2030]])
prediction_2030 = model.predict(future_year)[0]

# グラフ作成: 市場規模の推移と予測
plt.figure(figsize=(10, 6))
plt.scatter(df['year'], df['market_size_billion_yen'], color='blue', label='実績/予測値(CA調査)')
plt.plot(df['year'], model.predict(X), color='red', linestyle='--', label='線形回帰トレンド線')
plt.scatter(2030, prediction_2030, color='green', marker='x', s=100, label=f'2030年予測: {prediction_2030:.0f}億円')

plt.title('国内動画広告市場の線形回帰分析と2030年予測', fontsize=14)
plt.xlabel('年')
plt.ylabel('市場規模 (億円)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

output_path = '/home/ubuntu/project/assets/market_regression_analysis.png'
plt.savefig(output_path)
plt.close()

# 縦型動画の成長分析
X_v = df[['year']]
y_v = df['vertical_video_size_billion_yen']
model_v = LinearRegression()
model_v.fit(X_v, y_v)
prediction_v_2030 = model_v.predict(future_year)[0]

plt.figure(figsize=(10, 6))
plt.bar(df['year'], df['vertical_video_size_billion_yen'], color='orange', alpha=0.7, label='縦型動画市場規模')
plt.plot(df['year'], model_v.predict(X_v), color='darkorange', linestyle='-', marker='o', label='トレンド予測')

plt.title('縦型動画広告市場の急成長分析', fontsize=14)
plt.xlabel('年')
plt.ylabel('市場規模 (億円)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

output_path_v = '/home/ubuntu/project/assets/vertical_video_growth.png'
plt.savefig(output_path_v)
plt.close()

# 結果の出力
print(f"Analysis complete.")
print(f"2030 Market Size Prediction: {prediction_2030:.2f} billion yen")
print(f"2030 Vertical Video Prediction: {prediction_v_2030:.2f} billion yen")
print(f"Regression Coefficient (Market): {model.coef_[0]:.2f}")
print(f"Regression Coefficient (Vertical): {model_v.coef_[0]:.2f}")
