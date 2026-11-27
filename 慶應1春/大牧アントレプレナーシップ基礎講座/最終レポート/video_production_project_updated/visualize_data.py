import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import os

# 日本語フォントの設定（Noto Sans CJK JPを使用）
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if not os.path.exists(font_path):
    plt.rcParams['font.family'] = 'sans-serif'
    prop = None
else:
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

# 保存先ディレクトリ
output_dir = '/home/ubuntu/project/assets'
os.makedirs(output_dir, exist_ok=True)

def create_market_growth_chart():
    # 動画広告市場の推移
    years = ['2023', '2024', '2025', '2026', '2027', '2028', '2029']
    market_size = [6253, 7249, 8855, 10437, 12400, 14700, 16336]

    plt.figure(figsize=(10, 6))
    plt.bar(years, market_size, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.plot(years, market_size, marker='o', color='red', linestyle='--')
    
    plt.title('日本の動画広告市場規模の推移と予測 (2023-2029)', fontsize=14)
    plt.xlabel('年')
    plt.ylabel('市場規模 (億円)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(market_size):
        plt.text(i, v + 200, f'{v}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/market_growth.png')
    plt.close()

def create_regression_analysis():
    # データの読み込み
    data = {
        'year': [2023, 2024, 2025, 2026, 2027, 2028, 2029],
        'market_size': [6253, 7249, 8855, 10437, 12400, 14700, 16336]
    }
    df = pd.DataFrame(data)

    X = df[['year']]
    y = df['market_size']
    model = LinearRegression()
    model.fit(X, y)

    # 2030年の予測
    future_years = np.array([[2030]])
    prediction_2030 = model.predict(future_years)[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(df['year'], df['market_size'], color='blue', label='市場調査データ')
    plt.plot(df['year'], model.predict(X), color='red', linestyle='--', label='線形回帰トレンド線')
    plt.scatter(2030, prediction_2030, color='green', marker='x', s=100, label=f'2030年予測: {prediction_2030:.0f}億円')

    plt.title('動画広告市場の線形回帰分析と2030年予測', fontsize=14)
    plt.xlabel('年')
    plt.ylabel('市場規模 (億円)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/market_regression_analysis.png')
    plt.close()
    return prediction_2030

def create_tam_sam_som_chart():
    # TAM/SAM/SOM の可視化
    labels = ['TAM: 9,000億円\n(国内動画広告市場)', 'SAM: 1,800億円\n(中小企業・D2C市場)', 'SOM: 9億円\n(3年目獲得目標)']
    sizes = [100, 60, 20]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#e1f5fe', '#81d4fa', '#0288d1']
    
    for i in range(len(sizes)):
        circle = plt.Circle((0.5, 0.5), (sizes[i]/200), color=colors[i], label=labels[i])
        ax.add_artist(circle)
        ax.text(0.5, 0.5 + (sizes[i]/200) - 0.05, labels[i], ha='center', va='center', 
                fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('市場規模分析 (TAM / SAM / SOM)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tam_sam_som.png')
    plt.close()

def create_revenue_model_chart():
    # 収益モデルの内訳
    labels = ['受託制作 (B2B)', 'D2C製品販売', 'コミュニティ/教育', '広告収益 (YouTube)']
    sizes = [50, 25, 15, 10]
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('想定収益ポートフォリオ (創業3年目)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/revenue_model.png')
    plt.close()

if __name__ == '__main__':
    create_market_growth_chart()
    pred = create_regression_analysis()
    create_tam_sam_som_chart()
    create_revenue_model_chart()
    print(f"Charts generated. 2030 Prediction: {pred:.0f} billion yen")
