import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import os

# 日本語フォントの設定（Noto Sans CJK JPを使用）
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if not os.path.exists(font_path):
    # フォントが見つからない場合はデフォルトを使用（環境に応じて調整が必要な場合あり）
    plt.rcParams['font.family'] = 'sans-serif'
else:
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()

# 保存先ディレクトリ
output_dir = '/home/ubuntu/output_images'
os.makedirs(output_dir, exist_ok=True)

def create_market_growth_chart():
    # 動画広告市場の推移 (サイバーエージェントデータ)
    years = ['2023', '2024', '2025(予)', '2026(予)', '2027(予)', '2028(予)', '2029(予)']
    market_size = [6253, 7249, 8855, 10500, 12400, 14700, 16336] # 2026-2028は推計

    plt.figure(figsize=(10, 6))
    plt.bar(years, market_size, color='skyblue', edgecolor='navy')
    plt.plot(years, market_size, marker='o', color='red', linestyle='--')
    
    plt.title('日本の動画広告市場規模の推移と予測', fontsize=14, fontproperties=prop)
    plt.xlabel('年', fontsize=12, fontproperties=prop)
    plt.ylabel('市場規模 (億円)', fontsize=12, fontproperties=prop)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(market_size):
        plt.text(i, v + 200, f'{v}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/market_growth.png')
    plt.close()

def create_tam_sam_som_chart():
    # TAM/SAM/SOM の可視化 (同心円状)
    labels = ['TAM: 9,000億円\n(国内動画広告市場)', 'SAM: 1,800億円\n(中小企業・D2C市場)', 'SOM: 9億円\n(3年目獲得目標)']
    sizes = [100, 60, 20] # 相対的な大きさ
    
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['#e1f5fe', '#81d4fa', '#0288d1']
    
    for i in range(len(sizes)):
        circle = plt.Circle((0.5, 0.5), (sizes[i]/200), color=colors[i], label=labels[i])
        ax.add_artist(circle)
        # テキスト配置
        ax.text(0.5, 0.5 + (sizes[i]/200) - 0.05, labels[i], ha='center', va='center', 
                fontsize=10, fontproperties=prop, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('市場規模分析 (TAM / SAM / SOM)', fontsize=16, fontproperties=prop)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tam_sam_som.png')
    plt.close()

def create_revenue_model_chart():
    # 収益モデルの内訳
    labels = ['受託制作 (B2B)', 'D2C製品販売', 'コミュニティ/教育', '広告収益 (YouTube)']
    sizes = [50, 25, 15, 10]
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
            textprops={'fontproperties': prop})
    plt.title('想定収益ポートフォリオ (創業3年目)', fontsize=14, fontproperties=prop)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/revenue_model.png')
    plt.close()

if __name__ == '__main__':
    create_market_growth_chart()
    create_tam_sam_som_chart()
    create_revenue_model_chart()
    print("Charts generated successfully in /home/ubuntu/output_images")
