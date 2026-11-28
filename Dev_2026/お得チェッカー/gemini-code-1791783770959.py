import pandas as pd

# 1. 各メーカーが実際に展開しているカテゴリ
brand_categories = {
    "Sony": ["テレビ", "カメラ", "オーディオ・スピーカー", "スマートフォン"],
    "Panasonic": [
        "テレビ",
        "エアコン",
        "洗濯機",
        "冷蔵庫",
        "調理家電",
        "掃除機・生活家電",
        "カメラ",
        "オーディオ・スピーカー",
    ],
    "SHARP": [
        "テレビ",
        "エアコン",
        "洗濯機",
        "冷蔵庫",
        "調理家電",
        "掃除機・生活家電",
        "スマートフォン",
    ],
    "Mitsubishi Electric": ["エアコン", "冷蔵庫", "掃除機・生活家電", "調理家電"],
    "Hitachi": ["エアコン", "洗濯機", "冷蔵庫", "掃除機・生活家電", "調理家電"],
    "Toshiba/TCL": ["テレビ", "洗濯機", "冷蔵庫", "エアコン", "調理家電"],
    "Dyson": ["掃除機・生活家電", "エアコン", "オーディオ・スピーカー"],
    "Apple": ["スマートフォン", "パソコン", "オーディオ・スピーカー"],
    "Samsung": ["スマートフォン", "テレビ"],
    "LG Electronics": ["テレビ", "洗濯機", "冷蔵庫", "パソコン"],
    "Canon": ["カメラ"],
    "Nikon": ["カメラ"],
    "Fujifilm": ["カメラ"],
    "Bose": ["オーディオ・スピーカー"],
    "Sennheiser": ["オーディオ・スピーカー"],
    "Yamaha": ["オーディオ・スピーカー"],
    "Balmuda": ["調理家電", "掃除機・生活家電", "スマートフォン"],
    "Iris Ohyama": [
        "テレビ",
        "エアコン",
        "洗濯機",
        "冷蔵庫",
        "調理家電",
        "掃除機・生活家電",
    ],
    "Yamazen": ["エアコン", "調理家電", "掃除機・生活家電", "テレビ"],
    "Twinbird": ["冷蔵庫", "調理家電", "掃除機・生活家電"],
    "Haier": ["洗濯機", "冷蔵庫", "エアコン"],
    "AQUA": ["洗濯機", "冷蔵庫"],
    "Hisense": ["テレビ", "冷蔵庫", "洗濯機", "エアコン"],
    "Anker": ["オーディオ・スピーカー", "掃除機・生活家電"],
    "Elecom": ["オーディオ・スピーカー", "パソコン"],
    "BUFFALO": ["パソコン"],
    "ASUS": ["パソコン", "スマートフォン"],
    "Lenovo": ["パソコン"],
    "Dell": ["パソコン"],
    "HP": ["パソコン"],
    "Acer": ["パソコン"],
    "MSI": ["パソコン"],
    "Xiaomi": ["スマートフォン", "掃除機・生活家電", "テレビ"],
    "OPPO": ["スマートフォン"],
    "Google": ["スマートフォン", "オーディオ・スピーカー"],
    "Philips": ["掃除機・生活家電", "調理家電"],
    "De'Longhi": ["調理家電", "掃除機・生活家電"],
    "Zojirushi": ["調理家電"],
    "Tiger": ["調理家電"],
    "Siroca": ["調理家電", "掃除機・生活家電"],
}

# 2. ブランド別の価格コントロール・維持力（1.00が標準）
brand_power = {
    "Apple": 1.25,
    "Sony": 1.15,
    "Canon": 1.18,
    "Nikon": 1.15,
    "Fujifilm": 1.18,
    "Dyson": 1.15,
    "Bose": 1.15,
    "Sennheiser": 1.12,
    "Balmuda": 1.10,
    "Panasonic": 1.10,  # 指定価格制度など
    "Hitachi": 1.05,
    "Mitsubishi Electric": 1.03,
    "SHARP": 1.00,
    "Toshiba/TCL": 0.98,
    "LG Electronics": 0.95,
    "Samsung": 1.02,
    "Yamaha": 1.08,
    "Google": 1.05,
    "Zojirushi": 1.05,
    "Tiger": 1.03,
    "De'Longhi": 1.08,
    "Philips": 0.98,
    "ASUS": 0.98,
    "Dell": 0.95,
    "HP": 0.95,
    "Lenovo": 0.92,
    "Iris Ohyama": 0.88,
    "Yamazen": 0.85,
    "Twinbird": 0.88,
    "Haier": 0.85,
    "AQUA": 0.90,
    "Hisense": 0.88,
    "Anker": 0.92,
    "Elecom": 0.88,
    "BUFFALO": 0.90,
    "Acer": 0.88,
    "MSI": 0.95,
    "Xiaomi": 0.88,
    "OPPO": 0.88,
    "Siroca": 0.90,
}

# 3. カテゴリごとの価格帯と基準残価率
category_config = {
    "テレビ": {
        "price_range": (30000, 500000),
        "rates": [62.0, 48.0, 36.0, 26.0, 18.0],
    },
    "エアコン": {
        "price_range": (40000, 300000),
        "rates": [65.0, 50.0, 38.0, 28.0, 20.0],
    },
    "洗濯機": {
        "price_range": (35000, 350000),
        "rates": [63.0, 48.0, 36.0, 26.0, 18.0],
    },
    "冷蔵庫": {
        "price_range": (40000, 400000),
        "rates": [65.0, 50.0, 38.0, 28.0, 20.0],
    },
    "パソコン": {
        "price_range": (50000, 400000),
        "rates": [70.0, 52.0, 38.0, 26.0, 18.0],
    },
    "カメラ": {
        "price_range": (40000, 600000),
        "rates": [85.0, 72.0, 60.0, 50.0, 40.0],
    },
    "スマートフォン": {
        "price_range": (20000, 250000),
        "rates": [78.0, 60.0, 45.0, 32.0, 22.0],
    },
    "オーディオ・スピーカー": {
        "price_range": (5000, 200000),
        "rates": [80.0, 68.0, 56.0, 46.0, 36.0],
    },
    "調理家電": {
        "price_range": (5000, 120000),
        "rates": [65.0, 48.0, 35.0, 25.0, 18.0],
    },
    "掃除機・生活家電": {
        "price_range": (8000, 150000),
        "rates": [62.0, 46.0, 34.0, 24.0, 16.0],
    },
}

# 4. 各メーカー×展開カテゴリの組み合わせで算出
rows = []
for brand, categories in brand_categories.items():
  power = brand_power.get(brand, 1.0)
  for cat in categories:
    cfg = category_config.get(cat)
    if not cfg:
      continue

    base_rates = cfg["rates"]

    # 残価率(%) = 100 - (100 - 基準残価率) / ブランド維持力
    r1 = round(100.0 - (100.0 - base_rates[0]) / power, 1)
    r2 = round(100.0 - (100.0 - base_rates[1]) / power, 1)
    r3 = round(100.0 - (100.0 - base_rates[2]) / power, 1)
    r4 = round(100.0 - (100.0 - base_rates[3]) / power, 1)
    r5 = round(100.0 - (100.0 - base_rates[4]) / power, 1)

    # 値下がり率(%) = 100 - 残価率
    d1 = round(100.0 - r1, 1)
    d2 = round(100.0 - r2, 1)
    d3 = round(100.0 - r3, 1)
    d4 = round(100.0 - r4, 1)
    d5 = round(100.0 - r5, 1)

    rows.append({
        "メーカー・ブランド": brand,
        "製品カテゴリ": cat,
        "ブランド維持力(係数)": power,
        "1年後残価率(%)": r1,
        "2年後残価率(%)": r2,
        "3年後残価率(%)": r3,
        "4年後残価率(%)": r4,
        "5年後残価率(%)": r5,
        "1年後値下がり率(%)": d1,
        "2年後値下がり率(%)": d2,
        "3年後値下がり率(%)": d3,
        "4年後値下がり率(%)": d4,
        "5年後値下がり率(%)": d5,
    })

df_summary = pd.DataFrame(rows)

# CSVファイルへ出力（UTF-8 with BOM）
output_filename = "brand_category_price_decline_summary.csv"
df_summary.to_csv(output_filename, index=False, encoding="utf-8-sig")

print(
    f"計算完了: 全 {len(df_summary)} 行の組合せを出力しました。({output_filename})"
)
