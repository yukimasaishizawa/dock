import pandas as pd
import json

df = pd.read_csv('/Users/yukimasaishizawa/Documents/GCI/04.最終課題（6-9公開済み・一部準備中）/GCI_最終課題作成/S社離職予測ツール/data.csv')

column_info = {}
for col in df.columns:
    if col == 'Attrition':
        continue
    
    # 実際に数値として解釈できるか確認
    try:
        pd.to_numeric(df[col])
        is_numeric = True
    except:
        is_numeric = False

    if is_numeric:
        column_info[col] = {
            'type': 'numerical',
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean())
        }
    else:
        column_info[col] = {
            'type': 'categorical',
            'values': sorted([str(x) for x in df[col].unique().tolist()])
        }

with open('/Users/yukimasaishizawa/Documents/GCI/04.最終課題（6-9公開済み・一部準備中）/GCI_最終課題作成/S社離職予測ツール/info.json', 'w', encoding='utf-8') as f:
    json.dump(column_info, f, ensure_ascii=False, indent=2)

print("Target distribution (Attrition):")
print(df['Attrition'].value_counts())
