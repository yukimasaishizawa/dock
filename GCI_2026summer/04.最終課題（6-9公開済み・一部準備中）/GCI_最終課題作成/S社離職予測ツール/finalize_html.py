import json

# index.htmlのテンプレートを読み込む
with open('/home/ubuntu/index.html', 'r', encoding='utf-8') as f:
    template = f.read()

# モデルデータを読み込む
with open('/home/ubuntu/model_data.json', 'r', encoding='utf-8') as f:
    model_data = json.load(f)

# プレースホルダーを実際のデータで置換
final_html = template.replace('MODEL_DATA_PLACEHOLDER', json.dumps(model_data))

# 最終的なindex.htmlを書き出す
with open('/home/ubuntu/index.html', 'w', encoding='utf-8') as f:
    f.write(final_html)

print("index.html finalized with model data.")
