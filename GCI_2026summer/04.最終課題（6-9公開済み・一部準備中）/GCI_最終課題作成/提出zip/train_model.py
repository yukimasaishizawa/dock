import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import json

# データの読み込み
df = pd.read_csv('/home/ubuntu/upload/data.csv')

# 不要なカラムの削除
drop_cols = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# ターゲットの変換
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# カテゴリ変数のエンコーディング
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# 特徴量とターゲットの分離
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# 特徴量名の保存
feature_names = X.columns.tolist()

# モデルの訓練
model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X, y)

# 決定木の構造をエクスポートする関数
def export_tree(tree, feature_names):
    tree_ = tree.tree_
    def recurse(node):
        if tree_.feature[node] != -2: # Not a leaf
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            return {
                "type": "split",
                "feature": name,
                "threshold": float(threshold),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            return {
                "type": "leaf",
                "value": float(tree_.value[node][0, 0])
            }
    return recurse(0)

# 全ての木の構造を保存
trees = []
for estimator in model.estimators_:
    trees.append(export_tree(estimator[0], feature_names))

# init_valueの取得（バージョンによって異なる可能性があるため安全に取得）
try:
    init_value = float(model.init_.constant_[0][0])
except AttributeError:
    # 別の属性名や方法を試す
    if hasattr(model.init_, 'predict'):
        # 初期予測値を取得
        dummy_X = np.zeros((1, X.shape[1]))
        init_pred = model.init_.predict_proba(dummy_X)
        # 対数オッズに変換
        init_value = float(np.log(init_pred[0, 1] / init_pred[0, 0]))
    else:
        init_value = 0.0

# モデル情報の保存
model_data = {
    "init_value": init_value,
    "learning_rate": float(model.learning_rate),
    "trees": trees,
    "feature_names": feature_names,
    "categorical_info": {}
}

# カテゴリ変数の情報を保存
for col in categorical_cols:
    model_data["categorical_info"][col] = sorted(df[col].unique().tolist())

with open('/home/ubuntu/model_data.json', 'w') as f:
    json.dump(model_data, f)

print("Model trained and exported to model_data.json")
