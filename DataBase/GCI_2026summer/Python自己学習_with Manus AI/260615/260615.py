import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------
# 1. データ読み込みと初期確認
# -----------------------------------------------------------
df = pd.read_csv("exercise_data_day30.csv")
print("=" * 30)
print("1. データ確認")
print("=" * 30)
print("データ先頭部分:")
print(df.head())
print("欠損値の数:")
print(df.isnull().sum())
print("データ概要:")
print(df.describe())

# -----------------------------------------------------------
# 2. 相関行列の可視化 (特徴量間の相関を確認)
# -----------------------------------------------------------
# 対象となる列は、モデルの学習に使う特徴量と、目的変数 (y) を含めます
feature_cols = ["sku", "stock_level", "incoming_shipment"]
target_col = "sales_volume"

print("\n" + "=" * 30)
print("2. 相関行列の可視化")
print("=" * 30)

# 相関係数の計算
corr = df[feature_cols + [target_col]].corr()

# ヒートマッププロット
plt.figure(figsize=(10, 6))
ax = sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('特徴量と目的変数の相関行列')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()
plt.close()  # メモリ解放

# 相関数値を console にも表示
print("相関行列の数値表示:")
print(corr)

# 相関の判断基準を表示
print("\n相関の判断基準:")
print("- 0.7 以上：強い正の相関")
print("- 0.3 ~ 0.7: 中程度の相関")
print("- 0.3 以下：弱い相関または相関なし")

# -----------------------------------------------------------
# 3. 欠損値の補完
# -----------------------------------------------------------
print("\n" + "=" * 30)
print("3. 欠損値の補完")
print("=" * 30)

# 欠損値が 0 になることを確認
if df.isnull().sum().sum() > 0:
    df["sku"] = df["sku"].fillna(df["sku"].mean())
    df["stock_level"] = df["stock_level"].fillna(df["stock_level"].mean())
    df["incoming_shipment"] = df["incoming_shipment"].fillna(df["incoming_shipment"].mean())
    df["sales_volume"] = df["sales_volume"].fillna(df["sales_volume"].mean())
    print("欠損値を補完しました。")
else:
    print("欠損値がなかったため、補完は省略されました。")

print("補完後の欠損値:", df.isnull().sum())

# -----------------------------------------------------------
# 4. モデルの構築と学習 (クロスバリデーション)
# -----------------------------------------------------------
print("\n" + "=" * 30)
print("4. クロスバリデーションでの学習")
print("=" * 30)

x = df[feature_cols]
y = df[target_col]

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42) # n_splits=5 に変更（標準は 10）

# R2 スコアのクロスバリデーション
cv_scores = cross_val_score(model, x, y, cv=kf, scoring="r2")
print(f"クロスバリデーション R2 スコア：{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# -----------------------------------------------------------
# 5. 学習済みモデルでテストデータの予測
# -----------------------------------------------------------
print("\n" + "=" * 30)
print("5. テストデータの予測と評価")
print("=" * 30)

# 訓練データとテストデータを分割
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# 特徴量を標準化
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)

# モデルの訓練
model.fit(x_train_std, y_train)

# 予測
y_pred = model.predict(x_test_std)

# 評価指標
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"平均二乗誤差 (MSE): {mse:.4f}")
print(f"決定係数 (R2 Score): {r2:.4f}")

# モデルの係数（重み）
print("係数 (モデルの重み):")
for i, col in enumerate(feature_cols):
    print(f"{col}: {model.coef_[i]:.4f}")

# -----------------------------------------------------------
# 6. 残渣プロットの可視化 (モデルの評価)
# -----------------------------------------------------------
print("\n" + "=" * 30)
print("6. 残渣プロットの可視化")
print("=" * 30)

# 予測値と残差を計算
residuals = y_test - y_pred

# グラフ作成
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data={"predicted": y_test, "residual": residuals}, 
                     x="predicted", y="residual", color="blue", alpha=0.7)

# 残差=0 のライン
plt.axhline(y=0, color="red", linestyle="--", linewidth=2, label="残差=0")

# (オプション) 残差の回帰直線 (残差が 0 に近くあるか確認)
# sns.regplot(data={"predicted": y_test, "residual": residuals}, 
#             x="predicted", y="residual", color="gray", line_kws={"linewidth": 1})

plt.xlabel("予測値 (Predicted Values)")
plt.ylabel("残差 (Residuals)")
plt.title("残渣プロット (残差 vs 予測値)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.show()
plt.close()

print("残渣プロットの可視化が完了しました。")
print("理想の残渣プロット: 点が y=0 のラインの周りにランダムに散らばっている状態")
