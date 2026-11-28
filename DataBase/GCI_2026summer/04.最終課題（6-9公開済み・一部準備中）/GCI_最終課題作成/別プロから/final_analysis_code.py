import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# フォント設定（日本語表示用）
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def load_and_preprocess(path):
    df = pd.read_csv(path)
    
    # 離職フラグの数値化
    df['Attrition_Flag'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 不要な列の削除（定数やIDなど）
    # EmployeeCount, Over18, StandardHours などは通常定数
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    return df

def basic_stats(df):
    print("--- Data Info ---")
    print(df.info())
    print("\n--- Descriptive Stats ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # 離職率の確認
    attrition_rate = df['Attrition_Flag'].mean() * 100
    print(f"\nOverall Attrition Rate: {attrition_rate:.2f}%")
    
    # 生産性の確認
    print(f"Mean Performance Index: {df['PerformanceIndex'].mean():.2f}")

def save_initial_plots(df):
    os.makedirs('plots', exist_ok=True)
    
    # 1. 離職率の分布 (図1)
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Attrition', palette='viridis')
    plt.title('図1: 離職(Attrition)の分布')
    plt.savefig('plots/fig1_attrition_dist.png')
    plt.close()
    
    # 2. 生産性指標の分布 (図2)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PerformanceIndex'], kde=True, color='blue')
    plt.title('図2: 生産性指標(PerformanceIndex)の分布')
    plt.savefig('plots/fig2_performance_dist.png')
    plt.close()
    
    # 3. 相関ヒートマップ (図3)
    plt.figure(figsize=(15, 12))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('図3: 数値変数の相関ヒートマップ')
    plt.tight_layout()
    plt.savefig('plots/fig3_correlation_heatmap.png')
    plt.close()

if __name__ == "__main__":
    data_path = "/home/ubuntu/upload/data.csv"
    df = load_and_preprocess(data_path)
    basic_stats(df)
    save_initial_plots(df)
    df.to_csv("/home/ubuntu/project/preprocessed_data.csv", index=False)
    print("\nStep 1 Completed: Data loaded and preprocessed.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# フォント設定
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
os.makedirs('plots', exist_ok=True)

def perform_10_analyses(df):
    # 外部データ（ベンチマーク値）の定義（調査結果に基づく）
    benchmarks = {
        'it_attrition_rate': 10.2, # 厚労省 雇用動向調査 情報通信業 平均
        'avg_monthly_income_it': 400000, # 賃金構造基本統計調査 概算
    }

    # 1. 属性分析 (Age vs Attrition)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df[df['Attrition']=='No'], x='Age', label='在職', fill=True)
    sns.kdeplot(data=df[df['Attrition']=='Yes'], x='Age', label='離職', fill=True)
    plt.title('図4: 年齢層別の離職傾向')
    plt.legend()
    plt.savefig('plots/fig4_age_attrition.png')
    plt.close()

    # 2. 働き方 (OverTime vs Attrition)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='OverTime', y='Attrition_Flag', palette='magma')
    plt.axhline(y=benchmarks['it_attrition_rate']/100, color='red', linestyle='--', label='業界平均')
    plt.title('図5: 残業の有無と離職率の関係')
    plt.savefig('plots/fig5_overtime_attrition.png')
    plt.close()

    # 3. 報酬 (MonthlyIncome vs Performance)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='MonthlyIncome', y='PerformanceIndex', hue='Attrition', alpha=0.5)
    plt.title('図6: 月収と生産性・離職の関係')
    plt.savefig('plots/fig6_income_performance.png')
    plt.close()

    # 4. 職場環境 (EnvironmentSatisfaction vs Attrition)
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df, x='EnvironmentSatisfaction', y='Attrition_Flag', color='green')
    plt.title('図7: 職場環境満足度と離職率')
    plt.savefig('plots/fig7_env_satisfaction.png')
    plt.close()

    # 5. キャリア (YearsAtCompany vs Attrition)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Attrition', y='YearsAtCompany')
    plt.title('図8: 勤続年数と離職の関係')
    plt.savefig('plots/fig8_years_at_company.png')
    plt.close()

    # 6. 関与度 (JobInvolvement vs Performance)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='JobInvolvement', y='PerformanceIndex', palette='viridis')
    plt.title('図9: 仕事への関与度と生産性指標')
    plt.savefig('plots/fig9_involvement_performance.png')
    plt.close()

    # 7. ストレス (StressRating vs Attrition)
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='Attrition', y='StressRating', palette='coolwarm')
    plt.title('図10: ストレス評価と離職の関係')
    plt.savefig('plots/fig10_stress_attrition.png')
    plt.close()

    # 8. 採用チャネル (HowToEmploy vs Attrition)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='HowToEmploy', y='Attrition_Flag')
    plt.title('図11: 採用チャネル別の離職率')
    plt.xticks(rotation=45)
    plt.savefig('plots/fig11_hiring_attrition.png')
    plt.close()

    # 9. リモートワーク (RemoteWork vs Performance)
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='RemoteWork', y='PerformanceIndex', marker='o')
    plt.title('図12: リモートワーク水準と生産性指標')
    plt.savefig('plots/fig12_remote_performance.png')
    plt.close()

    # 10. 勤続年数と昇進 (YearsSinceLastPromotion vs Attrition)
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df, x='Attrition', y='YearsSinceLastPromotion', jitter=True, alpha=0.3)
    plt.title('図13: 前回の昇進からの年数と離職')
    plt.savefig('plots/fig13_promotion_attrition.png')
    plt.close()

if __name__ == "__main__":
    df = pd.read_csv("/home/ubuntu/project/preprocessed_data.csv")
    perform_10_analyses(df)
    print("Step 2 Completed: 10 detailed analyses performed.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import os

def run_modeling_fallback(df):
    os.makedirs('plots', exist_ok=True)
    
    # カテゴリカル変数のエンコーディング
    df_model = df.copy()
    le = LabelEncoder()
    cat_cols = df_model.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'Attrition':
            df_model[col] = le.fit_transform(df_model[col].astype(str))
    
    # 特徴量とターゲットの分離
    X = df_model.drop(columns=['Attrition', 'Attrition_Flag'])
    y = df_model['Attrition_Flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
    
    # モデル構築 (RandomForest)
    print("Training RandomForest...")
    rf = RandomForestClassifier(random_state=123)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # 重要因子の可視化 (図14)
    importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
    plt.title('図14: 特徴量重要度 (Random Forest)')
    plt.tight_layout()
    plt.savefig('plots/fig14_feature_importance.png')
    plt.close()
    
    # 混同行列 (図15)
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['在職', '離職']).plot(cmap='Blues')
    plt.title('図15: 混同行列')
    plt.savefig('plots/fig15_confusion_matrix.png')
    plt.close()
    
    # 評価レポートの保存
    report = classification_report(y_test, y_pred)
    with open('model_summary.txt', 'w') as f:
        f.write(f"Best Parameters: {grid_search.best_params_}\n")
        f.write(f"Classification Report:\n{report}\n")

if __name__ == "__main__":
    df = pd.read_csv("/home/ubuntu/project/preprocessed_data.csv")
    run_modeling_fallback(df)
    print("Step 3 Fallback Completed: Modeling and evaluation finished.")
