import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set font for Japanese characters
plt.rcParams['font.family'] = 'Noto Sans CJK JP'

def analyze_age_trends():
    df_husband_age = pd.read_csv("/home/ubuntu/wedding_analysis/mg120001.csv", encoding='cp932', skiprows=11)
    df_wife_age = pd.read_csv("/home/ubuntu/wedding_analysis/mg120002.csv", encoding='cp932', skiprows=11)
    
    df_husband_age = df_husband_age.rename(columns={df_husband_age.columns[0]: 'Prefecture'})
    df_wife_age = df_wife_age.rename(columns={df_wife_age.columns[0]: 'Prefecture'})
    
    latest_year = '2023'
    df_husband_2023 = df_husband_age[['Prefecture', latest_year]].dropna()
    df_wife_2023 = df_wife_age[['Prefecture', latest_year]].dropna()
    
    df_husband_pref = df_husband_2023[df_husband_2023['Prefecture'] != '全　国'].copy()
    df_wife_pref = df_wife_2023[df_wife_2023['Prefecture'] != '全　国'].copy()
    
    df_husband_pref[latest_year] = pd.to_numeric(df_husband_pref[latest_year], errors='coerce')
    df_wife_pref[latest_year] = pd.to_numeric(df_wife_pref[latest_year], errors='coerce')
    
    df_husband_top = df_husband_pref.sort_values(by=latest_year, ascending=False).head(10)
    df_wife_top = df_wife_pref.sort_values(by=latest_year, ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=latest_year, y='Prefecture', data=df_husband_top, palette='Blues_r')
    plt.title('夫の平均初婚年齢が高い上位10都道府県 (2023)')
    plt.xlabel('年齢')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=latest_year, y='Prefecture', data=df_wife_top, palette='Reds_r')
    plt.title('妻の平均初婚年齢が高い上位10都道府県 (2023)')
    plt.xlabel('年齢')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/wedding_analysis/avg_age_by_pref.png')
    plt.close()
    
    return df_husband_pref, df_wife_pref

def analyze_marriage_counts():
    df_counts = pd.read_csv("/home/ubuntu/wedding_analysis/12-25 保管第15表（12月分）.csv", encoding='cp932', skiprows=5)
    df_counts.columns = ['AgeGroup', 'Total', 'Husband_First_Wife_First', 'Husband_First_Wife_Remarried', 
                         'Husband_First_Total', 'Husband_First_Wife_First_2', 'Husband_First_Wife_Remarried_2',
                         'Husband_Remarried_Total', 'Husband_Remarried_Wife_First', 'Husband_Remarried_Wife_Remarried',
                         'Wife_Total', 'Wife_Husband_First', 'Wife_Husband_Remarried',
                         'Wife_First_Total', 'Wife_First_Husband_First', 'Wife_First_Husband_Remarried',
                         'Wife_Remarried_Total', 'Wife_Remarried_Husband_First', 'Wife_Remarried_Husband_Remarried']
    
    df_counts = df_counts[df_counts['AgeGroup'].notna()]
    df_counts = df_counts[df_counts['AgeGroup'] != '総数']
    df_counts['Total'] = pd.to_numeric(df_counts['Total'], errors='coerce')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='AgeGroup', y='Total', data=df_counts, palette='viridis')
    plt.title('年齢階級別婚姻件数 (2025年12月概数)')
    plt.xlabel('年齢階級')
    plt.ylabel('婚姻件数 (組)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/wedding_analysis/marriage_counts_by_age.png')
    plt.close()
    
    return df_counts

if __name__ == "__main__":
    h_pref, w_pref = analyze_age_trends()
    df_counts = analyze_marriage_counts()
    print("Analysis completed and charts saved.")
    
    # Regional Insight: High age = likely urban areas
    print("\nRegional Insights (Highest Average Age):")
    print("Husbands:", h_pref.sort_values(by='2023', ascending=False).head(5))
    print("Wives:", w_pref.sort_values(by='2023', ascending=False).head(5))
    
    # Age Insight: Peak marriage age
    peak_age = df_counts.sort_values(by='Total', ascending=False).iloc[0]
    print(f"\nPeak Marriage Age Group: {peak_age['AgeGroup']} with {peak_age['Total']} marriages.")
