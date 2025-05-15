# -*- coding: utf-8 -*-
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # 使用词形还原
import re
import string # 用于去除标点符号

# --- 机器学习相关库 ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# --- 可视化相关库 ---
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- NLTK 数据下载 ---
# 确保必要的 NLTK 数据已下载
def download_nltk_data():
    required_data = {
        'sentiment/vader_lexicon.zip': 'vader_lexicon',
        'corpora/stopwords.zip': 'stopwords',
        'tokenizers/punkt': 'punkt',
        'corpora/wordnet.zip': 'wordnet',
        'corpora/omw-1.4.zip': 'omw-1.4'
    }
    for path, pkg_id in required_data.items():
        try:
            nltk.data.find(path)
            print(f"'{pkg_id}' 已存在.")
        except LookupError:
            print(f"下载 NLTK 数据包: '{pkg_id}'...")
            nltk.download(pkg_id)

print("--- 检查和下载 NLTK 数据 ---")
download_nltk_data()
print("-" * 30)

# --- 1. 数据准备 ---
print("--- 1. 数据加载与初步清理 ---")
# 从CSV文件读取数据
df = pd.read_csv("IMDBDataset.csv")

# 移除HTML标签（例如 <br />）
df['review_cleaned'] = df['review'].apply(lambda x: re.sub(r'<br\s*/?>', ' ', x))
print(f"数据加载完成，共 {len(df)} 条评论。")
print("-" * 30)

# --- 2. 集成模型准备 ---
print("--- 2. 集成模型准备 ---")

# 2.1 初始化VADER分析器
print("   2.1 初始化VADER分析器...")
analyzer = SentimentIntensityAnalyzer()

# 2.2 创建sentiment_label列
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 检查并移除包含NaN的行
nan_count_before = df['sentiment_label'].isna().sum()
if nan_count_before > 0:
    print(f"   发现 {nan_count_before} 条带有NaN的记录")
    df = df.dropna(subset=['sentiment_label', 'review_cleaned'])
    print(f"   清理后剩余 {len(df)} 条记录")

# 2.3 训练集与测试集划分
print("   2.3 划分训练集和测试集...")
# 确保数据集有足够的数据
if len(df) >= 50000:
    # 使用前45000条作为训练集
    train_df = df.iloc[:45000]
    # 使用后5000条作为测试集
    test_df = df.iloc[45000:50000]
    
    X_train_text = train_df['review_cleaned']
    y_train = train_df['sentiment_label']
    X_test_text = test_df['review_cleaned']
    y_test = test_df['sentiment_label']
    
    print(f"   训练集大小: {len(X_train_text)}, 测试集大小: {len(X_test_text)}")
else:
    # 如果数据集不足50000条，则使用随机划分方法
    print(f"   警告：数据集只有{len(df)}条记录，少于预期的50000条")
    print(f"   使用随机划分方法代替...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['review_cleaned'], df['sentiment_label'], test_size=0.25, random_state=42, stratify=df['sentiment_label'])
    print(f"   训练集大小: {len(X_train_text)}, 测试集大小: {len(X_test_text)}")

# 2.4 计算VADER特征
print("   2.4 计算VADER特征...")
# 为所有评论计算VADER分数
df['vader_score'] = df['review_cleaned'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# 2.5 定义SentiML特征提取器
print("   2.5 定义SentiML特征提取器...")
class SentiMLFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        for text in X:
            # 基础文本特征
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                     if word not in self.stop_words and word not in string.punctuation]
            
            # 词汇特征
            text_length = len(text)
            word_count = len(tokens)
            avg_word_length = sum(len(word) for word in tokens) / max(1, word_count)
            
            # VADER情感分数
            vs = self.analyzer.polarity_scores(text)
            
            # 情感词汇特征
            positive_words = sum(1 for word in tokens if self.analyzer.polarity_scores(word)['compound'] > 0.5)
            negative_words = sum(1 for word in tokens if self.analyzer.polarity_scores(word)['compound'] < -0.5)
            
            # 句法特征
            exclamation_count = text.count('!')
            question_count = text.count('?')
            
            # 组合特征
            feature_dict = {
                'text_length': text_length,
                'word_count': word_count,
                'avg_word_length': avg_word_length,
                'vader_pos': vs['pos'],
                'vader_neg': vs['neg'],
                'vader_neu': vs['neu'],
                'vader_compound': vs['compound'],
                'positive_words': positive_words,
                'negative_words': negative_words,
                'positive_ratio': positive_words / max(1, word_count),
                'negative_ratio': negative_words / max(1, word_count),
                'exclamation_count': exclamation_count,
                'question_count': question_count
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)

# 2.6 提取SentiML特征
print("   2.6 提取SentiML特征...")
sentiml_extractor = SentiMLFeatureExtractor()
X_train_sentiml = sentiml_extractor.transform(X_train_text)
X_test_sentiml = sentiml_extractor.transform(X_test_text)
print(f"   SentiML特征提取完成，每个样本提取了 {X_train_sentiml.shape[1]} 个特征")

# 2.7 文本预处理 (为 TF-IDF 准备)
print("   2.7 文本预处理 (为 TF-IDF 准备)...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text_ml(text):
    # 转换为小写
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 去除数字 (可选，根据任务决定)
    text = re.sub(r'\d+', '', text)
    # 分词
    tokens = word_tokenize(text)
    # 去除停用词并进行词形还原
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1] # 保留长度大于1的词
    # 重新组合成字符串 (TF-IDF 需要字符串输入)
    return ' '.join(processed_tokens)

# 应用预处理函数
X_train_ml = X_train_text.apply(preprocess_text_ml)
X_test_ml = X_test_text.apply(preprocess_text_ml)
print("   文本预处理完成。")

# 2.8 TF-IDF特征提取
print("   2.8 TF-IDF特征提取...")
vectorizer = TfidfVectorizer(max_features=5000) # 限制特征数量，防止维度过高
X_train_tfidf = vectorizer.fit_transform(X_train_ml)
X_test_tfidf = vectorizer.transform(X_test_ml)
print("   TF-IDF 特征提取完成。")

# --- 3. 集成模型构建 ---
print("--- 3. 集成模型构建 ---")

# 3.1 准备集成特征
print("   3.1 准备集成特征...")

# 确定训练集和测试集的索引
if len(df) >= 50000:
    train_indices = train_df.index
    test_indices = test_df.index
else:
    train_indices = X_train_text.index
    test_indices = X_test_text.index

# 为训练集和测试集创建VADER特征
train_vader_features = pd.DataFrame({
    'vader_compound': df.loc[train_indices, 'vader_score'],
    'vader_pos': df.loc[train_indices].apply(lambda x: analyzer.polarity_scores(x['review_cleaned'])['pos'], axis=1),
    'vader_neg': df.loc[train_indices].apply(lambda x: analyzer.polarity_scores(x['review_cleaned'])['neg'], axis=1),
    'vader_neu': df.loc[train_indices].apply(lambda x: analyzer.polarity_scores(x['review_cleaned'])['neu'], axis=1)
})

test_vader_features = pd.DataFrame({
    'vader_compound': df.loc[test_indices, 'vader_score'],
    'vader_pos': df.loc[test_indices].apply(lambda x: analyzer.polarity_scores(x['review_cleaned'])['pos'], axis=1),
    'vader_neg': df.loc[test_indices].apply(lambda x: analyzer.polarity_scores(x['review_cleaned'])['neg'], axis=1),
    'vader_neu': df.loc[test_indices].apply(lambda x: analyzer.polarity_scores(x['review_cleaned'])['neu'], axis=1)
})

# 3.2 从TF-IDF中选择最重要的特征
print("   3.2 选择TF-IDF中最重要的特征...")
# 选择前1000个最重要的TF-IDF特征
selector = SelectKBest(chi2, k=1000)
X_train_tfidf_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_tfidf_selected = selector.transform(X_test_tfidf)

# 将稀疏矩阵转换为DataFrame
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [f'tfidf_{i}' for i in selected_feature_indices]
X_train_tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_train_tfidf_selected, 
                                                    index=train_indices,
                                                    columns=selected_feature_names)
X_test_tfidf_df = pd.DataFrame.sparse.from_spmatrix(X_test_tfidf_selected, 
                                                   index=test_indices,
                                                   columns=selected_feature_names)

# 3.3 组合所有特征
print("   3.3 组合所有特征...")
X_train_ensemble = pd.concat([X_train_sentiml.reset_index(drop=True), 
                             train_vader_features.reset_index(drop=True),
                             X_train_tfidf_df.reset_index(drop=True)], axis=1)

X_test_ensemble = pd.concat([X_test_sentiml.reset_index(drop=True), 
                            test_vader_features.reset_index(drop=True),
                            X_test_tfidf_df.reset_index(drop=True)], axis=1)

print(f"   集成特征维度: {X_train_ensemble.shape}")

# 3.4 训练集成模型
print("   3.4 训练集成模型...")
# 使用梯度提升决策树作为集成模型
ensemble_model = GradientBoostingClassifier(random_state=42)

# 简单的参数网格
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

# 使用网格搜索找到最佳参数
print("   执行网格搜索以找到最佳参数...")
grid_search = GridSearchCV(ensemble_model, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train_ensemble, y_train)

# 获取最佳模型
best_ensemble_model = grid_search.best_estimator_
print(f"   最佳参数: {grid_search.best_params_}")

# 3.5 评估集成模型
print("   3.5 评估集成模型...")
y_pred_ensemble = best_ensemble_model.predict(X_test_ensemble)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
ensemble_cm = confusion_matrix(y_test, y_pred_ensemble)
ensemble_report = classification_report(y_test, y_pred_ensemble, target_names=['negative', 'positive'])

print(f"集成模型在测试集上的准确率: {accuracy_ensemble:.2%}")
print("集成模型分类报告:\n", ensemble_report)

# 3.6 特征重要性分析
print("   3.6 集成模型特征重要性分析...")
feature_importance_ensemble = pd.DataFrame({
    'feature': X_train_ensemble.columns,
    'importance': best_ensemble_model.feature_importances_
})
feature_importance_ensemble = feature_importance_ensemble.sort_values('importance', ascending=False)
print("最重要的10个特征:")
print(feature_importance_ensemble.head(10))

# --- 4. 可视化 ---
print("--- 4. 可视化 ---")

# 创建保存图像的目录
import os
if not os.path.exists('visualization_results'):
    os.makedirs('visualization_results')
    print("创建了保存可视化结果的目录: visualization_results/")

plt.style.use('seaborn-v0_8-darkgrid') # 使用 seaborn 风格

# 4.1 集成模型混淆矩阵
ensemble_cm_labels = confusion_matrix(y_test.map({1: 'positive', 0: 'negative'}),
                                pd.Series(y_pred_ensemble).map({1: 'positive', 0: 'negative'}),
                                labels=['positive', 'negative'])
plt.figure(figsize=(6, 5))
sns.heatmap(ensemble_cm_labels, annot=True, fmt='d', cmap='Purples',
            xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.title('Ensemble Model Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('visualization_results/ensemble_confusion_matrix.png', dpi=300)
plt.show()

# 4.2 集成模型特征重要性可视化
plt.figure(figsize=(14, 10))
top_features = feature_importance_ensemble.head(20)
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 20 Important Features in Ensemble Model')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig('visualization_results/ensemble_feature_importance.png', dpi=300)
plt.show()

# 4.3 预测概率分布
plt.figure(figsize=(10, 6))
pred_proba = best_ensemble_model.predict_proba(X_test_ensemble)[:, 1]  # 获取正类的概率
sns.histplot(pred_proba, bins=50, kde=True)
plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')
plt.title('Ensemble Model Prediction Probability Distribution')
plt.xlabel('Probability of Positive Sentiment')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('visualization_results/ensemble_probability_distribution.png', dpi=300)
plt.show()

print("   所有可视化结果已保存到 visualization_results/ 目录")
print("-" * 30)

print("=== 分析完成 ===")