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

# --- 2. VADER 情感分析 (原始方法) ---
print("--- 2. VADER 情感分析 ---")
# 初始化 VADER
analyzer = SentimentIntensityAnalyzer()

# 定义函数来获取 VADER 分数和情感标签
def get_vader_sentiment(text):
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']
    if compound_score >= 0.05:
        return compound_score, 'positive'
    elif compound_score <= -0.05:
        return compound_score, 'negative'
    else:
        # 注意：VADER可以输出'neutral'，但IMDB数据集只有 'positive'/'negative'
        # 为了与原始标签比较，我们可能需要强制将 neutral 归类
        # 这里我们暂时保留 VADER 的原始判断，但在计算准确率时只与 'positive'/'negative' 比较
        return compound_score, 'neutral'

# 应用函数
results = df['review_cleaned'].apply(get_vader_sentiment)
df['vader_score'] = [res[0] for res in results]
df['vader_sentiment_raw'] = [res[1] for res in results] # 保留原始VADER输出 (pos/neg/neu)

# 为了与原始标签比较，将 VADER 的 neutral 结果视为不匹配（或根据需要分配）
# 这里我们创建一个仅包含 pos/neg 的列用于比较
df['vader_sentiment_compare'] = df['vader_sentiment_raw'].apply(lambda x: x if x in ['positive', 'negative'] else 'neutral_vader')

# 计算 VADER 的准确率 (仅对比 positive/negative)
# 过滤掉 VADER 判断为 neutral 的情况进行比较
vader_comparable = df[df['vader_sentiment_compare'] != 'neutral_vader']
correct_predictions_vader = (vader_comparable['sentiment'] == vader_comparable['vader_sentiment_compare']).sum()
total_comparable_vader = len(vader_comparable)
accuracy_vader = correct_predictions_vader / total_comparable_vader if total_comparable_vader > 0 else 0

print("VADER 分析完成。")
print(f"VADER 准确率 (仅比较 positive/negative 预测): {accuracy_vader:.2%}")
print(f"({correct_predictions_vader} 正确预测，共 {total_comparable_vader} 条 VADER 非中性评论)")

# 生成 VADER 混淆矩阵所需的数据 (只考虑 pos/neg vs pos/neg)
vader_true_labels = vader_comparable['sentiment']
vader_pred_labels = vader_comparable['vader_sentiment_compare']
vader_cm = confusion_matrix(vader_true_labels, vader_pred_labels, labels=['positive', 'negative'])

print("-" * 30)

# --- 3. SentiML 细粒度标注框架 ---
print("--- 3. SentiML 细粒度标注框架 ---")

# 定义SentiML特征提取器
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

print("   3.1 准备SentiML特征提取器...")
sentiml_extractor = SentiMLFeatureExtractor()

# 3.2 数据准备
print("   3.2 数据清理和特征准备...")
# 创建sentiment_label列
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 检查并移除包含NaN的行
nan_count_before = df['sentiment_label'].isna().sum()
if nan_count_before > 0:
    print(f"   发现 {nan_count_before} 条带有NaN的记录")
    df = df.dropna(subset=['sentiment_label', 'review_cleaned'])
    print(f"   清理后剩余 {len(df)} 条记录")

# 3.3 训练集与测试集划分
print("   3.3 划分训练集和测试集...")
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

# 3.4 提取SentiML特征
print("   3.4 提取SentiML特征...")
X_train_sentiml = sentiml_extractor.transform(X_train_text)
X_test_sentiml = sentiml_extractor.transform(X_test_text)
print(f"   SentiML特征提取完成，每个样本提取了 {X_train_sentiml.shape[1]} 个特征")

# 3.5 训练SentiML模型
print("   3.5 训练SentiML模型...")
sentiml_model = LogisticRegression(max_iter=1000)
sentiml_model.fit(X_train_sentiml, y_train)
print("   SentiML模型训练完成")

# 3.6 评估SentiML模型
print("   3.6 评估SentiML模型...")
y_pred_sentiml = sentiml_model.predict(X_test_sentiml)
accuracy_sentiml = accuracy_score(y_test, y_pred_sentiml)
sentiml_cm = confusion_matrix(y_test, y_pred_sentiml)
sentiml_report = classification_report(y_test, y_pred_sentiml, target_names=['negative', 'positive'])

print(f"SentiML模型在测试集上的准确率: {accuracy_sentiml:.2%}")
print("SentiML模型分类报告:\n", sentiml_report)

# 3.7 特征重要性分析
print("   3.7 SentiML特征重要性分析...")
feature_importance = pd.DataFrame({
    'feature': X_train_sentiml.columns,
    'importance': abs(sentiml_model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("最重要的5个特征:")
print(feature_importance.head(5))

print("-" * 30)

# --- 4. 机器学习流程 (TF-IDF + 逻辑回归) ---
print("--- 4. 机器学习流程 (TF-IDF + 逻辑回归) ---")

# 4.1 文本预处理 (为 ML 模型准备)
print("   4.1 文本预处理 (小写、去标点、分词、去停用词、词形还原)...")
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

# 应用预处理函数 (这步可能比较耗时)
df['review_processed_ml'] = df['review_cleaned'].apply(preprocess_text_ml)
print("   文本预处理完成。")

# 4.2 特征提取 (TF-IDF)
print("   4.2 特征提取 (TF-IDF + n-gram)...")
# 添加n-gram特征 (1-gram和2-gram的组合)
vectorizer = TfidfVectorizer(
    max_features=5000,  # 限制特征数量，防止维度过高
    ngram_range=(1, 2),  # 使用1-gram和2-gram
    min_df=5,           # 至少在5个文档中出现
    max_df=0.9          # 在超过90%的文档中出现的词被视为停用词
)

# 使用相同的训练集和测试集划分
X_train_ml = train_df['review_processed_ml'] if len(df) >= 50000 else X_train_text.apply(preprocess_text_ml)
X_test_ml = test_df['review_processed_ml'] if len(df) >= 50000 else X_test_text.apply(preprocess_text_ml)

# 在训练集上拟合 TF-IDF 并转换训练集和测试集
X_train_tfidf = vectorizer.fit_transform(X_train_ml)
X_test_tfidf = vectorizer.transform(X_test_ml)
print(f"   TF-IDF 特征提取完成。特征维度: {X_train_tfidf.shape[1]}")

# 获取一些顶部的n-gram特征
feature_names = vectorizer.get_feature_names_out()
print("   部分n-gram特征示例:")
# 随机选择10个特征展示
import random
sample_features = random.sample(list(feature_names), min(10, len(feature_names)))
print("   ", sample_features)

# 4.3 模型构建 (逻辑回归)
print("   4.3 模型训练 (逻辑回归)...")
model_ml = LogisticRegression(max_iter=1000) # 增加迭代次数以确保收敛
model_ml.fit(X_train_tfidf, y_train)
print("   模型训练完成。")

# 4.4 模型评估
print("   4.4 模型评估...")
y_pred_ml = model_ml.predict(X_test_tfidf)
accuracy_ml = accuracy_score(y_test, y_pred_ml)
ml_cm = confusion_matrix(y_test, y_pred_ml) # 标签是 1/0
ml_report = classification_report(y_test, y_pred_ml, target_names=['negative', 'positive']) # target_names 对应 0 和 1

print(f"TF-IDF + 逻辑回归模型在测试集上的准确率: {accuracy_ml:.2%}")
print("TF-IDF + 逻辑回归模型分类报告:\n", ml_report)
print("-" * 30)




# --- 5. 模型比较 ---
print("--- 5. 模型比较 ---")
print(f"VADER 准确率: {accuracy_vader:.2%}")
print(f"SentiML 准确率: {accuracy_sentiml:.2%}")
print(f"TF-IDF + 逻辑回归准确率: {accuracy_ml:.2%}")

# 创建比较表格
comparison_df = pd.DataFrame({
    'Model': ['VADER', 'SentiML', 'TF-IDF + Logistic Regression'],
    'Accuracy': [accuracy_vader, accuracy_sentiml, accuracy_ml]
})
print("\n模型比较表格:")
print(comparison_df)
print("-" * 30)



# --- 6. 可视化 ---
print("--- 6. 可视化 ---")

# 创建保存图像的目录
import os
if not os.path.exists('visualization_results'):
    os.makedirs('visualization_results')
    print("创建了保存可视化结果的目录: visualization_results/")

plt.style.use('seaborn-v0_8-darkgrid') # 使用 seaborn 风格

# 6.1 模型准确率比较图
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=comparison_df)
plt.title('Accuracy Comparison of Different Sentiment Analysis Models')
plt.ylim(0, 1)
for i, v in enumerate(comparison_df['Accuracy']):
    plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
plt.tight_layout()
plt.savefig('visualization_results/model_comparison.png', dpi=300)
plt.show()

# 6.2 VADER 分数分布图
plt.figure(figsize=(10, 5))
sns.histplot(df['vader_score'], kde=True, bins=30)
plt.title('VADER Compound Score Distribution (Full Dataset)')
plt.xlabel('VADER Compound Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('visualization_results/vader_score_distribution.png', dpi=300)
plt.show()

# 6.3 VADER 混淆矩阵
plt.figure(figsize=(6, 5))
sns.heatmap(vader_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.title('VADER Confusion Matrix (Positive/Negative Predictions Only)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('visualization_results/vader_confusion_matrix.png', dpi=300)
plt.show()

# 6.4 SentiML 混淆矩阵
# 将 0/1 标签映射回 negative/positive 用于绘图
sentiml_cm_labels = confusion_matrix(y_test.map({1: 'positive', 0: 'negative'}),
                                pd.Series(y_pred_sentiml).map({1: 'positive', 0: 'negative'}),
                                labels=['positive', 'negative'])
plt.figure(figsize=(6, 5))
sns.heatmap(sentiml_cm_labels, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.title('SentiML Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('visualization_results/sentiml_confusion_matrix.png', dpi=300)
plt.show()

# 6.5 TF-IDF + 逻辑回归混淆矩阵
# 将 0/1 标签映射回 negative/positive 用于绘图
ml_cm_labels = confusion_matrix(y_test.map({1: 'positive', 0: 'negative'}),
                                pd.Series(y_pred_ml).map({1: 'positive', 0: 'negative'}),
                                labels=['positive', 'negative'])
plt.figure(figsize=(6, 5))
sns.heatmap(ml_cm_labels, annot=True, fmt='d', cmap='Greens',
            xticklabels=['positive', 'negative'], yticklabels=['positive', 'negative'])
plt.title('TF-IDF + Logistic Regression Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('visualization_results/ml_confusion_matrix.png', dpi=300)
plt.show()

# 6.6 SentiML 特征重要性可视化
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 10 Most Important Features in SentiML Model')
plt.xlabel('Feature Importance (Absolute Coefficient Value)')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.savefig('visualization_results/sentiml_feature_importance.png', dpi=300)
plt.show()

# 6.7 词云
print("   生成词云...")

# 合并所有正面和负面评论文本 (使用清理过的文本，但未做ML的过度处理，保留更多词汇)
positive_text = ' '.join(df[df['sentiment'] == 'positive']['review_cleaned'])
negative_text = ' '.join(df[df['sentiment'] == 'negative']['review_cleaned'])

# 创建词云对象
wordcloud_positive = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(positive_text)
wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds', collocations=False).generate(negative_text) # collocations=False 避免重复短语

# 显示词云
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud', fontsize=16)

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud', fontsize=16)

plt.tight_layout(pad=2.0)
plt.savefig('visualization_results/wordclouds.png', dpi=300)
plt.show()
print("   词云生成完成。")
print("   所有可视化结果已保存到 visualization_results/ 目录")
print("-" * 30)

print("=== 分析完成 ===")