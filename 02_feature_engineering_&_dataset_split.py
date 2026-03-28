#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data = pd.read_csv('train.csv')
data.head()


# In[2]:


raw_model_df = data.copy()


# In[4]:


data_columns = raw_model_df.columns.tolist()
data_columns


# In[5]:


data['issueDate_clean'] = pd.to_datetime(data['issueDate']) # 把原来字符串形式的日期，转换成pandas能识别的真正日期格式

data['issueYear'] = data['issueDate_clean'].dt.year # 提取年份和月份
data['issueMonth'] = data['issueDate_clean'].dt.month
data[['issueYear','issueMonth','issueDate']].head(10)

data['earliesCreditLine_clean'] = pd.to_datetime(data['earliesCreditLine'], format = '%b-%Y')

data['earliesYear'] = data['earliesCreditLine_clean'].dt.year
data['earliesMonth'] = data['earliesCreditLine_clean'].dt.month
data[['earliesYear','earliesMonth','earliesCreditLine']].head(10)

# 构建信用历史长度'credit_history_years'

raw_model_df['credit_history_years'] = data['issueYear'] - data['earliesYear']


# In[6]:


raw_model_df['dti_cleaned'] = data['dti'].copy()
raw_model_df.loc[raw_model_df['dti_cleaned'] < 0,['dti_cleaned']] = np.nan
raw_model_df.loc[raw_model_df['dti_cleaned'] == 999, ['dti_cleaned']] = np.nan
raw_model_df = raw_model_df.drop(columns = 'dti', errors = 'ignore')


# In[7]:


def clean_employmentLength(x):
    if pd.isna(x):
        return np.nan
    elif x == '10+ years':
        return 10
    elif x == '< 1 year':
        return 0
    else:
        return int(str(x).replace('years','').replace('year',''))
    
raw_model_df['employmentLength_clean'] = raw_model_df['employmentLength'].apply(clean_employmentLength)
raw_model_df = raw_model_df.drop(columns = 'employmentLength', errors = 'ignore')


# In[8]:


raw_model_df.shape


# In[9]:


drop_cols = ['id','issueDate','earliesCreditLine',
             'ficoRangeHigh',
             'initialListStatus',
            'policyCode',
            'title',
            'employmentTitle',
            'grade']
raw_model_df = raw_model_df.drop(columns = drop_cols, errors = 'ignore')


# In[10]:


raw_model_df.shape


# In[11]:


type(raw_model_df['subGrade'])


# In[12]:


subgrade_order = {}
letters = ['A','B','C','D','E','F','G']
idx = 1
for letter in letters:
    for num in range(1,6):
        subgrade_order[f'{letter}{num}'] = idx
        idx += 1
raw_model_df['subGrade_coded'] = raw_model_df['subGrade'].map(subgrade_order)
raw_model_df = raw_model_df.drop(columns = 'subGrade', errors = 'ignore')


# In[13]:


raw_model_df.shape


# In[14]:


def ratio_calculatior(num, den):
    return np.where((den.isna()) | (den == 0), np.nan, num/den)

raw_model_df['installment_income_ratio'] = ratio_calculatior(data['installment'],data['annualIncome'])
raw_model_df['loan_income_ratio'] = ratio_calculatior(data['loanAmnt'],data['annualIncome'])


# In[15]:


raw_model_df.shape


# In[16]:


47-8+2


# In[17]:


raw_model_df.dtypes


# # 拆分变量

# In[18]:


x = raw_model_df.drop(columns = 'isDefault')
y = raw_model_df['isDefault']

type(y)


# In[19]:


from sklearn.model_selection import train_test_split

x_temp, x_test, y_temp, y_test = train_test_split(x, y,
                                                   test_size = 0.1,
                                                 stratify = y,
                                                 random_state = 42)
x_train, x_valid, y_train, y_valid = train_test_split(x_temp, y_temp,
                                                     test_size = 2/9,
                                                     stratify = y_temp,
                                                     random_state = 42)


# In[20]:


# 检查划分结果
print('x_train:',x_train.shape)
print('x_valid:',x_valid.shape)
print('x_test:',x_test.shape)
print("train:", y_train.mean())
print("valid:", y_valid.mean())
print("test :", y_test.mean())


# In[23]:


# 保存划分结果
import os
os.makedirs('saved_models', exist_ok = True)
import joblib

joblib.dump(x_train, "saved_models/x_train.pkl")
joblib.dump(x_test, "saved_models/x_test.pkl")
joblib.dump(y_train, "saved_models/y_train.pkl")
joblib.dump(y_test, "saved_models/y_test.pkl")
joblib.dump(x_valid, "saved_models/x_valid.pkl")
joblib.dump(y_valid, "saved_models/y_valid.pkl")


# In[24]:


# 手动划分变量类型：


ordinal_categorical_features = ['term', 'subGrade_coded', 'employmentLength_clean']

nominal_categorical_features = ['homeOwnership', 'verificationStatus', 'purpose',
                               'regionCode', 'applicationType', 
                               'postCode']

discrete_numerical_features = ['delinquency_2years','pubRec','pubRecBankruptcies']

serial_numerical_features = ['loanAmnt','interestRate','installment','annualIncome',
                            'dti_cleaned','ficoRangeLow','openAcc','revolBal','revolUtil','totalAcc',
                            'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 
                             'n11', 'n12', 'n13', 'n14',
                            'credit_history_years','installment_income_ratio',
                            'loan_income_ratio']

len(ordinal_categorical_features+nominal_categorical_features+discrete_numerical_features+serial_numerical_features)


# In[25]:


# 定义不同的preprocessor组

# 低基数无序分类变量(独热编码)
low_card_cat_features = ['homeOwnership', 'verificationStatus','applicationType']

# 高基数无序分类变量(独热编码)
high_card_cat_simple_features = ['purpose']

# 高基数无序分类变量(先合并低频类别，再独热编码)
high_card_cat_rare_features = ['regionCode','postCode']

# 连续型数值变量（对数变换）
log_serial_num_features = ['annualIncome', 'installment']

# 连续型数值变量（99分位数截尾）
winsor_serial_num_features = ['loanAmnt', 'interestRate', 'dti_cleaned', 'revolUtil']

remaining_num_features = [i for i in discrete_numerical_features+serial_numerical_features if i not in 
                          log_serial_num_features+winsor_serial_num_features]+ordinal_categorical_features


# In[26]:


# 构建合并低频变量处理器：

from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    def __init__ (self, min_freq = 0.01, other_label = 'Other'):
        self.min_freq = min_freq
        self.other_label = other_label
        self.frequent_categories_ = {}
        
    def fit(self, x, y = None):
        x = pd.DataFrame(x).copy()
        for col in x.columns:
            freq = x[col].value_counts(normalize = True, dropna = True) # 统计出现频率，而非数值
            self.frequent_categories_[col] = set(freq[freq > self.min_freq].index)
        return self
    
    def transform(self, x):
        x = pd.DataFrame(x).copy()
        for col in x.columns:
            
            x[col] = x[col].where(x[col].isin(self.frequent_categories_.get(col,set())), self.other_label)
        return x 

# 将转化后的列都变为字符串
from sklearn.preprocessing import FunctionTransformer
def to_string_func(x):
    return pd.DataFrame(x).astype(str)

to_string_transformer = FunctionTransformer(to_string_func,
                                           feature_names_out = 'one-to-one')


# In[27]:


# 构建99分位数截尾器：

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile = 0.00, upper_quantile = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None 
        self.upper_bounds_ = None
    
    def fit(self, x, y = None):
        x = pd.DataFrame(x).copy()
        self.lower_bounds_ = x.quantile(self.lower_quantile)
        self.upper_bounds_ = x.quantile(self.upper_quantile)
        return self
    def transform(self, x):
        x = pd.DataFrame(x).copy()
        x = x.clip(lower = self.lower_bounds_,
                  upper = self.upper_bounds_,
                  axis = 1)
        return x


# In[28]:


# 构建对数转换处理器：

from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log1p, feature_names_out = 'one-to-one')


# In[29]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 低基数无序分类变量(独热编码)
low_card_cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')), #缺失值填充（都在这里做，每一组都要）
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

# 高基数无序分类变量(独热编码)
high_card_cat_simple_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

# 高基数无序分类变量(先合并低频类别，再独热编码)
high_card_cat_rare_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('rare_grouper', RareCategoryGrouper(min_freq = 0.01)),
    ('to_string',to_string_transformer),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

# 连续型数值变量（对数变换）
log_serial_num_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('log', log_transformer)
])

# 连续型数值变量（99分位数截尾）
winsor_serial_num_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('winsor', Winsorizer(upper_quantile=0.99))
])

# 普通数值型+分类变量，只用填充缺失值
normal_num_transformer = Pipeline(steps= [
    ('imputer', SimpleImputer(strategy='median'))
])


# In[30]:


# 构建数据预处理器

preprocessor = ColumnTransformer(
    transformers = [
        ('normal_num_transformer', normal_num_transformer, remaining_num_features),
        ('low_card_cat_transformer', low_card_cat_transformer, low_card_cat_features),
        ('high_card_cat_simple_transformer', high_card_cat_simple_transformer, high_card_cat_simple_features),
        ('high_card_cat_rare_transformer', high_card_cat_rare_transformer, high_card_cat_rare_features),
        ('log_serial_num_transformer', log_serial_num_transformer, log_serial_num_features),
        ('winsor_serial_num_transformer', winsor_serial_num_transformer, winsor_serial_num_features)
    ],
    remainder = 'drop')


# In[ ]:




