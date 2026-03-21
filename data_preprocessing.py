#!/usr/bin/env python
# coding: utf-8

# # 数据加载与初步探索

# In[1]:


import pandas as pd
import numpy as np


# In[44]:


data = pd.read_csv('train.csv')
data.head()


# ### 数据形状

# In[5]:


data.shape


# In[6]:


data_columns = data.columns.tolist()
data_columns


# ### 数据类型

# In[7]:


# 根据.info()的类型分析结果初步进行类型划分

data.info()


# In[45]:


numerical_cols = data.select_dtypes(include = ['int64','float64']).columns.tolist()
categorical_cols = data.select_dtypes(include = ['object']).columns.tolist()

print('数值型变量')
print(numerical_cols)
print('分类变量')
print(categorical_cols)


# In[93]:


# 根据以上结果，手动划分变量

target_cols = 'isDefault'
id_cols = 'id'
date_cols = ['issueDate','earliesCreditLine']

categorical_cols = ['homeOwnership','verificationStatus',
                   'purpose','regionCode','applicationType',
                   'initialListStatus','postCode','policyCode','title','employmentTitle']
ordinal_cols = ['term','grade','subGrade','employmentLength']

# 捕捉到employmentLength可能是职业代码编号，后续要特殊处理

numerical_cols_df = [col for col in data.columns
                 if col not in date_cols + categorical_cols + ordinal_cols + [target_cols,id_cols]]

# 因为每当修改dataframe之后numerical_cols都会变动，故创建新的列表用于后续数据处理
# 每次创建新的变量都在这里手动添加并删除旧列
numerical_cols = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'delinquency_2years', 
 'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 'revolBal', 'revolUtil', 
 'totalAcc', 
 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14'
                 ,'dti_cleaned','credit_history_years']


print('目标变量：',target_cols)
print('ID变量：',id_cols)
print('日期变量：',date_cols)
print('分类变量：',f'{len(categorical_cols)}个')
print(categorical_cols)
print('有序变量：')
print(ordinal_cols)
print('数值型变量：',f'{len(numerical_cols)}个')
print(numerical_cols)


# In[91]:


print((len(date_cols) + len(categorical_cols) + len(ordinal_cols) + len(numerical_cols) + 2)==47)  


# In[81]:


# 发现可以通过日期变量： ['issueDate', 'earliesCreditLine']构建新的变量，即信用历史长度，是很好的指标

# 'issueDate' 处理
data['issueDate_clean'] = pd.to_datetime(data['issueDate']) # 把原来字符串形式的日期，转换成pandas能识别的真正日期格式
data[['issueDate_clean','issueDate']].head(10)
data['issueYear'] = data['issueDate_clean'].dt.year # 提取年份和月份
data['issueMonth'] = data['issueDate_clean'].dt.month
data[['issueYear','issueMonth','issueDate']].head(10)


# In[82]:


# 'earliesCreditLine' 处理

data['earliesCreditLine'].head(10)


# In[83]:


data['earliesCreditLine_clean'] = pd.to_datetime(data['earliesCreditLine'], format = '%b-%Y')
# %b：英文缩写月份，比如 Jan, Feb, Aug, %Y：四位年份，比如 2001
data[['earliesCreditLine_clean','earliesCreditLine']].head(10)


# In[84]:


data['earliesYear'] = data['earliesCreditLine_clean'].dt.year
data['earliesMonth'] = data['earliesCreditLine_clean'].dt.month
data[['earliesYear','earliesMonth','earliesCreditLine']].head(10)


# In[94]:


# 构建信用历史长度'credit_history_years'

data['credit_history_years'] = data['issueYear'] - data['earliesYear'] 
data[['issueDate_clean','earliesCreditLine_clean','credit_history_years']].head(10)

# 将'credit_history_years'放入数值型变量
numerical_cols.append('credit_history_years')
len(numerical_cols)


# ### 缺失值统计

# In[11]:


# 统计缺失值数量和比例

missing_data = pd.DataFrame({
    'Missing Count': data.isnull().sum(),
    'Missing Ratio': data.isnull().mean()
}).sort_values('Missing Ratio', ascending = False)

missing_data.head(20)


# In[13]:


# 缺失值统计可视化

import matplotlib.pyplot as plt
import seaborn as sns

missing_ratio = data.isnull().mean().sort_values(ascending = False)
missing_ratio = missing_ratio[missing_ratio > 0]

plt.figure(figsize = (20,8))
sns.barplot(x = missing_ratio.index, y = missing_ratio.values)
plt.title('Data Missing Ratio')
plt.xlabel('Numerical Features')
plt.ylabel('Missing Ratio')
plt.show()


# 解释：上述结果表明，单纯对于缺失值进行删除，并不是理想的选择。
# 考虑数值型变量用中位数代替，分类变量用众数代替，因此最好是在训练集和测试集划分之后归于preprocessor中，避免样本数据清洗从测试集数据中学习分布。

# ## 阶段二：数据预处理

# ### 任务2.3:重复值处理与数据一致性检查

# In[14]:


# 重复值检测

data.duplicated().sum()


# In[15]:


data['id'].duplicated().sum()


# ### 任务2.2:异常值检测与处理(箱线图、3σ原则、业务规则)

# In[58]:


# 异常值检测

data[numerical_cols].describe().T


# In[50]:


# 业务规则检测

print('dti < 0:',f"{(data['dti'] < 0).sum()}个")
print('dti < 0:',data[data['dti'] < 0]['dti'].value_counts())

print('dti > 100:',f"{(data['dti'] > 100).sum()}个")
print('dti > 100:',data[data['dti'] > 100]['dti'].value_counts().head(20))


# In[100]:


# 直方图

for col in numerical_cols:
    plt.figure(figsize = (8,5))
    sns.histplot(data[col], bins = 50, kde = True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[59]:


# 因为大部分数据呈现偏态（右偏），考虑用IQR检测（代替3σ原则）

result = []
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outlier_count = data[(data[col] < lower) | (data[col] > upper)][col].count()
    outlier_ratio = outlier_count / data[col].notnull().sum()
    
    result.append({
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower': lower,
        'upper': upper,
        'outlier_count': outlier_count,
        'outlier_ratio': outlier_ratio,
    })

IQR_check_df = pd.DataFrame(result, index = numerical_cols).sort_values('outlier_ratio', ascending = False)
IQR_check_df


# 解释：
# 
# 1. dti 需要进行修正
# 2. annualIncome, revolBal , revolUtil , installment , loanAmnt 先保留，后续考虑对数变换
# 3. ficoRangeLow, ficoRangeHigh, totalAcc 保留

# In[101]:


# 异常值处理
# (1) dti 异常值改为缺失值
data['dti_cleaned'] = data['dti'].copy()
data.loc[data['dti_cleaned'] < 0,['dti_cleaned']] = np.nan
data.loc[data['dti_cleaned'] == 999, ['dti_cleaned']] = np.nan


# In[102]:


print((data['dti_cleaned'] < 0).sum())
print((data['dti_cleaned'] == 999).sum())


# 解释：初步的变量分类做了以下改动：
# 1. numerical_cols 新增'credit_history_years'
# 2. numerical_cols 将'dti'变为'dti_cleaned'

# ### 任务2.4:探索性数据分析(分布分析、相关性初步分析)

# In[23]:


# 虽然已经识别出isDefault是目标变量，但是无法判断0，1代表的含义，故通过探索性分析进行推测

# 因变量分布

counts = data['isDefault'].value_counts()
ratio = data['isDefault'].value_counts(normalize = True)

print(f'样本（0）：{counts[0]}，占比{ratio[0]:.2%}')
print(f'样本（1）：{counts[1]}，占比{ratio[1]:.2%}')


# In[28]:


# 假设'interestRate'越高，越容易违约

plt.figure(figsize = (8,5))
sns.boxplot(x = 'isDefault', y = 'interestRate', data = data)
plt.title('interestRate by isDefault')
plt.xlabel('isDefault')
plt.ylabel('interestRate')
plt.show()


# In[25]:


data.groupby('isDefault')['interestRate'].agg(['count','mean','median','max','min'])


# 解释：因为1组的利率平均高于0组，初步判断1为违约

# In[26]:


#ficoRangeLow

data.groupby('isDefault')['ficoRangeLow'].agg(['count','mean','median','max','min'])


# In[104]:


#dti_cleaned

data.groupby('isDefault')['dti_cleaned'].agg(['count','mean','median','max','min'])


# 解释：（考虑做一下假设检验，不够严谨）
# 
# 1组借款人FICO信用评分区间的下界明显低于0组，加强了对于1为违约组判断的验证
# 
# 1组dti（债务收入比）明显高于0组，加强了对于1为违约组判断的验证

# In[97]:


# 数值型数据EDA(去掉n0-n14)
key_numerical_cols = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti_cleaned', 'delinquency_2years', 
                      'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 
                      'revolBal', 'revolUtil', 'totalAcc','credit_history_years' 
                      ]

data.groupby('isDefault')[key_numerical_cols].agg(['mean','median']).T


# In[98]:


for col in key_numerical_cols:
    plt.figure(figsize = (8,5))
    sns.boxplot(x = 'isDefault', y = col, data = data)
    plt.title(f'{col} by isDefault')
    plt.xlabel('isDefault')
    plt.ylabel(col)
    plt.show()


# In[99]:


# 对数值变量做相关性分析，画热力图

corr_cols = ['loanAmnt', 'interestRate', 'installment', 'annualIncome', 'dti_cleaned', 'delinquency_2years', 
                      'ficoRangeLow', 'ficoRangeHigh', 'openAcc', 'pubRec', 'pubRecBankruptcies', 
                      'revolBal', 'revolUtil', 'totalAcc','credit_history_years' 
                      ,'isDefault']

plt.figure(figsize = (10,8))
corr_matrix = data[corr_cols].corr()
sns.heatmap(corr_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm')
plt.title('Correlation Map of Key Numerical Data')
plt.show()


# In[75]:


# 分类数据EDA
# 分类变量： 10个
# ['homeOwnership', 'verificationStatus', 'purpose', 'regionCode', 'applicationType', 'initialListStatus', 'postCode', 'policyCode', 'title', 'employmentTitle']
# 有序变量：4个
# ['term', 'grade', 'subGrade', 'employmentLength']

# 首先要对'employmentLength'进行清洗
data['employmentLength'].value_counts(dropna = False)


# In[76]:


def clean_employmentLength(x):
    if pd.isna(x):
        return np.nan
    elif x == '10+ years':
        return 10
    elif x == '< 1 year':
        return 0
    else:
        return int(str(x).replace('years','').replace('year',''))
        
            
data['employmentLength_clean'] = data['employmentLength'].apply(clean_employmentLength)
data[['employmentLength_clean','employmentLength']].head(10)

ordinal_cols = ['employmentLength_clean' if col == 'employmentLength' else col for col in ordinal_cols]
ordinal_cols


# In[78]:


# 有序变量EDA

for col in ordinal_cols:
    group_default_rate = data.groupby(col)['isDefault'].mean()
    plt.figure(figsize = (8,5))
    sns.barplot(x = group_default_rate.index, y = group_default_rate.values)
    plt.title(f'Default Rate by {col}')
    plt.xlabel(f'{col}')
    plt.ylabel(f'Default Rate')
    plt.show()


# In[79]:


# 无序分类变量(划分为)

cat_cols_small = ['homeOwnership', 'verificationStatus', 'applicationType', 'initialListStatus', 'policyCode']
cat_cols_high = ['purpose', 'regionCode', 'postCode', 'title','employmentTitle']

for col in cat_cols_small:
    plt.figure(figsize=(8,5))
    sns.countplot(x=data[col], order=data[col].value_counts().index)
    plt.title(f'Count Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

    default_rate = data.groupby(col)['isDefault'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=default_rate.index, y=default_rate.values)
    plt.title(f'Default Rate by {col}')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)
    plt.show()
    
for col in cat_cols_high:
    print(f'{col} unique values:', data[col].nunique())
    print(data[col].value_counts().head(10))

    top_n = data[col].value_counts().head(10).index
    temp = data[data[col].isin(top_n)]

    plt.figure(figsize=(10,5))
    sns.countplot(x=temp[col], order=temp[col].value_counts().index)
    plt.title(f'Top 10 Category Counts of {col}')
    plt.xticks(rotation=45)
    plt.show()

    default_rate = temp.groupby(col)['isDefault'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=default_rate.index, y=default_rate.values)
    plt.title(f'Default Rate by Top 10 {col}')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45)
    plt.show()

# 可以看出postcode类别非常多，regioncode类别中等, title


# In[ ]:


# 考虑之后对低基数的无序分类变量做独热编码
# 考虑对['purpose', 'regionCode', 'postCode', 'title','employmentTitle']高基数分类变量的几种处理策略：
# 1.合并低频数类别后做独热编码 2.频数编码 3.目标编码 4.直接删除


# 问题：
# 第一张图是根据EDA得出的结论，第二张图是异常值检测时考虑用对数变换。和分箱、多项式有什么实质差异吗？没有的话，我右偏的变量都做对数变换好了？
# 编码是在划分前还是后做好？（不同编码方式之间，在划分前后做编码有什么差异？）
# 

# In[ ]:





# In[ ]:




