#!/usr/bin/env python
# coding: utf-8

# # 逻辑回归模型

# In[79]:


# 定义模型

from sklearn.linear_model import LogisticRegression

logit_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter = 1000,
                                     class_weight = 'balanced',
                                     random_state = 42))
])


# In[80]:


# 训练模型
logit_pipeline.fit(x_train, y_train)


# In[81]:


# 模型评估


# In[82]:


from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, confusion_matrix)

def model_evaluation(model, x, y, datasetname = 'dataset', threshold = 0.5):
    y_prob = model.predict_proba(x)[:,1]
    y_pred = (y_prob > threshold).astype(int)
    
    print(datasetname)
    print("Accuracy :", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall   :", recall_score(y, y_pred))
    print("F1       :", f1_score(y, y_pred))
    print("ROC-AUC  :", roc_auc_score(y, y_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))


# In[83]:



model_evaluation(logit_pipeline, x_train, y_train, datasetname = 'Train')
model_evaluation(logit_pipeline, x_valid, y_valid, datasetname = 'Valid')


# 评价：
# 
# AUC 还可以
# 
# 泛化稳定
# 
# Recall 较高
# 
# Precision 偏低

# In[84]:


# 阈值调整

y_valid_prob = logit_pipeline.predict_proba(x_valid)[:,1]
# 阈值评估函数

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
result = []
for t in thresholds:
    y_valid_pred = (y_valid_prob > t).astype(int)
    result.append({
        'Accuracy':accuracy_score(y_valid, y_valid_pred),
        'Precision':precision_score(y_valid, y_valid_pred),
        'Recall':recall_score(y_valid, y_valid_pred),
        'F1':f1_score(y_valid, y_valid_pred),
        'ROC-AUC':roc_auc_score(y_valid, y_valid_prob)
    })

threshold_classifier_df = pd.DataFrame(result,
                                      index = thresholds)
threshold_classifier_df.sort_values('F1', ascending = False)


# In[85]:


# 模型保存

import os
os.makedirs('saved_models', exist_ok = True)


# In[86]:


import joblib

joblib.dump(logit_pipeline, "saved_models/logit_pipeline.pkl")


# ## 逻辑回归模型性能可视化

# In[37]:


import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                            confusion_matrix, ConfusionMatrixDisplay)

y_valid_pred_04 = (y_valid_prob > 0.4).astype(int)


# In[38]:


# ROC 曲线

fpr, tpr, _ = roc_curve(y_valid, y_valid_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize = (6,5))
plt.plot(fpr, tpr, label = f'Logistic Regression (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc = 'lower right')
plt.show()


# In[39]:


# PR曲线

precision_curve, recall_curve, _ = precision_recall_curve(y_valid, y_valid_prob)
ap_score = average_precision_score(y_valid, y_valid_prob)

plt.figure(figsize = (6,5))
plt.plot(recall_curve, precision_curve, label = f'PR Curve (AP = {ap_score:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Logistic Regression')
plt.legend(loc = 'lower left')
plt.show()


# In[40]:


# 混淆矩阵

cm = confusion_matrix(y_valid, y_valid_pred_04)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(values_format = 'd')
plt.title('Confusion Matrix for Logistic Regression (threshold = 0.4)')
plt.show()


# ## 逻辑回归模型优化

# In[ ]:




# 连续型数值变量（对数变换）
log_serial_num_transformer_v2 = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('log', log_transformer),
    ('scaler', StandardScaler())
])

# 连续型数值变量（99分位数截尾）
winsor_serial_num_transformer_v2 = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='median')),
    ('winsor', Winsorizer(upper_quantile=0.99)),
    ('scaler', StandardScaler())
])

# 普通数值型+分类变量，只用填充缺失值
normal_num_transformer_v2 = Pipeline(steps= [
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

logit_preprocessor = ColumnTransformer(
    transformers = [
        ('normal_num_transformer', normal_num_transformer_v2, remaining_num_features),
        ('low_card_cat_transformer', low_card_cat_transformer, low_card_cat_features),
        ('high_card_cat_simple_transformer', high_card_cat_simple_transformer, high_card_cat_simple_features),
        ('high_card_cat_rare_transformer', high_card_cat_rare_transformer, high_card_cat_rare_features),
        ('log_serial_num_transformer', log_serial_num_transformer_v2, log_serial_num_features),
        ('winsor_serial_num_transformer', winsor_serial_num_transformer_v2, winsor_serial_num_features)
    ],
    remainder = 'drop')


# In[107]:


logit_pipeline_v2 = Pipeline(steps = [
    ('preprocessor', logit_preprocessor),
    ('classifier', LogisticRegression(max_iter = 1000,
                                     class_weight = 'balanced',
                                     random_state = 42))
])

logit_pipeline_v2.fit(x_train, y_train)


# In[108]:


model_evaluation(logit_pipeline_v2, x_train, y_train, datasetname = 'Train')
model_evaluation(logit_pipeline_v2, x_valid, y_valid, datasetname = 'Valid')


# In[111]:


# 阈值调整

y_valid_prob_logit_pipeline_v2 = logit_pipeline_v2.predict_proba(x_valid)[:,1]
# 阈值评估函数

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
result = []
for t in thresholds:
    y_valid_pred = (y_valid_prob > t).astype(int)
    result.append({
        'Accuracy':accuracy_score(y_valid, y_valid_pred),
        'Precision':precision_score(y_valid, y_valid_pred),
        'Recall':recall_score(y_valid, y_valid_pred),
        'F1':f1_score(y_valid, y_valid_pred),
        'ROC-AUC':roc_auc_score(y_valid, y_valid_prob)
    })

threshold_classifier_df = pd.DataFrame(result,
                                      index = thresholds)
threshold_classifier_df.sort_values('F1', ascending = False)


# In[110]:


joblib.dump(logit_pipeline_v2, "saved_models/logit_pipeline_v2.pkl")


# # 随机森林模型

# In[41]:


# 定义模型

from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators = 200,
                                         max_depth = 200,
                                         min_samples_split = 50,
                                         min_samples_leaf = 20,
                                         class_weight = 'balanced_subsample',
                                         random_state = 42,
                                         n_jobs = -1))
])

rf_pipeline.fit(x_train, y_train)


# In[42]:


# 模型评估

model_evaluation(rf_pipeline, x_train, y_train, datasetname = 'Train')
model_evaluation(rf_pipeline, x_valid, y_valid, datasetname = 'Valid')


# In[43]:


# 存在过拟合问题--调整模型参数

rf_pipeline_v2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline_v2.fit(x_train, y_train)


# In[44]:


model_evaluation(rf_pipeline_v2, x_train, y_train, datasetname = 'Train')
model_evaluation(rf_pipeline_v2, x_valid, y_valid, datasetname = 'Valid')


# In[88]:


# 仍选择第一版模型
# 保存模型

joblib.dump(logit_pipeline, "saved_models/rf_pipeline.pkl")


# In[87]:






# 阈值调整

rf_y_valid_prob = rf_pipeline.predict_proba(x_valid)[:,1]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
result = []
for t in thresholds:
    y_valid_pred = (rf_y_valid_prob > t).astype(int)
    result.append({
        'Accuracy':accuracy_score(y_valid, y_valid_pred),
        'Precision':precision_score(y_valid, y_valid_pred),
        'Recall':recall_score(y_valid, y_valid_pred),
        'F1':f1_score(y_valid, y_valid_pred),
        'ROC-AUC':roc_auc_score(y_valid, rf_y_valid_prob)
    })

threshold_classifier_df = pd.DataFrame(result,
                                      index = thresholds)
threshold_classifier_df.sort_values('F1', ascending = False)


# In[46]:


rf_y_valid_pred_04 = (rf_y_valid_prob > 0.4).astype(int)


# In[47]:


# ROC曲线

fpr, tpr, _ = roc_curve(y_valid, rf_y_valid_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize = (6,5))
plt.plot(fpr, tpr, label = f'Random Forest Model (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Model')
plt.legend(loc = 'lower right')
plt.show()


# In[48]:


# PR曲线

precision_curve, recall_curve, _ = precision_recall_curve(y_valid, rf_y_valid_prob)
ap_score = average_precision_score(y_valid, rf_y_valid_prob)

plt.figure(figsize = (6,5))
plt.plot(recall_curve, precision_curve, label = f'PR Curve (AP = {ap_score:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Random Forest Model')
plt.legend(loc = 'lower left')
plt.show()


# In[49]:


# 混淆矩阵

cm = confusion_matrix(y_valid, rf_y_valid_pred_04)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(values_format = 'd')
plt.title('Confusion Matrix for Random Forest Model (threshold = 0.4)')
plt.show()


# # 模型对比评估

# In[112]:




result = []

y_valid_pred_04 = (y_valid_prob_logit_pipeline_v2 > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_valid, y_valid_pred_04),
    'Precision':precision_score(y_valid, y_valid_pred_04),
    'Recall':recall_score(y_valid, y_valid_pred_04),
    'F1':f1_score(y_valid, y_valid_pred_04),
    'ROC-AUC':roc_auc_score(y_valid, y_valid_prob)})

rf_y_valid_pred_04 = (rf_y_valid_prob > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_valid, rf_y_valid_pred_04),
    'Precision':precision_score(y_valid, rf_y_valid_pred_04),
    'Recall':recall_score(y_valid, rf_y_valid_pred_04),
    'F1':f1_score(y_valid, rf_y_valid_pred_04),
    'ROC-AUC':roc_auc_score(y_valid, rf_y_valid_prob)})

compare_df = pd.DataFrame(result,
                         index = ['Logic Regression','Random Forest Model'])
compare_df



# # XGBoost

# In[51]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')


# In[52]:


from xgboost import XGBClassifier


# In[53]:


xgb_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators = 200,
                                max_depth = 4,
                                learning_rate = 0.05,
                                subsample = 0.8,
                                colsample_bytree = 0.8,
                                reg_alpha = 0,
                                reg_lambda = 1,
                                objective = 'binary:logistic',
                                eval_metric = 'logloss',
                                random_state = 42,
                                n_jobs = -1))
])


# In[54]:


xgb_pipeline.fit(x_train, y_train)


# In[55]:


model_evaluation(xgb_pipeline, x_train, y_train, datasetname = 'Train')
model_evaluation(xgb_pipeline, x_valid, y_valid, datasetname = 'Valid')


# In[56]:


# 阈值调整

# 阈值调整

xgb_y_valid_prob = xgb_pipeline.predict_proba(x_valid)[:,1]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
result = []
for t in thresholds:
    y_valid_pred = (xgb_y_valid_prob > t).astype(int)
    result.append({
        'Accuracy':accuracy_score(y_valid, y_valid_pred),
        'Precision':precision_score(y_valid, y_valid_pred),
        'Recall':recall_score(y_valid, y_valid_pred),
        'F1':f1_score(y_valid, y_valid_pred),
        'ROC-AUC':roc_auc_score(y_valid, xgb_y_valid_prob)
    })

threshold_classifier_df = pd.DataFrame(result,
                                      index = thresholds)
threshold_classifier_df.sort_values('F1', ascending = False)


# In[57]:


# 阈值调整

thresholds = [0.15, 0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3]
result = []
for t in thresholds:
    y_valid_pred = (xgb_y_valid_prob > t).astype(int)
    result.append({
        'Accuracy':accuracy_score(y_valid, y_valid_pred),
        'Precision':precision_score(y_valid, y_valid_pred),
        'Recall':recall_score(y_valid, y_valid_pred),
        'F1':f1_score(y_valid, y_valid_pred),
        'ROC-AUC':roc_auc_score(y_valid, xgb_y_valid_prob)
    })

threshold_classifier_df = pd.DataFrame(result,
                                      index = thresholds)
threshold_classifier_df.sort_values('F1', ascending = False)


# In[58]:


xgb_y_valid_pred_023 = (xgb_y_valid_prob > 0.23).astype(int)


# In[59]:


# ROC曲线

fpr, tpr, _ = roc_curve(y_valid, xgb_y_valid_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize = (6,5))
plt.plot(fpr, tpr, label = f'XGBoost Model (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1], linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost Model')
plt.legend(loc = 'lower right')
plt.show()


# In[60]:


# PR曲线

precision_curve, recall_curve, _ = precision_recall_curve(y_valid, xgb_y_valid_prob)
ap_score = average_precision_score(y_valid, xgb_y_valid_prob)

plt.figure(figsize = (6,5))
plt.plot(recall_curve, precision_curve, label = f'PR Curve (AP = {ap_score:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for XGBoost Model')
plt.legend(loc = 'lower left')
plt.show()


# In[61]:


# 混淆矩阵

cm = confusion_matrix(y_valid, xgb_y_valid_pred_023)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(values_format = 'd')
plt.title('Confusion Matrix for XGBoost Model (threshold = 0.23)')
plt.show()


# # 模型优化

# In[62]:


from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report


# In[124]:


# 1. 定义基础模型
xgb_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42))
])



# 2. 定义参数空间
params = {
    'classifier__n_estimators': [100, 200, 300], # 树的数量，可理解为 boosting 迭代次数
    'classifier__max_depth': [3, 4, 5, 6], # 每棵树的最大深度
    'classifier__learning_rate': [0.03, 0.05, 0.1], # 学习率，控制每棵树对最终结果的贡献大小
    'classifier__min_child_weight': [1, 3, 5], # 子节点最小样本权重和
    'classifier__subsample': [0.8, 0.9, 1.0], # 每棵树训练时随机抽取的样本比例
    'classifier__colsample_bytree': [0.8, 0.9, 1.0], # 每棵树训练时随机抽取的特征比例
    'classifier__scale_pos_weight': [3, 4, 5]    # 给正类更高权重，用于缓解类别不平衡
}

# 3. 定义分层K折交叉验证
skf = StratifiedKFold(n_splits=3,  # 将训练数据分成3折，每次用其中1折做验证，其余4折做训练
                      shuffle=True, # 在分折之前先将样本随机打乱，避免原始数据顺序带来偏差
                      random_state=42)

# 4. 随机搜索
random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=params,
    n_iter=4,               
    scoring='roc_auc',
    cv=skf,
    verbose=2, # 输出较详细的训练过程信息，便于观察搜索进度
    random_state=42,
    n_jobs=1
)

# 5. 开始训练
random_search.fit(x_train, y_train)

# 6. 输出最优参数和最优AUC
print("Best Params:", random_search.best_params_)
print("Best CV AUC:", random_search.best_score_)


# In[98]:


best_xgb = random_search.best_estimator_

# 保存模型

joblib.dump(logit_pipeline, "saved_models/best_xgb.pkl")


# In[90]:


best_xgb_valid_prob = best_xgb.predict_proba(x_valid)[:,1]
print('AUC score:', roc_auc_score(y_valid, best_xgb_valid_prob))


# In[91]:


# 阈值调整

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
result = []
for t in thresholds:
    y_valid_pred = (best_xgb_valid_prob > t).astype(int)
    result.append({
        'Accuracy':accuracy_score(y_valid, y_valid_pred),
        'Precision':precision_score(y_valid, y_valid_pred),
        'Recall':recall_score(y_valid, y_valid_pred),
        'F1':f1_score(y_valid, y_valid_pred),
        'ROC-AUC':roc_auc_score(y_valid, best_xgb_valid_prob)
    })

threshold_classifier_df = pd.DataFrame(result,
                                      index = thresholds)
threshold_classifier_df.sort_values('F1', ascending = False)


# # Logistic, RandomForest, XGBoost模型对比

# In[92]:


result = []

y_valid_pred_04 = (y_valid_prob > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_valid, y_valid_pred_04),
    'Precision':precision_score(y_valid, y_valid_pred_04),
    'Recall':recall_score(y_valid, y_valid_pred_04),
    'F1':f1_score(y_valid, y_valid_pred_04),
    'ROC-AUC':roc_auc_score(y_valid, y_valid_prob)})

rf_y_valid_pred_04 = (rf_y_valid_prob > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_valid, rf_y_valid_pred_04),
    'Precision':precision_score(y_valid, rf_y_valid_pred_04),
    'Recall':recall_score(y_valid, rf_y_valid_pred_04),
    'F1':f1_score(y_valid, rf_y_valid_pred_04),
    'ROC-AUC':roc_auc_score(y_valid, rf_y_valid_prob)})

xgb_y_valid_pred_04 = (best_xgb_valid_prob > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_valid, xgb_y_valid_pred_04),
    'Precision':precision_score(y_valid, xgb_y_valid_pred_04),
    'Recall':recall_score(y_valid, xgb_y_valid_pred_04),
    'F1':f1_score(y_valid, xgb_y_valid_pred_04),
    'ROC-AUC':roc_auc_score(y_valid, best_xgb_valid_prob)})

compare_df = pd.DataFrame(result,
                         index = ['Logic Regression','Random Forest Model','XGBoost Model'])
compare_df


# # SHAP模型解释

# In[93]:


get_ipython().system('pip install shap')


# In[94]:


get_ipython().run_line_magic('pip', 'uninstall -y shap numpy pandas scipy scikit-learn matplotlib contourpy slicer')
get_ipython().run_line_magic('pip', 'install numpy==1.26.4 pandas==2.2.3 scipy==1.13.1 scikit-learn==1.5.2 matplotlib==3.8.4 shap==0.46.0')


# In[3]:


get_ipython().run_line_magic('pip', 'install numexpr==2.10.1 bottleneck==1.4.2')


# In[2]:


import shap
import matplotlib.pyplot as plt


# In[100]:


explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(x_valid)


# In[101]:


shap.summary_plot(shap_values, x_valid)


# In[102]:


shap.summary_plot(shap_values, x_valid, plot_type="bar")


# # 测试集最终评估

# In[117]:


best_xgb_test_prob = best_xgb.predict_proba(x_test)[:, 1]
print('AUC score:', roc_auc_score(y_test, best_xgb_test_prob))

y_test_pred_04 = (best_xgb_test_prob > 0.4).astype(int)

print('Accuracy:', accuracy_score(y_test, y_test_pred_04))
print('Precision:', precision_score(y_test, y_test_pred_04))
print('Recall:', recall_score(y_test, y_test_pred_04))
print('F1:', f1_score(y_test, y_test_pred_04))


# In[119]:


# 混淆矩阵
cm = confusion_matrix(y_test, y_test_pred_04)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(values_format = 'd')
plt.title('Confusion Matrix for XGBoost Model\'s Final Test(threshold = 0.4)')
plt.show()


# In[126]:


# 泛化能力分析

y_train_prob = best_xgb.predict_proba(x_train)[:, 1]
y_valid_prob = best_xgb.predict_proba(x_valid)[:, 1]
y_test_prob = best_xgb.predict_proba(x_test)[:, 1]

result = []

y_train_pred_04 = (y_train_prob > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_train, y_train_pred_04),
    'Precision':precision_score(y_train, y_train_pred_04),
    'Recall':recall_score(y_train, y_train_pred_04),
    'F1':f1_score(y_train, y_train_pred_04),
    'ROC-AUC':roc_auc_score(y_train, y_train_prob)})

y_valid_pred_04 = (y_valid_prob > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_valid, y_valid_pred_04),
    'Precision':precision_score(y_valid, y_valid_pred_04),
    'Recall':recall_score(y_valid, y_valid_pred_04),
    'F1':f1_score(y_valid, y_valid_pred_04),
    'ROC-AUC':roc_auc_score(y_valid, y_valid_prob)})

y_test_pred_04 = (y_test_prob > 0.4).astype(int)
result.append({'Accuracy':accuracy_score(y_test, y_test_pred_04),
    'Precision':precision_score(y_test, y_test_pred_04),
    'Recall':recall_score(y_test, y_test_pred_04),
    'F1':f1_score(y_test, y_test_pred_04),
    'ROC-AUC':roc_auc_score(y_test, y_test_prob)})

compare_df = pd.DataFrame(result,
                         index = ['Train','Valid','Test'])
compare_df


# In[ ]:




