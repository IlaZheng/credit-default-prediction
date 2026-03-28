#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


st.set_page_config(page_title = '信贷违约预测业务仪表盘', layout = 'wide')

# 基础数据：后续可以换为真实数据读取
final_model = 'XGBoost'
final_threshold = 0.4

metrics_df = pd.DataFrame({
        "Dataset": ["Train", "Valid", "Test"],
        "Accuracy": [0.556382, 0.545906, 0.546900],
        "Precision": [0.289922, 0.281642, 0.281790],
        "Recall": [0.844263, 0.822912, 0.820688],
        "F1": [0.431624, 0.419656, 0.419530],
        "ROC_AUC": [0.748864, 0.726465, 0.725810],
    })

shap_importance_df = pd.DataFrame({"feature": [
            "subGrade_coded",
            "term",
            "homeOwnership",
            "dti_cleaned",
            "revolBal",
            "interestRate",
            "ficoRangeLow",
            "n14",
            "loan_income_ratio",
            "employmentLength_clean",
            "regionCode",
            "n2",
            "totalAcc",
            "revolUtil",
            "annualIncome"],"importance": [
            0.47,
            0.19,
            0.14,
            0.12,
            0.095,
            0.091,
            0.089,
            0.081,
            0.080,
            0.056,
            0.055,
            0.054,
            0.052,
            0.043,
            0.037]})

conf_matrix = np.array([[30653, 33386],[2862, 13099]])


# In[9]:


# 标题区

st.title("信贷违约预测业务仪表板")
st.caption('For displaying final model performance, threshold selection strategy and key features for judgement.')

col1, col2, col3 = st.columns(3)
col1.metric('最终模型', final_model)
col2.metric('业务阈值', final_threshold)
col3.metric("测试集 ROC-AUC", f"{metrics_df.loc[metrics_df['Dataset']=='Test', 'ROC_AUC'].iloc[0]:.4f}")
st.divider()


# In[ ]:


# 侧边栏

st.sidebar.header('仪表盘设置')
selected_dataset = st.sidebar.selectbox("选择数据集", metrics_df["Dataset"].tolist(), index=2)
show_top_n = st.sidebar.slider("展示前 N 个关键特征", min_value=5, max_value=15, value=10)
threshold_note = st.sidebar.checkbox("显示阈值说明", value=True)


# In[ ]:


# 模型表现

st.subheader("一、模型表现总览")
selected_row = metrics_df[metrics_df["Dataset"] == selected_dataset].iloc[0]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accuracy", f"{selected_row['Accuracy']:.4f}")
k2.metric("Precision", f"{selected_row['Precision']:.4f}")
k3.metric("Recall", f"{selected_row['Recall']:.4f}")
k4.metric("F1", f"{selected_row['F1']:.4f}")
k5.metric("ROC-AUC", f"{selected_row['ROC_AUC']:.4f}")

st.dataframe(metrics_df, use_container_width=True)
fig_perf, ax_perf = plt.subplots(figsize=(8, 4))
plot_df = metrics_df.set_index("Dataset")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
plot_df.T.plot(kind="bar", ax=ax_perf)
ax_perf.set_title("Train / Valid / Test 指标对比")
ax_perf.set_ylabel("Score")
ax_perf.legend(title="Dataset")
plt.xticks(rotation=0)
st.pyplot(fig_perf)


# In[ ]:


# 阈值说明

st.subheader("二、风险阈值选择")
if threshold_note:
    st.info(
        "最终阈值设置为 0.4。该设定强调提高 Recall，尽量减少对高风险客户的漏判，"
        "更符合信贷违约预测中的风控目标。"
    )

st.markdown(
    """
- 默认阈值 0.5 虽然通常带来更高的 Accuracy / Precision，但会降低违约客户识别率。  
- 阈值进一步降到 0.3 或 0.2 时，Recall 会继续升高，但误判正常客户的数量会明显增加。  
- 0.4 是当前项目中在 Recall、Precision 和 F1 之间较稳妥的折中方案。  
"""
)


# In[ ]:


# 混淆矩阵
st.subheader("三、测试集混淆矩阵（阈值 = 0.4）")
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
im = ax_cm.imshow(conf_matrix)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax_cm.text(j, i, conf_matrix[i, j], ha="center", va="center")
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["预测非违约", "预测违约"])
ax_cm.set_yticklabels(["实际非违约", "实际违约"])
ax_cm.set_title("Confusion Matrix")
fig_cm.colorbar(im, ax=ax_cm)
st.pyplot(fig_cm)

st.markdown(
    f"""
- 正确识别违约客户（TP）：**{conf_matrix[1,1]}**  
- 漏判违约客户（FN）：**{conf_matrix[1,0]}**  
- 将正常客户误判为违约（FP）：**{conf_matrix[0,1]}**  
- 正确识别正常客户（TN）：**{conf_matrix[0,0]}**  
"""
)    


# In[ ]:


# SHAP关键特征
st.subheader("四、关键风险特征（SHAP）")
show_df = shap_importance_df.head(show_top_n).sort_values("importance", ascending=True)

fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
ax_shap.barh(show_df["feature"], show_df["importance"])
ax_shap.set_title(f"前 {show_top_n} 个关键特征")
ax_shap.set_xlabel("mean(|SHAP value|)")
st.pyplot(fig_shap)

feature_explanations = {
    "subGrade_coded": "信用等级细分信息，是模型识别违约风险的核心依据。",
    "term": "贷款期限越长，借款人面临的不确定性通常越高。",
    "homeOwnership": "住房状况反映借款人资产稳定性与财务基础。",
    "dti_cleaned": "债务收入比越高，偿债压力通常越大。",
    "revolBal": "循环授信余额反映借款人的资金占用情况。",
    "interestRate": "较高利率通常意味着更高风险定价与更大还款负担。",
    "ficoRangeLow": "信用评分下限越低，通常意味着信用资质越弱。",
    "loan_income_ratio": "贷款金额相对收入越高，财务压力通常越强。",
    "annualIncome": "收入水平体现借款人的还款能力和财务稳定性。",
}

selected_feature = st.selectbox("选择一个特征查看业务解释", shap_importance_df["feature"].tolist())
st.write(feature_explanations.get(selected_feature, "该特征对模型具有一定影响，可结合业务编码规则进一步解释。"))


# In[ ]:




