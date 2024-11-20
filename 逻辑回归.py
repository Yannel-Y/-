#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


# In[4]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# In[5]:


#加载sklearn自带的乳腺癌分类数据集
X, y = load_breast_cancer(return_X_y=True)
#自己探索数据集大小和列的情况


# In[6]:


#以指定比例将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    train_size=0.875,test_size=0.125,random_state=188)


# # 创建逻辑回归模型，训练

# In[10]:


#使用Lr类，初始化模型
clf = LogisticRegression(
    penalty="l2",C=1.0,random_state=None,solver="lbfgs",max_iter=3000,
    multi_class='ovr', verbose=0,
)

#使用训练数据来信息（拟合）
clf.fit(X_train,y_train)


# # 使用测试数据进行预测

# In[11]:


#使用测试数据来预测，返回值预测分类数据
y_pred = clf.predict(X_test)


# # 评估模型性能

# In[12]:


# 打印主要分类指标文本报告
print('--- report ---')
print(classification_report(y_test,y_pred))

#打印模型的参数
print('--- params ---')
print(clf.coef_, clf.intercept_)

#打印准确率
print('--- accuracy ---')
print(accuracy_score(y_test,y_pred))


# In[ ]:




