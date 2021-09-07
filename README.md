# cross-lingual-MRC
cross-lingual doc_QA and passage_QA based MRC model , chinese passage and  vi &amp; ru question

单文档自由提问系统                                                                                                                                                          

推理过程代码封装在doc_based_cross_lingual_qa.py
主要包含两个阶段：

# 调用方法
## 初始化和加载模型
### 需要给出MRC模型、相关性排序模型和句子切分模型的配置文件(json)
qa_model = MrcBasedDocModel(mrc_config_fpath, rel_config_fpath, doc_split_config_fpath)
qa_model.load_model()
根据输入的<question, context>，进行inference
## 模型推理
### 预处理和tokenize部分已封装在predict()中
qa_model.predict(question, context)
