# cross-lingual-MRC
cross-lingual doc_QA and passage_QA based MRC model , chinese passage and  vi &amp; ru question

单文档自由提问系统                                                                                                                                                          

推理过程代码封装在doc_based_cross_lingual_qa.py  
主要包含两个阶段：load_model(), predict()

## 调用方法

### 初始化和加载模型
模型初始化只需要model_dir即可，model_dir必需包含：MRC模型、相关性排序模型和句子切分模型的配置文件(json)等文件

qa_model = MrcBasedDocModel(model_dir)  
qa_model.load_model()

### 模型推理，根据输入的<question, context>，进行inference
预处理和tokenize部分已封装在predict()中  
qa_model.predict(question, context)

## 输入输出格式

### 输入
question    问句，文本  str  
context 段落，文本  str

### 输出
<限定passage模式> : (start, end, text)  元组  
start   答案的起始位置  int  
end 答案的结束位置  int  
text    答案，文本  str
