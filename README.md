# cross-lingual-MRC
cross-lingual doc_QA and passage_QA based MRC model , chinese passage and  vi &amp; ru question

## 两种模式，提供两种推理接口
### 单passage自由提问系统                                                                                                                                               
单MRC模型，给定passage，提出question，基于passage给出答案
### 单doc自由提问系统
处理大型文档document，将document经过切分、排序和MRC，基于document给出答案

推理过程代码封装在doc_based_cross_lingual_qa.py  
主要包含两个阶段：
load_model()  
passage_qa_predict() # 单passage自由提问系统  
passage_qa_predict() # 单doc自由提问系统

## 调用方法
### 初始化和加载模型
模型初始化只需要model_dir即可，model_dir  
必需包含：MRC模型、相关性排序模型和句子切分模型的配置文件(json)等文件

qa_model = MrcBasedDocModel(model_dir)  
qa_model.load_model()

### 模型推理，根据输入的<question, context>，进行inference
预处理和tokenize部分已封装在predict()中  
qa_model.predict(question, context)

## 输入输出格式
### 输入
#### <限定passage模式>， <限定doc模式>
question    问句，文本  str  
context 段落，文本  str

### 输出
#### <限定passage模式>:  
返回结果格式：OrderDict()  
举例：OrderedDict([(0, (10, 13, '308', 9.74964427947998)), (1, (10, 13, '308', 9.74964427947998))])  
其中每个元素格式为：(index, (start, end, text, score))  元组  
index   下标，从0开始。索引第index个<question, context>    int  
start   答案的起始位置  int  
end 答案的结束位置  int  
text    答案，文本  str  
score   分数，浮点数    float  
#### <限定doc模式>:  
同上

### GPU调用
#### 如有需要，请自行指定具体GPU环境的机器编号device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model.to(device)  
input.to(device)

### 批测脚本
在doc_based_cross_lingual_qa.py的main()加入批测脚本，方便离线自测

## 启动web服务代码
python app.py
