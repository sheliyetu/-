from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import evaluate
from transformers import DataCollatorWithPadding
from transformers import pipeline

# 加载数据集
data_set = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")

# 过滤掉没有评论的样本
data_set = data_set.filter(lambda x: x["review"] is not None)

# 划分训练集和测试集
split_datasets = data_set.train_test_split(test_size=0.1)

# 加载分词器
tokenizer_model = AutoTokenizer.from_pretrained("hfl/rbt3")

# 数据预处理函数
def preprocess_data(examples):
    tokenized_data = tokenizer_model(examples["review"], max_length=128, truncation=True)
    tokenized_data["labels"] = examples["label"]
    return tokenized_data

# 对数据集进行分词处理
tokenized_data_set = split_datasets.map(preprocess_data, batched=True, remove_columns=split_datasets["train"].column_names)

# 加载模型
classification_model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

# 评价指标
accuracy_metric = evaluate.load("accuracy")
f1_score_metric = evaluate.load("f1")

def compute_metrics(evaluation_predictions):
    pred, labels = evaluation_predictions
    pred = pred.argmax(axis=-1)
    accuracy = accuracy_metric.compute(predictions=pred, references=labels)
    f1_score = f1_score_metric.compute(predictions=pred, references=labels)
    accuracy.update(f1_score)
    return accuracy

# 训练参数
training_args = TrainingArguments(output_dir="./checkpoints",  # 输出文件夹
                                  per_device_train_batch_size=16,  # 训练时的batch_size
                                  per_device_eval_batch_size=32,  # 验证时的batch_size
                                  logging_steps=10,  # log 打印的频率
                                  evaluation_strategy="epoch",  # 评估策略
                                  save_strategy="epoch",  # 保存策略
                                  save_total_limit=3,  # 最大保存数
                                  learning_rate=2e-5,  # 学习率
                                  weight_decay=0.01,  # weight_decay
                                  metric_for_best_model="f1",  # 设定评估指标
                                  load_best_model_at_end=True)  # 训练完成后加载最优模型

# 创建Trainer
trainer_instance = Trainer(model=classification_model,
                           args=training_args,
                           train_dataset=tokenized_data_set["train"],
                           eval_dataset=tokenized_data_set["test"],
                           data_collator=DataCollatorWithPadding(tokenizer=tokenizer_model),
                           compute_metrics=compute_metrics)

# 训练模型
trainer_instance.train()

# 评估模型
trainer_instance.evaluate(tokenized_data_set["test"])

# 标签映射
label_map = {0: "差评！", 1: "好评！"}
classification_model.config.id2label = label_map

# 创建分类管道
classification_pipeline = pipeline("text-classification", model=classification_model, tokenizer=tokenizer_model)

# 测试一个例子
test_sentence = "我觉得不错！"

result = classification_pipeline(test_sentence)
print(result)