import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# 从基础模型加载分词器并保存到检查点目录
model_name = "hfl/rbt3"  # 基础模型名称
checkpoint_path = "./checkpoints/checkpoint-2622"  # 检查点目录

# 加载基础模型的分词器
tokenizer_model = AutoTokenizer.from_pretrained(model_name)

# 保存分词器到检查点目录
tokenizer_model.save_pretrained(checkpoint_path)

# 加载模型
classification_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

# 加载分词器
tokenizer_model = AutoTokenizer.from_pretrained(checkpoint_path)

# 创建分类管道
classification_pipeline = pipeline("text-classification", model=classification_model, tokenizer=tokenizer_model)

# 定义标签映射
label_mapping = {'LABEL_1': '好评', 'LABEL_0': '差评'}

# 评价函数
def classify_review(review_text):
    prediction = classification_pipeline(review_text)
    mapped_prediction = [{'label': label_mapping[item['label']], 'score': item['score']} for item in prediction]
    return f"评价: {mapped_prediction[0]['label']} 😊" if mapped_prediction[0]['label'] == '好评' else f"评价: {mapped_prediction[0]['label']} 😞", f"置信度: {mapped_prediction[0]['score']:.2f}"

# 创建Gradio界面
interface = gr.Interface(
    fn=classify_review,  # 绑定的函数
    inputs=gr.Textbox(label="请输入评价", lines=2, placeholder="请输入您的评价..."),  # 输入框，提示语为“请输入您的评价...”
    outputs=[gr.Textbox(label="评价结果"), gr.Textbox(label="置信度")],  # 输出框标签
    title="文本评价 🎉",  # 标题，可以根据需要更改
    description="输入文本并点击“提交”按钮来获取评价结果（好评或差评）。"  # 描述，可以根据需要更改
)

# 添加背景图片和自定义CSS
interface.css = """
body {
    background-image: url('https://imgur.la/images/2024/07/29/wallhaven-9dp3y1_2560x1440.png');  # 替换为您的背景图片URL
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    margin: 0;
    padding: 0;
    height: 100vh;
    width: 100vw;
}
#title {
    color: #ff73b3;  # 标题颜色
}
#input_text {
    font-size: 20px;  # 输入文本的字体大小
    background-color: #f0f0f0;  # 输入文本框的背景颜色
    color: #333333;  # 输入文本框的字体颜色
}
#output_text {
    font-size: 20px;  # 输出文本的字体大小
    background-color: #e0e0e0;  # 输出文本框的背景颜色
    color: #000000;  # 输出文本框的字体颜色
}
"""

# 启动Gradio界面
interface.launch(share=True)