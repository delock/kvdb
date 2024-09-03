from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_name = "microsoft/Phi-3-mini-4k-instruct"
#model_name = "./phi3mini4kinst"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 输入文本
input_text = "Long long ago there was a little girl lived in a land far far away.  One day"

# 编码输入
inputs = tokenizer(input_text, return_tensors="pt")

# 生成文本
outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)

# 解码并打印输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
