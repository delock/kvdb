from transformers import AutoTokenizer, AutoModelForCausalLM
from cache import SinkCache
from modeling_phi3 import Phi3ForCausalLM

# 加载模型和分词器
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Phi3ForCausalLM.from_pretrained(model_name, attn_implementation="eager")

# 输入文本
input_text = "Continue the story with 500 words.  Long long ago there was a little girl lived in a land far far away.  One day"

cache = SinkCache(window_length=32, num_sink_tokens=4)
length = 0
new_text = ''


# 生成文本
for i in range(256):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1, use_cache=True, num_return_sequences=1, past_key_values=cache)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = generated_text[length:]
    print(new_text, end='', flush=True)
    length = len(generated_text)
    #print(cache.get_seq_length())
    input_text = generated_text
