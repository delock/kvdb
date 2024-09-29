import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cache import PersistentCache
from transformers.cache_utils import DynamicCache, SinkCache
from modeling_phi3 import Phi3ForCausalLM
from tqdm import tqdm

debug = False

def gen_text(input_text, model, tokenizer, cache):
    length = 0
    new_text = ''
    # Generate text
    inputs = tokenizer(input_text, return_tensors="pt")
    input_tokens = inputs['input_ids']
    for i in tqdm(range(256)):
        outputs = model.generate(input_tokens, max_new_tokens=1, use_cache=True, num_return_sequences=1, past_key_values=cache)
        if debug:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[length:]
            if length == 0:
                print('----------------------------')
            print(new_text, end='', flush=True)
            length = len(generated_text)
        input_tokens = outputs
    if debug:
        print('')
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def cross_perplexity(input_text, model, tokenizer):
    # Compute cross perplexity
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']

    # Get the model's output logits and calculate log-likelihood
    with torch.no_grad():
        outputs = model_ref(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross entropy loss (negative log-likelihood)

    # Compute perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

# load model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Phi3ForCausalLM.from_pretrained(model_name, attn_implementation="eager")

model_ref = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")

# Input text
prompts = [
          "Earth, mostly harmless,",
          "Long long ago there was a little girl lived in a land far far away.  One day",
          ]

total_ppl = 1
total_ppl_ref = 1
for prompt in prompts:
    cache = PersistentCache(window_length=32, num_sink_tokens=8, replace_sink_tokens=4)
    if debug:
        cache_ref = DynamicCache()
    print('Generating text for test configuration')
    result = gen_text(prompt, model, tokenizer, cache)
    if debug:
        print('Generating text for reference(default) configuration')
        result_ref = gen_text(prompt, model_ref, tokenizer, cache_ref)
    print('Compute cross perplexity for test configuration against reference configuration')
    ppl = cross_perplexity(result, model_ref, tokenizer)
    if debug:
        print('Compute self perplexity for reference configuration against itself')
        ppl_ref = cross_perplexity(result_ref, model_ref, tokenizer)
    if debug:
        print(f'result={result}')
        print(f'result_ref={result_ref}')
        print(f'ppl={ppl}')
        print(f'ppl_ref={ppl_ref}')
    total_ppl *= ppl
    if debug:
        total_ppl_ref *= ppl_ref

print(f'==============================================')
print(f'geomean ppl = {total_ppl**(1/len(prompts))}')
if debug:
    print(f'geomean ppl_ref = {total_ppl_ref**(1/len(prompts))}')
