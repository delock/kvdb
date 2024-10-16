import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from cache import SinkCacheExt
from datasets import load_dataset
from transformers.cache_utils import DynamicCache
from modeling_phi3 import Phi3ForCausalLM
from tqdm import tqdm
import argparse

# define the following arguments:
# debug -- print debug result, default false
# full -- do a full perplexity computation
# fast -- do a fast perplexity computation
# ref -- test reference perplexity, default false
# sink_size -- size of sink, defult 8
# window_length -- length of window, default 32
# replace_sink_size -- number of elements in sink that can be replaced, default 4

# debug and ref are boolean arguments
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--ref', action='store_true', default=False)
parser.add_argument('--full', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--sink_size', type=int, default=8)
parser.add_argument('--window_length', type=int, default=32)
parser.add_argument('--replace_sink_size', type=int, default=4)
parser.add_argument('--regression', type=float, default=1.0)
args = parser.parse_args()

debug = args.debug
ref = args.ref

def gen_text(input_text, model, tokenizer, cache):
    length = 0
    new_text = ''
    # Generate text
    inputs = tokenizer(input_text, return_tensors="pt")
    input_tokens = inputs['input_ids']
    for i in tqdm(range(256)):
        outputs = model.generate(input_tokens, max_new_tokens=1, use_cache=True, num_return_sequences=1, past_key_values=cache)
        input_tokens = outputs
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def cross_perplexity(input_text, model, tokenizer):
    # Compute cross perplexity
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']

    # Get the model's output logits and calculate log-likelihood
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # Cross entropy loss (negative log-likelihood)

    # Compute perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()

def perplexity(input_text, model, tokenizer, cache=None):
    # Compute average cross perplexity over each decode step
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']
    total_loss = 0.0
    num_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(1, input_ids.size(1))):
            # Slice input to predict the next token step by step
            input_slice = input_ids[:, i-1].unsqueeze(1)
            label = input_ids[:, i]

            # print input_slice and labsls size
            outputs = model(input_slice, past_key_values=cache)
            step_loss = F.cross_entropy(outputs.logits[:, -1, :], label, reduction='mean')
            if i > 5:
                total_loss += step_loss
                num_tokens += 1

    # Compute average perplexity
    average_loss = total_loss / num_tokens
    perplexity = torch.exp(average_loss)
    print(f"Average loss: {average_loss}, Perplexity: {perplexity}")
    return perplexity.item()

# load model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Phi3ForCausalLM.from_pretrained(model_name, attn_implementation="eager")

model_ref = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")


# Input text
prompts = [
          "Continue the story with 500 words: Long long ago there was a little girl lived in a land far far away.\nAssistant:",
          "Continue the story with 500 words: Everything we cherished will perish.\nAssistant:",
          "Continue the story with 500 words: Planet Earth: Harmless (mostly).\nAssistant:",
          ]

if args.debug:
    total_ppl = 1
    for prompt in prompts:
        cache = SinkCacheExt(window_length=args.window_length, num_sink_tokens=args.sink_size, replace_sink_tokens=args.replace_sink_size, regression=args.regression)
        print('Generating text for test configuration')
        result = gen_text(prompt, model, tokenizer, cache)
        print('Compute cross perplexity for test configuration against reference configuration')
        ppl = cross_perplexity(result, model, tokenizer)
        print(f'result={result}')
        print(f'ppl={ppl}')
        total_ppl *= ppl

    print(f'==============================================')
    print(f'geomean ppl = {total_ppl**(1/len(prompts))}')

else:
    if args.fast:
        text = "Long long ago there was a little girl lived in a land far far away.  One day she went to the forest to pick some flowers.  She saw a rabbit hopping around and she followed it.  The rabbit led her to a magical place where she met a fairy.  The fairy told her that she was the chosen one to save the kingdom from the evil witch."
    else:
        # Load the WikiText dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = '\n\n'.join(dataset["text"])
        if not args.full:
            text = text[:1500]

    if args.ref:
        cache = DynamicCache()
    else:
        cache = SinkCacheExt(window_length=args.window_length, num_sink_tokens=args.sink_size, replace_sink_tokens=args.replace_sink_size, regression=args.regression)
    ppl = perplexity(text, model, tokenizer, cache)