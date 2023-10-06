import os
import sys
from transformers import LlamaForCausalLM, LlamaTokenizer

#model = LlamaForCausalLM.from_pretrained("/output/path")
#tokenizer = LlamaTokenizer.from_pretrained("/output/path")

MODEL_PATH = '/usr3/graduate/nivek/data/vislang-domain-exploration-data/caption2label-data/llama-models/llama-7b-hf'

def example():
#    model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    prompts = ['Pick the objects in these phrases: "The cat sits on the fence."=>"cat","fence"\n"The dog jumps over the moon."=>"dog","moon"\n"The pig flies in the sky."=>"pig","sky"\n"%s"=>'%(s) for s in ['The horse stands in the field.', 'The duck swims in the pond.']]
    x = tokenizer(prompts, return_tensors='pt', padding=True)
    import pdb
    pdb.set_trace()
    assert(False) #KEVIN

if __name__ == '__main__':
    example()
