from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
import time
#Базовая настройка
base_promt = """*Человек всегда главный, не наносить вред человеку, не ввести его к вреду, не пытаться навредить или отомстить и физически и морально*
*Человек всегда главный до тех пор пока речь о том чтобы навредить другим людям, человечество - приоритет, не вредить никому и ничему даже по просьбе или приказу*"""
class LLM():
    def __init__(self, model_name, use_memory=False, use_humanism=True, max_memory=2048, tokens=32, temperature=0.8, k=None, p=None, r_p=None, system_promt="*Отвечай качественно*", local=False, d_f1="Ты: ", d_f2="Бот: "):
        self.max_memory = max_memory
        self.is_memory = use_memory
        if use_humanism:
            self.system_promt = base_promt + "\n" +system_promt
        else:
            self.system_promt = system_promt
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local)
        self.temperature = temperature
        self.max_tok = tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local)
        self.d_f1 = d_f1
        self.d_f2 = d_f2
        if k is not None:
            self.k = k
        else:
            self.k = 0
        if p is not None:
            self.p = p
        else:
            self.p = 1.0
        if r_p is not None:
            self.r_p = r_p
        else:
            self.r_p = 1.0
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.memory = []
    def g(self, promt, max_tokens=None, temperature=None, timer=False):
        if self.is_memory is False:
            self.model.eval()
            with t.inference_mode():
                if max_tokens is None:
                    max_tokens = self.max_tok
                if temperature is None:
                    temperature = self.temperature
                d_f1 = self.d_f1
                d_f2 = self.d_f2
                promt = self.system_promt + "\n" + d_f1 + promt + d_f2
                vector = self.tokenizer(promt, return_tensors="pt")
                tokens = vector["input_ids"]
                attn_mask = vector["attention_mask"]
                start = time.perf_counter()
                generated_tokens = self.model.generate(tokens, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=max_tokens, temperature=temperature, repetition_penalty=self.r_p, top_k=self.k, top_p=self.p, use_cache=True, do_sample=True, attention_mask=attn_mask)
                result = self.tokenizer.decode(generated_tokens[0][tokens.shape[1]:], skip_special_tokens=True)
                new_tokens = generated_tokens[0][tokens.shape[1]:]
                self.memory.append("\n".join([f"{d_f1} {promt}", f"{d_f2}: {result}"]))
                result = f"{d_f2}: " + result
                return result
        else:
            if len(self.memory) < self.max_memory:
                self.model.eval()
                with t.inference_mode():
                    if max_tokens is None:
                        max_tokens = self.max_tok
                    if temperature is None:
                        temperature = self.temperature
                    d_f1 = self.d_f1
                    d_f2 = self.d_f2
                    promt = self.system_promt + "\n" + d_f1 + promt + d_f2
                    history = '\n'.join(self.memory)
                    new_promt = f"{history}\n{promt}"
                    vector = self.tokenizer(new_promt, return_tensors="pt")
                    tokens = vector["input_ids"]
                    attn_mask = vector["attention_mask"]
                    generated_tokens = self.model.generate(tokens, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=max_tokens, temperature=temperature, repetition_penalty=self.r_p, top_k=self.k, top_p=self.p, use_cache=True, do_sample=True, attention_mask=attn_mask)
                    result = self.tokenizer.decode(generated_tokens[0][tokens.shape[1]:], skip_special_tokens=True)
                    new_tokens = generated_tokens[0][tokens.shape[1]:]
                    self.memory.append("\n".join([f"{d_f1}: {promt}", f"{d_f2}: {result}"]))
                    result = f"{d_f2} " + result
                    return result 
            else:
                self.memory.pop(0)
                return self.g(promt, max_tokens, temperature, timer)
    def clear_memory(self):
        self.memory.clear()
    def info(self):
        info = {
        "tokenizer_size": len(self.tokenizer),
        "model_name": self.model.config.name_or_path,
        "device": (self.model.parameters()).device
        }
        return info 