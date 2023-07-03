import logging
from dataclasses import dataclass
import time
from typing import Optional, Union, List, Dict, Any, Tuple

from get_models import ALPACA_ID, LLAMA_ID, TOKENIZER_ID
from peft import PeftModel
from simple_ai.api.grpc.completion.server import LanguageModel
from transformers import GenerationConfig, LLaMAForCausalLM, LLaMATokenizer
from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria

import copy
import inspect
import torch 
import torch.distributed as dist
import json 

@dataclass(unsafe_hash=True)
class AlpacaModel(LanguageModel):
    try:
        tokenizer = LLaMATokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as ex:
        logging.exception(f"Could not load tokenizer: {ex}")
        tokenizer = None
    try:
        model = LLaMAForCausalLM.from_pretrained(
            LLAMA_ID,
            load_in_8bit=True,
            device_map="auto",
        )
    except Exception as ex:
        logging.exception(f"Could not load pretrained LlaMa model: {ex}")
        model = None
    try:
        model = PeftModel.from_pretrained(model, ALPACA_ID)
    except Exception as ex:
        logging.exception(f"Could not load pretrained Peft model: {ex}")
        model = None

    if tokenizer is None or model is None:
        raise Exception("Could not load model or tokenizer")
    # tokenizer.add_special_tokens({"pad_token": " "})
    # tokenizer.pad_token_id = (
    # 0  # unk. we want this to be different from the eos token
    # )
    def _score_base_prompt(self, base_prompt: str) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(base_prompt, return_tensors="pt")#, padding='max_length', max_length=max_tokens)
        input_ids = inputs.input_ids.cuda()
        token_ids_list = input_ids.tolist()[0]
        output = self.model(
            input_ids = inputs.input_ids.cuda(),
            attention_mask = inputs.attention_mask.cuda(),
            output_attentions = True,
            use_cache = True,
        )
        past_key_values = self.model._extract_past_from_model_output(output, standardize_cache_format=False)
        return input_ids, past_key_values
    
    def _score_tokens(self, base_prompt_ids, past_key_values, action) -> List[Dict[str, Any]]:
        # add padding to the prompt
        timestamp = time.time()
        inputs = self.tokenizer(action, return_tensors="pt")#, padding='max_length', max_length=max_tokens)
        input_ids = inputs.input_ids.cuda()
        # tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        input_ids = base_prompt_ids + inputs.input_ids[0][0]
        # token_ids_list = input_ids.tolist()[0]
        local_past_key_values = copy.deepcopy(past_key_values)
        scored_tokens_dict = []
        for existed_token in inputs.input_ids[0][1:]:
            output = self.model(
                input_ids = input_ids,
                # attention_mask = inputs.attention_mask.cuda(),
                output_attentions = True,
                use_cache = True,
                # past_key_values = local_past_key_values
            )
            next_token_logits = output.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            # get existed_token's score and replace next_token with existed_token
            existed_token_reshaped = existed_token.repeat(next_tokens.shape[0])
            existed_token_score = next_token_logits[0][existed_token_reshaped]
            next_tokens = existed_token_reshaped

            # print("=========================\n" * 50, existed_token_score)
            scored_tokens_dict.append({"token": existed_token, "score": existed_token_score})
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids = input_ids.cuda()
            # input_ids = torch.cat([input_ids, existed_token.unsqueeze(0)], dim=1)
            local_past_key_values = self.model._extract_past_from_model_output(output, standardize_cache_format=False)
        print(f"Time to score tokens: {time.time() - timestamp}")
        return scored_tokens_dict, None# token_ids_list

    def complete(
        self,
        prompt: str = "<|endoftext|>",
        suffix: str = "",
        max_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        logprobs: int = 0,
        echo: bool = False,
        stop: Union[str, list] = "",
        presence_penalty: float = 0.0,
        frequence_penalty: float = 0.0,
        best_of: int = 0,
        logit_bias: dict = {},
    ) -> str:
        self._base_prompt_cache = None
        base_prompt, variants = prompt.split("<|endofprompt|>")
        actions = list(variants.split("<|endofvariant|>"))
        base_prompt_ids = self.tokenizer(base_prompt, return_tensors="pt").input_ids.tolist()[0]
        base_prompt_tokens = self.tokenizer.convert_ids_to_tokens(base_prompt_ids)
        response = []
        action_ids_lens = []
        for action in actions:
            ids = self.tokenizer(action, return_tensors="pt").input_ids.tolist()[0]
            action_ids_lens.append(len(ids))
        max_action_ids_len = max(action_ids_lens) + len(base_prompt_ids)
        base_prompt_ids, past_key_values = self._score_base_prompt(base_prompt)
        for action in actions:
            scored_tokens_dict, prompt_with_action_ids = self._score_tokens(base_prompt_ids, past_key_values, action)
            # remove base prompt tokens from the scored tokens
            scored_tokens_dict = scored_tokens_dict[len(base_prompt_ids):]
            generated_logprobs = {"tokens": [token['token'] for token in scored_tokens_dict],
                        "token_logprobs": [token['score'] for token in scored_tokens_dict]} if logprobs else None
            response.append({"text": self.tokenizer.decode(prompt_with_action_ids).strip(), 
                            "logprobs": generated_logprobs,
                            "finish_reason": "ok"})
        return json.dumps(response, ensure_ascii=False)
