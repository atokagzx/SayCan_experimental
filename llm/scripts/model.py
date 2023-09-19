import logging
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Tuple

from get_models import ALPACA_ID, LLAMA_ID, TOKENIZER_ID
from peft import PeftModel
from simple_ai.api.grpc.completion.server import LanguageModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import LogitsProcessorList, StoppingCriteriaList, MaxLengthCriteria

import copy
import inspect
import torch 
import torch.distributed as dist
import json 

@dataclass(unsafe_hash=True)
class AlpacaModel(LanguageModel):
    base_prompt = None
    try:
        tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as ex:
        logging.exception(f"Could not load tokenizer: {ex}")
        tokenizer = None
    try:
        model = LlamaForCausalLM.from_pretrained(
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

    def get_logits_for_query(self, instruction):
        inputs = self.tokenizer(instruction, return_tensors='pt')
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        offsets = inputs.attention_mask.sum(dim=1)
        attention_mask = inputs.attention_mask
        return offsets, attention_mask, outputs.past_key_values
    
    def _score_tokens(self, prompt: str, offsets_query, attention_mask_query, past_key_values_query) -> List[Dict[str, Any]]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        num_completion_tokens = inputs.input_ids.shape[1]
        suffixes_position_ids = (torch.arange(0, num_completion_tokens) +
                                offsets_query[:, None]) # broadcast
        attention_mask = torch.cat((attention_mask_query,
                                    inputs.attention_mask),
                                    dim=1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        input_ids = inputs.input_ids.cuda()
        token_ids_list = input_ids.tolist()[0]
        output = self.model(
            input_ids = inputs.input_ids.cuda(),
            attention_mask = attention_mask.cuda(),
            past_key_values = past_key_values_query,
            position_ids = suffixes_position_ids.cuda(),
        )
        # get the scores of all (32000) ids for every token in the prompt
        tokens_scores_all = output.logits[0].tolist()
        # add the score of the first token
        tokens_scores_selected = [None]
        # select only the scores of the tokens in the prompt, skipping the first token (last score is for the new token, so we skip it)
        tokens_scores_selected.extend([score[idx] for idx, score in zip(token_ids_list[1:], tokens_scores_all)])
        # zip the tokens and scores together and create a list of dictionaries 
        scored_tokens_dict = [dict(zip(['token', 'score'], token_score)) for token_score in zip(tokens, tokens_scores_selected)]
        return scored_tokens_dict, token_ids_list, output.past_key_values

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
        base_prompt, variants = prompt.split("<|endofprompt|>")
        history, variants = variants.split("<|endofhistory|>")
        actions = list(variants.split("<|endofvariant|>"))
        history_ids = self.tokenizer(history, return_tensors="pt").input_ids.tolist()[0]
        response = []
        action_ids_lens = []
        if self.base_prompt is None:
            offsets_query, attention_mask_query, past_key_values_query = self.get_logits_for_query(base_prompt)
            self.base_prompt = {
                "offsets_query": offsets_query,
                "attention_mask_query": attention_mask_query,
                "past_key_values_query": past_key_values_query,
            }
        else:
            offsets_query = self.base_prompt["offsets_query"]
            attention_mask_query = self.base_prompt["attention_mask_query"]
            past_key_values_query = self.base_prompt["past_key_values_query"]
        for action in actions:
            prompt_to_model = history + action
            scored_tokens_dict, prompt_with_action_ids, _past_key_values = self._score_tokens(prompt_to_model, offsets_query, attention_mask_query, past_key_values_query)
            scored_tokens_dict = scored_tokens_dict[len(history_ids):]
            generated_logprobs = {"tokens": [token['token'] for token in scored_tokens_dict],
                        "token_logprobs": [token['score'] for token in scored_tokens_dict]} if logprobs else None
            response.append({"text": self.tokenizer.decode(prompt_with_action_ids).strip(), 
                            "logprobs": generated_logprobs,
                            "finish_reason": "ok"})
        return json.dumps(response, ensure_ascii=False)
