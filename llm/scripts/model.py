import logging
from dataclasses import dataclass
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
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor = None,
        prefix_allowed_tokens_fn = None,
        existed_tokens = None,
        **kwargs,
    ):
        '''This is a copy of the generate function from transformers.generation_utils.GenerationMixin'''

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        generation_config.validate()
        self.model._validate_model_kwargs(model_kwargs.copy())
        # 2. Set generation parameters if not already defined

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )

        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.model.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self.model._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )
        
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        return self.complete_search(
                input_ids,
                logits_processor=logits_processor,
                existed_tokens = existed_tokens,
                **model_kwargs,
            )
        
    def complete_search(self,
        input_ids: torch.LongTensor,
        logits_processor = None,
        existed_tokens: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ):
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        tokens, token_logprobs = [], []
        for existed_token in existed_tokens[0][1:]:

            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=self.model.generation_config.output_attentions,
                output_hidden_states=self.model.generation_config.output_hidden_states,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # get existed_token's score and replace next_token with existed_token
            existed_token_reshaped = existed_token.repeat(next_tokens.shape[0])
            existed_token_score = next_tokens_scores[0][existed_token_reshaped]
            # variant_token_scores.append(float(existed_token_score.cpu().numpy()[0]))
            tokens.append(self.tokenizer.decode(existed_token.cpu().numpy()))
            token_logprobs.append(float(existed_token_score.cpu().numpy()[0]))
            next_tokens = existed_token_reshaped
            
            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
        return input_ids, tokens, token_logprobs
    
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
        prompt, variants = prompt.split("<|endofprompt|>")
        variants = list(variants.split("<|endofvariant|>"))
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            do_sample=True,
            beam_group_size=1,
        )
        # generation_config = GenerationConfig(
        #     temperature=temperature,
        #     top_p=top_p,
        #     num_beams=4,
        #     do_sample=True,
        #     beam_group_size=1,
            
        # )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        input_ids = input_ids.cuda()

        
        variant = variants[0]
        if echo and logprobs:
            existed_tokens = self.tokenizer(variant, return_tensors="pt")["input_ids"].cuda()
            ids, tokens, token_logprobs = self.generate(input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_tokens,
                existed_tokens=existed_tokens,
            )

        # # TODO: add completion
        # if max_tokens > 0:
        #     output = self.model.generate(
        #         input_ids=input_ids,
        #         generation_config=generation_config,
        #         return_dict_in_generate=True,
        #         output_scores=True,
        #         max_new_tokens=max_tokens,
        #     )
        results = []
        # for sequence in output.sequences:
        #     results.append(self.tokenizer.decode(sequence).strip())
        for idss in ids:
            results.append(self.tokenizer.decode(idss).strip())
        generated_logprobs = {"tokens": tokens,
                    "token_logprobs": token_logprobs} if logprobs else None
        return json.dumps({"text": results[0], 
                           "logprobs": generated_logprobs,
                            "finish_reason": ""})