def generate_request(prompt, variants):
    prompt += "<|endofprompt|>"
    prompt += "<|endofvariant|>".join(variants)
    return prompt