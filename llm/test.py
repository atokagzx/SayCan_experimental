#!/usr/bin/env python3

import openai

# Put anything you want in `API key`
openai.api_key = 'API key'

# Point to your own url
openai.api_base = "http://127.0.0.1:8080"

def generate_list(pick_obj):
    place_obj = pick_obj.copy()
    place_obj.extend(['table'])
    variants = []
    for pick in pick_obj:
        for place in place_obj:
            if pick is place:
                continue
            variants.append({"pick": pick, "place": place, "text": f"Pick the {pick} and place it on the {place}."})
    return variants


# prompt = '''There is a blue block, a red block, a green block, a yellow block, a fish on the table.
# Imagine that you are a robotic arm with a suction cup and can only take the top block from the stack.
# Describe the action plan for building a tower of maximum height:
# '''
prompt = '''On the table are {available_objects}.
List the order of stacking the vertical tower from these colored blocks.
Write each action as: Pick the "some block" and place it on the "another block".
Separate actions with a new line. At the end of the work, type "done()". 
'''
available_objs = ["red block", "green block", "yellow block", "fish", "blue block"]

while True:
    variants = generate_list(available_objs)
    pick_objs = list(map(lambda x: x["pick"], variants))
    place_objs = list(map(lambda x: x["place"], variants))
    available_objs_text = ", ".join(available_objs)
    new_prompt = prompt.format(available_objects=available_objs_text)
    print(new_prompt) 
    new_prompt = new_prompt + "<|endofprompt|>"
    new_prompt += "<|endofvariant|>".join(list(map(lambda x: x["text"], variants)))
    completion = openai.Completion.create(model="llama-7B-4b", prompt=new_prompt, max_tokens=20, logprobs=True, echo=True)
    # print(completion)
    logprobs_avgs = [sum(choice.logprobs.token_logprobs[1:]) / len(choice.logprobs.token_logprobs)-1 for choice in completion.choices]
    rated = [{"text": text, "logprobs": logprobs_avg} for text, logprobs_avg in zip(variants, logprobs_avgs)]
    rated.sort(key=lambda x: x["logprobs"], reverse=True)
    print(*rated, sep="\n")
    index = int(input("Choose index: "))
    print(f"picked: {rated[index]['text']['pick']}, placed: {rated[index]['text']['place']}")
    available_objs.remove(rated[index]['text']["place"])
    prompt += rated[index]["text"]['text'] + "\n"
