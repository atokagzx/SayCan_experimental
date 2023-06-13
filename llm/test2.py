#!/usr/bin/env python3

import openai

# Put anything you want in `API key`
openai.api_key = "API key"

# Point to your own url
#openai.api_base = "http://127.0.0.1:8080"

pick_obj = ["red block", "green block", "yellow block", "fish", "blue block"]
place_obj = pick_obj.copy()
place_obj.extend(['table'])
variants = ["done()", "<|endoftext|>"]
for pick in pick_obj:
    for place in place_obj:
        if pick is place:
            continue
        variants.append(f"Pick the {pick} and place it on the {place}")

# prompt = '''There is a blue block, a red block, a green block, a yellow block, a fish on the table.
# Imagine that you are a robotic arm with a suction cup and can only take the top block from the stack.
# Describe the action plan for building a tower of maximum height:
# '''
# prompt = '''On the table are a blue block, a red block, a green block, a yellow block, and a fish.
# Block can be on the table or on the another block. You can pick block and place block on the another block. 
# If block is placed you can not pick it. You can not pick placed blocks. You can only pick block on which you have not yet placed other block.
# List the order of stacking the vertical tower from all these colored blocks.
# Write each action as: Pick the "some block" and place it on the "another block". After all actions type "done()".
# '''
# You can pick only blocks. Dont take the fish.
# You can pick block and place block on the another block.  
# If block is placed you can not pick it. You can not pick placed blocks. 
# You can only pick block on which you have not yet placed other block.
# prompt = '''You have blue block, a red block, a green block, a yellow block, and a fish.
# Indicate the order in which all colored blocks are arranged sequentially one on top of the other.
# Write each action as: On the "another block" put the "some block". After all actions type "done()".'''
prompt = '''On the table are a blue block, a red block, a green block, a yellow block, mashrooms and a fish.
List the order of stacking the vertical tower from these colored blocks.
Write each action as: Pick the "some block" and place it on the "another block".
Separate actions with a new line. At the end of the work, type "done()". 
Pick the'''
#completion = openai.Completion.create(model="llama-7B-4b", prompt=prompt, max_tokens=100, logprobs=True, echo=True)
completion = openai.Completion.create(engine="text-davinci-002", prompt=prompt, max_tokens=100, echo=True)
print(completion.choices[0].text)
# print(completion)