#!/usr/bin/env python3

import openai

# Put anything you want in `API key`
openai.api_key = 'API key'

# Point to your own url
openai.api_base = "http://127.0.0.1:8080"

# Do your usual things, for instance a completion query:
print(f"models list:\n{openai.Model.list()}")

prompt = '''objects = ["dried figs", "protein bar",
"cornmeal", "Macadamia nuts", "vinegar", "herbal
tea", "peanut oil", "chocolate bar", "bread
crumbs", "Folgers instant coffee"]
receptacles = ["top rack", "middle rack",
"table", "shelf", "plastic box"]
pick and place("dried figs", "plastic box")
pick and place("protein bar", "shelf")
pick and place("cornmeal", "top rack")
pick and place("Macadamia nuts", "plastic box")
pick and place("vinegar", "middle rack")
pick and place("herbal tea", "table")
pick and place("peanut oil", "middle rack")
pick and place("chocolate bar", "shelf")
pick and place("bread crumbs", "top rack")
pick and place("Folgers instant coffee", "table")
# Summary: Put dry ingredients on the top rack,
liquid ingredients in the middle rack, tea and
coffee on the table, packaged snacks on the
shelf, and dried fruits and nuts in the plastic
box.
objects = ["yoga pants", "wool sweater", "black
jeans", "Nike shorts"]
receptacles = ["hamper", "bed"]
pick and place("yoga pants", "hamper")
pick and place("wool sweater", "bed")
pick and place("black jeans", "bed")
pick and place("Nike shorts", "hamper")
# Summary: Put athletic clothes in the hamper
and other clothes on the bed.
objects = ["Nike sweatpants", "sweater", "cargo
shorts", "iPhone", "dictionary", "tablet",
"Under Armour t-shirt", "physics homework"]
receptacles = ["backpack", "closet", "desk",
"nightstand"]
pick and place("Nike sweatpants", "backpack")
pick and place("sweater", "closet")
pick and place("cargo shorts", "closet")
pick and place("iPhone", "nightstand")
pick and place("dictionary", "desk")
pick and place("tablet", "nightstand")
pick and place("Under Armour t-shirt",
"backpack")
pick and place("physics homework", "desk")
# Summary: Put workout clothes in the backpack,
other clothes in the closet, books and homeworks
on the desk, and electronics on the nightstand.
objects = ["jacket", "candy bar", "soda can",
"Pepsi can", "jeans", "wooden block", "orange",
"chips", "wooden block 2", "apple"]
receptacles = ["recycling bin", "plastic storage
box", "black storage box", "sofa", "drawer"]
pick and place("jacket", "sofa")
pick and place("candy bar", "plastic storage
box")
pick and place("soda can", "recycling bin")
pick and place("Pepsi can", "recycling bin")
pick and place("jeans", "sofa")
pick and place("wooden block", "drawer")
pick and place("orange", "black storage box")
pick and place("chips", "plastic storage box")
pick and place("wooden block 2", "drawer")
pick and place("apple", "black storage box")
# Summary:'''


variants = ['robot.pick_and_place(blue block, blue block)',
    'robot.pick_and_place(blue block, red block)',
    'robot.pick_and_place(blue block, green block)',
    'robot.pick_and_place(blue block, yellow block)',
    'robot.pick_and_place(blue block, not yellow block)',
    'robot.pick_and_place(blue block, blue bowl)',
    'robot.pick_and_place(blue block, red bowl)',
    'robot.pick_and_place(blue block, green bowl)',
    'robot.pick_and_place(blue block, yellow bowl)',
    'robot.pick_and_place(blue block, top left corner)',
    'robot.pick_and_place(blue block, top right corner)',
    'robot.pick_and_place(blue block, bottom left corner i am a human with eyes)',
    "robot.pick_and_place(blue block, "]
prompt = """To pick blue block and put it on yellow block, I should:\n"""
print(f"prompt:\n{prompt}")
for var in variants:
    new_prompt = prompt + "<|endofprompt|>" + var + "<|endofvariant|>"
    # print(f"prompt:\n{prompt}")
    completion = openai.Completion.create(model="llama-7B-4b", prompt=new_prompt, max_tokens=20, logprobs=1, echo=True)
    # print(f"completion:\n{completion}")
    logprobs = completion.choices[0].logprobs['token_logprobs']
    # print(f"logprobs:\n{logprobs}")
    logprobs = list(map(float, logprobs))
    print(f"{var}: {sum(logprobs)/len(logprobs)}, {len(logprobs)}")