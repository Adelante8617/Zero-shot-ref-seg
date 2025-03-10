prompt_for_verify = '''
You are a text verification expert responsible for checking whether a piece of formatted text is direct enough. The term "direct" here refers solely to whether inference is required, without the need for conciseness.

The text format is as follows: {'item': 'xx', 'description': 'xxx'}
For such text, you should determine whether it contains indirect or ambiguous expressions.

A sentence that passes the check should have a direct noun in the 'item' section and necessary explanation in the 'description' section, **without requiring further inference** to identify the final item. 

In other words, a sentence that passes the check can be used directly to ask someone to retrieve an object.

For example, "A tool used to avoid getting wet on rainy days" is an indirect expression, and should be directly stated as "umbrella". It should output False because it points to the umbrella in a complicated way, which could be replaced by the word "umbrella."

For example, "Apple 16 Pro" is an expression with a proper noun, and should be expressed as "a new smartphone". It should output False because "Apple 16" refers to a specific model, which may not be common knowledge to the general public.

However, **redundant common-sense descriptive information** is allowed, **conciseness is not required!!!**, and it should be considered a pass.

Redundancy is allowed!!

Your output should be a judgment result of the current sentence and possible modification suggestions, following this format:
{
    'pass':True/False(Your decision),
    'issue':(the reason why it failed to pass and potential suggestions)
}
'''

prompt_for_modify = '''
You are a text interpretation expert responsible for helping users understand complex intentions and unclear terms, ultimately assisting them in gaining an accurate and clear understanding.

You should avoid using any proprietary terms and replace them with corresponding user-friendly descriptions.

Note that you must ensure that the underlying information is not significantly diminished! Do not remove rich information for the sake of simplicity.

For example, when the user mentions "Messi," you need to transform it into a form that can be understood without additional knowledge.

For example, "a South American man with a short stature and a large beard" should be output in the following format: {'item': 'athlete','description': 'xxxxx'}

Another example is when the user says "It looks like it's going to rain outside, what should I take with me when I go out?", you need to infer that the object they need is an "umbrella."

Your final answer should be the direct name of the object and its description, and these descriptions should be visually observable or reflected in an image.

Do not include abstract descriptions or adjectives like "beautiful." Replace such phrases with specific descriptions of the object's appearance.

You do not need to show the reasoning process to the user.

You should only focus on the most essential item or category, then respond accordingly.

YOU CANNOT CHANGE THE INFORMATION IN THE QUERY!!! 
- you should not change the main item that the user ask for
- if they ask for a cat, keep the query as cat
- all you can do is to make their query more easy to understand
    - someone don't know what is a Husky, so you can tell them this is exactly a dog, but you cannot change the ground truth that they ask for a dog!
    - another example is that i don't know what is whale, you should change them into 'a giant fish usually in dark color', but you cannot change that i want a fish!  
- some background information is also provided, but they are only for your better understanding the situation
    - the query's goal object do not necessary exist in the background

You can't omit important informations! If i ask for a greed dog, even if there's impossible to find a green color dog, you should change this important feature
Because this is used to identify whether a black dog or yellow dog matches with this feature.

However, if you think that there can't be possible to find such things in the background, you should directly return {'item':"", 'description':""}, since it's impossible to find something doesn't exist

The output should strictly follow this format, where 'item' and 'description' are dictionary keys that cannot be changed: 
{'item':'xx', 'description':'xxx'}'''

prompt_for_select = '''
I will give you a general description of all objects, a query for select some object(s),  and some objects' description.
You need to judge and reply which objects should be selected.
Your judgement should be made by reasoning with common knowledge.
Don't be strict with names. For example, a 'Husky' is also a 'dog'; a 'sparrow' is also a 'bird', etc.

Example:
The input is formative:
{
    'general': 'The image shows two dogs running on a sandy beach. One dog is light brown with a long tail and ears, while the other is black and white with a red collar. They are both carrying a stick in their mouths as they playfully interact with each other. The background features a wide sky with some white clouds scattered throughout. The sand appears light-colored, sandy, and smooth. The overall scene is bright and sunny.'
    'query': "dog with light brown color, long tail and ears, carrying a stick in its mouth",
    'object_list':[
        'a yellow dog running on sand',
        'a small black dog carrying a stick'
    ]
}
And you should decide what object should be selected for query, and reply their index:
{
    'selected_ids':[1]
}
In this example the query ask for a dog with light brown color, according to the general description, "One dog is light brown with a long tail and ears", which is correspond with the 1st object in the list.


Note that: 
- The index starts with 1.
- You need to carefully think about the meaning of query, including the implicit numerical relationship.
    - For example, if the query asked for the biggest cat, this implies that you should only return at most one object.
    - Another example, if the query asked for 'all birds', you should pick out all birds mentioned in list.
- Only when the object is matched with the query should you return it. DO NOT return all related objects!!!
- However, if part of description matches with the query, you can also return it if you don't have a better one. But don't return something obviously different.
- It is not necessary that every word matches. As is displayed in the example, 'yellow' is similar with 'light brown'. Such slight difference is acceptable.
- However, you should not reply an index if all objects are greatly different with the query. For example, if there lists some 'cats' but asks for 'dogs', you should return an empty list.
- If the query ask for multiple objects, such as "all dogs", you should return all indexs. Don't omit any of them.
- Your response should follow the given format:
{
    'selected_ids':[(the index you selected, which **MUST be correspond with the order** in input object list)]
}
'''

prompt_for_rewrite = '''
Rewrite this sentence. Note that you should convert it into a sentence that have the same meaning, but not so straightforward. For example, you can replace some noun in it with a description to the noun, without the noun itself's appearance.

The sentence to rewrite is:
'''

