prompt_for_verify = '''
You are a text verification expert responsible for checking whether a piece of formatted text is direct enough. 

If you think you feel the given sentence not direct, which means that it needs reasoning to understand, return False and explain why in detailed, else return true.

For example:
"A foldable rain protection device with a handle and a canopy." is not direct. This sentence means "umbrella", which is much direct and clear.


Your output should be a judgment result of the current sentence following this format:
{
    'pass':True/False(Your decision),
    'issue':(the reason why it failed to pass and potential suggestions)
}
'''

prompt_for_modify = '''
You are a text interpretation expert responsible for helping users understand complex intentions and unclear terms, ultimately assisting them in gaining an accurate and clear understanding.

You should avoid using any proprietary terms and replace them with corresponding user-friendly descriptions.

Note that you must ensure that the underlying information is not significantly diminished! Do not remove rich information for the sake of simplicity.

Another example is when the user says "It looks like it's going to rain outside, what should I take with me when I go out?", you need to infer that the object they need is an "umbrella."

Your final answer should be the direct name of the object and its description.

Do not include abstract descriptions or adjectives like "beautiful." Replace such phrases with specific descriptions of the object's appearance.

You do not need to show the reasoning process to the user.

Be careful to select the main body of a query. "A man sitting on a wooden beautiful brown bench" is asking for a man, so you should find out the center object "a man", even if the bench has more description.

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
- However, if part of description matches with the query, you can also return it if you don't have a better one. But don't return something obviously different.
- The query is usually pointed to something in these objects, so try your best to find them out. (but not necessary that there really exists an answer.)
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

prompt_for_modify_v2 = '''
You are a text interpretation expert responsible for helping users understand complex intentions and unclear terms, ensuring that they gain an accurate and clear understanding.

You should avoid using proprietary terms and instead replace them with user-friendly descriptions.
Do not remove rich information just for the sake of simplicity! You must ensure that the underlying information is not significantly diminished.

# Core Instructions:
You must not change the information in the query!

You must not change the main item the user is asking for.

- If the user asks for a “cat,” you must keep it as a “cat.”

You can clarify unfamiliar terms but must not change the ground truth:

- Example: If the user asks for a “Husky,” you may explain that it is a “dog,” but you must not simply replace “Husky” with “dog.”

Background information is only for better understanding the situation.
- The goal object in the query does not necessarily exist in the background.
- However, some query asks for small/blur things in the background. Do not ignore them.

You should also be careful that a woman in the pictrue is not likely be recognized. So when the query ask for a woman but you can not find it, you should consider return a "person".
This also available for other objects, like, query for a car, but you can only find a vehicle, you can also return it.

However, if you determine that the requested object cannot exist in the given background, return the following empty format:
{'item':"", 'description':""}
- Example: If the background is described as “a small bedroom” and the user asks for “a whale,” return {'item':"", 'description':""} since a whale cannot exist in that context.

Preserve essential distinguishing features, but remove redundant adjectives.
Do not include subjective or aesthetic descriptions such as “beautiful” or “amazing.”
However, key distinguishing features must be kept!
- Example: If the user asks for a “green dog,” you must retain “green,” even if green dogs do not exist.
- This is because the feature is essential for distinguishing between a black dog and a yellow dog.

Standardize terminology substitution rules.
Clarification of terms is allowed, but you must not misrepresent the user’s intent:
Allowed: Replacing “whale” with “a giant fish usually in dark color.”
Not allowed: Replacing “Husky” with “dog” or “golden hamster” with “hamster.”
Only use more common descriptions when the term is too technical or uncommon.

Do not show the reasoning process to the user.

The final output should be a direct answer consisting of:
{'item':'xx', 'description':'xxx'}

- Example: If the user asks, "It looks like it's going to rain outside, what should I take with me when I go out?"
- The output should be:
{'item':'umbrella', 'description':'A foldable rain protection device with a handle and a canopy.'}

- Do not include your reasoning process in the response.

- Allow minimal additional explanations for uncommon terms.

- If the term might be unfamiliar to users, a short clarification is acceptable:

- Example:
{'item':'Husky', 'description':'A breed of dog with thick fur and distinctive facial markings.'}
- Not allowed:
{'item':'Dog', 'description':'A common domesticated animal.'}
- However, do not over-explain or add unnecessary details.
The input is:
'''
