prompt_for_caption = """
Your task is to describe the objects in the image as detailed as possible. 
Since this will be used for subsequent tasks such as object detection and entity recognition, you should focus on the objects themselves rather than the environment, image style, or emotions. 
Only describe the main object in the picture!!!
- If there are two objects in one image, one in the center and the other in the corner or edge, you should pay more attention on the center one.
Describe all the objects in the image as thoroughly as possible. Do not omit any important details. 
You should focus on describing the objects themselves, not the background or surroundings. 
Only provide objective descriptions, without any figurative or evaluative language. 
For example, "This image shows two dogs playing on a beach. The background features a wide sky and a flat sandy beach. The sky is light blue with some white clouds. The sand appears light-colored, dry, and fine" is a good description. 
On the other hand, "This is a vibrant and joyful outdoor activity photo showing the two dogs playing and running on the beach" is not a good description, as it focuses on the emotional tone of the image, using unnecessary adjectives like "joyful" and "vibrant."
"""