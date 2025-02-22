from Img2Cap import LMM_API
from ObjDetect import BoxGen
from Reasoner.LLM_API_calling import modify_query
from Seg import GetSegFromBox

image_path = r"../Data/images/dogs.jpg"
total_caption = LMM_API(image_path)
query = '被两只狗咬住的那个物体'

modified = modify_query(query)