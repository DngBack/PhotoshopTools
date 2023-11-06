from rembg import remove
from PIL import Image

# set some config
input_url = "./Image/Test3.jpg"
output_url = "./Output/Output.png"

input_path = input_url
output_path = output_url

input = Image.open(input_path)
output = remove(input)
output.save(output_path)
