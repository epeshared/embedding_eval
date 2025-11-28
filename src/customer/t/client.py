# import requests

# url = f"http://localhost:{30000}/v1/embeddings"

# data = {
#     "model": "openai/clip-vit-base-patch32",
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "What’s in this image?"},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": "https://github.com/sgl-project/sglang/blob/main/examples/assets/example_image.png?raw=true"
#                     },
#                 },
#             ],
#         }
#     ],
#     "max_tokens": 300,
# }

# response = requests.post(url, json=data)
# print_highlight(response.text)


import requests

url = "http://127.0.0.1:30000"

text_input = "<image> Represent this image in embedding space."
image_path = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main/images/023.jpg"

payload = {
    "model": "openai/clip-vit-large-patch14-336",
    "input": [{"text": text_input}, {"image": image_path}],
}

response = requests.post(url + "/v1/embeddings", json=payload).json()

print("Embeddings:", [x.get("embedding") for x in response.get("data", [])])