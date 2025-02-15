from llms import HuggingFaceClient


model = HuggingFaceClient(model_name="meta-llama/Llama-3.2-1B", mode="cpu")

print(model.make_request("Hello, how are you?"))