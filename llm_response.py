from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load DeepSeek model locally
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

def generate_response(transcribed_text):
    prompt = f"""You are an interview assistant. Help formulate a strong, confident response to the following interview question:
    
    "{transcribed_text}"
    
    Be concise and use technical terms where needed.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)