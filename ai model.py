# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="XingChen-AGI/Xing4.0-29B-A4B", trust_remote_code=True)
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe(messages)

# Load model directly
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("XingChen-AGI/Xing4.0-29B-A4B", trust_remote_code=True, device_map="auto")