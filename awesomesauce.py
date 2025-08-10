from transformers import GPT2Tokenizer, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList

# Custom stopping criteria to stop on newline
class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer):
        self.newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    def __call__(self, input_ids, scores, **kwargs):
        return self.newline_token_id in input_ids[0]

model_path = "./gpt2-math-finetuned"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Test prompt
prompt = "Solve: 23 + 50 ="
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with stopping on newline
outputs = model.generate(
    **inputs,
    max_length=20,
    temperature=0.0,
    top_p=0.9,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    stopping_criteria=StoppingCriteriaList([StopOnNewline(tokenizer)])
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
