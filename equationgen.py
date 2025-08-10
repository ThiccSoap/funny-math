import random

# How many examples to generate
num_examples = 600  # change to any number you want

output_file = "math_data.txt"

with open(output_file, "w") as f:
    for _ in range(num_examples):
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        f.write(f"Solve: {a} + {b} = {a + b}\n")

print(f"✅ Generated {num_examples} addition problems in {output_file}")
