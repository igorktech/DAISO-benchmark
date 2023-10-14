from datasets import load_dataset
dataset = load_dataset(
  "igorktech/daiso",
  revision="dev",
  split="dev"# tag name, or branch name, or commit hash
)
print(dataset)