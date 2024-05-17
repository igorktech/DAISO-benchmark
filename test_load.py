from datasets import load_dataset
dataset = load_dataset(
  "igorktech/daiso",
  revision="dev",
  split="test",# tag name, or branch name, or commit hash
  cache_dir = 'cache'
)
print(dataset[1])