import requests
import numpy as np
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

import torch
from transformers.pipelines import pipeline
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

# Ensure kvpress is correctly imported
# from kvpress import ExpectedAttentionPress  # Uncomment this line if kvpress is available

# Key observation
# We start by observing that the hidden states in a transformer model follow a unimodal distribution.
# This is not true for the very first layers (for instance, layer #0 is simply the distribution of token embeddings)
# but it is true for the deeper layers (e.g. starting layer #2)

def load_pipeline_and_data():
    device = "cuda:0"
    ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipe = pipeline("kv-press-text-generation", model=ckpt, device=device, torch_dtype="auto", model_kwargs={"attn_implementation":"flash_attention_2"})

    # Load data
    url = "https://en.wikipedia.org/wiki/Nvidia"
    content = requests.get(url).content
    soup = BeautifulSoup(content, "html.parser")
    context = "".join([p.text for p in soup.find_all("p")]) + "\n\n"
    tokens = pipe.tokenizer(context, return_tensors="pt").to(device)
    n_tokens = tokens.size(1)
    print(f"Number of tokens: {n_tokens}")

    return pipe, tokens, n_tokens


def get_hidden_states(pipe, tokens):
    with torch.no_grad():
        outputs = pipe.model(tokens, output_hidden_states=True)
    return outputs


def display_hidden_states_distribution(outputs, layer_idx=12, n_dims=32, n_sink=4):
    H = outputs.hidden_states[layer_idx][0, n_sink:]
    H = H.cpu().float().numpy().T
    dims = np.random.randint(0, len(H), size=n_dims)
    H = H[dims]

    x_min = np.percentile(H, 0.1)
    x_max = np.percentile(H, 99.9)

    plt.figure(figsize=(10, 3))
    for i in range(n_dims):
        y, bin_edges = np.histogram(H[i], bins=50, density=True, range=(x_min, x_max))
        x = (bin_edges[1:] + bin_edges[:-1]) / 2

        plt.fill_between(x, y + i, i, zorder=n_dims - i, color="#76B900")
        plt.plot(x, y + i, color="black", zorder=n_dims - i)
        plt.axis("off")
    plt.title(f"Hidden state distributions, layer #{layer_idx}");
    plt.show()


def main():
    pipe, tokens, n_tokens = load_pipeline_and_data()
    outputs = get_hidden_states(pipe, tokens)
    display_hidden_states_distribution(outputs)


if __name__ == "__main__":
    main() 