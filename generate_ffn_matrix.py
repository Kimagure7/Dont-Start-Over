import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute FFN activation matrix from source soft prompts")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the source LLM (must match the model used to train soft prompts)")
    parser.add_argument("--soft_prompt_path", type=str, required=True,
                        help="Path to the source soft prompt checkpoint (checkpoint_model_best.pth)")
    parser.add_argument("--output_path", type=str, default="activation_matrix.pt",
                        help="Output path for the activation matrix (.pt file)")
    parser.add_argument("--top_k_layers", type=int, default=3,
                        help="Number of top transformer layers to extract FFN activations from")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def precompute_ffn_activations(model, tokenizer, prompt_embeddings, top_k_layers, device):
    num_prompts = prompt_embeddings.shape[0]
    layer_indices = [-(i + 1) for i in range(top_k_layers)]

    captured_activations = {}
    hooks = []

    def _get_hook(layer_name):
        def hook(module, input_tensor, output_tensor):
            captured_activations[layer_name] = output_tensor.detach().clone()
        return hook

    for layer_idx in layer_indices:
        layer_obj = model.model.layers[layer_idx]
        handle = layer_obj.mlp.act_fn.register_forward_hook(_get_hook(f"layer_{layer_idx}"))
        hooks.append(handle)

    bos_token_id = torch.tensor([tokenizer.bos_token_id], dtype=torch.long, device=device)
    bos_embedding = model.get_input_embeddings()(bos_token_id)  # (1, hidden_dim)

    all_activations = []
    with torch.no_grad():
        for i in tqdm(range(num_prompts), desc="Computing FFN activations"):
            current_prompt = prompt_embeddings[i].unsqueeze(0).to(device)  # (1, prompt_len, hidden_dim)
            input_embeds = torch.cat([bos_embedding.unsqueeze(1), current_prompt], dim=1)

            model(inputs_embeds=input_embeds)

            layer_acts = []
            for layer_idx in layer_indices:
                act = captured_activations[f"layer_{layer_idx}"][0, -1, :]  # last token position
                binary_act = (act > 0).float()
                layer_acts.append(binary_act)

            combined = torch.cat(layer_acts)
            all_activations.append(combined.cpu())

    for h in hooks:
        h.remove()

    return torch.stack(all_activations)  # (num_prompts, top_k_layers * intermediate_size)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading soft prompts from {args.soft_prompt_path} ...")
    ckpt = torch.load(args.soft_prompt_path, map_location="cpu", weights_only=True)
    user_embeddings = ckpt["user_embedding"]["weight"]  # (num_users, hidden_dim)
    user_embeddings = user_embeddings.unsqueeze(1)       # (num_users, 1, hidden_dim) — prompt_len=1

    print(f"Total users: {user_embeddings.shape[0]}, extracting from top {args.top_k_layers} layers ...")
    activation_matrix = precompute_ffn_activations(model, tokenizer, user_embeddings, args.top_k_layers, device)

    activation_matrix = activation_matrix.to(torch.int8)
    torch.save(activation_matrix, args.output_path)
    print(f"Saved activation matrix {tuple(activation_matrix.shape)} -> {args.output_path}")


if __name__ == "__main__":
    main()
