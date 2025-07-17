import torch, os

from torchvision.transforms import ToPILImage
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.datasets import get_loader
from net.adapters.llama_adapter import LLaMAAdapter
from build.build_encoder import build_encoder
# from build.build_decoder import build_decoder
# from net.cloud.llama3 import LLaMAModel
class SharedState:
    def __init__(self):
        self.H = 0
        self.W = 0
        self.downsample = 4

def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Test single image
    image_path = "GenSC-Testbed/GT_Images_Classification/Test/Armored_Personnel_Carrier/C_Test_Armored_Personnel_Carrier_2.png"
    train_path = "GenSC-Testbed/GT_Images_Classification/Train/Armored_Personnel_Carrier"
    test_path = "GenSC-Testbed/GT_Images_Classification/Test/Armored_Personnel_Carrier"
    
    state = SharedState()

    train_loader, test_loader = get_loader(train_dirs=[train_path], test_dirs=[test_path], batch_size=1, num_workers=4)

    # print(f"Train_loader size: {train_loader.__len__()}, Test_loader size: {test_loader.__len__()}")
    encoder = build_encoder(encoder_name='swin', device=device).to(device)

    # ——— load small LLM & tokenizer ———
    llm_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)

    # ——— adapter to map encoder outputs → llm hidden_size ———
    adapter = LLaMAAdapter(
        input_dim=encoder.embed_dims[-1],
        llama_dim=llm.config.hidden_size,
        num_visual_tokens=32
    ).to(device)

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            x = batch.to(device)                    # [1,3,H,W]
            _, _, H, W = x.shape
            if H!=state.H or W!=state.W:
                encoder.update_resolution(H, W)
                state.H, state.W = H, W

            feats = encoder(x)                      # [1, N, C]
            adapted = adapter(feats)               # [1, num_tokens, hidden_size]

            # build text prompt embeddings
            prompt = "Describe the image:"
            enc = tokenizer(prompt, return_tensors="pt")
            txt_emb = llm.get_input_embeddings()(enc.input_ids.to(device))

            # prepend visual tokens
            inputs_embeds = torch.cat([adapted, txt_emb], dim=1)

            # generate
            outputs = llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
            caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Caption: {caption}")
            break

if __name__ == "__main__":    
    main()