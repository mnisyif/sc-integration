# # %%
# import torch, os

# from torchvision.transforms import ToPILImage
# from transformers import AutoTokenizer, AutoModelForCausalLM

# from data.datasets import get_loader
# from net.adapters.llama_adapter import LLaMAAdapter
# from build.build_encoder import build_encoder
# # from build.build_decoder import build_decoder
# # from net.cloud.llama3 import LLaMAModel

# # %%
# class SharedState:
#     def __init__(self):
#         self.H = 0
#         self.W = 0
#         self.downsample = 4

# state = SharedState()

# # %%
# # Initialize device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Test single image
# image_path = "GenSC-Testbed/GT_Images_Classification/Test/Armored_Personnel_Carrier/C_Test_Armored_Personnel_Carrier_2.png"
# train_path = "GenSC-Testbed/GT_Images_Classification/Train/Armored_Personnel_Carrier"
# test_path = "GenSC-Testbed/GT_Images_Classification/Test/Armored_Personnel_Carrier"
# train_loader, test_loader = get_loader(train_dirs=[train_path], test_dirs=[test_path], batch_size=1, num_workers=4)

# # %%
# encoder = build_encoder(encoder_name='swin', device=device).to(device)

# # ——— load small LLM & tokenizer ———
# llm_name = "EleutherAI/gpt-neo-125M"
# tokenizer = AutoTokenizer.from_pretrained(llm_name)
# llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)

# # ——— adapter to map encoder outputs → llm hidden_size ———
# adapter = LLaMAAdapter(
#     input_dim=encoder.embed_dims[-1],
#     llama_dim=llm.config.hidden_size,
#     num_visual_tokens=256         # up from 128
# ).to(device)          # [1, …]

# # %%
# with torch.no_grad():
#     for b_idx, batch in enumerate(test_loader):
#         x = batch.to(device)                    # [1,3,H,W]
#         _, _, H, W = x.shape
#         if H!=state.H or W!=state.W:
#             encoder.update_resolution(H, W)
#             state.H, state.W = H, W

#         feats = encoder(x)                      # [1, N, C]
#         adapted = adapter(feats)               # [1, num_tokens, hidden_size]

#         # improved prompt
#         prompt = (
#             "You are a helpful assistant that describes images in detail. "
#             "Provide a thorough description of objects, colors, actions and scene context:"
#         )
#         enc = tokenizer(prompt, return_tensors="pt")
#         txt_emb = llm.get_input_embeddings()(enc.input_ids.to(device))

#         # prepend visual tokens
#         inputs_embeds = torch.cat([adapted, txt_emb], dim=1)

#         # more powerful decode settings
#         outputs = llm.generate(
#             inputs_embeds=inputs_embeds,
#             max_new_tokens=100,
#             num_beams=4,
#             early_stopping=True,
#             no_repeat_ngram_size=2,
#             repetition_penalty=1.2,
#             temperature=0.7,
#             top_p=0.9
#         )
#         caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"Caption: {caption}")
#         break
# # %%

# %%
import torch, os
from torchvision.transforms import ToPILImage
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.datasets import get_loader  # Updated import
# from net.adapters.llama_adapter import LLaMAAdapter
from build.build_encoder import build_encoder
from build.build_decoder import build_decoder
from net.cloud.classifier import GenericClassifier
# from net.cloud.llama3 import LLaMAModel

# %%
class SharedState:
    def __init__(self):
        self.H = 0
        self.W = 0
        self.downsample = 4

state = SharedState()

# %%
# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Classification loader for class folder structure - returns (image, label) tuples
train_path = ["GenSC-Testbed/GT_Images_Classification/Train"]  # Note: needs to be a list
test_path = ["GenSC-Testbed/GT_Images_Classification/Test"]    # Note: needs to be a list

train_loader, test_loader = get_loader(
    train_dirs=train_path,
    test_dirs=test_path,
    batch_size=8,
    num_workers=4
)

print(f"Train loader: {len(train_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")

# Get class mappings from the dataset
idx_to_class = test_loader.dataset.idx_to_class
class_to_idx = test_loader.dataset.class_to_idx
print(f"Classes found: {list(class_to_idx.keys())}")
print(f"Number of classes: {test_loader.dataset.num_classes}")

# %%
encoder = build_encoder(encoder_name='swin', device=device).to(device)
decoder = build_decoder(decoder_name='swin', device=device).to(device)

# Create output directory
os.makedirs('run', exist_ok=True)


# %%
with torch.no_grad():
    for b_idx, (x, label) in enumerate(test_loader):  # Simple tuple unpacking
        x = x.to(device)  # [1,3,H,W]
        label = label.to(device)  # [1]
        
        # Get class name from label using the reverse mapping
        class_name = idx_to_class[label.item()]
        print(f"Processing: {class_name} (label: {label.item()})")
        
        _, _, H, W = x.shape
        if H != state.H or W != state.W:
            encoder.update_resolution(H, W)
            decoder.update_resolution(H // (2 ** state.downsample), W // (2 ** state.downsample))
            state.H, state.W = H, W
            
        feats = encoder(x)  # [1, N, C]
        recon = decoder(feats)  # [1, N, C] - assuming decoder outputs same shape as encoder input
        
        # Convert tensors to PIL images for saving
        to_pil = ToPILImage()
        
        # Handle original image (no normalization expected in generic mode)
        orig_clamped = torch.clamp(x, 0, 1)
        
        # Handle reconstructed image
        recon_clamped = torch.clamp(recon, 0, 1)
        
        # Convert to PIL images
        orig_pil = to_pil(orig_clamped.squeeze(0).cpu())  # Remove batch dim and move to CPU
        recon_pil = to_pil(recon_clamped.squeeze(0).cpu())  # Remove batch dim and move to CPU
        
        # Create filenames
        filename_base = f"image_{b_idx:04d}"
        orig_path = os.path.join('run', f"{filename_base}_original.png")
        recon_path = os.path.join('run', f"{filename_base}_reconstructed.png")
        
        # Save images
        orig_pil.save(orig_path)
        recon_pil.save(recon_path)
        
        print(f"Saved original: {orig_path}")
        print(f"Saved reconstruction: {recon_path}")
        print(f"Original shape: {x.shape}, Reconstructed shape: {recon.shape}")
        print("-" * 50)
        
        # Process only first few images for testing
        if b_idx >= 4:  # Save first 5 images
            break