import torch, os

from torchvision.transforms import ToPILImage

from data.datasets import get_loader
from build.build_encoder import build_encoder
# from build.build_decoder import build_decoder
from net.adapters.llama_adapter import LLaMAAdapter
from net.cloud.llama3 import LLaMAModel
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
    
    # adapter = LLaMAAdapter(
    #     input_dim=encoder.embed_dims[-1],       # or the last encoder dimension
    #     llama_dim=LLaMAModel().get_embedding_dim(),
    #     num_visual_tokens=256
    # ).to(device)
    # # llama model
    # llama = LLaMAModel(device=device)

    with torch.no_grad():

        for b_idx, batch in enumerate(test_loader):
            # batch has shape [1, 3, H, W]
                x = batch.to(device)

                _, _, H, W = x.shape
                if H != state.H or W != state.W:
                    encoder.update_resolution(H, W) # type: ignore
                    # decoder.update_resolution(H // (2 ** state.downsample), W // (2 ** state.downsample)) # type: ignore
                    state.H, state.W = H, W

                # forward through encoder + decoder
                feats       = encoder(x)               # [1, …]
                encoder = encoder.to("cpu")
                torch.cuda.empty_cache()

                adapter = LLaMAAdapter(
                    input_dim=320,       # or the last encoder dimension
                    llama_dim=LLaMAModel().get_embedding_dim(),
                    num_visual_tokens=256
                ).to(device)
                # llama model
                llama = LLaMAModel(device=device)
                # 4) adapt to fixed‐size visual tokens
                vis_tokens = adapter(feats)        # [1, num_tokens, llama_dim]

                # 5) generate caption
                caption = llama.generate_from_features(
                    vis_tokens,
                    prompt="Describe the image in one phrase:"
                )
                print("Caption:", caption)

if __name__ == "__main__":    
    main()