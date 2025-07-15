import torch, requests

from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# from semantic_pipeline import SemanticPipeline
from build.build_encoder import build_encoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "lmms-lab/llama3-llava-next-8b"
    # encoder = build_encoder(encoder_name='swin', device=device)

    proc  = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=True,              # INT4 to fit in 8 GB
            torch_dtype=torch.float16,
            use_flash_attention_2=True)

    img = Image.open("GenSC-Testbed/GT_Images_Classification/Test/Armored_Personnel_Carrier/C_Test_Armored_Personnel_Carrier_0.png")
    prompt = proc.apply_chat_template(
        [{"role": "user",
        "content": [{"type":"image"},{"type":"text","text":"Give me a detailed description."}]}],
        add_generation_prompt=True
    )

    inputs = proc(images=img, text=prompt, return_tensors="pt").to(model.device)
    out    = model.generate(**inputs, max_new_tokens=120)
    print(proc.decode(out[0], skip_special_tokens=True))
    
if __name__ == "__main__":    
    main()