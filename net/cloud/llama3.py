# import os
# import torch
# from PIL import Image
# from transformers import AutoTokenizer, AutoModelForCausalLM

# class ImageCaptioner:
#     def __init__(self, model_name="qresearch/llama-3.1-8B-vision-378"):
#         """
#         Initialize the Image Captioner with LLaMA 3.1 Vision model
#         """
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_name = model_name
#         self.model = None
#         self.tokenizer = None
        
#         print(f"Using device: {self.device}")
#         self.load_model()
    
#     def load_model(self):
#         """
#         Load the pre-trained LLaMA 3.1 Vision model and tokenizer
#         """
#         print(f"Loading pre-trained model: {self.model_name}...")
#         try:
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 trust_remote_code=True,
#                 torch_dtype=torch.float16,
#                 device_map="auto"
#             )
            
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
#             print("LLaMA 3.1 Vision model loaded successfully!")
            
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise
    
#     def describe_image(self, image_path, prompt="Caption the image", max_tokens=128):
#         """
#         Function to get image description using LLaMA 3.1 Vision model
#         """
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image not found: {image_path}")
        
#         # Open and load image
#         image = Image.open(image_path).convert("RGB")
        
#         # Generate description using LLaMA 3.1 Vision model
#         description = self.model.answer_question(
#             image, 
#             prompt, 
#             self.tokenizer, 
#             max_new_tokens=max_tokens, 
#             do_sample=True, 
#             temperature=0.3
#         )
        
#         return description
    
#     def batch_describe_images(self, image_paths, prompt="Caption the image"):
#         """
#         Process multiple images at once
#         """
#         descriptions = []
#         for image_path in image_paths:
#             try:
#                 description = self.describe_image(image_path, prompt)
#                 descriptions.append({
#                     "image_path": image_path,
#                     "description": description
#                 })
#             except Exception as e:
#                 descriptions.append({
#                     "image_path": image_path,
#                     "error": str(e)
#                 })
#         return descriptions

# def main():
#     # Initialize the LLaMA 3.1 Vision captioner
#     captioner = ImageCaptioner()
    
#     # Get image path from user
#     image_path = input("Enter image path: ")
    
#     try:
#         print("Analyzing image with LLaMA 3.1 Vision...")
#         description = captioner.describe_image(image_path)
#         print(f"\nImage Description: {description}")
#     except Exception as e:
#         print(f"Error: {e}")

# if __name__ == "__main__":    
#     main()
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLaMAModel:
    """Wrapper for LLaMA model"""
    
    def __init__(self, model_name="qresearch/llama-3.1-8B-vision-378", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self._device = device
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        
        # Load model and tokenizer (no auto‐offload → keep full model on `device`)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=None,            # disable splitting onto meta/CPU
            low_cpu_mem_usage=False
        ).to(self._device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully!")
    
    def generate_from_features(self, adapted_features, prompt="Caption the image", max_tokens=128):
        """Generate text from adapted visual features"""
        with torch.no_grad():
            response = self.model.generate(
                image_embeds   = adapted_features.to(self._device),
                prompt         = prompt,
                tokenizer      = self.tokenizer,
                max_new_tokens = max_tokens,
                do_sample      = True,
                temperature    = 0.3
            )
        # response is typically a string or list of strings
        if isinstance(response, (list, tuple)):
            return response[0]
        return response
    
    def get_embedding_dim(self):
        """Get model embedding dimension from embedding layer"""
        try:
            # Get embedding dimension from the embedding layer weight
            embed_weight = self.model.get_input_embeddings().weight
            embedding_dim = embed_weight.shape[1]  # This should be 4096
            print(f"Embedding dimension: {embedding_dim}")
            return embedding_dim
        except Exception as e:
            print(f"Could not get embedding dimension: {e}")
            # Fallback to known dimension for this model
            return 4096