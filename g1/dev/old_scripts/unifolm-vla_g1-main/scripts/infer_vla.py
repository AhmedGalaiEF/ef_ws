import torch
from PIL import Image
from transformers import CLIPProcessor
from train_vla import VLAModel

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = VLAModel().to(device)
model.load_state_dict(torch.load("vla_model.pth"))
model.eval()

def predict(image_path, text):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        action = model(inputs)

    return action.cpu().numpy()


if __name__ == "__main__":
    result = predict("test.png", "pick the red cup")
    print("Predicted pose:", result)
