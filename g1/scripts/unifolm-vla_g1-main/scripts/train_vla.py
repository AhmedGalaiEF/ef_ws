import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -------------------------------
# Dataset
# -------------------------------
class VLADataset(Dataset):
    def __init__(self, json_path, image_dir, processor):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(f"{self.image_dir}/{item['image']}").convert("RGB")
        text = item["text"]
        action = torch.tensor(item["action"], dtype=torch.float32)

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        )

        return inputs, action


# -------------------------------
# Model
# -------------------------------
class VLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze CLIP (important initially)
        for param in self.clip.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # (x,y,z,r,p,y)
        )

    def forward(self, inputs):
        outputs = self.clip(**inputs)

        # Use pooled embedding
        image_emb = outputs.image_embeds
        text_emb = outputs.text_embeds

        fused = (image_emb + text_emb) / 2.0

        action = self.fc(fused)
        return action


# -------------------------------
# Training
# -------------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = VLADataset(
        json_path="dataset/data.json",
        image_dir="dataset/images",
        processor=processor
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = VLAModel().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(20):
        total_loss = 0

        for inputs, target in dataloader:
            inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()}
            target = target.to(device)

            pred = model(inputs)

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "vla_model.pth")


if __name__ == "__main__":
    train()
