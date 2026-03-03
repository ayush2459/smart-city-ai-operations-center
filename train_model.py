import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ============================================================
# DEVICE SETUP (CUDA / MPS / CPU)
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ============================================================
# DATASET
# ============================================================
class VideoDataset(Dataset):
    def __init__(self, samples, seq_len=16):
        self.samples = samples
        self.seq_len = seq_len

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)

        frames = []
        count = 0

        while count < self.seq_len:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)
            count += 1

        cap.release()

        while len(frames) < self.seq_len:
            frames.append(torch.zeros(3, 224, 224))

        frames = torch.stack(frames)

        return frames, torch.tensor(label)


# ============================================================
# LOAD DATA PATHS
# ============================================================
def load_dataset(root_dir):
    classes = ["normal", "accident"]
    samples = []

    for label, cls in enumerate(classes):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.exists(cls_path):
            continue

        for file in os.listdir(cls_path):
            if file.lower().endswith(".mp4"):
                samples.append((os.path.join(cls_path, file), label))

    print("Total samples:", len(samples))
    return samples


all_samples = load_dataset("dataset/train")

train_samples, val_samples = train_test_split(
    all_samples,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_dataset = VideoDataset(train_samples)
val_dataset = VideoDataset(val_samples)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True if device.type != "cpu" else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=2,
    pin_memory=True if device.type != "cpu" else False
)

# ============================================================
# MODEL (Pretrained CNN + LSTM)
# ============================================================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # 🔥 Use pretrained weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze CNN for stability (optional)
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(512, 256, 2, batch_first=True)
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        cnn_out = []

        for t in range(seq_len):
            out = self.cnn(x[:, t])
            out = out.view(batch_size, -1)
            cnn_out.append(out)

        cnn_out = torch.stack(cnn_out, dim=1)

        lstm_out, _ = self.lstm(cnn_out)
        final = lstm_out[:, -1, :]

        return self.fc(final)


model = CNN_LSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ============================================================
# TRAINING LOOP
# ============================================================
epochs = 10
best_val_acc = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for frames, labels in train_loader:
        frames = frames.to(device)
        labels = labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # ---------------- VALIDATION ----------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "incident_model.pth")
        print("✅ Best model saved")

print("🎯 Training complete")