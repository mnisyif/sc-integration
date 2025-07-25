# %% ===============================================
# Load libraries
import torch
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from data.datasets import get_loader  # Updated import
# from net.adapters.llama_adapter import LLaMAAdapter
from build.build_encoder import build_encoder
from build.build_decoder import build_decoder
from net.cloud.classifier import GenericClassifier, AttentionPoolingClassifier
# from net.cloud.llama3 import LLaMAModel

TARGET = 256

# %% ===============================================
# Intialize shared state
class SharedState:
    def __init__(self):
        self.H = 0
        self.W = 0
        self.downsample = 4

state = SharedState()

# %% ===============================================
# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% ===============================================
# Initialize dataloaders
# Classification loader for class folder structure - returns (image, label) tuples
train_path = ["GenSC-Testbed/GT_Images_Classification/Train"]  # Note: needs to be a list
test_path = ["GenSC-Testbed/GT_Images_Classification/Test"]    # Note: needs to be a list

train_loader, test_loader = get_loader(
    train_dirs=train_path,
    test_dirs=test_path,
    batch_size=32,
    num_workers=4
)

print(f"Train loader: {len(train_loader)} batches")
print(f"Test loader: {len(test_loader)} batches")

# Get class mappings from the dataset
idx_to_class = test_loader.dataset.idx_to_class
class_to_idx = test_loader.dataset.class_to_idx
print(f"Classes found: {list(class_to_idx.keys())}")
print(f"Number of classes: {test_loader.dataset.num_classes}")

# %% ===============================================
# Build models
encoder = build_encoder(encoder_name='swin', device=device).to(device)

# (Option) freeze the encoder for a quick baseline:
for p in encoder.parameters():
    p.requires_grad = False        # comment‑out to fine‑tune

encoder.eval()                    # required even if frozen (BatchNorm, Dropout)

# Peek at one batch to discover feature dimension C
x0, _ = next(iter(train_loader))          # (1,3,H,W)
H0, W0 = x0.shape[2:]
encoder.update_resolution(H0, W0)

with torch.no_grad():
    C = encoder(x0.to(device)).shape[-1]  # (B, N, C) → C

num_classes = len(idx_to_class)
# clf = GenericClassifier(in_features=C, num_classes=num_classes, hidden=256, agg="mean").to(device)
clf = AttentionPoolingClassifier(in_features=C, num_classes=num_classes, hidden=256, dropout=0.3).to(device)

# %% ===============================================
# Build optimizer and criterion

# If encoder is frozen only classifier params require_grad=True
optim_groups = [
    # {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-5},
    {"params": clf.parameters(), "lr": 1e-3},
]
optimizer = torch.optim.Adam(optim_groups, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()

# %% ===============================================
# Train loop
EPOCHS = 20
for epoch in trange(1, EPOCHS + 1, desc="Epochs"):
    encoder.train(); clf.train()
    running_loss, correct, total = 0, 0, 0

    train_bar = tqdm(train_loader, leave=False, desc=f"Train {epoch:02d}", unit="batch")
    for imgs, labels in train_bar: # type: ignore
        imgs, labels = imgs.to(device), labels.to(device)

        # -- keep your dynamic‑resolution logic ----------------------------
        # _, _, H, W = imgs.shape
        # if (H, W) != (state.H, state.W):
        #     encoder.update_resolution(TARGET, TARGET)
        #     state.H, state.W = TARGET, TARGET
        # ------------------------------------------------------------------

        feats = encoder(imgs)              # (B, N, C)
        # print(f"Features shape: {feats.shape}")
        # print(f"Features mean: {feats.mean():.4f}, std: {feats.std():.4f}")
        # print(f"Features min: {feats.min():.4f}, max: {feats.max():.4f}")

        logits = clf(feats)                # (B, num_classes)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = logits.max(1)
        total   += labels.size(0)
        correct += (preds == labels).sum().item()

        train_bar.set_postfix(
            loss = f"{running_loss / total:.4f}",
            acc  = f"{100 * correct / total:.2f}%"
        )

    encoder.eval();  clf.eval()
    v_loss, v_correct, v_total = 0, 0, 0

    with torch.no_grad():
        val_bar = tqdm(test_loader, leave=False, desc=f"Val   {epoch:02d}", unit="batch")
        for imgs, labels in val_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            feats  = encoder(imgs)
            logits = clf(feats)
            loss   = criterion(logits, labels)

            v_loss += loss.item()
            _, preds = logits.max(1)
            v_total   += labels.size(0)
            v_correct += (preds == labels).sum().item()

    train_acc = 100 * correct / total
    val_acc   = 100 * v_correct / v_total
    val_loss  = v_loss / len(test_loader)

    # print a one‑liner summary that stays after the bars disappear
    print(f"[{epoch:02d}/{EPOCHS}] "
          f"train loss {running_loss/len(train_loader):.4f} "
          f"train acc {train_acc:.2f}% | "
          f"val loss {val_loss:.4f} val acc {val_acc:.2f}%")
