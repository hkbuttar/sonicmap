"""CPU training loop for the purpose-built triplet embedding."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from embeddings.triplet_model import TripletEmbeddingCNN


def train_triplet(dataset, epochs=15, batch_size=64, margin=0.2, lr=2e-3, seed=42):
    torch.manual_seed(seed)
    model = TripletEmbeddingCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    history = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        total_loss, n_samples = 0.0, 0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            anchor_z, positive_z, negative_z = model(anchor, positive, negative)
            loss = loss_fn(anchor_z, positive_z, negative_z)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(anchor)
            n_samples += len(anchor)
        mean_loss = total_loss / n_samples
        history.append({"epoch": epoch + 1, "triplet_loss": mean_loss})
        print(f"[epoch {epoch + 1}/{epochs}] triplet_loss={mean_loss:.6f}")
    return model, history
