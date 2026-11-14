import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from model import DeepfakeDetectionModel


class DeepfakeDetectionPipeline:
    def __init__(self, dir, lr=1e-4, weight_decay=1e-4, label_smoothing=0.05,
                 consistency_weight=0.0, supcon_weight=0.0, temperature=0.1, **model_kwargs):
        self.dir = dir
        self.train_dir = os.path.join(dir, "train")
        self.test_dir = os.path.join(dir, "test")
        self.train_csv = os.path.join(dir, "ground_truth.csv")
        self.test_csv = os.path.join(dir, "sample_submission.csv")

        self.label_smoothing = label_smoothing
        self.consistency_weight = consistency_weight
        self.supcon_weight = supcon_weight
        self.temperature = temperature
        if supcon_weight > 0:
            model_kwargs.setdefault("proj_dim", 128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepfakeDetectionModel(**model_kwargs).to(self.device)

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCEWithLogitsLoss()

    def _compute_loss_and_probs(self, logits, labels=None, smooth=False):
        if logits.ndim == 2 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
            if labels is None:
                return None, probs
            smoothing = self.label_smoothing if smooth else 0.0
            loss = nn.functional.cross_entropy(logits, labels.long(), label_smoothing=smoothing)
            return loss, probs

        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        if labels is None:
            return None, probs

        targets = labels.float()
        if smooth and self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = self.criterion(logits, targets)
        return loss, probs

    def _supcon_loss(self, emb, labels):
        # Supervised contrastive loss (Khosla et al. 2020) on L2-normalised embeddings:
        # pull together samples with the same label (incl. the other view of the same image), push apart the rest.
        sim = emb @ emb.T / self.temperature
        eye = torch.eye(len(emb), dtype=torch.bool, device=emb.device)
        sim = sim.masked_fill(eye, float("-inf"))
        pos = (labels[:, None] == labels[None, :]) & ~eye
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return -(log_prob.masked_fill(~pos, 0.0).sum(1) / pos.sum(1).clamp(min=1)).mean()

    def evaluate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss, probs = self._compute_loss_and_probs(logits, labels, smooth=False)

                running_loss += loss.item() * images.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        val_auc = roc_auc_score(all_labels, all_probs)
        val_f1 = f1_score(all_labels, preds)
        val_acc = accuracy_score(all_labels, preds)
        return val_loss, val_auc, val_f1, val_acc

    def fit(self, train_loader, val_loader=None, epochs=10, save_path="./models/best_model.pt"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        best_val_auc = 0.0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] Train", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                batch_size = labels.size(0)

                # Two augmented views per image: [B, 2, C, H, W] -> [2B, C, H, W] (views of one image are adjacent)
                two_views = images.ndim == 5
                if two_views:
                    images = images.flatten(0, 1)
                    labels = labels.repeat_interleave(2)

                self.optimiser.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    if self.supcon_weight > 0:
                        logits, emb = self.model(images, return_embedding=True)
                    else:
                        logits = self.model(images)
                    loss, probs = self._compute_loss_and_probs(logits, labels, smooth=True)
                    if two_views and self.consistency_weight > 0:
                        # Both views of an image should get the same fake-probability, whatever the degradation
                        p = probs.float().view(-1, 2)
                        loss = loss + self.consistency_weight * nn.functional.mse_loss(p[:, 0], p[:, 1])
                    if self.supcon_weight > 0:
                        loss = loss + self.supcon_weight * self._supcon_loss(emb.float(), labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimiser)
                scaler.update()

                running_loss += loss.item() * batch_size
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            train_loss = running_loss / len(train_loader.dataset)

            if val_loader is not None:
                val_loss, val_auc, val_f1, val_acc = self.evaluate(val_loader)
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_auc:.4f} | "
                    f"Val F1: {val_f1:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
                if save_path and val_auc > best_val_auc:
                    best_val_auc = val_auc
                    self.save(save_path)
                    print(f"  -> Saved best model to {save_path} (Val AUC: {best_val_auc:.4f})")
            else:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f}")
                if save_path:
                    self.save(save_path)

    def predict(self, data_loader, threshold=0.5):
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                images = images.to(self.device)
                logits = self.model(images)
                _, probs = self._compute_loss_and_probs(logits)
                all_probs.extend(probs.cpu().numpy())

        preds = [1 if p >= threshold else 0 for p in all_probs]
        return preds, all_probs

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
