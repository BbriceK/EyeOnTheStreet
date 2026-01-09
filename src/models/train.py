import argparse
import os
import torch
import numpy as np
import torch.optim as optim

from .full_model import FullModel
from .loss import AsymmetricLoss
from .model_utils import seed_everything, load_embeddings, EmbeddingDataset, build_loaders, train_one_epoch, evaluate, evaluate_predictions_and_save

def main(emb_path, save_path, out_path, num_batch=32, num_classes=4, lr=1e-4, weight_decay=1e-3, num_epochs=150, device=None):
    # ---- Device setup ----
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")    
    seed_everything()

    # ---- Load data ----
    data = load_embeddings(emb_path)
    train_loader, val_loader, test_loader = build_loaders(data, batch_size=num_batch)
    CATEGORY_NAMES = ["Curb extensions", "Cycle Tracks", "Median island", "Speed Hump"]

    # ---- Initialize model, optimizer, loss ----
    model = FullModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = AsymmetricLoss()

    best_val_loss = float('inf')
    os.makedirs(save_path, exist_ok=True)
    
    # ---- Training loop ----
    for epoch in range(num_epochs):
        # Train
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # Validate
        avg_val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_file = os.path.join(save_path, "best.pth")
            torch.save(model.state_dict(), model_file)
            print(f"Best model saved → {model_file}")

    # ---- Load best model and test ----
    print(f"Loading best model from {model_file}")
    model.load_state_dict(torch.load(model_file, map_location=device))

    avg_test_loss, y_pred_full, y_true_full = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # Save predictions
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, "y_pred_full.npy"), y_pred_full)
    np.save(os.path.join(out_path, "y_true_full.npy"), y_true_full)
    print(f"Saved y_pred_full.npy and y_true_full.npy in {out_path}")

    # ---- Compute & save metrics ---- 
    evaluate_predictions_and_save(
        y_true=y_true_full,
        y_pred=y_pred_full,
        category_names=CATEGORY_NAMES,
        out_path=out_path,
        thresholds=(0.5, 0.9),
    )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate FullModel")
    parser.add_argument('emb_path', type=str, help='Path to the embedding folder')
    parser.add_argument('save_path', type=str, help='Path to save the best model weight')
    parser.add_argument('out_path', type=str, help='Path to save predictions')
    parser.add_argument('--num_batch', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=150)
    args = parser.parse_args()

    main(
        emb_path=args.emb_path,
        save_path=args.save_path,
        out_path=args.out_path,
        num_batch=args.num_batch,
        num_classes=args.num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs
    )
