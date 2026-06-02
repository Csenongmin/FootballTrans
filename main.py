from model.model import MorphologySequenceDataset, LSTMMorphologyModel, TransformerMorphologyModel
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import train
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

batch_size = 64
defensive_df = pd.read_csv("tracking_first_defensive.csv")

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.append(preds.reshape(-1))
            all_labels.append(y_batch.numpy().reshape(-1))
    print("Input shape:", X_batch.shape)
    print("Logits shape:", logits.shape)
    print("Pred shape:", preds.shape)
    print("Target shape:", y_batch.shape)

    return np.concatenate(all_labels), np.concatenate(all_preds)

if __name__ == '__main__':    
    # ------A. Home 11 + Away 11 + ball 1 ----- #
    feature_cols_A = [
        col for col in defensive_df.columns
        if col not in [
            "morphology",
            "morphology_id",
            "possession",
            "frame",
            "frame_diff",
            "segment_id"
        ]
    ]
    # -----B. Home 7(no defense) + Away 11 + ball 1 ---- #
    # Home 4-back 수비수 목록
    home_4back_players = ["H2", "H3", "H7", "H10"]
   
    # # 제거할 4-back 좌표 column
    exclude_4back_cols = []
    
    for p in home_4back_players:
        exclude_4back_cols.extend([f"{p}_x", f"{p}_y"])
    
    print("제거할 Home 4-back columns:")
    print(exclude_4back_cols)

    non_feature_cols = [
        "morphology",
        "morphology_id",
        "possession",
        "frame",
        "frame_diff",
        "segment_id"
    ]

    # 실험 B feature columns
    feature_cols_B = [
        col for col in defensive_df.columns
        if col not in non_feature_cols
        and col not in exclude_4back_cols
    ]
    # --------- C. Away 4 + Ball 1 -------- #
    away_4attack_players = ["A3", "A5" ,"A8", "A10"]
    include_4attk_cols = []
    for p in away_4attack_players:
        include_4attk_cols.extend([f"{p}_x", f"{p}_y"])
    include_4attk_cols.append('ball_x')
    include_4attk_cols.append('ball_y')
    feature_cols_C = [
        col for col in defensive_df.columns
        if col not in non_feature_cols
        and col in include_4attk_cols
    ]
    print(feature_cols_C)
    # --------------lavel encoder --------------#
    le = LabelEncoder()
    defensive_df["morphology"] = defensive_df["morphology"].astype(str)
    defensive_df["morphology_id"] = le.fit_transform(defensive_df["morphology"])
    X_list = []
    y_list = []

    #----------------- Sliding Window ------------------------#
    # 75 frame per sequences
    SEQ_LEN = 75
    STRIDE = 5
    for seg_id, seg_df in defensive_df.groupby("segment_id"):
        seg_df = seg_df.reset_index(drop=True)
        
        if len(seg_df) < SEQ_LEN:
            continue
        
        for start in range(0, len(seg_df) - SEQ_LEN + 1, STRIDE):
            end = start + SEQ_LEN
            
            X_seq = seg_df.iloc[start:end][feature_cols_A].values
            y_seq = seg_df.iloc[start:end]["morphology_id"].values
            
            X_list.append(X_seq)
            y_list.append(y_seq)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(X.shape)
    print(y.shape)

    # --------------- train, validate, test random split -----------#
    num_samples = len(X)

    indices = np.arange(len(X))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    train_dataset = MorphologySequenceDataset(X_train, y_train)
    val_dataset = MorphologySequenceDataset(X_val, y_val)
    test_dataset = MorphologySequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------- Model ---------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim_a = 46
    input_dim_b = 38
    
    input_dim_c = 10
    # LSTM
    # model = LSTMMorphologyModel(
    #     input_dim=input_dim_c,
    #     hidden_dim=128,
    #     num_layers=2,
    #     num_classes=45,
    #     dropout=0.2
    # ).to(device)
    
    # Transformer
    model = TransformerMorphologyModel(
        input_dim=input_dim_a,
        num_classes=45,
        seq_len=75,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # -------------- loss - cross entrophy --------------#
    num_classes = 45
    classes, counts = np.unique(y_train, return_counts=True)

    class_counts = np.zeros(num_classes, dtype=np.float32)
    class_counts[classes] = counts

    weights = np.sqrt(class_counts.sum() / (num_classes * class_counts))
    weights = np.clip(weights, 0.5, 10.0)

    # 핵심: numpy -> torch tensor로 변환
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    # ------------------ Train epoch 20 -------------------#
    num_epochs = 20

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train.train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = train.evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # ------------ Model Test -----------#
    test_loss, test_acc = train.evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    #----------F1 score ------------------#
    y_true, y_pred = get_predictions(model, test_loader, device)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print("Weighted F1:", f1_score(y_true, y_pred, average="weighted"))

    print(classification_report(y_true, y_pred, digits=4))

    # ------------ confusion matrix --------------#
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(45))

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.colorbar()
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_tr.png')
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm_norm, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized Confusion Matrix")

    plt.savefig("confusion_matrix_normalized_tr.png", dpi=300, bbox_inches="tight")