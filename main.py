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
df1 = pd.read_csv("processed_dataset/combined_defensive1.csv")
df2 = pd.read_csv("processed_dataset/combined_defensive2.csv")
all_morphology_classes = [
    'line','2000','0200','0020','0002',
    '2220','2202','2022','0222','0022','2200','0202','2020',
    '1002','2001','1020','0201','2100','0012','0120','0210',
    '0021','1200','2010','0102','0221','2021','0122','2102',
    '2201','0212','2120','1022','2012','2210','1202','1220',
    '0112','1012','0211','1201','1102','0121','1210','2011',
    '1021','1120','2101','2110','2002','0220'
]

label_to_id = {
    label: idx for idx, label in enumerate(all_morphology_classes)
}

id_to_label = {
    idx: label for label, idx in label_to_id.items()
}

def make_windows_from_segments(df, feature_cols, target_col="morphology_id"):
    X_list = []
    y_list = []

    for seg_id, seg_df in df.groupby("global_segment_id"):
        seg_df = seg_df.sort_values("frame").reset_index(drop=True)

        if len(seg_df) < SEQ_LEN:
            continue

        for start in range(0, len(seg_df) - SEQ_LEN + 1, STRIDE):
            end = start + SEQ_LEN

            X_seq = seg_df.iloc[start:end][feature_cols].values
            y_seq = seg_df.iloc[start:end][target_col].values

            X_list.append(X_seq)
            y_list.append(y_seq)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y

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
    df1["match_id"] = "match1"
    df2["match_id"] = "match2"
    dataset_df = pd.concat([df1,df2], axis=0, ignore_index=True)
    dataset_df["global_segment_id"] = (
    dataset_df["match_id"].astype(str)
        + "_"
        + dataset_df["global_segment_id"].astype(str)
    )

    print(dataset_df.shape)
    print(dataset_df["match_id"].value_counts())
    print(dataset_df["global_segment_id"].nunique())

    feature_cols_A = [
        col for col in dataset_df.columns
        if col not in [
            "morphology",
            "morphology_id",
            "possession",
            "frame",
            "frame_diff",
            "segment_id",
            "global_segment_id",
            "defense_team",
            "match_id",'O_STZ_x',
            'O_STZ_y', 'O_ZO_x', 'O_ZO_y', 'O_ORM_x', 'O_ORM_y', 'O_OLM_x','O_OLM_y'
        ]
    ]

    # -----B. Home 7(no defense) + Away 11 + ball 1 ---- #
    feature_cols_B = []

    # 수비팀 나머지 7명만 사용
    for i in range(1, 8):
        feature_cols_B += [f"D_other{i}_x", f"D_other{i}_y"]

    # 공격팀 11명
    for i in range(1, 12):
        feature_cols_B += [f"O{i}_x", f"O{i}_y"]

    feature_cols_B += ["ball_x", "ball_y"]

    #print(len(feature_cols_B))  # 38

    # --------- C. Away 4 + Ball 1 -------- #
    attacker_roles = ["STZ", "ZO", "ORM", "OLM"]
    feature_cols_C = []
    for role in attacker_roles:
        feature_cols_C += [f"O_{role}_x", f"O_{role}_y"]

    feature_cols_C += ["ball_x", "ball_y"]

    #print(feature_cols_C)
    #print(len(feature_cols_C)) # 10

    # --------------label encoding --------------#
   
    dataset_df["morphology"] = dataset_df["morphology"].astype(str)
    dataset_df["morphology_id"] = dataset_df["morphology"].map(label_to_id)
    missing = dataset_df[dataset_df["morphology_id"].isna()]["morphology"].unique()
    print("mapping에 없는 label:", missing)


    # --------------- train, validate, test segment split -----------#
  
    segments = dataset_df["global_segment_id"].unique()

    train_segments, temp_segments = train_test_split(
        segments,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    val_segments, test_segments = train_test_split(
        temp_segments,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    train_df = dataset_df[dataset_df["global_segment_id"].isin(train_segments)].copy()
    val_df = dataset_df[dataset_df["global_segment_id"].isin(val_segments)].copy()
    test_df = dataset_df[dataset_df["global_segment_id"].isin(test_segments)].copy()

    print(train_df.shape)
    print(val_df.shape)
    print(test_df.shape)

    #----------------- Sliding Window ------------------------#
    # 75 frame per sequences
    SEQ_LEN = 75
    STRIDE = 5
    X_train, y_train = make_windows_from_segments(train_df, feature_cols_A)
    X_val, y_val = make_windows_from_segments(val_df, feature_cols_A)
    X_test, y_test = make_windows_from_segments(test_df, feature_cols_A)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print(X_test.shape, y_test.shape)

    #----------------- Construct dataset ------------------------#
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
        num_classes=51,
        seq_len=75,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # -------------- loss - cross entrophy --------------#
    num_classes = 51
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
    # cm = confusion_matrix(y_true, y_pred, labels=np.arange(45))

    # plt.figure(figsize=(12, 10))
    # plt.imshow(cm)
    # plt.colorbar()
    # plt.xlabel("Predicted class")
    # plt.ylabel("True class")
    # plt.title("Confusion Matrix")
    # plt.savefig('confusion_tr.png')
    # cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    # cm_norm = np.nan_to_num(cm_norm)

    # plt.figure(figsize=(12, 10))
    # plt.imshow(cm_norm, vmin=0, vmax=1)
    # plt.colorbar()
    # plt.xlabel("Predicted class")
    # plt.ylabel("True class")
    # plt.title("Normalized Confusion Matrix")

    # plt.savefig("confusion_matrix_normalized_tr.png", dpi=300, bbox_inches="tight")

    #--------------- split class distribution --------- #
    print("split 별 클래스 분포")
    train_classes = set(np.unique(y_train))
    val_classes = set(np.unique(y_val))
    test_classes = set(np.unique(y_test))

    print("num train classes:", len(train_classes))
    print("num val classes:", len(val_classes))
    print("num test classes:", len(test_classes))

    print("Val classes not in train:", sorted(val_classes - train_classes))
    print("Test classes not in train:", sorted(test_classes - train_classes))
    def show_dist(y_split, name):
        classes, counts = np.unique(y_split, return_counts=True)
        df = pd.DataFrame({
            "class_id": classes,
            "count": counts,
            "ratio": counts / counts.sum()
        }).sort_values("count", ascending=False)

        print(name)
        print("num classes:", len(classes))
        print(df.head(20))

    show_dist(y_train, "train")
    show_dist(y_val, "val")
    show_dist(y_test, "test")