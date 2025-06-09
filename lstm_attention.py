import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import wandb  # Для логирования экспериментов
from early_stopping import EarlyStopping  # Кастомный ранняя остановка

# --- Конфигурация с улучшениями ---
config = {
    "features": ["TFEED_SET", "TFEED", "UADD", "NOISE", "USET"],
    "target": "TEMPERATURE",
    "seq_len": 288,                # 24 часа при 5-минутных данных
    "batch_size": 64,              # Увеличенный батч для стабильности
    "hidden_dim": 128,             # Большая размерность для LSTM
    "n_layers": 2,                 # 2 слоя LSTM
    "lr": 0.0005,                  # Пониженный learning rate
    "dropout": 0.2,                # Регуляризация
    "train_ratio": 0.8,
    "epochs": 100,                 # Максимальное число эпох
    "patience": 15,                # Для ранней остановки
    "grad_clip": 1.0,              # Обрезка градиентов
    "use_wandb": True              # Логирование в Weights & Biases
}

# --- 1. Улучшенная загрузка данных с кешированием ---
def load_data():
    try:
        df = pd.read_pickle("processed_data.pkl")  # Кешируем обработанные данные
    except:
        df = pd.read_csv("minute_data.csv", parse_dates=["timestamp"])
        df = df.sort_values(by=["objt_id", "timestamp"])
        
        # Улучшенная обработка пропусков
        df[config["features"]] = df.groupby("objt_id")[config["features"]].transform(
            lambda x: x.interpolate().fillna(method='ffill').fillna(method='bfill')
        )
        
        # RobustScaler вместо StandardScaler для устойчивости к выбросам
        scaler = RobustScaler()
        df[config["features"]] = scaler.fit_transform(df[config["features"]])
        df.to_pickle("processed_data.pkl")
    return df

# --- 2. Улучшенное разбиение на последовательные train/test ---
def train_test_split_by_group(df):
    train_dfs, test_dfs = [], []
    
    for objt_id in df["objt_id"].unique():
        group = df[df["objt_id"] == objt_id]
        split_idx = int(len(group) * config["train_ratio"])
        
        # Гарантируем, что тестовые данные идут ПОСЛЕ тренировочных
        train_dfs.append(group.iloc[:split_idx])
        test_dfs.append(group.iloc[split_idx:])
    
    return pd.concat(train_dfs), pd.concat(test_dfs)

# --- 3. Улучшенная модель с LayerNorm и Residual ---
class EnhancedLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.residual = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Residual connection from input
        res = self.residual(x[:, -1, :])
        
        # LSTM + Attention
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attm_weights * lstm_out, dim=1)
        
        return self.fc(context) + res  # Добавляем residual connection

# --- 4. Улучшенный DataLoader с доп. функциями ---
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, df, seq_len):
        self.X, self.y = self.create_sequences(df, seq_len)
        
    def create_sequences(self, df, seq_len):
        X, y = [], []
        for objt_id in df["objt_id"].unique():
            group = df[df["objt_id"] == objt_id]
            group_X = group[config["features"]].values
            group_y = group[config["target"]].values
            
            for i in range(len(group_X) - seq_len):
                X.append(group_X[i:i+seq_len])
                y.append(group_y[i+seq_len])
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

# --- 5. Улучшенный цикл обучения ---
def train_model():
    # Инициализация логирования
    if config["use_wandb"]:
        wandb.init(project="lstm-temperature-forecast", config=config)
    
    # Загрузка данных
    df = load_data()
    train_df, test_df = train_test_split_by_group(df)
    
    # Датасеты и загрузчики
    train_dataset = SequenceDataset(train_df, config["seq_len"])
    test_dataset = SequenceDataset(test_df, config["seq_len"])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"] * 2,
        num_workers=4,
        pin_memory=True
    )
    
    # Модель и оптимизатор
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedLSTMAttention(
        input_dim=len(config["features"]),
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        dropout=config["dropout"]
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5
    )
    early_stopping = EarlyStopping(patience=config["patience"])
    
    # Цикл обучения
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = nn.HuberLoss()(outputs, batch_y)  # Более устойчивый loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                test_loss += nn.HuberLoss()(outputs, batch_y).item()
        
        # Логирование
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, LR = {lr:.6f}")
        
        if config["use_wandb"]:
            wandb.log({
                "train_loss": train_loss,
                "test_loss": test_loss,
                "lr": lr
            })
        
        # Ранняя остановка и планировщик
        scheduler.step(test_loss)
        early_stopping(test_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Сохранение модели
    torch.save(model.state_dict(), "best_model.pth")
    if config["use_wandb"]:
        wandb.save("best_model.pth")

# --- Запуск ---
if __name__ == "__main__":
    train_model()
