from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM, MultiHeadAttention, Flatten, Concatenate, LayerNormalization
from tensorflow.keras.models import Model

def build_fixed_model(window_size=24, forecast_horizon=24, n_obj=46, embed_dim=10, n_head=4):
    # Основной вход — признаки с историей
    input_seq = Input(shape=(window_size, 9), name='history_features')
    # 9 — число признаков (обнови, если у тебя другое количество)
    
    # Embedding для OBJT_ID (отдельный вход)
    input_obj_id = Input(shape=(1,), name='obj_id')
    emb_obj = Embedding(n_obj + 1, embed_dim, name='embedding_obj')(input_obj_id)
    emb_obj = Flatten()(emb_obj)  # (batch, embed_dim)
    
    # Первый LSTM слой — возвращает последовательность
    lstm_1 = LSTM(64, return_sequences=True)(input_seq)
    lstm_1 = LayerNormalization()(lstm_1)
    
    # Второй LSTM слой — возвращает последовательность
    lstm_2 = LSTM(32, return_sequences=True)(lstm_1)
    lstm_2 = LayerNormalization()(lstm_2)
    
    # MultiHeadAttention: Query, Key, Value — из lstm_2
    attn_output = MultiHeadAttention(num_heads=n_head, key_dim=16)(lstm_2, lstm_2)
    attn_output = LayerNormalization()(attn_output)
    
    # Аггрегация по времени — flatten + dense
    flat = Flatten()(attn_output)
    
    # Соединяем с эмбеддингом объекта
    concat = Concatenate()([flat, emb_obj])
    
    # Полносвязная часть
    dense_1 = Dense(128, activation='relu')(concat)
    dense_1 = Dropout(0.3)(dense_1)
    dense_1 = BatchNormalization()(dense_1)
    
    dense_2 = Dense(64, activation='relu')(dense_1)
    dense_2 = Dropout(0.2)(dense_2)
    dense_2 = BatchNormalization()(dense_2)
    
    # Финальный выход — прогноз температуры на forecast_horizon
    output = Dense(forecast_horizon, name='temp_pred')(dense_2)
    
    model = Model(inputs=[input_seq, input_obj_id], outputs=output)
    return model
