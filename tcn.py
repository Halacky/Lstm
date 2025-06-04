def build_tcn_model():
    # Входные данные - последовательность признаков
    input_seq = Input(shape=(window_size, 11), name='history_features')  # 11 признаков с временными фичами
    
    # Вход ID объекта
    input_obj_id = Input(shape=(1,), name='obj_id')
    emb_obj = Embedding(n_obj + 1, embed_dim, name='embedding_obj')(input_obj_id)
    emb_obj = Flatten()(emb_obj)
    
    # TCN слой (возвращает последовательность)
    tcn_layer = tfa.layers.TCN(
        nb_filters=tcn_filters,
        kernel_size=tcn_kernel_size,
        dilations=[1, 2, 4, 8],
        return_sequences=True,
        name='tcn_layer'
    )(input_seq)
    
    tcn_layer = LayerNormalization()(tcn_layer)
    
    # MultiHeadAttention
    attn_output = MultiHeadAttention(num_heads=n_head, key_dim=16)(tcn_layer, tcn_layer)
    attn_output = LayerNormalization()(attn_output)
    
    # Flatten + объединение с эмбеддингом
    flat = Flatten()(attn_output)
    concat = Concatenate()([flat, emb_obj])
    
    # Полносвязная часть
    dense_1 = Dense(128, activation='relu')(concat)
    dense_1 = Dropout(0.3)(dense_1)
    dense_1 = BatchNormalization()(dense_1)
    
    dense_2 = Dense(64, activation='relu')(dense_1)
    dense_2 = Dropout(0.2)(dense_2)
    dense_2 = BatchNormalization()(dense_2)
    
    # Выход — прогноз температуры на forecast_horizon
    output = Dense(forecast_horizon, name='temp_pred')(dense_2)
    
    model = Model(inputs=[input_seq, input_obj_id], outputs=output)
    return model
