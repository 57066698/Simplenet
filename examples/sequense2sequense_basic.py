"""
    keras 的 s2s 模型本地化
    model:  [N, len, char_eng]
            -> LSTM(latent)
            -> [N, len, latent], [N, latent], [N, latent] (输出序列，最后单元H, 最后单元S)
                                 -> and [N, len, char_fra]
                                 -> LSTM(latent)
                                 -> [N, len, latent], _, _
                                 -> Dense("softmax")

    train:  input_text -> encode_input[t]
            target_text -> decode_input[t]
                        -> decode_target[t-1]


"""