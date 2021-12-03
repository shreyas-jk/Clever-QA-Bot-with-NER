import pickle
import numpy as np
import pandas as pd
from config import *
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFBertModel

def masked_ce_loss(real, pred):
    spare_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )
    
    mask = tf.math.logical_not(tf.math.equal(real, 17))
    loss_ = spare_loss(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def header_text(value):
    st.markdown("<h3 style='font-family:Georgia; font-size:25px;'>{0}</h3>".format(value), unsafe_allow_html=True)

def input_transform_tokenize(sentence):
    dataset_dict = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    word_token_mapping = []
    # for sentence in texts:
    input_ids = []
    for idx, word in enumerate(sentence.split()):
        ids = bert_tokenizer.encode(word, add_special_tokens=False)
        input_ids.extend(ids.ids)
        # num_tokens = len(ids)
        word_token_mapping.append((word, ids.ids))
        
    # Pad and create attention masks.
    # Skip if truncation is needed
    input_ids = input_ids[:max_len - 2]

    input_ids = [101] + input_ids + [102]
    no_of_tokens = len(input_ids)
    token_type_ids = [0] * len(input_ids)
    attention_mask = [1] * len(input_ids)
    padding_len = max_len - len(input_ids)

    input_ids = input_ids + ([0] * padding_len)
    attention_mask = attention_mask + ([0] * padding_len)
    token_type_ids = token_type_ids + ([0] * padding_len)
    
    dataset_dict["input_ids"].append(input_ids)
    dataset_dict["token_type_ids"].append(token_type_ids)
    dataset_dict["attention_mask"].append(attention_mask)
        
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["token_type_ids"],
        dataset_dict["attention_mask"]
    ]
    return x, no_of_tokens, word_token_mapping

def build_model():
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    token_type_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
    embedding = layers.Dropout(0.3)(embedding)
    output = layers.Dense(20, activation='softmax')(embedding)
    
    model = keras.Model(
        inputs=[input_ids, token_type_ids, attention_mask],
        outputs=[output],
    )
    optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=masked_ce_loss, metrics=['accuracy'])
    return model

def encoded_tags_to_dict(tags_encoder_path):
    le_encoder = pickle.load(open(tags_encoder_path, "rb"))
    return dict(zip(le_encoder.transform(le_encoder.classes_), le_encoder.classes_))

def get_result_dataframe(tags_dict, word_map, pred_tags):
    predict_entities = [tags_dict.get(_, '[pad]') for _ in pred_tags]

    input_list = []
    for word_item in word_map:
        if len(word_item[1]) != 1:
            for token_list in word_item[1]:
                input_list.append([word_item[0], token_list])
        else:
            token = word_item[1]
            input_list.append([word_item[0], token[0]])

    df_map = pd.DataFrame(input_list, columns=['word', 'token_id'])
    df_map['entity'] = predict_entities[1:-1]
    df_map.to_csv('generated_entities.csv')
    return df_map

def entity_mapping(value):
    if value == 'O':
        return 'Others'

    if value == 'C-MAN':
        return 'Manufacturer'

    if value == 'C-MOD':
        return 'Model'


def add_description_field(df):
    df['description'] = df['entity'].apply(entity_mapping)
    return df

def custom_filter(data, manufacturer_list, model_list):
    df_filter = data[data['Manufacturer'].str.lower().isin(manufacturer_list) | data['Model'].isin(model_list)]
    if len(df_filter) == 0:
        return data
    else:
        return df_filter 

def get_manufacturer_list(df):
    unique_values = list(df.query("entity == 'C-MAN'")['word'].unique())
    return list(map(str.lower, unique_values))

def get_model_list(df):
    unique_values= list(df.query("entity == 'C-MOD'")['word'].unique())
    return list(map(str.lower, unique_values))

def ner_filter(data, question):

    # build model and load weights
    ner_model = build_model()
    ner_model.load_weights(model_weights_path)

    # tokenize input text
    tokenize_text, no_of_tokens, word_map = input_transform_tokenize(question)

    # get prediction
    pred = ner_model.predict(tokenize_text)

    # ignore predictions of padding tokens
    pred_tags = np.argmax(pred, 2)[0][:no_of_tokens] 

    # import and load dictionary of tags
    tags_dict = encoded_tags_to_dict(tags_encoder_path)

    # get final results
    df_res = get_result_dataframe(tags_dict, word_map, pred_tags)
    
    df_res = add_description_field(df_res)

    header_text('2) After applying NER')
    st.table(df_res[['word', 'entity', 'description']].T)

    manufacturer_list = get_manufacturer_list(df_res)
    model_list = get_model_list(df_res)
    return custom_filter(data, manufacturer_list, model_list)