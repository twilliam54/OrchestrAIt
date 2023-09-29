import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import BertTokenizer

import os
import shutil

from .text_functions import word_segmenter
from.directory_maker import doc_type_df

def prep_data(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokens=tokenizer.encode_plus(text,max_length=512,
                                 truncation=True, padding='max_length',
                                 add_special_tokens=True, return_token_type_ids=False,
                                 return_tensors='tf')
    return {
        'input_ids': tf.cast(tokens['input_ids'], tf.float64),
        'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)
    }

def ml_sort_data(df, seg_size, model):
    doc_segments_df=word_segmenter(df, seg_size)
    doc_segments_df['tokenized_segments']=doc_segments_df['doc_segments'].apply(lambda x: prep_data(x))
    doc_segments_df['probability']=[model.predict(i)[0] for i in doc_segments_df.tokenized_segments]
    agg_probs_df=pd.DataFrame(doc_segments_df.groupby(['doc_path','doc_name'])['probability'].sum()).reset_index()
    agg_probs_df['doc_type_no']=agg_probs_df.probability.apply(lambda x: np.argmax(x))
    agg_probs_df=agg_probs_df.merge(doc_type_df, left_on='doc_type_no', right_on=doc_type_df.index)
    for index, row in agg_probs_df.iterrows():
        shutil.move(os.path.join(row.doc_path,row.doc_name), os.path.join(row.doc_path,row.doc_type))