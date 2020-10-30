import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys

sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib
from matplotlib import pyplot as plt

matplotlib.rc('font',family='monospace')
plt.style.use('ggplot')
fig, ax = plt.subplots()

#DOWNLOAD BERT MODEL
CategoryLabels=[i for i in range(0,2)]
max_tok_sequence=512# maximum length of (token) input sequences
batchsize=32


# Get BERT layer and tokenizer:

# More details here: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2
bert_layer=hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2',trainable=True)
vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()#### is the Bert layer case sensitive?
tokenizer=tokenization.FullTokenizer(vocab_file,do_lower_case)

#### This needs to know the category labels
def to_feature(text, label, label_list=CategoryLabels, max_seq_length=max_tok_sequence, tokenizer=tokenizer):
    example=classifier_data_lib.InputExample(guid=None,text_a=text.numpy(), text_b=None, label=label.numpy())#### text_b is for Sentence prediction/classification
    feature=classifier_data_lib.convert_single_example(0,example, label_list,max_seq_length,tokenizer)
    return feature.input_ids,feature.input_mask,feature.segment_ids,feature.label_id

def to_feature_map(text, label):### Need to map data to features for BERT formatting
  #### Problem is that graph tensors do not have a value so we need this wrapper function
  input_ids,input_mask,segment_ids,label_id=tf.py_function(to_feature, inp=[text,label],Tout=[tf.int32,tf.int32,tf.int32,tf.int32])
  input_ids.set_shape([max_tok_sequence])
  input_mask.set_shape([max_tok_sequence])
  segment_ids.set_shape([max_tok_sequence])
  label_id.set_shape([])
  x={'input_word_ids':input_ids,'input_masks':input_mask,'input_type_ids':segment_ids}
  return x,label_id
def create_model():
   input_word_ids = tf.keras.layers.Input(shape=(max_tok_sequence,), dtype=tf.int32,
                                        name="input_word_ids")
   input_mask = tf.keras.layers.Input(shape=(max_tok_sequence,), dtype=tf.int32,
                                    name="input_mask")
   input_type_ids = tf.keras.layers.Input(shape=(max_tok_sequence,), dtype=tf.int32,
                                     name="input_type_ids")
   pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
   drop=tf.keras.layers.Dropout(0.4)(pooled_output)#### hyperparameter
   output=tf.keras.layers.Dense(1,activation='sigmoid',name='output')(drop)#### classifier values between 0,1
   #output=tf.keras.layers.Dense(1,activation='softmax',name='output')(drop)#### classifier values between 0,1
   model=tf.keras.Model(inputs={'input_word_ids':input_word_ids,'input_masks':input_mask,'input_type_ids':input_type_ids},outputs=output)#### keras modelformatted
   return model


DebateDF=pd.read_csv("../LabeledCSV/candidate.csv")
DebateDF=DebateDF.sample(frac=1)
### Need integer topics
DebateDF.party.plot(kind='hist', title="Target Distribution")
plt.show()

#### Split train/test samples
train_df,residual=train_test_split(DebateDF,random_state=42,train_size=0.75,stratify=DebateDF.party.values)
valid_df,_=train_test_split(residual,random_state=42,train_size=0.25,stratify=residual.party.values)
print(train_df.shape, valid_df.shape)
with tf.device('/cpu:0'):#### Make data pipeline more efficient
	train_data=tf.data.Dataset.from_tensor_slices((train_df['Response'].values,train_df['party'].values))
	valid_data=tf.data.Dataset.from_tensor_slices((valid_df['Response'].values, valid_df['party'].values))

	for text,label in train_data.take(10):
    		print(text)
    		print(label)

#print(tokenizer.wordpiece_tokenizer.tokenize("i grew up in the segregated south, thankfully raised by a grandfather with almost no formal education but with a heart of gold who taught me early that all people were equal in the eyes of god."))

with tf.device('/cpu:0'):##### Glue the above functions together for train and test BERT model
	train_data=(train_data.map(to_feature_map,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(32,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))### Shuffle and parse into batches, prefetch data for validation

	valid_data=(valid_data.map(to_feature_map,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(1000).batch(32,drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
  #### the above is the input pipeline, with output pairs of pooled_output (representation for full input) and sequence output  (representation for all tokens)
# Building the model


model=create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),loss=tf.keras.losses.BinaryCrossentropy(),metrics=tf.keras.metrics.BinaryAccuracy())
model.summary()
history=model.fit(train_data,validation_data=valid_data,epochs=4,verbose=1)

sample_example=["no. there are many people who believe that the only way we can get this country turned around is to tax the middle class more and punish them more, but the truth is that middle-class americans are basically the only group of americans who’ve been taxed more in the 1980s and during the last 12 years, even though their incomes have gone down. the wealthiest americans have been taxed much less, even though their incomes have gone up. middle-class people will have their fair share of changing to do, and many challenges to face, including the challenge of becoming constantly re-educated. but my plan is a departure from trickle-down economics, just cutting taxes on the wealthiest americans and getting out of the way. it’s also a departure from tax-and- spend economics, because you can’t tax and divide an economy that isn’t growing. i propose an american version of what works in other countries — i think we can do it better: invest and grow. i believe we can increase investment and reduce the deficit at the same time, if we not only ask the wealthiest americans and foreign corporations to pay their share; we also provide over $100 billion in tax relief, in terms of incentives for new plants, new small businesses, new technologies, new housing, and for middle class families; and we have $140 billion of spending cuts. invest and grow. raise some more money, spend the money on tax incentives to have growth in the private sector, take the money from the defense cuts and reinvest it in new transportation and communications and environmental clean-up systems. this will work. on this, as on so many other issues, i have a fundamental difference from the present administration. i don’t believe trickle down economics will work. unemployment is up. most people are working harder for less money than they were making 10 years ago. i think we can do better if we have the courage to change."]

test_data=tf.data.Dataset.from_tensor_slices((sample_example,[0]*len(sample_example)))
test_data=test_data.map(to_feature_map).batch(1)
preds=model.predict(test_data)
print(preds)
threshold=0.5
print(['Conservative' if pred>=threshold else 'Liberal' for pred in preds])
