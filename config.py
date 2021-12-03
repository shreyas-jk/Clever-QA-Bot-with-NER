from transformers import AutoTokenizer, AutoModelForTableQuestionAnswering
from tokenizers import BertWordPieceTokenizer

data_path = './car_sales.csv'
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = AutoModelForTableQuestionAnswering.from_pretrained("./model")
aggregations = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3:"COUNT"}

max_len = 384
tags_encoder_path = "./saved/tags.pkl"
model_weights_path = "./saved/weights.h5"
bert_tokenizer = BertWordPieceTokenizer("./bert_base_uncased/vocab.txt", lowercase=True)