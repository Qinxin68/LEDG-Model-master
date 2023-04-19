import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = 'data\\data_size5'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE = 'fold%s_test.json'


class Config(object):
    def __init__(self):
        self.split = 'split10'
        self.bert_cache_path = 'Bert-base-chinese'
        self.feat_dim = 768
        self.num_layers = 1
        self.dropout = 0
        self.pos_dim = 192
        self.gnn_hidden = 50
        self.pairwise_loss = False
        self.epochs = 20
        self.lr = 1e-5
        self.batch_size = 4
        self.gradient_accumulation_steps = 2
        self.l2 = 1e-5
        self.l2_bert = 0.01
        self.warmup_proportion = 0.1
        self.adam_epsilon = 1e-8
        self.graph_size = 5
        self.ref_size = 30


