import sys

sys.path.append('..')
from os.path import join
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from config import *
from utils.utils import *

torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=configs.batch_size,
                                              shuffle=False, collate_fn=bert_batch_preprocessing)
    return data_loader


class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split
        self.graph_s = configs.graph_size
        self.data_type = data_type
        self.train_file = join('..\\', data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join('..\\', data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join('..\\', data_dir, self.split, TEST_FILE % fold_id)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list, self.true_m_list, self.emo_ref_list = self.read_data(self.data_type)

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[
            idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        true_m = self.true_m_list[idx]
        emo_ref = self.emo_ref_list[idx]
        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)
        true_m = torch.LongTensor(np.array(true_m))
        emo_ref = torch.FloatTensor(emo_ref)
        assert doc_len == len(y_emotions)
        return doc_couples, y_emotions, y_causes, doc_len, doc_id, \
               bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens, true_m, emo_ref

    def read_data(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []
        true_couple_matrix_list = []
        emotion_reference_list = []
        data_list = read_json(data_file)
        data_flag = 0
        data_len = len(data_list)
        for doc in data_list:
            doc_id = doc['doc_id']
            graph_id = doc['graph_id']

            graph_len = doc['graph_len']
            doc_couples = doc['pairs']
            if any(abs(cou[0] - cou[1]) > graph_len for cou in doc_couples):
                data_len = data_len - 1
                pass
            else:
                data_flag = data_flag + 1
                true_couple_matrix = np.zeros(shape=(self.graph_s, self.graph_s), dtype=float)
                doc_emotions, doc_causes = zip(*doc_couples)
                doc_id_list.append(doc_id)
                doc_len_list.append(graph_len)
                doc_couples = list(map(lambda x: list(x), doc_couples))
                doc_couples_list.append(doc_couples)
                y_emotions, y_causes = np.zeros((graph_len, 2)), np.zeros((graph_len, 2))
                doc_clauses = doc['clauses']
                doc_str = ''
                graph_emo_ref = []
                for i in range(len(doc_couples)):
                    if max(doc_couples[i][0], doc_couples[i][1]) < graph_len + 1 + graph_id and \
                            min(doc_couples[i][0], doc_couples[i][1]) > graph_id + 1:
                        true_couple_matrix[doc_couples[i][0] - graph_id - 1,
                        doc_couples[i][1] - graph_id - 1:doc_couples[i][1] - graph_id] = 1.0
                true_couple_matrix_list.append(true_couple_matrix)
                for i in range(graph_len):
                    emotion_label = int(i + 1 + graph_id in doc_emotions)
                    y_emotions[i][0] = 1 - emotion_label
                    y_emotions[i][1] = emotion_label
                    cause_label = int(i + 1 + graph_id in doc_causes)
                    y_causes[i][0] = 1 - cause_label
                    y_causes[i][1] = cause_label
                    clause = doc_clauses[i]
                    clause_id = clause['clause_id']
                    graph_emo_ref.append(clause['emotion_reference'])
                    assert int(clause_id) == i + graph_id + 1
                    doc_str += ' [CLS] ' + clause['clause'] + ' [SEP] '
                emo_ref = np.array(graph_emo_ref)
                indexed_tokens = self.bert_tokenizer.encode(doc_str.strip(), add_special_tokens=False)
                clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                doc_token_len = len(indexed_tokens)
                segments_ids = []
                segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                segments_indices.append(len(indexed_tokens))
                for i in range(len(segments_indices) - 1):
                    semgent_len = segments_indices[i + 1] - segments_indices[i]
                    if i % 2 == 0:
                        segments_ids.extend([0] * semgent_len)
                    else:
                        segments_ids.extend([1] * semgent_len)
                assert len(clause_indices) == graph_len
                assert len(segments_ids) == len(indexed_tokens)
                bert_token_idx_list.append(indexed_tokens)
                bert_clause_idx_list.append(clause_indices)
                bert_segments_idx_list.append(segments_ids)
                bert_token_lens_list.append(doc_token_len)
                y_emotions_list.append(y_emotions)
                y_causes_list.append(y_causes)
                emotion_reference_list.append(emo_ref)

        return doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, \
               bert_token_idx_list, bert_clause_idx_list, bert_segments_idx_list, bert_token_lens_list, \
               true_couple_matrix_list, emotion_reference_list


def bert_batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, doc_len_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b, true_m, emo_ref = zip(*batch)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)

    bsz, max_len = bert_token_b.size()

    bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)

    for index, seq_len in enumerate(bert_token_lens_b):
        bert_masks_b[index][:seq_len] = 1
    bert_masks_b = torch.FloatTensor(bert_masks_b)
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape
    return np.array(doc_len_b), np.array(y_emotions_b), np.array(y_causes_b), \
           doc_couples_b, doc_id_b, bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, true_m, emo_ref