import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import BertModel
from config import DEVICE
from utils.utils import *


class NodeUpdateNetwork(nn.Module):
    def __init__(self, configs):
        super(NodeUpdateNetwork, self).__init__()
        self.in_dim = configs.feat_dim
        self.batch_size = configs.batch_size
        self.cou_feat_dim = 2 * self.in_dim
        self.g_size = configs.graph_size
        self.pos_dim = configs.pos_dim
        self.pos_emb = nn.Embedding(self.g_size, self.pos_dim)
        self.dropout = configs.dropout
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=self.in_dim * 3 + self.pos_dim,
                      out_channels=self.in_dim,
                      kernel_size=1,
                      bias=False),
            nn.LayerNorm((self.in_dim, self.g_size)),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
        )
        self.init_weight()

    def forward(self, node_feat, edge_feat):
        batch, N, _, _ = edge_feat.size()
        edge_feat = F.normalize(edge_feat, p=1, dim=1)
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, -1), 1).squeeze(-1), node_feat)
        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(N, 1), -1)], -1)
        pos_emb = self.pos_emb(torch.arange(N).cuda()).unsqueeze(0).repeat(batch, 1, 1)
        node_feat = torch.cat([node_feat, pos_emb], -1)
        node_feat = self.network(node_feat.permute(0, 2, 1))
        node_feat = node_feat.permute(0, 2, 1)
        return node_feat

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue


class EdgeUpdateNetwork(nn.Module):
    def __init__(self, configs):
        super(EdgeUpdateNetwork, self).__init__()
        # set size
        self.in_dim = configs.feat_dim
        self.batch_size = configs.batch_size
        self.cou_feat_dim = 2 * self.in_dim
        self.g_size = configs.graph_size
        self.dropout = configs.dropout
        self.network_ec = nn.Sequential(
            nn.Conv2d(in_channels=self.cou_feat_dim + 2,
                      out_channels=self.in_dim,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.in_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(in_channels=self.in_dim,
                      out_channels=self.in_dim,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.in_dim),
        )
        self.pair_network = nn.Sequential(
            nn.Linear(self.in_dim, 2),
        )
        self.init_weight()

    def forward(self, node_feat, edge_feat):
        batch, N, _, _ = edge_feat.size()
        couple_ec = self.ec_generator(node_feat)
        edge_feat = torch.cat([edge_feat, couple_ec], -1)
        edge_feat = self.network_ec(edge_feat.permute(0, 3, 2, 1))
        edge_feat = self.pair_network(edge_feat.permute(0, 3, 2, 1))
        return edge_feat

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue

    def ec_generator(self, H):
        batch, seq_len, feat_dim = H.size()
        P_left = H.reshape(-1, seq_len, 1, feat_dim)
        P_left = P_left.repeat(1, 1, seq_len, 1)
        P_right = H.reshape(-1, 1, seq_len, feat_dim)
        P_right = P_right.repeat(1, seq_len, 1, 1)
        P = torch.cat([P_left, P_right], dim=3).reshape(batch, seq_len, seq_len, feat_dim * 2)
        return P


class GRUNet1(nn.Module):
    def __init__(self, configs):
        super(GRUNet1, self).__init__()
        self.in_dim = configs.feat_dim
        self.g_size = configs.graph_size
        self.dropout = configs.dropout
        self.hidden_size = 192
        self.ref_size = configs.ref_size
        self.net_bi_lstm = nn.Sequential(
            nn.GRU(input_size=self.in_dim, hidden_size=int(self.in_dim / 2),
                   bias=False, batch_first=True, dropout=0.5, bidirectional=True),

        )
        self.attention_layer = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
            nn.Linear(self.hidden_size, 1)
        )
        self.attention_layer_ref = nn.Sequential(
            nn.Linear(self.ref_size, 2),
            nn.Linear(2, 1)
        )
        self.edge_e_net = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(self.in_dim, 2),
        )
        self.init_weight()

    def forward(self, node_feat, emo_ref):
        # get size
        batch, N, in_dim = node_feat.size()
        node_feat, _ = self.net_bi_lstm(node_feat)
        attention_w = F.softmax(self.attention_layer(node_feat), -1)
        emo_ref = torch.stack(emo_ref, 0).reshape(batch, N, self.ref_size).cuda()
        attention_r = F.softmax(self.attention_layer_ref(emo_ref), -1)
        node_feat_x = node_feat * attention_w * attention_r
        edge_e = self.edge_e_net(node_feat_x)
        return node_feat_x, edge_e

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue


class GRUNet2(nn.Module):
    def __init__(self, configs):
        super(GRUNet2, self).__init__()
        self.in_dim = configs.feat_dim
        self.g_size = configs.graph_size
        self.dropout = configs.dropout
        self.hidden_size = 192
        self.ref_size = configs.ref_size
        self.net_bi_lstm = nn.Sequential(
            nn.GRU(input_size=self.in_dim + 2, hidden_size=int(self.in_dim / 2),
                   bias=False, batch_first=True, dropout=0.1, bidirectional=True),
        )

        self.edge_c_net = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.LeakyReLU(),
            nn.Sigmoid(),
            nn.Linear(self.in_dim, 2),

        )
        self.init_weight()

    def forward(self, node_feat):
        batch, N, in_dim = node_feat.size()
        node_feat, _ = self.net_bi_lstm(node_feat)
        edge_c = self.edge_c_net(node_feat)
        return node_feat, edge_c

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue


class Classification(nn.Module):
    def __init__(self, configs):
        super(Classification, self).__init__()
        self.in_dim = configs.feat_dim
        self.out_e_net = nn.Sequential(
            nn.Linear(self.in_dim * 2, 2),
        )
        self.out_c_net = nn.Sequential(
            nn.Linear(self.in_dim * 2, 2),
        )
    def forward(self, node_feat_e, node):
        pre_e = self.out_e_net(torch.cat([node_feat_e, node], -1))
        pre_c = self.out_c_net(torch.cat([node_feat_e, node], -1))
        return pre_e, pre_c

class GraphNetwork(nn.Module):
    def __init__(self, configs):
        super(GraphNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.in_dim = configs.feat_dim
        self.num_layers = configs.num_layers
        self.dropout = configs.dropout
        self.g_size = configs.graph_size
        self.GRU_net1 = GRUNet1(configs)
        self.GRU_net2 = GRUNet2(configs)
        self.node2edge_nets = nn.ModuleList()
        self.edge2node_nets = nn.ModuleList()
        for l in range(self.num_layers):
            self.edge2node_nets.append(EdgeUpdateNetwork(configs))
        for l in range(self.num_layers):
            self.node2edge_nets.append(NodeUpdateNetwork(configs))
        self.Pre_EC = Classification(configs)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len, emo_ref):
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE),
                                token_type_ids=bert_segment_b.to(DEVICE))
        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        node_feat = doc_sents_h
        node_feat_e, edge_e = self.GRU_net1(node_feat, emo_ref)
        node_feat_e = torch.cat([edge_e, node_feat], -1)

        node_feat_e, edge_c = self.GRU_net2(node_feat_e)
        edge = self.init_edge(edge_e, edge_c)
        for i, (node_net, edge_net) in enumerate(zip(self.node2edge_nets, self.edge2node_nets), 1):
            node = node_net(node_feat_e, edge)
            edge = edge_net(node, edge)
        pre_e, pre_c = self.Pre_EC(node_feat_e, node)
        return edge, pre_c, pre_e

    def batched_index_select(self, bert_output, bert_clause_b):
        hidden_state = bert_output[0]
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        doc_sents_h = hidden_state.gather(1, dummy)
        return doc_sents_h

    def init_edge(self, e, c):
        b, N, dim = e.size()
        e_0 = e[:, :, 0].unsqueeze(-1).repeat(1, 1, self.g_size)
        c_0 = c[:, :, 0].unsqueeze(-1).repeat(1, 1, self.g_size).permute(0, 2, 1)
        e_1 = e[:, :, 1].unsqueeze(-1).repeat(1, 1, self.g_size)
        c_1 = c[:, :, 1].unsqueeze(-1).repeat(1, 1, self.g_size).permute(0, 2, 1)
        edge_0 = (e_0 * c_0).unsqueeze(-1)
        edge_1 = (e_1 * c_1).unsqueeze(-1)
        edge = torch.cat([edge_0, edge_1], dim=-1).reshape(b, N, N, dim)
        edge = F.softmax(edge, -1)
        return edge

    def loss_couple(self, couples_pred, true_matrix):

        criterion = focal_loss(alpha=0.25, gamma=2, num_classes=2)
        batch, N, _, _ = couples_pred.size()
        couples_pred = couples_pred.reshape(batch, N * N, 2)
        true_matrix = torch.stack(true_matrix, 0).long().to(DEVICE)
        true_matrix = true_matrix.view(batch, N * N)
        loss_cou = criterion(couples_pred, true_matrix)
        return loss_cou

    def loss_pre(self, pred_e, pred_c, y_emotions, y_causes):
        batch, N, _ = pred_e.size()
        y_emotions = torch.LongTensor(y_emotions).to(DEVICE)
        y_causes = torch.LongTensor(y_causes).to(DEVICE)
        y_emotions = y_emotions[:, :, 1].view(batch, N)
        y_causes = y_causes[:, :, 1].view(batch, N)
        criterion = focal_loss(alpha=0.25, gamma=2, num_classes=2)
        loss_e = criterion(pred_e, y_emotions)
        loss_c = criterion(pred_c, y_causes)
        return loss_e, loss_c

