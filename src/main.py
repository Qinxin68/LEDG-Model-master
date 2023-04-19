import sys, os, warnings, time

# from tensorboardX import SummaryWriter

sys.path.append('..')
warnings.filterwarnings("ignore")
from data_loader import *
from networks.egnn_layer import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *


def main(configs, fold_id):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')
    model = GraphNetwork(configs).to(DEVICE)

    params = model.parameters()
    params_bert = model.bert.parameters()
    params_rest = list(model.GRU_net1.parameters()) + list(model.GRU_net2.parameters()) + \
                  list(model.edge2node_nets.parameters()) + list(model.node2edge_nets.parameters()) + \
                  list(model.Pre_EC.parameters())

    assert sum([param.nelement() for param in params]) == \
           sum([param.nelement() for param in params_bert]) + sum([param.nelement() for param in params_rest])
    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': configs.l2_bert, 'eps': configs.adam_epsilon},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'eps': configs.adam_epsilon},
        {'params': params_rest,
         'weight_decay': configs.l2}
    ]
    optimizer = AdamW(params, lr=configs.lr)
    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_steps_all)
    # model.zero_grad()
    max_ec, max_e, max_c = (-1, -1, -1), None, None
    metric_ec, metric_e, metric_c = (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)
    early_stop_flag = None
    # writer = SummaryWriter()
    for epoch in range(1, configs.epochs + 1):
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, y_emotions_b, y_causes_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, true_m, emo_ref = batch
            couples_pred, pred_c, pred_e = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                 bert_clause_b, doc_len_b, emo_ref)
            loss_couple = model.loss_couple(couples_pred, true_m)
            # loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b)
            loss = loss_couple
            loss = loss / configs.gradient_accumulation_steps
            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                # model.zero_grad()
                optimizer.zero_grad()
            print("\rEpoch: {:d} batch: {:d} loss: {:.4f} "
                  "loss_cou: {:.4f} ".format(epoch, train_step + 1, loss,
                                                                        loss_couple / 2,), end='')
        print("\rEpoch: {:d}/{:d} epoch_loss: {:.4f} "
              "loss_cou: {:.4f} ".format(epoch, configs.epochs, loss, loss_couple / 2, end='\n'))
        with torch.no_grad():
            model.eval()
            model.bert.eval()
            with open("result.txt", "w") as f:
                if configs.split == 'split10':
                    _, _, _, test_ec, test_e, test_c = update_epoch(configs, test_loader, model)
                    if test_ec[2] > metric_ec[2]:
                        early_stop_flag = 1
                        metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                    else:
                        early_stop_flag += 1
                    print("\rEpoch: {:d}/{:d} f1: {:.4f} ".format(epoch, configs.epochs, test_ec[2], end='\n'))
                    print("\rEpoch: {:d}/{:d} f1: {:.4f} ".format(epoch, configs.epochs, test_e[2], end='\n'))
                    print("\rEpoch: {:d}/{:d} f1: {:.4f} ".format(epoch, configs.epochs, test_c[2], end='\n'))
                if configs.split == 'split20':
                    _, _, _, valid_ec, valid_e, valid_c = update_epoch(configs, valid_loader, model)
                    _, _, _, test_ec, test_e, test_c = update_epoch(configs, test_loader, model)
                    if valid_ec[2] > max_ec[2]:
                        early_stop_flag = 1
                        max_ec, max_e, max_c = valid_ec, valid_e, valid_c
                        metric_ec, metric_e, metric_c = test_ec, test_e, test_c
                    else:
                        early_stop_flag += 1
                    print("\rEpoch: {:d}/{:d} f1: {:.4f} ".format(epoch, configs.epochs, valid_ec[2], end='\n'))
                    print("\rEpoch: {:d}/{:d} f1: {:.4f} ".format(epoch, configs.epochs, valid_e[2], end='\n'))
                    print("\rEpoch: {:d}/{:d} f1: {:.4f} ".format(epoch, configs.epochs, valid_c[2], end='\n'))
                    f.write('fold{} epoch{}: pair: {}, emo: {}, cau: {}'.format(fold_id, epoch, metric_ec, metric_e,
                                                                                metric_c, end='\n'))
                f.close()

    return metric_ec, metric_e, metric_c


def update_epoch(configs, batches, model):
    m_ec_list, m_e_list, m_c_list = [], [], []
    couple_list, true_list = [], []
    e_list, te_list = [], []
    c_list, tc_list = [], []
    for t, batch in enumerate(batches, 1):
        doc_len_b, y_emotions_b, y_causes_b, doc_couples_b, doc_id_b, \
        bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b, true_m, emo_ref = batch
        doc_couples_pred, pred_c, pred_e = model(bert_token_b, bert_segment_b, bert_masks_b,
                                                 bert_clause_b, doc_len_b, emo_ref)
        couple_list.append(doc_couples_pred)
        true_list.append(true_m)
        e_list.append(pred_e)
        te_list.append(y_emotions_b)
        c_list.append(pred_c)
        tc_list.append(y_causes_b)
        loss_e, loss_c = model.loss_pre(pred_e, pred_c, y_emotions_b, y_causes_b)
        loss_couple = model.loss_couple(doc_couples_pred, true_m)
        loss = loss_couple
    metric_ec = pair_prf_CR(couple_list, true_list, configs)
    metric_e = cal_prf(e_list, te_list, configs)
    metric_c = cal_prf(c_list, tc_list, configs)
    print("\repoch_loss: {:.4f} ".format(loss, end='\n'))
    return to_np(loss_couple), to_np(loss_e), to_np(loss_c), \
           metric_ec, metric_e, metric_c


if __name__ == '__main__':
    configs = Config()
    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 20
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 15
    else:
        print('Unknown data split.')
        exit()
    metric_folds = {'ecp': [], 'emo': [], 'cau': []}

    for fold_id in range(1, n_folds + 1):
        print('===== fold {} ====='.format(fold_id))
        metric_ec, metric_e, metric_c = main(configs, fold_id)
        print('F_ecp: {} R{} P{}'.format(metric_ec[2], metric_ec[1], metric_ec[0]))
        print('F_ep: {} R{} P{}'.format(metric_e[2], metric_e[1], metric_e[0]))
        print('F_cp: {} R{} P{}'.format(metric_c[2], metric_c[1], metric_c[0]))
        metric_folds['ecp'].append(metric_ec)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)
    metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()
    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
