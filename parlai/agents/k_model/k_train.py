import numpy as np
import re, os, json
import k_model
import torch
import sys
import torch.autograd as autograd
import datetime
import torch.nn.functional as F
import os

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

UNKNOWN_TOKEN = '<unnown>'
PADDING_TOKEN = '<paddingword>'

def extract_text_from_line_numb(line):
    match = re.search('\d your persona:', line)
    if match is None:
        match = re.search('\d', line)
    if match is not None:
        end = match.end()
        rm_text = line[end: ].strip()
        return rm_text
    else:
        print ('cannot detect line number: %s' % line)
    return line


def preprocess_rm_period(text):
    if text.endswith('.'):
        text = text[: -1].strip()
        return text
    return text


def update_dic(dic, item):
    if item not in dic:
        dic[item] = len(dic)


def get_item_ids(dic, items):
    for item in items:
        if item not in dic:
            dic[item] = len(dic)
    return [dic[item] for item in items]


def add_count_items(dic, items):
    for item in items:
        if item not in dic:
            dic[item] = 0
        dic[item] +=1

def find_true_index(answer, cands):
    for i in range(len(cands)):
        if cands[i] == answer:
            return i
    return -1


def read_training_data(data_path, build_dic=True):
    f = open(data_path, 'r')
    convers = []
    profile = []
    question = []
    answer = []
    cands = []
    true_indexs = []
    item_count = {}
    for line in f:
        temp_line = line.strip()
        if len(temp_line) > 2:
            p_text = extract_text_from_line_numb(temp_line)
            p_text = preprocess_rm_period(p_text)
            if temp_line.startswith('1 your persona'):
                if len(profile) > 0:
                    if len(question) == 0:
                        print ('empty question: ', temp_line)
                    convers.append({'profile': profile, 'question': question, 'answer': answer, 'cand': cands, 'indexs': true_indexs})
                temp_p = p_text.split(' ')
                add_count_items(item_count, temp_p)
                profile = [temp_p]
                question = []
                answer = []
                cands = []
                true_indexs = []
            else:
                if '\t' not in temp_line:
                    temp_p = p_text.split(' ')
                    add_count_items(item_count, temp_p)
                    profile.append(temp_p)
                else:
                    tgs = p_text.split('\t')
                    item_question = preprocess_rm_period(tgs[0].strip())
                    #print ('question: ', item_question)
                    item_question = item_question.split(' ')
                    add_count_items(item_count, item_question)
                    question.append(item_question)
                    item_answer = preprocess_rm_period(tgs[1].strip())
                    #print ('item answer: ', item_answer)
                    cand_items = tgs[3].split('|')
                    #print ('number of cand_items: ', len(cand_items))
                    t_index = find_true_index(item_answer, cand_items)
                    #print ('index: ', t_index)
                    true_indexs.append(t_index)
                    if t_index == -1:
                        print ('cannot find true answer for this case: %s' % temp_line)
                    item_answer = item_answer.split(' ')
                    add_count_items(item_count, item_answer)
                    answer.append(item_answer)
                    rm_items = []
                    for item in cand_items:
                        temp_item = preprocess_rm_period(item)
                        temp_item = temp_item.split(' ')
                        add_count_items(item_count, temp_item)
                        rm_items.append(temp_item)
                    cands.append(rm_items)
    f.close()
    vocab = None
    if build_dic:
        vocab = cut_off_low_frequency(item_count, None, 50000)
    #for conver in convers:
    #    if len(conver['question']) == 0:
    #        print ('why there is empty question: ', conver)
    return convers, vocab


def get_wids_from_list_sens(vocab_dic, p_sens, max_length):
    p_ids = []
    for sen in p_sens:
        temp_ids = [vocab_dic[tok] if tok in vocab_dic else vocab_dic[UNKNOWN_TOKEN] for tok in sen]
        for i in range(len(sen), max_length):
            temp_ids.append(vocab_dic[PADDING_TOKEN])
        p_ids.append(temp_ids)
    return p_ids


def padding_list_sen_with_leng(sen_inds, vocab_dic, max_leng):
    result = []
    for sen in sen_inds:
        temp_res = list(sen)
        for i in range(len(sen), max_leng):
            temp_res.append(vocab_dic[PADDING_TOKEN])
        result.append(temp_res)
    return result



def convert_converse_to_wid(vocab_dic, conver):
    result = {}

    result['profile_leng'] = [len(sen) for sen in conver['profile']]
    p_max_leng = max(result['profile_leng'])
    result['profile'] = get_wids_from_list_sens(vocab_dic, conver['profile'], p_max_leng)

    result['question_leng'] = [len(sen) for sen in conver['question']]
    q_max_leng = max(result['question_leng'])
    result['question'] = get_wids_from_list_sens(vocab_dic, conver['question'], q_max_leng)

    if 'answer' in conver:
        result['answer_leng'] = [len(sen) for sen in conver['answer']]
        a_max_leng = max(result['answer_leng'])
        result['answer'] = get_wids_from_list_sens(vocab_dic, conver['answer'], a_max_leng)

    cand_lengths = []
    total_max_leng = 0
    for cand in conver['cand']:
        temp_cand_length = [len(item) for item in cand]
        total_max_leng = max(total_max_leng, max(temp_cand_length))
        cand_lengths.append(temp_cand_length)
    cands = []
    for cand in conver['cand']:
        temp_cand = get_wids_from_list_sens(vocab_dic, cand, total_max_leng)
        cands.append(temp_cand)
    result['cand'] = cands
    result['cand_leng'] = cand_lengths
    if 'indexs' in conver:
        result['index'] = conver['indexs']
    return result


def extract_data_from_w2vec(vocab_dic):
    from w2vec import redis_lookup
    id2w = {}
    for key in vocab_dic:
        id2w[vocab_dic[key]] = key
    keys = id2w.keys()
    keys.sort()
    v_list = []
    for key in keys:
        word = id2w[key]
        vector = redis_lookup.get_embedded_vecto(word)
        v_list.append(vector)
    matrix = np.array(v_list)
    return matrix


def save_w2vec_matrix(matrix):
    w2vec_path = get_w2vec_path()
    np.savetxt(w2vec_path, matrix)


def get_w2vec_path():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    w2vec_path = os.path.join(current_folder, 'w2vec.out')
    return w2vec_path


def load_w2vec_matrix():
    w2vec_path = get_w2vec_path()
    return np.loadtxt(w2vec_path)


def cut_off_low_frequency(item_count, threshold = None, max_vocab = 100000):
    result = {UNKNOWN_TOKEN: 1, PADDING_TOKEN: 0}
    if threshold is not None:
        for key in item_count:
            if item_count[key] >= threshold:
                result[key] = len(result)
    else:
        pairs = sorted(item_count.items(), key=lambda x: x[1])
        leng = len(pairs)
        for i in range(len(pairs)):
            index = leng - 1 - i
            key = pairs[index]
            result[key[0]] = len(result)
            if i > max_vocab:
                break
    return result


def get_random_of_list(l):
    numb = len(l)
    per = np.random.permutation(numb)
    result = [l[index] for index in per]
    return result

def get_loss(use_cuda, scores, indexs, lamda):
    """
    :param scores: N * k
    :param indexs: N
    :return:
    """
    k = scores.size(1)
    N = scores.size(0)
    true_arr = autograd.Variable(torch.zeros(N))
    if use_cuda:
        true_arr = true_arr.cuda()
    for i in range(len(indexs)):
        index = indexs[i]
        #print ('type of scores: ', type(scores[i][index]))
        true_arr[i] = scores[i][index]
    true_pre = 0
    _, max_indexs = torch.max(scores, 1) # N
    max_indexs = max_indexs.data.tolist()
    for i in range(len(indexs)):
        if max_indexs[i] == indexs[i]:
            true_pre += 1

    true_arr = torch.unsqueeze(true_arr, 1)
    true_arr = true_arr.expand(-1, k)
    if use_cuda:
        true_arr = true_arr.cuda()
    loss_m = lamda + scores - true_arr
    loss_m = F.relu(loss_m)
    return torch.sum(loss_m), true_pre


def get_loss_from_converse(conver, use_cuda, model, lamda):
    scores = get_prediction_from_converse(conver, use_cuda, model)
    true_indexs = conver['index']
    loss, true_pre_count = get_loss(use_cuda, scores, true_indexs, lamda)
    return loss, true_pre_count


def get_prediction_from_converse(conver, use_cuda, model):
    profile = conver['profile']
    var_profile = autograd.Variable(torch.LongTensor(profile))  # N *
    if use_cuda:
        var_profile = var_profile.cuda()
    # en_profile = model.encode_profile(var_profile, conver['profile_leng'])
    ### question and candidate ####
    question_length = conver['question_leng']
    question = conver['question']
    var_question = autograd.Variable(torch.LongTensor(question))
    if use_cuda:
        var_question = var_question.cuda()
    cands = conver['cand']
    var_cand = autograd.Variable(torch.LongTensor(cands))
    if use_cuda:
        var_cand = var_cand.cuda()
    cand_lengths = conver['cand_leng']
    scores = model(use_cuda, var_profile, conver['profile_leng'], var_question, question_length, var_cand, cand_lengths)
    return scores



def get_random_permute(N):
    per_path = '%d_permutation.out' % N
    if not os.path.exists(per_path):
        permute = np.random.permutation(N)
        indexs = [str(index) for index in permute]
        f = open(per_path, 'w')
        f.write(' '.join(indexs))
        f.close()
    f = open(per_path, 'r')
    text = f.read()
    f.close()
    tgs = text.strip().split(' ')
    indexs = [int(tg) for tg in tgs]
    return indexs


def get_random_permutation(items):
    N = len(items)
    permute = get_random_permute(N)
    result = [items[index] for index in permute]
    return result


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def get_model_path(epo, count):
    return os.path.join(CURRENT_FOLDER, 'model/model_%d_%d.dat' % (epo, count))


def get_final_model_path():
    return os.path.join(CURRENT_FOLDER, 'model/final_model.dat')


def save_setting(setting):
    setting_path = os.path.join(CURRENT_FOLDER, 'model/setting.json')
    save_json(setting, setting_path)


def read_setting():
    setting_path = os.path.join(CURRENT_FOLDER, 'model/setting.json')
    return read_json(setting_path)


def read_json(fpath):
    f = open(fpath, 'r')
    text = f.read()
    f.close()
    return json.loads(text)


def save_json(json_data, save_path):
    f = open(save_path, 'w')
    f.write(json.dumps(json_data, indent=4, ensure_ascii=False))
    f.close()



def get_vocab_path():
    return os.path.join(CURRENT_FOLDER, 'model/vocab.txt')


def save_vocab(vocab, save_path):
    vocab_str = json.dumps(vocab, indent=4, ensure_ascii=False)
    f = open(save_path, 'w')
    f.write(vocab_str)
    f.close()


def load_vocab(vocab_path):
    return read_json(vocab_path)


def train_model(setting):
    #save_setting(setting)
    convers, vocab_dic = read_training_data('train_self_original.txt', True)
    #print ('sample convers: ', convers[0])
    if setting['build_dic']:
        print ('start building dictionary')
        w2vec_matrix = extract_data_from_w2vec(vocab_dic)
        save_w2vec_matrix(w2vec_matrix)
        save_vocab(vocab_dic, get_vocab_path())
        return
    w2vec_init = load_w2vec_matrix()
    convers = get_random_permutation(convers)
    convers = [convert_converse_to_wid(vocab_dic, conver) for conver in convers]
    print ('number of conversations: %d' % len(convers))
    print ('size of w2vec matrix: ', w2vec_init.shape)
    setting['vocab_size'] = w2vec_init.shape[0]
    setting['word_dim'] = w2vec_init.shape[1]
    save_setting(setting)
    setting['embed_init'] = w2vec_init
    lamda = setting['lamda']
    learning_rate = setting['learning_rate']
    model = k_model.ConRankModel(setting)
    use_cuda = setting['use_cuda']
    if use_cuda:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    epo_num = setting['epo_num']
    tr_1 = datetime.datetime.now()
    dev_convers = convers[: 100]
    tr_convers = convers[100: ]
    print ('training convers: %d' % len(tr_convers))
    for epo in range(epo_num):
        print ('at epoch: %d' % epo)
        count = 0
        inter_d1 = datetime.datetime.now()
        for conver in tr_convers:
            ### find loss from scores and true_indexs###
            optimizer.zero_grad()
            b1 = datetime.datetime.now()
            loss, _ = get_loss_from_converse(conver, use_cuda, model, lamda)
            loss.backward()
            optimizer.step()
            b2 = datetime.datetime.now()
            #print ('time for 1 conversation = %f' % (b2 - b1).total_seconds())
            if count % 5000 == 1:
                print ('compute loss at dev at %d of epo %d' % (count, epo))
                d_t1 = datetime.datetime.now()
                total_loss = 0
                total_true_pre = 0
                total_sens = 0
                for d_conver in dev_convers:
                    temp_loss, temp_true_count = get_loss_from_converse(d_conver, use_cuda, model, lamda)
                    total_loss += temp_loss
                    total_true_pre += temp_true_count
                    total_sens += len(d_conver['question_leng'])
                dev_loss = total_loss.data[0]
                d_t2 = datetime.datetime.now()
                delta_d = (d_t2 - d_t1).total_seconds()
                accuracy = float(total_true_pre) /float(total_sens)
                print ('loss of dev at %d of epo %d is %f, accuracy %d/%d = %f, time = %f' % (count, epo, dev_loss, total_true_pre, total_sens, accuracy, delta_d))
                inter_d2 = datetime.datetime.now()
                print ('tim for traingin 5000 convers: %f seconds' % (inter_d2 - inter_d1).total_seconds())
                inter_d1 = datetime.datetime.now()
            count += 1
        if count % 20000 == 1:
            save_model(model, get_model_path(epo, count))
    tr_2 = datetime.datetime.now()
    delta_time = (tr_2 - tr_1).total_seconds()
    print ('time for training %d epo: %f seconds' % (epo_num, delta_time))
    save_model(model, get_final_model_path())
    print ('start eval newest model ...')
    eval_on_valid(False)
    print ('finished ...')


def get_final_model(use_cuda):
    model_path = get_final_model_path()
    setting = read_setting()
    w2vec_init = load_w2vec_matrix()
    setting['vocab_size'] = w2vec_init.shape[0]
    setting['embed_init'] = w2vec_init
    setting['word_dim'] = w2vec_init.shape[1]
    #setting_new = k_model.AttRankModel.get_default_setting()
    #setting_new.update(setting)
    model = k_model.ConRankModel(setting)
    if use_cuda:
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        saved_state = torch.load(model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state)
    return model



def eval_on_valid(use_cuda):
    o_convers, _ = read_training_data('valid_self_original.txt', False)
    vocab_dic = load_vocab(get_vocab_path())
    model = get_final_model(use_cuda)
    convers = [convert_converse_to_wid(vocab_dic, conver) for conver in o_convers]
    total_sens = 0
    total_true = 0
    count = 0
    d1 = datetime.datetime.now()
    print ('start evaluating %d conversations' % len(convers))
    for j in range(len(convers)):
        conver = convers[j]
        #print ('for converse: ', conver)
        scores = get_prediction_from_converse(conver, use_cuda, model)
        indexs = conver['index']
        true_pre = 0
        _, max_indexs = torch.max(scores, 1)  # N
        max_indexs = max_indexs.data.tolist()
        for i in range(len(indexs)):
            if max_indexs[i] == indexs[i]:
                #print ('ok at conver: %d, record: %d' % (j, i))
                true_pre += 1
        total_true += true_pre
        total_sens += len(conver['question_leng'])
        if count == 1000:
            print ('count = %d' % count)
    d2 = datetime.datetime.now()
    acc_ratio = float(total_true)/float(total_sens)
    print ('time for evaluating all %d conversations: %f' % (len(convers), (d2 - d1).total_seconds()))
    print ('final accuracy: %d/%d = %f' % (total_sens, total_true, acc_ratio))


def run_train1():
    setting = {
        'sim_dim': 512,
        'retrain_emb': False,
        'use_cuda': True,
        'alpha': 0.5,
        'learning_rate': 0.001,
        'epo_num': 4,
        'build_dic': False,
        'lamda': 0.3
    }
    train_model(setting)


def run_train2():
    setting1 = {
        'sim_dim': 512,
        'retrain_emb': False,
        'use_cuda': True,
        'alpha': 0.5,
        'learning_rate': 0.001,
        'epo_num': 4,
        'build_dic': False,
        'lamda': 0.3
    }
    setting = k_model.AttRankModel.get_default_setting()
    setting.update(setting1)
    train_model(setting)


def main():
    if len(sys.argv) != 2:
        print ('usage: python k_train.py train/eval')
        sys.exit(1)
    mode = sys.argv[1]
    if mode == 'eval':
        eval_on_valid(False)
    elif mode == 'train':
        run_train2()
        #run_train()

if __name__ == '__main__':
    main()