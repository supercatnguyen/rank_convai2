import k_train, k_model
import torch
import numpy as np

class K_Rank(object):
    def __init__(self):
        self.model = k_train.get_final_model(False)
        self.vocab_dic = k_train.load_vocab(k_train.get_vocab_path())

    def rank_conversation(self, converse):
        """
        converse is a dictionary containing:
        'question': list of question [['a', 'b'], ['c', 'd']]
        'profile': list os sentences [['a', 'b'], ['c', 'd']]
        'cand': list of candidates [[['a', 'b', 'c'], []], [[], []]]
        :param converse: dic as above
        :return: ranked indexs
        """
        temp_converse = k_train.convert_converse_to_wid(self.vocab_dic, converse)
        scores = k_train.get_prediction_from_converse(temp_converse, False, self.model)
        ### rank by scores ####
        score_list = scores.data.tolist()
        score_np = np.array(score_list) # N * k (N = number of questions, k = number of candidates)
        score_indexs = np.argsort(score_np)
        result = []
        N = len(score_list)
        for i in range(N):
            items = list(score_indexs[i])
            items.reverse()
            result.append(items)
        return result


K_RANK_MODEL = K_Rank()

def tokenize_conver(conver):
    result = {}
    question = conver['question']
    question = k_train.preprocess_rm_period(question)
    result['question'] = [question.split(' ')]
    cand_items = conver['cand']
    cands = []
    rm_items = []
    for item in cand_items:
        temp_item = k_train.preprocess_rm_period(item)
        temp_item = temp_item.split(' ')
        rm_items.append(temp_item)
    cands.append(rm_items)
    result['cand'] = cands
    sens = conver['profile']
    temp_profile = []
    if type(sens[0]) is not list:
        temp_profile = []
        for sen in sens:
            temp_profile.append(sen.split(' '))
        result['profile'] = temp_profile
    else:
        result['profile'] = sens
    return result

def get_ranked_indices(conver):
    return K_RANK_MODEL.rank_conversation(conver)

def get_ranked_indices_for_one_question(conver):
    temp_conver = tokenize_conver(conver)
    indices = get_ranked_indices(temp_conver)
    return indices[0]


def prin_converse(converse, index):
    question = converse['question']
    cands = converse['cand']
    print ('question: ', ' '.join(question[index]))
    print ('candidates: ')
    for i in range(len(cands[index])):
        cand = cands[index][i]
        print ('%d: %s' % (i, ' '.join(cand)))


def test_ranked_model():
    converses, _ = k_train.read_training_data('valid_self_original.txt', False)
    true_count = 0
    total_count = 0
    for converse in converses:
        indices = get_ranked_indices(converse) # N * k
        true_indexs = converse['indexs']# N
        for i in range(len(true_indexs)):
            if true_indexs[i] == indices[i][0]:
                true_count += 1
        total_count += len(true_indexs)
    ratio = float(true_count)/total_count
    print ('accuracy %d/%d = %f' % (true_count, total_count, ratio))


if __name__ == '__main__':
    test_ranked_model()

