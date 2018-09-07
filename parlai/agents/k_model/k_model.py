import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def get_last_output_lstm(lengths, output, batch_first):
    idx = (torch.LongTensor(lengths) - 1).view(-1, 1).expand(
        len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    idx = idx.unsqueeze(time_dimension)
    if output.is_cuda:
        idx = idx.cuda(output.data.get_device())
    # Shape: (batch_size, rnn_hidden_dim)
    last_output = output.gather(
        time_dimension, Variable(idx)).squeeze(time_dimension)
    return last_output


def get_norm_l2(x):
    last_dim = x.size(1)
    batch_size = x.size(0)
    x_norm = torch.norm(x, p=2, dim=1)
    x_norm = x_norm.unsqueeze(1)
    x_norm = x_norm.expand(batch_size, last_dim)
    return x / x_norm


def get_max_sim_dim(v_query, v_answer):
    """
    max cosine score
    :param v_query: N * dim
    :param v_answer: N * k * dim
    :return:
    """
    v_query = torch.unsqueeze(v_query, 2)  # N * sim_dim * 1
    q_a_sim = torch.bmm(v_answer, v_query)  # N * k * 1
    q_a_sim = q_a_sim.squeeze(2)  # N * k
    max_val, _ = torch.max(q_a_sim, 1)  # N



class ConRankModel(nn.Module):
    def __init__(self, setting):
        nn.Module.__init__(self)
        self.vocab_size = setting['vocab_size']
        self.sim_dim = setting['sim_dim']
        #self.use_cuda = setting['use_cuda']
        self.alpha = setting['alpha']
        embed_init = None
        if 'embed_init' in setting:
            embed_init = setting['embed_init']
        retrain_emb = setting['retrain_emb']
        word_dim = setting['word_dim']
        self.embed_layer = nn.Embedding(self.vocab_size, word_dim)
        if embed_init is not None:
            self.embed_layer.weight = nn.Parameter(torch.Tensor(embed_init))
        if retrain_emb:
            self.embed_layer.weight.requires_grad = True
        else:
            self.embed_layer.weight.requires_grad = False

        self.query_encoder = nn.LSTM(word_dim, self.sim_dim, num_layers=1, batch_first=True)
        self.profile_item_encoder = nn.LSTM(word_dim, self.sim_dim, num_layers=1, batch_first=True)

    def init_hidden_query(self, use_cuda, batch_size):
        first = Variable(torch.zeros(1, batch_size, self.sim_dim))
        second = Variable(torch.zeros(1, batch_size, self.sim_dim))
        if use_cuda:
            first = first.cuda()
            second = second.cuda()
        return (first, second)

    def get_last_lstm_output(self, use_cuda, lstm_layer, input_batch, lengths):
        batch_size = input_batch.size(0)
        init_ve = self.init_hidden_query(use_cuda, batch_size)
        output, hidden = lstm_layer(input_batch, init_ve)
        l_output = get_last_output_lstm(lengths, output, True)
        return l_output


    def encode_profile(self, use_cuda, p_items, p_lengths):
        """
        p_items list of indexs: [[1,2,3], [1,2,4]]
        :param p_items: 2d long-tensor
        :param p_lengths: list [3, 3] lengths of tensors
        :return: 2d tensor
        """
        item_embeded = self.embed_layer(p_items)
        return self.get_last_lstm_output(use_cuda, self.profile_item_encoder, item_embeded, p_lengths)


    def forward(self, use_cuda, p_items, p_lengths, query, q_lengs, answer, a_lengs):
        """

        :param p_encoders: P * sim_dim
        :param query: N * max_q
        :param q_lengs: N
        :param answer: N * k * max_a
        :param a_lengs: N * k
        :return:
        """
        e_query = self.embed_layer(query)
        v_query = self.get_last_lstm_output(use_cuda, self.query_encoder, e_query, q_lengs) # N * sim_dim
        v_query = get_norm_l2(v_query)
        k = answer.size(1)
        N = answer.size(0)
        flat_answer = answer.view(k * N, -1) # view (N*k) * max_a
        e_answer = self.embed_layer(flat_answer)
        # flat a_lengs #
        flat_a_lengs = []
        for item in a_lengs:
            flat_a_lengs.extend(item)
        v_answer = self.get_last_lstm_output(use_cuda, self.query_encoder, e_answer, flat_a_lengs) # (N * K) * sim_dim
        v_answer = get_norm_l2(v_answer) # normalize to l2
        v_answer = v_answer.view(N, k, -1) # N * k * sim_dim
        ### get sim from v_answer to v_query
        v_query = torch.unsqueeze(v_query, 2) # N * sim_dim * 1
        q_a_sim = torch.bmm(v_answer, v_query) # N * k * 1
        q_a_sim = q_a_sim.squeeze(2) # N * k # sim of answer candiate with query
        ### get sim from v_answer to p_encoders: P * dim and v_answer: N * k * dim || dim * P ==> N * k * P ==>max N *k
        p_encoders = self.encode_profile(use_cuda, p_items, p_lengths)
        p_encoders = get_norm_l2(p_encoders)
        p_transpose = torch.transpose(p_encoders, 0, 1)# dim * P
        p_transpose = torch.unsqueeze(p_transpose, 0)# 1 * dim * P
        dim = p_transpose.size(1)
        P_leng = p_transpose.size(2)
        #print ('p_transpose size = ', p_transpose.size())
        p_transpose = p_transpose.expand(N, dim, P_leng)# N * dim * P
        a_p_sim = torch.bmm(v_answer, p_transpose) # N * k * P
        a_p_sim, _ = torch.max(a_p_sim, 2) # N*k
        #### finally combine sim = alpha * question_sim + (1-alpha) * profile_sim
        #print ('alpha = ', self.alpha)
        #print ('q_a_sim size = ', q_a_sim.size())
        #print ('a_p_sim size = ', a_p_sim.size())
        f_sim = q_a_sim * self.alpha + (1.0 - self.alpha) * a_p_sim
        return f_sim # N * k


class AttRankModel(nn.Module):
    @staticmethod
    def get_default_setting():
        return {
            'sim_dim': 512,
            'att_type': 'simple',
            'profile_dim': 512,
            'att_dim': 512
        }

    def __init__(self, setting):
        nn.Module.__init__(self)
        self.vocab_size = setting['vocab_size']
        self.sim_dim = setting['sim_dim']
        #self.use_cuda = setting['use_cuda']
        self.att_type = setting['att_type']
        self.p_dim = setting['profile_dim']
        if self.att_type == 'concat':
            self.att_dim = setting['att_dim']
            self.linear_att = nn.Linear(self.p_dim + self.sim_dim, self.att_dim)
            ini_fil = torch.randn(self.att_dim)
            abs_norm = torch.norm(ini_fil)
            ini_fil = ini_fil/abs_norm
            ini_fil = ini_fil.unsqueeze(1)
            self.va = nn.Parameter(ini_fil) # self.att_dim * 1
        elif self.att_type == 'matrix':
            self.temp_linear = nn.Linear(self.p_dim, self.p_dim)
        embed_init = None
        if 'embed_init' in setting:
            embed_init = setting['embed_init']
        retrain_emb = setting['retrain_emb']
        word_dim = setting['word_dim']
        self.embed_layer = nn.Embedding(self.vocab_size, word_dim)
        if embed_init is not None:
            self.embed_layer.weight = nn.Parameter(torch.Tensor(embed_init))
        if retrain_emb:
            self.embed_layer.weight.requires_grad = True
        else:
            self.embed_layer.weight.requires_grad = False
        ### sim of query --> attention on profile ==> a vector to compute sim with question ###
        self.profile_item_encoder = nn.LSTM(word_dim, self.p_dim, num_layers=1, batch_first=True)
        self.query_encoder = nn.LSTM(word_dim, self.p_dim, num_layers=1, batch_first=True)
        self.candidate_encoder = nn.LSTM(word_dim, self.sim_dim, num_layers=1, batch_first=True)
        self.linear_sim_dim = nn.Linear(self.p_dim * 2, self.sim_dim)

    def init_hidden_lstm_dim(self, use_cuda, batch_size, lstm_dim):
        first = Variable(torch.zeros(1, batch_size, lstm_dim))
        second = Variable(torch.zeros(1, batch_size, lstm_dim))
        if use_cuda:
            first = first.cuda()
            second = second.cuda()
        return (first, second)


    def get_last_lstm_output(self, use_cuda, lstm_layer, output_dim, input_batch, lengths):
        batch_size = input_batch.size(0)
        init_ve = self.init_hidden_lstm_dim(use_cuda, batch_size, output_dim) #self.init_hidden_query(use_cuda, batch_size)
        output, hidden = lstm_layer(input_batch, init_ve)
        l_output = get_last_output_lstm(lengths, output, True)
        return l_output


    def encode_profile(self, use_cuda, p_items, p_lengths):
        """
        p_items list of indexs: [[1,2,3], [1,2,4]]
        :param p_items: 2d long-tensor
        :param p_lengths: list [3, 3] lengths of tensors
        :return: 2d tensor
        """
        item_embeded = self.embed_layer(p_items)
        return self.get_last_lstm_output(use_cuda, self.profile_item_encoder, self.p_dim, item_embeded, p_lengths)


    def encode_candidate(self, use_cuda, answer, a_lengs):
        k = answer.size(1)
        N = answer.size(0)
        flat_answer = answer.view(k * N, -1)  # view (N*k) * max_a
        e_answer = self.embed_layer(flat_answer)
        # flat a_lengs #
        flat_a_lengs = []
        for item in a_lengs:
            flat_a_lengs.extend(item)
        v_answer = self.get_last_lstm_output(use_cuda, self.candidate_encoder, self.sim_dim, e_answer, flat_a_lengs)  # (N * K) * sim_dim
        v_answer = F.tanh(v_answer) #get_norm_l2(v_answer)  # normalize to l2
        v_answer = v_answer.view(N, k, -1)  # N * k * sim_dim
        return v_answer


    def get_attention_profiles(self, p_encoded, q_encoded):
        """

        :param p_encoded: P * p_dim
        :param q_encoded: N * p_dim
        :return: (N * P) --> score for each profile to attention
        """
        P = p_encoded.size(0)
        N = q_encoded.size(0)
        if self.att_type == 'concat':
            q_dim = q_encoded.size(1)
            temp_q_encoded = q_encoded.unsqueeze(1)  # N * 1 * p_dim
            temp_q_encoded = temp_q_encoded.expand(N, P, q_dim)
            flat_q = temp_q_encoded.contiguous().view((N*P), q_dim)

            p_dim = p_encoded.size(1)
            temp_p_encoded = p_encoded.unsqueeze(0) # 1 * P * p_dim
            temp_p_encoded = temp_p_encoded.expand(N, P, p_dim)
            flat_p = temp_p_encoded.contiguous().view((N*P), p_dim)
            concat_reps = torch.cat([flat_q, flat_p], dim=1) # (N*P) * (2*p_dim)

            con_reps = self.linear_att(concat_reps) # (N*P) * att_dim
            con_reps = F.tanh(con_reps)# (N*P) * att_dim
            score_reps = torch.mm(con_reps, self.va) #
            score_reps = score_reps.squeeze(1)
            score_reps = score_reps.view(N, P) # N * P
            score_reps = F.softmax(score_reps, dim=1) # N * P
            final_rep = torch.mm(score_reps, p_encoded)# (N, P)  * (P, p_dim) --> N * p_dim
            return final_rep
        elif self.att_type == 'simple':
            p_norm2 = get_norm_l2(p_encoded)
            q_norm2 = get_norm_l2(q_encoded)
            score_reps = torch.mm(q_norm2, torch.transpose(p_norm2, 0, 1)) # N * P
            score_reps = F.softmax(score_reps, dim=1)
            final_rep = torch.mm(score_reps, p_encoded)
            return final_rep
        elif self.att_type == 'matrix':
            temp_q = self.temp_linear(q_encoded) # N * dim
            p_norm2 = get_norm_l2(p_encoded)
            q_norm2 = get_norm_l2(temp_q)
            score_reps = torch.mm(q_norm2, p_norm2, 0, 1)  # N * P
            score_reps = F.softmax(score_reps, dim=1)
            final_rep = torch.mm(score_reps, p_encoded)
            return final_rep

    def forward(self, use_cuda, p_items, p_lengths, question, q_lengs, answer, a_lengs):
        p_encoders = self.encode_profile(use_cuda, p_items, p_lengths) # P * p_dim
        e_query = self.embed_layer(question) # N * max_leng * word_dim
        v_query = self.get_last_lstm_output(use_cuda, self.query_encoder, self.p_dim, e_query, q_lengs) # N * p_dim
        v_cands = self.encode_candidate(use_cuda, answer, a_lengs) # N * k * sim_dim

        p_att = self.get_attention_profiles(p_encoders, v_query) # N * p_dim
        comb_context = torch.cat([p_att, v_query], dim=1)
        comb_context = self.linear_sim_dim(comb_context) # N * sim_dim
        comb_context = F.tanh(comb_context) # N * sim_dim
        #comb_context = get_norm_l2(comb_context) # N * sim_dim
        comb_context = comb_context.unsqueeze(2) # N * sim_dim * 1
        scores = torch.bmm(v_cands, comb_context) # N * k * 1
        scores = scores.squeeze(2)
        return scores
        # compute similarity here
