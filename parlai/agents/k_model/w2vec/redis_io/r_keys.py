import json

W2ID = 'w2id'
ID2VEC = 'id2v'
CHANNEL = 'channel'
CHANNEL_SERVICE_OUT = 'channel_out'

def get_key_of_word(word):
    return '%s:::%s'%(W2ID, word)

def get_key_of_id(wid):
    return '%s:::%d'%(ID2VEC, wid)

def encode_vector(np_array):
    vec = list(np_array)
    list_str = [str(vec[i]) for i in range(len(vec))]
    temp = ' '.join(list_str)
    return temp

def decode_vector(np_a_str):
    tgs = np_a_str.split(' ')
    result = [float(tgs[i]) for i in range(len(tgs))]
    return result