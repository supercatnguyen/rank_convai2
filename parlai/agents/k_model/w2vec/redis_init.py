import w2vec_lookup
import datetime
from redis_io import re_connection, r_keys
from text_utility import text_util

def populate_data():
    LOOKUP_TAB = w2vec_lookup.get_lookup_table(text_util.Language.ENGLISH)
    w2id = LOOKUP_TAB.w2id
    t1 = datetime.datetime.now()
    print 'insert %d words to redis'%(len(w2id))
    for w in w2id:
        wid = w2id[w]
        w_key = r_keys.get_key_of_word(w)
        re_connection.r.set(w_key, '%d'%wid)
    t2 = datetime.datetime.now()
    print 'time for caching w2id: ', (t2 - t1).total_seconds()
    print 'insert vecto to redis'
    embed_table = LOOKUP_TAB.embedd_table
    data_size = len(w2id)
    t1 = datetime.datetime.now()
    for i in range(data_size):
        vec = embed_table[i]
        en_vec = r_keys.encode_vector(vec)
        wid_key = r_keys.get_key_of_id(i)
        re_connection.r.set(wid_key, en_vec)
        if i%50000 == 1:
            print 'i = ', i
    t2 = datetime.datetime.now()
    print 'time for caching in redis: ', (t2 - t1).total_seconds()


if __name__ == '__main__':
    populate_data()

