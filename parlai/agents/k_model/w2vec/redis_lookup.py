from redis_io import re_connection, r_keys
UNKNOWN_WORD = '<unnown>'
PADDING_WORD = '<paddingword>'
PADDING_CHAR = '<padchar>'
PADDING_LABEL = '<padd_label>'
UNKNOWN_CHAR = '<unknonw_char>'
PADDING_TAG = '<pad_tag>'
UNKNOWN_TAG = '<unknowntag>'

CACHED_DIC = {}

def get_id_of_word(word):
    w_key = r_keys.get_key_of_word(word)
    wid_str = re_connection.r.get(w_key)
    if wid_str is not None:
        return int(wid_str)
    return None

def get_vector_of_id(wid):
    wid_key = r_keys.get_key_of_id(wid)
    encode_v = re_connection.r.get(wid_key)
    vec = r_keys.decode_vector(encode_v)
    return vec


def get_embedded_vecto(word, language = 'en'):
    if word in CACHED_DIC:
        return CACHED_DIC[word]

    t_wid = get_id_of_word(word)
    if t_wid is not None:
        vec = get_vector_of_id(t_wid)
        CACHED_DIC[word] = vec
        return vec

    t_wid = get_id_of_word(word.title())
    if t_wid is not None:
        vec = get_vector_of_id(t_wid)
        CACHED_DIC[word] = vec
        return vec

    t_wid = get_id_of_word(word.lower())
    if t_wid is not None:
        vec = get_vector_of_id(t_wid)
        CACHED_DIC[word] = vec
        return vec

    t_wid = get_id_of_word(word.upper())
    if t_wid is not None:
        vec = get_vector_of_id(t_wid)
        CACHED_DIC[word] = vec
        return vec
    #t_wid = get_id_of_word(UNKNOWN_WORD)
    return CACHED_DIC[UNKNOWN_WORD]
    #return get_vector_of_id(t_wid)

def get_sen_matrix(tokens, max_leng):
    ax = []
    add_tokens = list(tokens)
    for i in range(len(add_tokens), max_leng):
        add_tokens.append(UNKNOWN_WORD)

    for index in range(max_leng):
        tok = add_tokens[index]
        ax.append(get_embedded_vecto(tok))
    return ax

CACHED_DIC[UNKNOWN_WORD] = get_vector_of_id(get_id_of_word(UNKNOWN_WORD))

def test():
    while True:
        input_str = raw_input('input: ').decode('utf-8')
        vec = get_embedded_vecto(input_str)
        print (vec)

if __name__ == '__main__':
    test()