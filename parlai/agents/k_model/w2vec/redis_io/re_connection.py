import redis
import os

ACCESS_FILE_PATH = os.path.dirname(os.path.abspath(__file__)) + '/access.txt'


def read_file_config():
    result = {}
    f = open(ACCESS_FILE_PATH, 'r')
    for line in f:
        temp = line.split(':')
        result[temp[0].strip()] = temp[1].strip()
    f.close()
    return result

CONFIG_DIC = read_file_config()

pool = redis.ConnectionPool(host=CONFIG_DIC['host'], port=CONFIG_DIC['port'], db=CONFIG_DIC['db'])

r = redis.Redis(connection_pool=pool)