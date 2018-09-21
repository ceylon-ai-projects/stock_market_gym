import configparser

config = configparser.ConfigParser()

import os

dirname = os.path.dirname(__file__)
from firebase_admin import credentials
import firebase_admin
from firebase_admin import storage

cred = credentials.Certificate("{}/{}".format(dirname, "fireconfig.json"))
firebase_admin.initialize_app(cred, {
    'storageBucket': 'financialassitance-3f7fe.appspot.com'
})


def get_firebase_storage():
    bucket = storage.bucket()
    return bucket


def get_firebase_credential():
    from firebase_admin import credentials
    cred = credentials.Certificate("{}/{}".format(dirname, "fireconfig.json"))
    return cred


def get_config(pair_name, interval, prefix='default'):
    prefix += '.'
    config.read("{}/{}{}.conf".format(dirname, prefix, pair_name))
    section_name = '{}-m'.format(interval)
    section = {}
    if section_name in config:
        section = config[section_name]
    else:
        section = config['any-m']

    section['pair_name'] = config['DEFAULT']['pair_name']
    return section, config
