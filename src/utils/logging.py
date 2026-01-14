import logging

def get_logger(name='applied-ml-platform'):
    lg=logging.getLogger(name)
    if not lg.handlers:
        h=logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg
