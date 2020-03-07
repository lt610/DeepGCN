import datetime
import os


def generate_path(folder, file_name, suffix):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestr = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    if file_name is None:
        file_name = 'tmp'
    file_name = file_name + '-' + timestr + suffix
    path = os.path.join(folder, file_name)
    return path