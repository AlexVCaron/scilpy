# -*- coding: utf-8 -*-

import hashlib
import inspect
import json
from importlib_resources import files
import logging
import os
import pathlib
import zipfile

import requests

from scilpy import SCILPY_HOME

DVC_URL = "https://scil.usherbrooke.ca/scil_test_data/dvc-store/files/md5"


def download_file_from_google_drive(url, destination):
    """
    Download large file from Google Drive.
    Parameters
    ----------
    id: str
        id of file to be downloaded
    destination: str
        path to destination file with its name and extension
    """
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                f.write(chunk)

    session = requests.Session()
    response = session.get(url, stream=True)

    save_response_content(response, destination)


def get_testing_files_dict():
    """ Get dictionary linking zip file to their md5sums computed by DVC """
    return json.load(files('data').joinpath('test_data.json').open())


def fetch_data(files_dict, keys=None):
    """
    Fetch data. Typical use would be with gdown.
    But with too many data accesses, downloaded become denied.
    Using trick from https://github.com/wkentaro/gdown/issues/43.
    """

    if not os.path.exists(SCILPY_HOME):
        os.makedirs(SCILPY_HOME)

    if keys is None:
        keys = files_dict.keys()
    elif isinstance(keys, str):
        keys = [keys]
    for f in keys:
        url_md5 = files_dict[f]
        full_path = os.path.join(SCILPY_HOME, f)
        full_path_no_ext, ext = os.path.splitext(full_path)

        CURR_URL = DVC_URL + "/" + url_md5[:2] + "/" + url_md5[2:]
        if not os.path.isdir(full_path_no_ext):
            if ext == '.zip' and not os.path.isdir(full_path_no_ext):
                logging.warning('Downloading and extracting {} from url {} to '
                                '{}'.format(f, CURR_URL, SCILPY_HOME))

                # Robust method to Virus/Size check from GDrive
                download_file_from_google_drive(CURR_URL, full_path)

                with open(full_path, 'rb') as file_to_check:
                    data = file_to_check.read()
                    md5_returned = hashlib.md5(data).hexdigest()
                if md5_returned != url_md5:
                    try:
                        zipfile.ZipFile(full_path)
                    except zipfile.BadZipFile:
                        raise RuntimeError("Could not fetch valid archive for "
                                           "file {}".format(f))
                    raise ValueError('MD5 mismatch for file {}.'.format(f))

                try:
                    # If there is a root dir, we want to skip one level.
                    z = zipfile.ZipFile(full_path)
                    zipinfos = z.infolist()
                    root_dir = pathlib.Path(
                        zipinfos[0].filename).parts[0] + '/'
                    assert all([s.startswith(root_dir) for s in z.namelist()])
                    nb_root = len(root_dir)
                    for zipinfo in zipinfos:
                        zipinfo.filename = zipinfo.filename[nb_root:]
                        if zipinfo.filename != '':
                            z.extract(zipinfo, path=full_path_no_ext)
                except AssertionError:
                    # Not root dir. Extracting directly.
                    z.extractall(full_path)
            else:
                raise NotImplementedError("Data fetcher was expecting to deal "
                                          "with a zip file.")

        else:
            # toDo. Verify that data on disk is the right one.
            logging.warning("Not fetching data; already on disk.")


def get_synb0_template_path():
    """
    Return MNI 2.5mm template in scilpy repository
    Returns
    -------
    path: str
        Template path
    """
    import scilpy  # ToDo. Is this the only way?
    module_path = inspect.getfile(scilpy)
    module_path = os.path.dirname(os.path.dirname(module_path))

    path = os.path.join(module_path, 'data/',
                        'mni_icbm152_t1_tal_nlin_asym_09c_masked_2_5.nii.gz')
    return path
