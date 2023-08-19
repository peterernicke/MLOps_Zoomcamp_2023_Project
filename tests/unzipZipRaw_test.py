import os.path

import scripts.unzipZipRaw

CVS_FILE = './data/raw/housing-prices-35.csv'


def test_unzipZipFile():
    if os.path.isfile(CVS_FILE):
        print("removing csv file...")
        os.remove(CVS_FILE)

    scripts.unzipZipRaw.unzip_raw_data()

    assert os.path.isfile(CVS_FILE)
