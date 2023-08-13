import os.path
from zipfile import ZipFile

ZIP_FILE = './data/raw/archive.zip'
RAW_FOLDER = './data/raw/'
CVS_FILE = './data/raw/housing-prices-35.csv'

def unzip_raw_data(zip_file=ZIP_FILE, extract_to=RAW_FOLDER):
    if not os.path.isfile(CVS_FILE):
        zipfile = ZipFile(zip_file)
        zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    unzip_raw_data()