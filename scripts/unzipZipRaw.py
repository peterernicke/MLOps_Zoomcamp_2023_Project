from zipfile import ZipFile

ZIP_FILE = './data/raw/archive.zip'
RAW_FOLDER = './data/raw/'

def unzip_raw_data(zip_file=ZIP_FILE, extract_to=RAW_FOLDER):
    zipfile = ZipFile(zip_file)
    zipfile.extractall(path=extract_to)

if __name__ == "__main__":
    unzip_raw_data()