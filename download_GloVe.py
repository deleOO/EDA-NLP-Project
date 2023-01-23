import wget
import zipfile
import os

# Download glove.6B from this link https://nlp.stanford.edu/projects/glove/ using wget
def download_glove():
    # if glove.6B.zip is not in the data folder, download it
    if not os.path.exists('data/glove.6B.zip'):
        print('Downloading glove.6B.zip...')
        url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        wget.download(url, 'data/glove.6B.zip')
        print('Done!')

    else:
        print('glove.6B.zip is already downloaded!')

# Extract glove.6B.zip only if it is not extracted
def extract_sentiment140():
    if not os.path.exists('data/glove.6B.100d.txt'):
        print('Extracting glove.6B.zip...')
        with zipfile.ZipFile('data/glove.6B.zip', 'r') as zip_ref:
            zip_ref.extractall('data/')
        print('Done!')

    else:
        print('glove.6B.zip is already extracted!')

if __name__ == '__main__':
    download_glove()
    extract_sentiment140()
