import os
import zipfile
import urllib.request
import argparse

def download_embeddings(emb="fasttext"):
    
    dest_dir = emb
    os.makedirs(dest_dir, exist_ok=True)

    if emb == "fasttext":
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip"
        zip_name = "wiki.en.zip"
        txt_name = "wiki.en.vec"
    elif emb == "glove":
        url = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
        zip_name = "glove.6B.zip"
        txt_name = "glove.6B.300d.txt"
    else:
        raise ValueError("Not Supported")

    zip_path = os.path.join(dest_dir, zip_name)
    txt_path = os.path.join(dest_dir, txt_name)

    if not os.path.exists(txt_path):
        if not os.path.exists(zip_path):
            try:
                urllib.request.urlretrieve(url, zip_path)
                print(f"Saved at {zip_path}")
            except urllib.error.URLError as e:
                print("Error")
                return None
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extract(txt_name, path=dest_dir)
            print(f"Ready at {txt_path}")
        except zipfile.BadZipFile:
            print("Error")
            return None
    else:
        print(f"Found existing embeddings: {txt_path}")

    return txt_path

if __name__ == "__main__":
    path_fasttext = download_embeddings("fasttext")
    path_glove = download_embeddings("glove")
    
    if path_fasttext:
        print(f"Ready at: {path_fasttext}")
    if path_glove:
        print(f"Ready at: {path_glove}")
