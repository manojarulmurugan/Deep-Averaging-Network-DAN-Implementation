# Step 0. Change this to your campus ID
CAMPUSID='9087549409'
mkdir -p $CAMPUSID

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings
mkdir -p fasttext
mkdir -p glove
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
unzip glove.6B.zip -d glove
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip -d fasttext

# Step 2. Train models on two datasets.
##  2.1. Run experiments on SST
PREF='sst'
python3 main.py --train "data/${PREF}-train.txt" --dev "data/${PREF}-dev.txt" --test "data/${PREF}-test.txt" --dev_output "${CAMPUSID}/${PREF}-dev-output.txt"  --test_output "${CAMPUSID}/${PREF}-test-output.txt" --emb_file "fasttext/wiki.en.vec"

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
python3 main.py --train "data/${PREF}-train.txt" --dev "data/${PREF}-dev.txt" --test "data/${PREF}-test.txt" --dev_output "${CAMPUSID}/${PREF}-dev-output.txt"  --test_output "${CAMPUSID}/${PREF}-test-output.txt" --emb_file "glove/glove.6B.300d.txt" --emb_size 300 --hid_size 512 --hid_layer 2 --word_drop 0.3 --emb_drop 0.2 --hid_drop 0.25 --grad_clip 5.0 --max_train_epoch 10 --batch_size 32 --lrate 0.001

# Step 3. Prepare submission:
##  3.1. Copy your code to the $CAMPUSID folder
for file in 'main.py' 'model.py' 'vocab.py' 'setup.py'; do
    cp $file ${CAMPUSID}/
done
##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip (containing only .py/.txt/.pdf/.sh files)
python prepare_submit.py ${CAMPUSID} ${CAMPUSID}
##  3.3. Submit the zip file to Canvas ([https://canvas.wisc.edu/courses/292771/assignments](https://canvas.wisc.edu/courses/292771/assignments))! Congrats!
