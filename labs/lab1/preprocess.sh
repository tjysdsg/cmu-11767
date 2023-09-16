for v in 1000 5000 10000; do
  python generate_bow.py SST-2/train.tsv vocab_${v}.json --k $v
  python preprocess.py SST-2/train.tsv train_bow${v}.npy --vocab vocab_${v}.json
  python preprocess.py SST-2/dev.tsv dev_bow${v}.npy --vocab vocab_${v}.json
done
