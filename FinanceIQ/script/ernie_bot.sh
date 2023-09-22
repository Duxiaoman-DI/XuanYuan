
cd ../src

# ERNIE-Bot-turbo
for i in {0,5}; do
python3 -u ernie.py \
    --version ERNIE-Bot-turbo \
    --data_dir ../data \
    --save_dir ../results/ErnieBot-turbo \
    --num_few_shot $i
done

# ERNIE-Bot
for i in {0,5}; do
python3 -u ernie.py \
    --version ERNIE-Bot \
    --data_dir ../data \
    --save_dir ../results/ErnieBot-turbo \
    --num_few_shot 0
done

