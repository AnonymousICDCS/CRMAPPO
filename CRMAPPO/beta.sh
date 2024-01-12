for seed in {1..5}
do
    for beta in 0.0 0.2 0.4 0.6 0.8
    do
        python main.py --beta=$beta --user_num=6 --seed=$seed --write=True
    done
done
