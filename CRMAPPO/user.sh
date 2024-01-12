for seed in {1..5}
do
  for user_num in {3..8}
  do
    for beta in 0.0 0.2 0.4 0.6 0.8
    do
        python main.py --seed=$seed --user_num=$user_num --write=True --beta=$beta --seed=1
    done
  done
done
