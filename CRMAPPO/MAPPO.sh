for seed in {1..5}
do
  for user_num in {3..8}
  do
        python main.py --seed=$seed --user_num=$user_num --write=True --beta=0.0 --seed=1
  done
done
