time_tag=$(date +%F-%H-%M-%S)
python simple_rl_train_with_detection.py -p dqn 2>&1|tee rl_log_${time_tag}.txt
