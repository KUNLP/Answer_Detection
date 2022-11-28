# Answer_Detection

# Dependencies
* python 3.7
* PyTorch 1.6.0
* Transformers 2.11.0
* AttrDict

# Model Architecture
# Data
* KLUE Machine Reading Comprehension Click

# Train & Test
* python3.7 run_ad --train_file [train file] --test_file [test_file] --from_init_weight --do_train
* python3.7 run_ad --test_file [test_file] --do_evaluate
