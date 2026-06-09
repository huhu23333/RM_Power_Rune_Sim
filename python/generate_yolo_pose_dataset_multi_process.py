import random
import subprocess
import os

def main():
    n_process = 10
    seed = 42
    train_num = 1000
    val_num = 200

    random.seed(seed)
    sub_seed_list = []
    while len(sub_seed_list) < n_process:
        new_seed = random.randint(0, 2**32-1)
        if new_seed in sub_seed_list:
            continue
        else:
            sub_seed_list.append(new_seed)

    val_n_process = max(1, int(n_process * val_num / (train_num + val_num)))
    train_n_process = max(1, n_process - val_n_process)
    train_num_list = []
    base_train_num = train_num // train_n_process
    extra_train_num = train_num - base_train_num * train_n_process
    accumulate_train_num = 0
    for i in range(train_n_process):
        this_train_num = base_train_num + (1 if i < extra_train_num else 0)
        train_num_list.append((accumulate_train_num, this_train_num))
        accumulate_train_num += this_train_num
    val_num_list = []
    base_val_num = val_num // val_n_process
    extra_val_num = val_num - base_val_num * val_n_process
    accumulate_val_num = 0
    for i in range(val_n_process):
        this_val_num = base_val_num + (1 if i < extra_val_num else 0)
        val_num_list.append((accumulate_val_num, this_val_num))
        accumulate_val_num += this_val_num

    sub_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_yolo_pose_dataset.py")
    sub_processes = []

    all_process_index = 0
    if train_num > 0:
        for i in range(len(train_num_list)):
            is_last_process = val_num == 0 and i == len(train_num_list) - 1
            sub_processes.append(subprocess.Popen(["python3", str(sub_script_path), 
                                                   "--seed", str(sub_seed_list[all_process_index]),
                                                   "--sub_process_index", str(all_process_index), 
                                                   "--sub_process_start_index", str(train_num_list[i][0]),
                                                   "--sub_process_train", str(train_num_list[i][1])] 
                                                   + (["--sub_process_yaml", "1"] 
                                                   if is_last_process else [])))
            all_process_index += 1
    if val_num > 0:
        for i in range(len(val_num_list)):
            is_last_process = i == len(val_num_list) - 1
            sub_processes.append(subprocess.Popen(["python3", str(sub_script_path), 
                                                   "--seed", str(sub_seed_list[all_process_index]),
                                                   "--sub_process_index", str(all_process_index), 
                                                   "--sub_process_start_index", str(val_num_list[i][0]),
                                                   "--sub_process_val", str(val_num_list[i][1])] 
                                                   + (["--sub_process_yaml", "1"] 
                                                   if is_last_process else [])))
            all_process_index += 1

    for t in sub_processes:
        t.wait()



if __name__ == "__main__":
    main()