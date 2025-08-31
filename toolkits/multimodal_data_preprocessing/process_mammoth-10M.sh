python convert_custom_dataset_to_wds_chatml.py \
--dataset-root /home/ma-user/work/Dataset/MAmmoTH-VL-Instruct-12M/ \
--json /home/ma-user/work/wza/ms-swift/train_space/mammoth_si_10M-train.json \
--train-split 99999 \
--val-split 1 \
--test-split 0 \
--max-samples-per-tar 100000
