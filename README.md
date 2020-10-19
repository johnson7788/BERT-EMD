# BERT-EMD
该repository包含EMNLP 2020中论文“BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover's Distance”中提出的模型的PyTorch实现。
下图说明了模型体系结构的高阶视图。
![BERT-EMD Model](BERT-EMD-model.png "BERT-EMD")
有关BERT-EMD技术的更多详细信息，请参阅我们的论文 :https://arxiv.org/pdf/2010.06133.pdf

### Installation

Run command below to install the environment (using python3).

```
pip install -r requirements.txt 
```

### Data and Pre-train Model Prepare

1. Get GLUE data:
```
python download_glue_data.py --data_dir glue_data --tasks all
```
2. Get BERT-Base offical model from [here](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip), download and unzip to directory  `./model/bert_base_uncased`. Convert tf model to pytorch model:
```
cd bert_finetune
python convert_bert_original_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path ../model/bert_base_uncased \
--bert_config_file ../model/bert_base_uncased/bert_config.json \
--pytorch_dump_path ../model/pytorch_bert_base_uncased
``` 
或者您可以直接从以下位置下载pytorch版本 [huggingface](https://huggingface.co/bert-base-uncased#).

3. Get finetune teacher model, take task QQP for example:
```

export MODEL_PATH=../model/pytorch_bert_base_uncased/
export TASK_NAME=QQP
python run_glue.py \
  --model_type bert \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../data/glue_data/$TASK_NAME/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --save_steps 2000 \
  --output_dir ../model/$TASK_NAME/teacher/ \
  --evaluate_during_training \
  --overwrite_output_dir
```
4. 获取预训练的常规蒸馏TinyBERT student模型: [4-layer](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z) and [6-layer](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF).
Unzip to directory  `model/student/layer4` and  `model/student/layer6` respectively.
5. Distill student model, take 4-layer student model for example:
```
cd bert_emd
export TASK_NAME=QQP
python emd_task_distill.py  \
--data_dir ../glue_data/$TASK_NAME/ \
--teacher_model ../model/$TASK_NAME/teacher/ \
--student_model ../model/student/layer4/ \
--task_name $TASK_NAME \
--output_dir ../model/$TASK_NAME/student/
```
