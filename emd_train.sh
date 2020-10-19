cd bert_emd
export TASK_NAME=QQP
python emd_task_distill.py  \
--data_dir ../glue_data/$TASK_NAME/ \
--teacher_model ../model/$TASK_NAME/teacher/ \
--student_model ../model/student/layer4/ \
--task_name $TASK_NAME \
--output_dir ../model/$TASK_NAME/student/