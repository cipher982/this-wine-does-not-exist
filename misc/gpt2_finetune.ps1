python transformers/examples/language-modeling/run_language_modeling.py \
    -output_dir gpt2_output \
    -model_type gpt2 \
    -model_name_or_path gpt2 \
    -do_train \
    -train_data_file "data/scraped/name_desc_nlp_ready_train.txt" \
    -do_eval \
    -eval_data_file "data/scraped/name_desc_nlp_ready_test.txt"