import logging
import random
import torch
import numpy as np
import os

from src.functions.processor_ad import (
    SquadV1Processor,
    SquadV2Processor,
    squad_convert_examples_to_features
)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

# tensor를 list 형으로 변환하기위한 함수
def to_list(tensor):
    return tensor.detach().cpu().tolist()

# dataset을 load 하는 함수
def load_examples(args, tokenizer, evaluate=False, output_examples=False, do_predict=False, input_dict=None,
                  f_name=None):
    '''

    :param args: 하이퍼 파라미터
    :param tokenizer: tokenization에 사용되는 tokenizer
    :param evaluate: 평가나 open test시, True
    :param output_examples: 평가나 open test 시, True / True 일 경우, examples와 features를 같이 return
    :param do_predict: open test시, True
    :param input_dict: open test시 입력되는 문서와 질문으로 이루어진 dictionary
    :return:
    examples : max_length 상관 없이, 원문으로 각 데이터를 저장한 리스트
    features : max_length에 따라 분할 및 tokenize된 원문 리스트
    dataset : max_length에 따라 분할 및 학습에 직접적으로 사용되는 tensor 형태로 변환된 입력 ids
    '''
    input_dir = args.data_dir
    # cached_features_file = os.path.join(
    #     input_dir,
    #     "cached_{}_{}_{}".format(
    #         "dev" if evaluate else "train",
    #         list(filter(None, args.model_name_or_path.split("/"))).pop(),
    #         str(args.max_seq_length),
    #     ),
    # )

    if not evaluate:
        cached_features_file = os.path.join(input_dir, "extra",
            "cached_{}_{}_{}_indomain".format("train", list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length))
        )
    else:
        cached_features_file = os.path.join(input_dir,
            "cached_{}_{}_{}_outdomain_aug".format("pred", list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)))
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
    # open test 시
    if do_predict:
        examples = processor.get_example_from_input(input_dict)
    # 평가 시
    elif evaluate:
        examples = processor.get_dev_examples(args.data_dir,
                                              filename=f_name)
    # 학습 시
    else:
        examples = processor.get_train_examples(args.data_dir,
                                                filename=f_name)

    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        if not evaluate:
            dataset = features_and_dataset["dataset"]
        else:
            features, dataset = (
                features_and_dataset["features"],
                features_and_dataset["dataset"]
            )
    else:
        print("Creating features from dataset file at {}".format(input_dir))

        # processor 선언

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )
        if not evaluate:
            torch.save({"dataset": dataset}, cached_features_file)
        else:
            torch.save({"features": features, "dataset": dataset}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset
def make_cache_file(args, tokenizer, f_name=None, flag = None):

    input_dir = os.path.join(args.data_dir, "extra")
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(flag)
        ),
    )
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

    examples = processor.get_train_examples(args.data_dir,
                                                filename=f_name, flag=True)


    print("Creating features from dataset file at {}".format(input_dir))

    # processor 선언

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=args.threads,
    )

    torch.save({"dataset": dataset}, cached_features_file)


def make_cache_file_v2(args, tokenizer, f_name=None, flag=None):
    input_dir = os.path.join(args.data_dir, "extra")
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(flag)
        ),
    )
    processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

    examples = processor.get_train_examples(args.data_dir,
                                            filename=f_name)

    print("Creating features from dataset file at {}".format(input_dir))

    # processor 선언

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=args.threads,
    )
    del (dataset)
    del (examples)


    examples = processor.get_train_examples(args.data_dir,
                                            filename="generated_data_silver.json", flag=True)

    print("Creating features from dataset file at {}".format(input_dir))

    # processor 선언

    features_2, dataset_2 = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=True,
        return_dataset="pt",
        threads=args.threads,
    )
    del (examples)
    del (dataset_2)

    features += features_2
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
    all_example_indices = torch.tensor([f.example_index for f in features], dtype=torch.long)
    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(
        all_input_ids,
        all_attention_masks,
        all_token_type_ids,
        all_start_positions,
        all_end_positions,
        all_cls_index,
        all_p_mask,
        all_is_impossible,
        all_example_indices,
        all_feature_index
    )
    torch.save({"dataset": dataset}, cached_features_file)