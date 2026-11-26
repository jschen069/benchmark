from mmengine.config import read_base
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import MMLUProDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import match_answer_pattern

with read_base():
    from .mmlu_pro_categories import categories

INSTRUCTION = (
    'Answer the following multiple choice question. The last line of your response '
    "should be of the following format: 'ANSWER: $LETTER' (without quotes) where "
    'LETTER is one of Options (e.g. one of ABCDEFGHIJKLMNOP). '
    'Think step by step before answering.'
)
QUESTION_AND_OPTIONS = 'Question:\n{question}\n\nOptions:\n{options_str}'

mmlu_pro_datasets = []

for category in categories:
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer',
        train_split='validation',
        test_split='test',
    )
    mmlu_pro_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=(
                f'{QUESTION_AND_OPTIONS}\n\n'
                "Let's think step by step.\n{cot_content}\n"
                'ANSWER: {answer}'
            ),
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template=(
                f'{INSTRUCTION}\n\n</E>{QUESTION_AND_OPTIONS}\n\n'
                "Let's think step by step."
            ),
            ice_token='</E>',
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r'(?i)ANSWER\s*:\s*([A-P])',
        ),
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='ais_bench/datasets/mmlu_pro',
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        )
    )
