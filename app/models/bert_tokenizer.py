from abc import ABC
from typing import Dict
import tensorflow as tf
from tensorflow_hub import KerasLayer
from official.nlp.bert.tokenization import FullTokenizer

from models.model import AbstractDataProcessor


class AbstractBertTokenizer(AbstractDataProcessor, ABC):
    """ Abstract BERT Tokenizer"""

    def __init__(self, encoder: KerasLayer, model_input_size: int):
        """ Create the BERT encoder and tokenizer """
        self.tokenizer = FullTokenizer(
            encoder.resolved_object.vocab_file.asset_path.numpy(),
            do_lower_case=encoder.resolved_object.do_lower_case.numpy(),
        )
        self.model_input_size = model_input_size

    def _format_bert_tokens(self, ragged_word_ids: tf.RaggedTensor) -> Dict[str, tf.Tensor]:
        """ Create, format and pad BERT's input tensors """
        # Generate mask, and pad word_ids and mask
        mask = tf.ones_like(ragged_word_ids).to_tensor()
        word_ids = ragged_word_ids.to_tensor()
        padding = tf.constant([[0, 0], [0, (self.model_input_size - mask.shape[1])]])
        word_ids = tf.pad(word_ids, padding, "CONSTANT")
        mask = tf.pad(mask, padding, "CONSTANT")
        type_ids = tf.zeros_like(mask)

        return {
            "input_word_ids": word_ids,
            "input_mask": mask,
            "input_type_ids": type_ids,
        }

    def _format_bert_word_piece_input(self, word_piece_tokens: tf.Tensor) -> tf.Tensor:
        word_piece_tokens.insert(0, "[CLS]")
        word_piece_tokens.append("[SEP]")
        return self.tokenizer.convert_tokens_to_ids(word_piece_tokens)
