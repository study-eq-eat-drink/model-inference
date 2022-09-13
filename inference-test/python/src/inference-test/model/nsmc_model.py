from transformers import TFElectraModel, ElectraTokenizer


class NsmcKoelectraSmallTokenizer:

    __tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")

    @classmethod
    def tokenize_token(cls, text: str):
        return cls.__tokenizer.tokenize(text)

    @classmethod
    def tokenize_token_id(cls, text: str):
        tokenizer = cls.__tokenizer
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    @classmethod
    def tokenize_model_input(cls, text, max_length=512):
        model_input = cls.__tokenizer(
            text,
            return_tensors='np',
            truncation=True,
            max_length=max_length,
            padding="max_length",
            add_special_tokens=True
        )
        return model_input
