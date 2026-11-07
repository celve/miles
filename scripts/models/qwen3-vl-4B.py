from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Qwen3-VL-4B's language tower is Qwen3-4B's, apart from config.json's rope_theta of 5,000,000.
    return load_sibling_model_args(__file__, "qwen3-4B", rotary_base="5000000")
