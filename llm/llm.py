# from langchain.llms import CTransformers
import box
import yaml
from langchain.llms import LlamaCpp


# Import config vars
with open('config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))
llm = LlamaCpp(
                model_path="models/dolphin-2.6-mistral-7b.Q5_K_M.gguf",
                n_gpu_layers=-1,
                n_batch=2048,
                n_ctx=8000,
                # max_tokens=2048,
                temperature=0,
                # callback_manager=callback_manager,
                verbose=False,
                streaming=True,
                )

def setup_llm():
    # llm = CTransformers(model=cfg.MODEL_BIN_PATH,
    #                     model_type=cfg.MODEL_TYPE,
    #                     max_new_tokens=cfg.MAX_NEW_TOKENS,
    #                     temperature=cfg.TEMPERATURE
    # )

    
    return llm
