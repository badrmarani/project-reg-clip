import re
from collections import OrderedDict
from typing import List

import yake


def update_state_dict(state_dict: OrderedDict) -> OrderedDict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[".".join(k.split(".")[1:])] = v
    return new_state_dict


def extract_keywords(captions: str, **kwargs) -> List[str]:
    custom_kw_extractor = yake.KeywordExtractor(**kwargs)
    keywords = custom_kw_extractor.extract_keywords(captions)
    keywords = [keyword[0] for keyword in keywords]
    return keywords


def is_keyword_in_caption(caption, keyword):
    result = re.findall(f"\\b{keyword}\\b", caption, flags=re.IGNORECASE)
    if len(result) > 0:
        return True
    else:
        return False
