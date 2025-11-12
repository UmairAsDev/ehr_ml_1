from bs4 import BeautifulSoup
import re

def clean_html(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def flag_any(t, keywords):
    print(f"DEBUG: Type of t in flag_any: {type(t)}, Value: {t}")
    t = t.lower()
    return int(any(k in t for k in keywords))





def mask_post_flare_terms(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'\b(flare|flares|flaring|flare-up|flare up|psoriasis flare)\b', ' ', text, flags=re.I)
    text = re.sub(r'\b(triamcinolone|clobetasol|hydrocortisone|ointment|apply|start|apply\s+\w+|prescribed|prescription|start\s+)\b', ' ', text, flags=re.I)

    text = re.sub(r'\b(apply|use)\b.*?(ointment|cream|gel)\b', ' ', text, flags=re.I)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


