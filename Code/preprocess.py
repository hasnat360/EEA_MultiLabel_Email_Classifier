import os
import re
import numpy as np
import pandas as pd
from Config import Config
from utils import keep_top_level_classes


def get_input_data() -> pd.DataFrame:
    frames = []
    for filename in Config.DATA_FILES:
        path = os.path.join(Config.DATA_DIR, filename)
        frames.append(pd.read_csv(path, skipinitialspace=True))

    df = pd.concat(frames, ignore_index=True)
    df.rename(columns=Config.COLUMN_RENAME, inplace=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    for col in [Config.Y1, Config.Y2, Config.Y3, Config.Y4]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    df["y"] = df[Config.CLASS_COL]
    df = df.loc[df["y"].notna() & (df["y"].astype(str).str.strip() != "")].copy()
    return df.reset_index(drop=True)


def de_duplication(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["ic_deduplicated"] = ""

    cu_template = {
        "english": [
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) Customer Support team,?",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.??",
            r"(?:Aspiegel|\*\*\*\*\*\(PERSON\)) SE is the provider of Huawei Mobile Services to Huawei and Honor device owners in (?:Europe|\*\*\*\*\*\(LOC\)), Canada, Australia, New Zealand and other countries\.??",
        ]
    }
    all_patterns = sum(list(cu_template.values()), [])
    cu_pattern = "|".join(f"({p})" for p in all_patterns)

    split_pattern = (
        r"(From\s?:\s?xxxxx@xxxx.com Sent\s?:.{30,70}Subject\s?:)"
        r"|(On.{30,60}wrote:)"
        r"|(Re\s?:|RE\s?:)"
        r"|(\*\*\*\*\*\(PERSON\) Support issue submit)"
        r"|(\s?\*\*\*\*\*\(PHONE\))*$"
    )

    for ticket_id in data[Config.TICKET_ID].value_counts().index:
        df_t = data.loc[data[Config.TICKET_ID] == ticket_id]
        seen = set()
        deduped = []
        for ic in df_t[Config.INTERACTION_CONTENT]:
            parts = re.split(split_pattern, str(ic))
            parts = [p for p in parts if p is not None]
            parts = [re.sub(split_pattern, "", p.strip()) for p in parts]
            parts = [re.sub(cu_pattern, "", p.strip()) for p in parts]
            unique_parts = []
            for part in parts:
                if part and part not in seen:
                    seen.add(part)
                    unique_parts.append(part + "\n")
            deduped.append(" ".join(unique_parts))
        data.loc[data[Config.TICKET_ID] == ticket_id, "ic_deduplicated"] = deduped

    data[Config.INTERACTION_CONTENT] = data["ic_deduplicated"]
    return data.drop(columns=["ic_deduplicated"])


def noise_remover(df: pd.DataFrame) -> pd.DataFrame:
    ts_noise = (
        r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*)"
        r"|(\[|\])|(aspiegel support issue submit)|(null)|(nan)"
        r"|((bonus place my )?support.pt 自动回复:)"
    )
    df[Config.TICKET_SUMMARY] = (
        df[Config.TICKET_SUMMARY]
        .str.lower()
        .replace(ts_noise, " ", regex=True)
        .replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df[Config.INTERACTION_CONTENT] = (
        df[Config.INTERACTION_CONTENT]
        .str.replace("&amp;", "&", regex=False)
        .str.replace("&lt;", "<", regex=False)
        .str.replace("&gt;", ">", regex=False)
        .str.replace("&quot;", '"', regex=False)
        .str.replace("&#39;", "'", regex=False)
        .str.lower()
    )

    ic_noise_patterns = [
        r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|\.)\d{2}",
        r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
        r"\S+@\S+\.\S+",
        r"http\S+|www\.\S+",
        r"dear ((customer)|(user))",
        r"dear",
        r"(hello)|(hallo)|(hi )|(hi there)",
        r"good morning",
        r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
        r"thank you for contacting us",
        r"thank you for your availability",
        r"thank you for providing us this information",
        r"thank you for contacting",
        r"thank you for reaching us (back)?",
        r"thank you for patience",
        r"thank you for (your)? reply",
        r"thank you for (your)? response",
        r"thank you for (your)? cooperation",
        r"thank you for providing us with more information",
        r"thank you very kindly",
        r"thank you( very much)?",
        r"i would like to follow up on the case you raised on the date",
        r"i will do my very best to assist you",
        r"in order to give you the best solution",
        r"could you please clarify your request with following information:",
        r"in this matter",
        r"we hope you(( are)|('re)) doing ((fine)|(well))",
        r"i would like to follow up on the case you raised on",
        r"we apologize for the inconvenience",
        r"sent from my huawei (cell )?phone",
        r"original message",
        r"customer support team",
        r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland\.",
        r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
        r"canada, australia, new zealand and other countries",
        r"\d+",
        r"[^0-9a-zA-Z]+",
        r"(\s|^).(\s|$)",
    ]

    for pattern in ic_noise_patterns:
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(pattern, " ", regex=True)

    df[Config.INTERACTION_CONTENT] = (
        df[Config.INTERACTION_CONTENT]
        .replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return keep_top_level_classes(df, Config.Y1, 10).reset_index(drop=True)


def create_chained_cols(df: pd.DataFrame) -> pd.DataFrame:
    sep = Config.CHAIN_SEPARATOR
    for chain_name, cols in Config.CHAINED_TARGETS.items():
        out_col = f"y_{chain_name}"
        df[out_col] = df[cols[0]].astype(str)
        for col in cols[1:]:
            df[out_col] = df[out_col] + sep + df[col].fillna("NA").astype(str)
        for col in cols:
            bad = df[col].isna() | (df[col].astype(str).str.strip() == "") | (df[col].astype(str) == "nan")
            df.loc[bad, out_col] = np.nan
    return df


def translate_to_en(texts: list[str]) -> list[str]:
    if not Config.ENABLE_TRANSLATION:
        return texts
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target="en")
        out = []
        for text in texts:
            try:
                if isinstance(text, str) and text.strip():
                    translated = translator.translate(text[:4500])
                    out.append(translated if translated else text)
                else:
                    out.append(text)
            except Exception:
                out.append(text)
        return out
    except ImportError:
        print("  [INFO] deep_translator not installed — skipping translation.")
        return texts
