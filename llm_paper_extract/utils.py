import random
from pathlib import Path
import re
import sys
import typing
import unicodedata
import warnings

import pandas as pd
from pydantic import BaseModel, Field, create_model

from .models.model import Explained, Model, PaperExtractions

PAPERS_TO_IGNORE={"data/cache/arxiv/2404.09932.txt",}

_MODE_ALIASES = {
    "trained": ["training", "evaluation", "pretraining"],
    "inference": [],
    "finetuned": [],
}

_MODE_ALIASES = {
    alias: k
    for k, v in _MODE_ALIASES.items()
    for alias in {k, *v}
}

_RESEARCH_FIELDS_ALIASES = {
    "3dreconstruction": [],
    "3dvision": [],
    "accentclassification": [],
    "aiconsciousness": [],
    "aiethics": [],
    "aiethicsandhumancomputerinteractionhci": [],
    "aiforhumanity": [],
    "anomalydetection": [],
    "artificialgeneralintelligence": ["agi"],
    "artificialintelligence": ["ai"],
    "attentionmechanisms": [],
    "autonomousvehiclesystems": [],
    "bayesianinferenceandgenerativemodels": [],
    "combinatorialoptimization": [],
    "computationalbiology": [],
    "computervision": ["cv"],
    "continuallearning": ["cl"],
    "crosslingualtransferlearning": [],
    "deeplearning": ["dl"],
    "deepreinforcementlearning": ["drl"],
    "differentiableprogramming": [],
    "efficientinference": [],
    "energymanagementinroboticsystems": [],
    "evaluationmetrics": [],
    "fairnessinai": [],
    "fairnessinrecommendersystems": [],
    "generativemodels": ["generativemodeling"],
    "goalconditionedreinforcementlearning": [],
    "graphneuralnetwork": ["gnn", "gnns", "graphneuralnetworks", "graphneuralnetworksgnns"],
    "humancomputerinteraction": ["hci"],
    "humanintheloopreinforcementlearning": [],
    "interpretability": [],
    "interpretablemachinelearning": [],
    "longtermmemory": [],
    "machinelearning": ["ml"],
    "mathematics": [],
    "medical": [],
    "medicalimageanalysis": [],
    "medicalimagesegmentation": [],
    "medicalimaging": [],
    "microscopyimageanalysis": [],
    "modelcompressionsparsetrainingpruning": [],
    "modeloptimization": [],
    "modelriskmanagement": [],
    "modelsafetyethicsinai": [],
    "molecularpropertyprediction": [],
    "multiagentreinforcementlearning": [],
    "multilingualandlowresourcelanguageprocessing": [],
    "multilingualdatasetsandlargelanguagemodels": [],
    "multilingualnlp": [],
    "musicrecommendationsystems": [],
    "naturallanguageprocessing": ["nlp"],
    "navigationagents": [],
    "neuraldifferentialequations": [],
    "neuralnetworkarchitectures": [],
    "neuralnetworkoptimization": [],
    "neuralsymboliclearning": [],
    "optimizationandmetaheuristics": [],
    "optimizationandtraining": [],
    "optimizationindeeplearning": [],
    "outofdistribution": [],
    "proteinstructureprediction": [],
    "recommendersystems": [],
    "reinforcementlearning": ["rl"],
    "representationlearning": [],
    "roboticphotography": [],
    "roboticplanningandcontrol": [],
    "robotics": [],
    "sampleefficientreinforcementlearning": [],
    "sceneunderstanding": [],
    "scientificmachinelearning": [],
    "speechprocessing": [],
    "speechrecognition": [],
    "textclassification": [],
    "theoremproving": [],
    "timeseriesanomalydetection": [],
    "timeseriesforecasting": [],
    "trajectoryprediction": [],
    "transitnetworkdesigngraphlearning": [],
    "ultrasoundimaging": [],
    "visualcomputing": [],
    "visualquestionanswering": ["vqa"],
    "weatherforecast": [],
}

_RESEARCH_FIELDS_ALIASES = {
    alias: k
    for k, v in _RESEARCH_FIELDS_ALIASES.items()
    for alias in {k, *v, *([f"{k}{v[0]}"] if v else [])}
}


def _mode_aliases(mode):
    return _MODE_ALIASES.get(mode, mode)


def _reasearch_field_alias(field):
    return _RESEARCH_FIELDS_ALIASES.get(field, field)


def build_validation_set(data_dir:Path, seed=42):
    random.seed(seed)

    all_papers = set()
    research_fields = sorted(
        [fn.name.split("_")[0] for fn in data_dir.glob("*_papers.txt")]
    )
    papers_by_field = {}

    for field in research_fields:
        papers_by_field.setdefault(field, set())
        field_papers:set = papers_by_field[field]
        all_field_papers = (data_dir / f"{field}_papers.txt").read_text().splitlines()
        all_field_papers = sorted([p for p in all_field_papers if p])
        while len(field_papers) < 10:
            _field_papers = set(random.sample(all_field_papers, 10 - len(field_papers)))
            field_papers.update(_field_papers - all_papers)
            all_papers.update(_field_papers)
        print(
            f"Selected {len(field_papers)} papers out of {len(all_field_papers)} papers for field {field}",
            file=sys.stderr
        )
        papers_by_field[field] = sorted(field_papers)

    # Try to minimize impact on random selections by filtering the particular
    # papers only at the end
    for field in research_fields:
        papers_by_field.setdefault(field, set())
        field_papers = set(papers_by_field[field])
        field_papers = field_papers - PAPERS_TO_IGNORE
        all_field_papers = (data_dir / f"{field}_papers.txt").read_text().splitlines()
        all_field_papers = sorted([p for p in all_field_papers if p])
        while len(field_papers) < 10:
            _field_papers = set(random.sample(all_field_papers, 10 - len(field_papers)))
            field_papers.update(_field_papers - all_papers - PAPERS_TO_IGNORE)
            all_papers.update(_field_papers)
        papers_by_field[field] = sorted(field_papers)

    validation_set = sum(papers_by_field.values(), [])
    # # Dev validation set
    # validation_set = sum(map(lambda _:random.sample(_, 1), papers_by_field.values()), [])
    return list(map(lambda p:Path(p).absolute(), validation_set))


def fix_explained_fields():
    import pdb ; pdb.set_trace()
    fields = _get_fields(PaperExtractions)
    return create_model(PaperExtractions.__name__, **fields)


def model2df(model:BaseModel):
    paper_1d_df = {}
    paper_references_df = {k:{} for k in Model.model_fields}

    # import ipdb ; ipdb.set_trace()

    for k, v in model:
        if isinstance(v, Explained):
            v = v.value

        if k in ("type",):
            v = str_normalize(v.split()[0])

        elif k in ("research_field", "sub_research_field",):
            v, *extra = [_.strip().rstrip("]") for _ in v.split("[[")]
            # print(v, extra)
            assert len(extra) <= 1
            # if extra:
            #     import ipdb ; ipdb.set_trace()
            extra = [_.strip() for _ in extra for _ in _.split(",")]
            v = [v, *extra]
            v = map(str_normalize, v)
            v = sorted(set(map(_reasearch_field_alias, v)))
            # print(v)

        if k in ("title", "type", "research_field",):
            paper_1d_df[k] = v

        elif k in ("sub_research_field",):
            # This will become a list
            paper_1d_df.setdefault("sub_research_fields", [])
            paper_1d_df["sub_research_fields"] += v

        if k in ("research_field", "sub_research_field",):
            paper_1d_df.setdefault("all_research_fields", [])
            paper_1d_df["all_research_fields"] += v

        elif k in ("models", "datasets", "libraries",):
            for i, entry in enumerate(v):
                for entry_k, entry_v in entry:
                    if isinstance(entry_v, Explained):
                        entry_v = entry_v.value

                    if entry_k in ("name", "type",):
                        entry_v = str_normalize(entry_v)
                    elif entry_k in ("mode",):
                        try:
                            entry_v = _mode_aliases(str_normalize(entry_v.split()[0]))
                        except IndexError:
                            entry_v = None
                    elif entry_k in ("role",):
                        entry_v = str_normalize(entry_v.split()[0])

                    paper_references_df[entry_k][(k, i)] = entry_v

    paper_1d_df["sub_research_fields"] = [pd.Series(paper_1d_df["sub_research_fields"])]
    paper_1d_df["all_research_fields"] = [pd.Series(paper_1d_df["all_research_fields"])]
    paper_1d_df, paper_references_df = (
        pd.DataFrame(paper_1d_df),
        pd.DataFrame(paper_references_df)
    )

    # import ipdb ; ipdb.set_trace()
    for group in ("models", "datasets", "libraries",):
        try:
            _l = paper_references_df.loc[group]["name"]
            _s = paper_references_df.loc[group]["name"].drop_duplicates()
        except KeyError:
            _l = []
            _s = set()

        if len(_l) != len(_s):
            warnings.warn(f"Possibly duplicated {group} in\n{_l}")

    return paper_1d_df, paper_references_df


def print_model(model_cls:BaseModel, indent = 0):
    for field, info in model_cls.model_fields.items():
        print(" " * indent, field, info)
        if typing.get_origin(info.annotation) == list:
            print_model(info.annotation.__args__[0], indent+2)
            continue

        try:
            print_model(info.annotation, indent+2)
        except AttributeError:
            pass


def str_normalize(string):
    string = unicodedata.normalize("NFKC", string).lower()
    # if "{{" in string:
    #     import ipdb ; ipdb.set_trace()
    string = [_s.split("}}") for _s in string.split("{{")]
    string = sum(string, [])
    exclude = string[1:2]
    # import ipdb ; ipdb.set_trace()
    string = list(map(
        lambda _s:re.sub(pattern=r"[^a-z0-9]", string=_s, repl=""),
        string[:1] + string[2:]
    ))
    string = "".join(string[:1] + exclude + string[1:])
    return string


def _get_fields(model_cls:BaseModel):
    fields = {}
    for field, info in model_cls.model_fields.items():
        if typing.get_origin(info.annotation) == list:
            try:
                sub_fields = _get_fields(info.annotation.__args__[0])
            except AttributeError:
                fields[field] = (info.annotation, Field(description=info.description))
                continue
            fields[field] = (
                typing.List[create_model(field, **sub_fields)],
                Field(description=info.description)
            )
            continue

        try:
            sub_fields = _get_fields(info.annotation)
        except AttributeError:
            fields[field] = (info.annotation, Field(description=info.description))
            continue

        if info.annotation.__base__ == Explained:
            sub_fields = {
                field:(sub_fields["value"][0], Field(description=info.description)),
                **{k:v for k,v in sub_fields.items() if k != "value"}
            }
            cls_name = f"{Explained.__name__}[{field}]"
        else:
            cls_name = info.annotation.__name__
        fields[field] = (create_model(cls_name, **sub_fields), Field(description=info.description))

    return fields
