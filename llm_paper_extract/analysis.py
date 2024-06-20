import sys
from typing import List

try:
    import numpy as np
    from mlcm import mlcm
    np.int
except AttributeError:
    # Fix AttributeError in mlcm.mlcm
    # *** AttributeError: module 'numpy' has no attribute 'int'.
    # `np.int` was a deprecated alias for the builtin `int`. To avoid this error
    # in existing code, use `int` by itself. Doing this will not modify any
    # behavior and is safe. When replacing `np.int`, you may wish to use e.g.
    # `np.int64` or `np.int32` to specify the precision. If you wish to review
    # your current use, check the release note link for additional information.
    # The aliases was originally deprecated in NumPy 1.20; for more details and
    # guidance see the original release note at:
    #     https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations. Did you mean: 'inf'?
    np.int = int
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import ROOT_DIR
from .model import ExtractionResponse, PaperExtractions
from .utils import build_validation_set, model2df


def _append_left_indices(df:pd.DataFrame, indices:List[tuple]):
    df = df.copy(True)
    if df.empty:
        return df

    try:
        index_names = list(range(len(df.index.levels)))
        df.index.names = index_names
        df.reset_index(inplace=True)
    except AttributeError:
        index_names = list(range(df.index.nlevels-1))
        if index_names:
            df.index.names = index_names

    inds = [ind for ind, _ in indices]
    for ind, val in indices:
        df.loc[:,ind] = val

    df.set_index([*inds,*index_names], inplace=True)
    df.index.names = [None] * (len(inds) + len(index_names))

    return df


def _cm(annotations:pd.DataFrame, predictions:pd.DataFrame):
    classes = pd.concat([annotations, predictions])
    classes.sort_values(inplace=True, ignore_index=True)
    classes.drop_duplicates(inplace=True, ignore_index=True)

    return confusion_matrix(annotations, predictions, labels=classes), classes


def _mlcm(annotations:pd.DataFrame, predictions:pd.DataFrame):
    classes = pd.concat(list(annotations) + list(predictions))
    classes.sort_values(inplace=True, ignore_index=True)
    classes.drop_duplicates(inplace=True, ignore_index=True)

    _ann, _pred = (
        [classes.isin(arr).astype(int) for arr in annotations],
        [classes.isin(arr).astype(int) for arr in predictions]
    )

    return mlcm.cm(_ann, _pred), classes


if __name__ == "__main__":
    validation_set = build_validation_set(ROOT_DIR / "data/")

    annotated = [[], []]
    predictions = [[], []]

    for f in validation_set:
        f = (ROOT_DIR / "data/merged/") / f.with_suffix(".json").name
        if not f.exists():
            continue

        print(f"Fetching data from {f}", file=sys.stderr)
        model = PaperExtractions.model_validate_json(f.read_text())
        paper_attr, paper_refs = map(
            lambda m:_append_left_indices(m, [("paper_id",f.stem)]),
            model2df(model)
        )

        annotated[0].append(paper_attr)
        annotated[1].append(paper_refs)

        queries_dir = (ROOT_DIR / "data/queries/")
        for i, query_f in enumerate(sorted(queries_dir.glob(f.with_stem(f"{f.stem}*").name))):
            print(f"Fetching data from {query_f}", file=sys.stderr)
            model = ExtractionResponse.model_validate_json(query_f.read_text()).extractions

            paper_attr, paper_refs = map(
                lambda m:_append_left_indices(m, [("paper_id",f.stem),("attempt",i)]),
                model2df(model)
            )

            predictions[0].append(paper_attr)
            predictions[1].append(paper_refs)

    annotated[0] = pd.concat(annotated[0])
    annotated[1] = pd.concat(annotated[1])

    predictions[0] = pd.concat(predictions[0])
    predictions[1] = pd.concat(predictions[1])

    _analysis_dir = (ROOT_DIR / f"data/analysis/")
    _analysis_dir.mkdir(parents=True, exist_ok=True)

    max_attempt = max(predictions[0].reset_index([0,],drop=True).index)

    for label in ("title", "type", "research_field"):
        for i in range(max_attempt + 1):
            mat, classes = _cm(
                annotated[0].loc[:,label],
                predictions[0].loc[:,i,:].loc[:,label]
            )
            # Title confusion matrix should be the identity
            # if label == "title":
            #     assert (_mat == np.identity(_mat.shape[0])).all()

            (_analysis_dir / f"{label}_{i:02}.csv").write_text(
                pd.DataFrame(mat, index=classes, columns=classes).to_csv()
            )

    for label in ("sub_research_fields", "all_research_fields",):
        for i in range(max_attempt + 1):
            ann, pred = (
                annotated[0].loc[:,label],
                predictions[0].loc[:,i,:].loc[:,label]
            )

            (conf_mat, normal_conf_mat), classes = _mlcm(ann, pred)

            (_analysis_dir / f"{label}_{i:02}.csv").write_text(
                pd.DataFrame(conf_mat, index=[*classes, ""], columns=[*classes, ""]).to_csv()
            )
            print(f"{label}:")
            print("Raw confusion Matrix:")
            print(conf_mat)
            print("Normalized confusion Matrix (%):")
            print(normal_conf_mat)

    for group in ("models", "datasets", "libraries",):
        for i in range(max_attempt + 1):
            ann, pred = (
                annotated[1].loc[:,group,:],
                predictions[1].loc[:,i,group,:]
            )
            ann:pd.DataFrame
            pred:pd.DataFrame

            papers = (
                ann.index.get_level_values(0).union(
                    pred.index.get_level_values(0)
                )
                .sort_values().drop_duplicates()
            )
            ann_per_paper, pred_per_paper = (
                [ann[ann.index.isin([p], level=0)] for p in papers],
                [pred[pred.index.isin([p], level=0)] for p in papers],
            )

            col = "name"
            (conf_mat, normal_conf_mat), names = _mlcm(
                [_ann[col] for _ann in ann_per_paper],
                [_pred[col] for _pred in pred_per_paper],
            )

            (_analysis_dir / f"{group}.{col}_{i:02}.csv").write_text(
                pd.DataFrame(conf_mat, index=[*names, ""], columns=[*names, ""]).to_csv()
            )

            print(f"{group}.{col}:")
            print("Raw confusion Matrix:")
            print(conf_mat)
            print("Normalized confusion Matrix (%):")
            print(normal_conf_mat)

            ann_matches = []
            pred_matches = []

            for ann_items, pred_items in zip(ann_per_paper, pred_per_paper):
                for _, row in ann_items.iterrows():
                    pred_match = pred_items[pred_items["name"] == row["name"]]
                    if not pred_match.empty:
                        assert len(pred_match) == 1, (
                            f"Too many matches for\n{row['name']}\nin\n{pred_items}"
                        )

                        ann_matches.append(pd.DataFrame([row]))
                        pred_matches.append(pred_match)

            ann_matches, pred_matches = (
                pd.concat(ann_matches).reset_index(drop=True),
                pd.concat(pred_matches).reset_index(drop=True)
            )

            for col in ann.columns:
                if col in ("role", "mode",):
                    (conf_mat, normal_conf_mat), classes = _mlcm(
                        [ann_matches.loc[idx:idx,col] for idx in ann_matches.index],
                        [pred_matches.loc[idx:idx,col] for idx in pred_matches.index],
                    )

                    (_analysis_dir / f"{group}.{col}_{i:02}.csv").write_text(
                        pd.DataFrame(conf_mat, index=[*classes, ""], columns=[*classes, ""]).to_csv()
                    )

                    print(f"{group}.{col}:")
                    print("Raw confusion Matrix:")
                    print(conf_mat)
                    print("Normalized confusion Matrix (%):")
                    print(normal_conf_mat)
                else:
                    pass
