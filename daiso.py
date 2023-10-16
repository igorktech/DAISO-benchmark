# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

import textwrap
import csv
import pandas as pd
import json
import os

import datasets

_VERSION = datasets.Version("1.1.0")

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_DAISO_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={Igor Kuzmin
},
year={2023}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DAISO_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URL = "https://raw.githubusercontent.com/igorktech/DAISO-benchmark/dev"


class DAISOConfig(datasets.BuilderConfig):
    """BuilderConfig for DAISO."""

    def __init__(self, label_classes, features, data_url, citation, url, **kwargs):
        """BuilderConfig for DAISO.
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the csv file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
          **kwargs: keyword arguments forwarded to super.
        """
        super(DAISOConfig, self).__init__(version=_VERSION, **kwargs)
        self.label_classes = label_classes
        self.features = features
        self.data_url = data_url
        self.citation = citation
        self.url = url


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class DAISO(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        DAISOConfig(
            name="dyda",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=
            {"commissive": {
                "base": "Commissive",
                "ISO": "commissive"
            },
                "directive": {
                    "base": "Directive",
                    "ISO": "directive"
                },
                "inform": {
                    "base": "Inform",
                    "ISO": "inform"
                },
                "question": {
                    "base": "Question",
                    "ISO": None
                }
            },
            features=[
                "Utterance",
                "Dialogue_Act",
                "Emotion",
                "Dialogue_ID",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/dyda/train.csv",
                "dev": _URL + "/dyda/dev.csv",
                "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """\
            @InProceedings{li2017dailydialog,
            author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
            title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
            booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
            year = {2017}
            }"""
            ),
            url="http://yanran.li/dailydialog.html",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),
        DAISOConfig(
            name="",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes={},
            features=[
            ],
            data_url={
                # "train": _URL + "/dyda/train.csv",
                # "dev": _URL + "/dyda/dev.csv",
                # "test": _URL + "/dyda/test.csv",
            },
            citation=textwrap.dedent(
                """"""
            ),
            url="",
        ),

    ]

    DEFAULT_CONFIG_NAME = "dyda"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = {feature: datasets.Value("string") for feature in self.config.features}
        if self.config.label_classes:
            features["Label"] = datasets.features.ClassLabel(names=list(self.config.label_classes.keys()))
            features["Label_ISO"] = datasets.features.ClassLabel(
                names=[map.get("ISO") for map in self.config.label_classes.values()])
        features["Idx"] = datasets.Value("int32")
        # if self.config.name == "":  # This is the name of the configuration selected in BUILDER_CONFIGS above
        #     features = datasets.Features(
        #         {
        #             "sentence": datasets.Value("string"),
        #             "option1": datasets.Value("string"),
        #             "answer": datasets.Value("string")
        #             # These are the features of your dataset like images, labels ...
        #         }
        #     )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DAISO_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(features),
            # Here we define them above because they are different between the two configurations
            # Homepage of the dataset for documentation
            homepage=self.config.url,
            # License for the dataset if available
            # license=_LICENSE,
            # Citation for the dataset
            citation=self.config.citation + "\n" + _DAISO_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_files = dl_manager.download(self.config.data_url)
        splits = []
        if "train" in data_files:
            splits.append(datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file": data_files["train"],
                    "split": "train",
                },
            ))
        if "dev" in data_files:
            splits.append(datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file": data_files["dev"],
                    "split": "dev",
                },
            ))
        if "test" in data_files:
            splits.append(datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file": data_files["test"],
                    "split": "test"
                },
            ))
        return splits

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, file, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        df = pd.read_csv(file, delimiter=",", header=0, quotechar='"', dtype=str)[
            self.config.features
        ]

        rows = df.to_dict(orient="records")

        for n, row in enumerate(rows):
            example = row
            example["Idx"] = n

            if "Dialogue_Act" in example:
                label = example["Dialogue_Act"]
                example["Label"] = label
                example["Label_ISO"] = self.config.label_classes.get(label, {}).get("ISO")

            yield example["Idx"], example
