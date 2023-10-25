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

LABELS_MAPPING = {
    "ami": {
        "bck": {
            "base": "Backchannel",
            "ISO": "feedback"
        },
        "stl": {
            "base": "Stall",
            "ISO": None
        },
        "fra": {
            "base": "Fragment",
            "ISO": None
        },
        "inf": {
            "base": "Inform",
            "ISO": "inform"
        },
        "sug": {
            "base": "Suggest",
            "ISO": "directive"
        },
        "ass": {
            "base": "Assess",
            "ISO": "feedback"
        },
        "el.inf": {
            "base": "Elicit-Inform",
            "ISO": None
        },
        "el.sug": {
            "base": "Elicit-Offer-Or-Suggestion",
            "ISO": "directive"
        },
        "el.ass": {
            "base": "Elicit-Assessment",
            "ISO": None
        },
        "el.und": {
            "base": "Elicit-Comment-Understanding",
            "ISO": None
        },
        "off": {
            "base": "Offer",
            "ISO": "commissive"
        },
        "und": {
            "base": "Comment-About-Understanding",
            "ISO": "feedback"
        },
        "be.pos": {
            "base": "Be-Positive",
            "ISO": None
        },
        "be.neg": {
            "base": "Be-Negative",
            "ISO": None
        },
        "oth": {
            "base": "Other",
            "ISO": None
        }
    },
    "oasis": {
        "inform": {
            "base": "Inform",
            "ISO": "inform"
        },
        "ackn": {
            "base": "Acknowledge",
            "ISO": "feedback"
        },
        "reqInfo": {
            "base": "Request Inform",
            "ISO": "directive"
        },
        "backch": {
            "base": "Backchannel",
            "ISO": "feedback"
        },
        "answ": {
            "base": "Answer",
            "ISO": "answer"
        },
        "init": {
            "base": "Initialise",
            "ISO": "discourse"
        },
        "thank": {
            "base": "Thank",
            "ISO": "thanking"
        },
        "greet": {
            "base": "Greet",
            "ISO": "greeting"
        },
        "accept": {
            "base": "Accept",
            "ISO": "agreement"
        },
        "answElab": {
            "base": "Answer Elaborate",
            "ISO": "inform"
        },
        "informIntent": {
            "base": "Inform Intention",
            "ISO": "commissive"
        },
        "bye": {
            "base": "Bye",
            "ISO": "goodbye"
        },
        "direct": {
            "base": "Direct",
            "ISO": "directive"
        },
        "confirm": {
            "base": "Confirm",
            "ISO": "answer"
        },
        "expressRegret": {
            "base": "Express Regret",
            "ISO": "apology"
        },
        "hold": {
            "base": "Hold",
            "ISO": "turn"
        },
        "expressOpinion": {
            "base": "Express Opinion",
            "ISO": "inform"
        },
        "offer": {
            "base": "Offer",
            "ISO": "commissive"
        },
        "echo": {
            "base": "Echo",
            "ISO": "feedback"
        },
        "appreciate": {
            "base": "Appreciate",
            "ISO": "feedback"
        },
        "refer": {
            "base": "Refer",
            "ISO": None
        },
        "suggest": {
            "base": "Suggest",
            "ISO": "directive"
        },
        "reqDirect": {
            "base": "Request Direct",
            "ISO": "directive"
        },
        "negate": {
            "base": "Negate",
            "ISO": "disagreement"
        },
        "exclaim": {
            "base": "Exclaim",
            "ISO": None
        },
        "pardon": {
            "base": "Pardon",
            "ISO": "apology"
        },
        "identifySelf": {
            "base": "Identify Self",
            "ISO": None
        },
        "expressPossibility": {
            "base": "Express Possibility",
            "ISO": "inform"
        },
        "raiseIssue": {
            "base": "Raise Issue",
            "ISO": None
        },
        "expressWish": {
            "base": "Express Wish",
            "ISO": "inform"
        },
        "reqModal": {
            "base": "Request Modal",
            "ISO": "directive"
        },
        "complete": {
            "base": "Complete",
            "ISO": None
        },
        "directElab": {
            "base": "Direct Elaborate",
            "ISO": "directive"
        },
        "correct": {
            "base": "Correct",
            "ISO": None
        },
        "refuse": {
            "base": "Refuse",
            "ISO": None
        },
        "informIntent-hold": {
            "base": "Inform Intent Hold",
            "ISO": None
        },
        "informDisc": {
            "base": "Inform Continue",
            "ISO": None
        },
        "informCont": {
            "base": "Inform Discontinue",
            "ISO": None
        },
        "selfTalk": {
            "base": "Self Talk",
            "ISO": None
        },
        "correctSelf": {
            "base": "Correct Self",
            "ISO": "disagreement"
        },
        "expressRegret-inform": {
            "base": "Express Regret Inform",
            "ISO": None
        },
        "thank-identifySelf": {
            "base": "Thank Identify Self",
            "ISO": None
        }
    },
    "maptask": {
        "acknowledge": {
            "base": "Acknowledge",
            "ISO": "feedback"
        },
        "instruct": {
            "base": "Instruct",
            "ISO": "directive"
        },
        "reply_y": {
            "base": "Yes-Reply",
            "ISO": "answer"
        },
        "explain": {
            "base": "Explain",
            "ISO": "inform"
        },
        "check": {
            "base": "Check",
            "ISO": "feedback"
        },
        "ready": {
            "base": "Ready",
            "ISO": "discourse"
        },
        "align": {
            "base": "Check Attention",
            "ISO": None
        },
        "query_yn": {
            "base": "Yes-No-Question",
            "ISO": "propq"
        },
        "clarify": {
            "base": "Clarify",
            "ISO": "inform"
        },
        "reply_w": {
            "base": "Non Yes-No-Reply",
            "ISO": "answer"
        },
        "reply_n": {
            "base": "No-Reply",
            "ISO": "answer"
        },
        "query_w": {
            "base": "Non Yes-No-Question",
            "ISO": "setq"
        }
    },
    "mrda": {
        "s": {
            "base": "Statement",
            "ISO": "inform"
        },
        "b": {
            "base": "Continuer (backchannel)",
            "ISO": "feedback"
        },
        "fh": {
            "base": "Floor Holder",
            "ISO": "turn"
        },
        "bk": {
            "base": "Acknowledge-answer",
            "ISO": "feedback"
        },
        "aa": {
            "base": "Accept",
            "ISO": "agreement"
        },
        "df": {
            "base": "Defending/Explanation",
            "ISO": "inform"
        },
        "e": {
            "base": "Expansions of y/n Answers",
            "ISO": "answer"
        },
        "%": {
            "base": "Interrupted/Abandoned/Uninterpretable",
            "ISO": None
        },
        "rt": {
            "base": "Rising Tone",
            "ISO": None
        },
        "fg": {
            "base": "Floor Grabber",
            "ISO": "turn"
        },
        "cs": {
            "base": "Offer",
            "ISO": "commissive"
        },
        "ba": {
            "base": "Assessment/Appreciation",
            "ISO": "feedback"
        },
        "bu": {
            "base": "Understanding Check",
            "ISO": "feedback"
        },
        "d": {
            "base": "Declarative-Question",
            "ISO": "propq"
        },
        "na": {
            "base": "Affirmative Non-yes Answers",
            "ISO": "answer"
        },
        "qw": {
            "base": "Wh-Question",
            "ISO": "setq"
        },
        "ar": {
            "base": "Reject",
            "ISO": "disagreement"
        },
        "2": {
            "base": "Collaborative Completion",
            "ISO": None
        },
        "no": {
            "base": "Other Answers",
            "ISO": "answer"
        },
        "h": {
            "base": "Hold Before Answer/Agreement",
            "ISO": "turn"
        },
        "co": {
            "base": "Action-directive",
            "ISO": "directive"
        },
        "qy": {
            "base": "Yes-No-question",
            "ISO": "propq"
        },
        "nd": {
            "base": "Dispreferred Answers",
            "ISO": "answer"
        },
        "j": {
            "base": "Humorous Material",
            "ISO": None
        },
        "bd": {
            "base": "Downplayer",
            "ISO": "apology"
        },
        "cc": {
            "base": "Commit",
            "ISO": "commissive"
        },
        "ng": {
            "base": "Negative Non-no Answers",
            "ISO": "answer"
        },
        "am": {
            "base": "Maybe",
            "ISO": None
        },
        "qrr": {
            "base": "Or-Clause",
            "ISO": "choiceq"
        },
        "fe": {
            "base": "Exclamation",
            "ISO": "feedback"
        },
        "m": {
            "base": "Mimic Other",
            "ISO": None
        },
        "fa": {
            "base": "Apology",
            "ISO": "apology"
        },
        "t": {
            "base": "About-task",
            "ISO": None
        },
        "br": {
            "base": "Signal-non-understanding",
            "ISO": "feedback"
        },
        "aap": {
            "base": "Accept-part",
            "ISO": None
        },
        "qh": {
            "base": "Rhetorical-Question",
            "ISO": "inform"
        },
        "tc": {
            "base": "Topic Change",
            "ISO": "discourse"
        },
        "r": {
            "base": "Repeat",
            "ISO": "inform"
        },
        "t1": {
            "base": "Self-talk",
            "ISO": None
        },
        "t3": {
            "base": "3rd-party-talk",
            "ISO": None
        },
        "bh": {
            "base": "Rhetorical-question Continue",
            "ISO": "propq"
        },
        "bsc": {
            "base": "Reject-part",
            "ISO": None
        },
        "arp": {
            "base": "Misspeak Self-Correction",
            "ISO": None
        },
        "bs": {
            "base": "Reformulate/Summarize",
            "ISO": "feedback"
        },
        "f": {
            "base": "Follow Me",
            "ISO": None
        },
        "qr": {
            "base": "Or-Question",
            "ISO": "choiceq"
        },
        "ft": {
            "base": "Thanking",
            "ISO": "thanking"
        },
        "g": {
            "base": "Tag-Question",
            "ISO": "propq"
        },
        "qo": {
            "base": "Open-Question",
            "ISO": None
        },
        "bc": {
            "base": "Correct-misspeaking",
            "ISO": None
        },
        "by": {
            "base": "Sympathy",
            "ISO": "apology"
        },
        "fw": {
            "base": "Welcome",
            "ISO": "thanking"
        }
    },
    "swda": {
        "sd": {
            "base": "Statement-non-opinion",
            "ISO": "inform"
        },
        "b": {
            "base": "Acknowledge (Backchannel)",
            "ISO": "feedback"
        },
        "sv": {
            "base": "Statement-opinion",
            "ISO": "inform"
        },
        "%": {
            "base": "Uninterpretable",
            "ISO": None
        },
        "aa": {
            "base": "Agree/Accept",
            "ISO": "agreement"
        },
        "ba": {
            "base": "Appreciation",
            "ISO": "feedback"
        },
        "qy": {
            "base": "Yes-No-Question",
            "ISO": "propq"
        },
        "ny": {
            "base": "Yes Answers",
            "ISO": "answer"
        },
        "fc": {
            "base": "Conventional-closing",
            "ISO": "discourse"
        },
        "qw": {
            "base": "Wh-Question",
            "ISO": "setq"
        },
        "nn": {
            "base": "No Answers",
            "ISO": "answer"
        },
        "bk": {
            "base": "Response Acknowledgement",
            "ISO": "feedback"
        },
        "h": {
            "base": "Hedge",
            "ISO": "answer"
        },
        "qy^d": {
            "base": "Declarative Yes-No-Question",
            "ISO": "propq"
        },
        "bh": {
            "base": "Backchannel in Question Form",
            "ISO": "propq"
        },
        "^q": {
            "base": "Quotation",
            "ISO": None
        },
        "bf": {
            "base": "Summarize/Reformulate",
            "ISO": "feedback"
        },
        "fo": {
            "base": "Other forward-looking functions",
            "ISO": "commissive"
        },
        "by": {
            "base": "Sympathy",
            "ISO": "apology"
        },
        "fw": {
            "base": "Welcome",
            "ISO": "thanking"
        },
        "o_\"_bc": {
            "base": "Other",
            "ISO": None
        },
        "na": {
            "base": "Affirmative Non-yes Answers",
            "ISO": "answer"
        },
        "ad": {
            "base": "Action-directive",
            "ISO": "directive"
        },
        "^2": {
            "base": "Collaborative Completion",
            "ISO": None
        },
        "b^m": {
            "base": "Repeat-phrase",
            "ISO": "feedback"
        },
        "qo": {
            "base": "Open-Question",
            "ISO": None
        },
        "qh": {
            "base": "Rhetorical-Question",
            "ISO": "inform"
        },
        "^h": {
            "base": "Hold Before Answer/Agreement",
            "ISO": "turn"
        },
        "ar": {
            "base": "Reject",
            "ISO": "disagreement"
        },
        "ng": {
            "base": "Negative Non-no Answers",
            "ISO": "answer"
        },
        "br": {
            "base": "Signal-non-understanding",
            "ISO": "feedback"
        },
        "no": {
            "base": "Other Answers",
            "ISO": "answer"
        },
        "fp": {
            "base": "Conventional-opening",
            "ISO": "discourse"
        },
        "qrr": {
            "base": "Or-Clause",
            "ISO": "choiceq"
        },
        "arp_nd": {
            "base": "Dispreferred Answers",
            "ISO": "answer"
        },
        "t3": {
            "base": "3rd-party-talk",
            "ISO": None
        },
        "oo": {
            "base": "Offers",
            "ISO": "directive"
        },
        "co_cc": {
            "base": "Options Commits",
            "ISO": "commissive"
        },
        "aap_am": {
            "base": "Maybe/Accept-part",
            "ISO": None
        },
        "t1": {
            "base": "Downplayer",
            "ISO": "apology"
        },
        "bd": {
            "base": "Self-talk",
            "ISO": None
        },
        "^g": {
            "base": "Tag-Question",
            "ISO": "propq"
        },
        "qw^d": {
            "base": "Declarative Wh-Question",
            "ISO": "setq"
        },
        "fa": {
            "base": "Apology",
            "ISO": "apology"
        },
        "ft": {
            "base": "Thanking",
            "ISO": "thanking"
        }
    },
    "frames": {
        "inform": {
            "base": "Inform",
            "ISO": "inform"
        },
        "sorry": {
            "base": "Sorry",
            "ISO": "apology"
        },
        "suggest": {
            "base": "Suggest",
            "ISO": "directive"
        },
        "negate": {
            "base": "Negate",
            "ISO": "disagreement"
        },
        "thankyou": {
            "base": "Thank you",
            "ISO": "thanking"
        },
        "greeting": {
            "base": "Greeting",
            "ISO": "greeting"
        },
        "request": {
            "base": "Request",
            "ISO": "directive"
        },
        "switch_frame": {
            "base": "Switch Frame",
            "ISO": None
        },
        "offer": {
            "base": "Offer",
            "ISO": "commissive"
        },
        "request_alts": {
            "base": "Request Alternative",
            "ISO": "directive"
        },
        "null": {
            "base": "Other",
            "ISO": None
        },
        "goodbye": {
            "base": "Goodbye",
            "ISO": "goodbye"
        },
        "moreinfo": {
            "base": "Request More information",
            "ISO": "directive"
        },
        "no_result": {
            "base": "No Result",
            "ISO": None
        },
        "affirm": {
            "base": "Affirm",
            "ISO": "answer"
        },
        "request_compare": {
            "base": "Request Compare",
            "ISO": "directive"
        },
        "confirm": {
            "base": "Confirm",
            "ISO": "answer"
        },
        "hearmore": {
            "base": "Hear More",
            "ISO": None
        },
        "canthelp": {
            "base": "Can not help",
            "ISO": None
        },
        "you_are_welcome": {
            "base": "Welcome",
            "ISO": "thanking"
        },
        "reject": {
            "base": "Reject",
            "ISO": "disagreement"
        }
    },
    "dyda": {
        "commissive": {
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
    "dstc3": {
        "welcomemsg": {
            "base": "Welcome",
            "ISO": "thanking"
        },
        "inform": {
            "base": "Inform",
            "ISO": "inform"
        },
        "select": {
            "base": "Select",
            "ISO": None
        },
        "expl-conf": {
            "base": "Explicit Confirmation",
            "ISO": "answer"
        },
        "affirm": {
            "base": "Affirmation",
            "ISO": "answer"
        },
        "canthelp": {
            "base": "Can not help",
            "ISO": None
        },
        "request": {
            "base": "Request",
            "ISO": "directive"
        },
        "bye": {
            "base": "Goodbye",
            "ISO": "goodbye"
        },
        "offer": {
            "base": "Offer",
            "ISO": "commissive"
        },
        "thankyou": {
            "base": "Thank you",
            "ISO": "thanking"
        },
        "negate": {
            "base": "Negate",
            "ISO": "disagreement"
        },
        "null": {
            "base": "Other",
            "ISO": None
        },
        "reqalts": {
            "base": "Request Alternative",
            "ISO": "directive"
        },
        "canthelp.missing_slot_value": {
            "base": "Can not help",
            "ISO": None
        },
        "restart": {
            "base": "Restart",
            "ISO": None
        },
        "ack": {
            "base": "Acknowledge",
            "ISO": "feedback"
        },
        "reqmore": {
            "base": "Request More",
            "ISO": "directive"
        },
        "confirm": {
            "base": "Confirm",
            "ISO": "answer"
        },
        "hello": {
            "base": "Hello",
            "ISO": "greeting"
        },
        "repeat": {
            "base": "Repeat",
            "ISO": "inform"
        },
        "deny": {
            "base": "Deny",
            "ISO": "answer"
        }
    },
    "dstc8-sgd": {
        "INFORM": {
            "base": "Inform",
            "ISO": "inform"
        },
        "REQUEST": {
            "base": "Request",
            "ISO": "directive"
        },
        "CONFIRM": {
            "base": "Confirm",
            "ISO": "answer"
        },
        "AFFIRM": {
            "base": "Affirmation",
            "ISO": "answer"
        },
        "NOTIFY_FAILURE": {
            "base": "Notify Failure",
            "ISO": "inform"
        },
        "THANK_YOU": {
            "base": "Thank you",
            "ISO": "thanking"
        },
        "REQ_MORE": {
            "base": "Request More",
            "ISO": "directive"
        },
        "NEGATE": {
            "base": "Negate",
            "ISO": "disagreement"
        },
        "GOODBYE": {
            "base": "Goodbye",
            "ISO": "goodbye"
        },
        "NOTIFY_SUCCESS": {
            "base": "Notify Success",
            "ISO": "inform"
        },
        "INFORM_INTENT": {
            "base": "Inform Intention",
            "ISO": "commissive"
        },
        "OFFER": {
            "base": "Offer",
            "ISO": "commissive"
        },
        "SELECT": {
            "base": "Select",
            "ISO": None
        },
        "OFFER_INTENT": {
            "base": "Offer Intent",
            "ISO": "commissive"
        },
        "NEGATE_INTENT": {
            "base": "Negate Intent",
            "ISO": "disagreement"
        },
        "REQUEST_ALTS": {
            "base": "Request Alternatives",
            "ISO": "directive"
        },
        "AFFIRM_INTENT": {
            "base": "Affirm Intent",
            "ISO": "answer"
        }
    }
}


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
            name="ami",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["ami"],
            features=[
                "Utterance",
                "Dialogue_Act",
                "Speaker",
                "Dialogue_Id",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/ami/train.csv",
                "test": _URL + "/ami/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @article{carletta2006ami,
                author = "Carletta, J.",
                title = "Announcing the AMI Meeting Corpus",
                journal = "The ELRA Newsletter",
                volume = "11",
                number = "1",
                year = "2006",
                pages = "3-5",
                month = "January-March"
                }"""
            ),
            url="https://groups.inf.ed.ac.uk/ami/corpus/",
        ),
        DAISOConfig(
            name="oasis",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["oasis"],
            features=[
                "Speaker",
                "Utterance",
                "Dialogue_Act",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/oasis/train.csv",
                "dev": _URL + "/oasis/dev.csv",
                "test": _URL + "/oasis/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @inproceedings{leech2003generic,
                title={Generic speech act annotation for task-oriented dialogues},
                author={Leech, Geoffrey and Weisser, Martin},
                booktitle={Proceedings of the corpus linguistics 2003 conference},
                volume={16},
                pages={441--446},
                year={2003},
                organization={Lancaster: Lancaster University}
                }"""
            ),
            url="http://groups.inf.ed.ac.uk/oasis/",
        ),
        DAISOConfig(
            name="maptask",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["maptask"],
            features=[
                "Speaker",
                "Utterance",
                "Dialogue_Act",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/maptask/train.csv",
                "dev": _URL + "/maptask/dev.csv",
                "test": _URL + "/maptask/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @inproceedings{thompson1993hcrc,
                title={The HCRC map task corpus: natural dialogue for speech recognition},
                author={Thompson, Henry S and Anderson, Anne H and Bard, Ellen Gurman and Doherty-Sneddon,
                Gwyneth and Newlands, Alison and Sotillo, Cathy},
                booktitle={HUMAN LANGUAGE TECHNOLOGY: Proceedings of a Workshop Held at Plainsboro, New Jersey, March 21-24, 1993},
                year={1993}
                }"""
            ),
            url="http://groups.inf.ed.ac.uk/maptask/",
        ),
        DAISOConfig(
            name="mrda",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["mrda"],
            features=[
                "Speaker",
                "Utterance",
                "Basic_DA",
                "General_DA",
                "Dialogue_Act",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/mrda/train.csv",
                "dev": _URL + "/mrda/dev.csv",
                "test": _URL + "/mrda/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @techreport{shriberg2004icsi,
                title={The ICSI meeting recorder dialog act (MRDA) corpus},
                author={Shriberg, Elizabeth and Dhillon, Raj and Bhagat, Sonali and Ang, Jeremy and Carvey, Hannah},
                year={2004},
                institution={INTERNATIONAL COMPUTER SCIENCE INST BERKELEY CA}
                }"""
            ),
            url="https://www.aclweb.org/anthology/W04-2319",
        ),
        DAISOConfig(
            name="swda",
            description=textwrap.dedent(
                """\
                Switchboard Dialogue Act Corpus. 
                Grouping procedure is different from original recommendations.
                Contains detailed split for specific labels for ISO mapping.
                """
            ),
            label_classes=LABELS_MAPPING["swda"],
            features=[
                "Speaker",
                "Utterance",
                "Dialogue_Act",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/swda/train.csv",
                "dev": _URL + "/swda/dev.csv",
                "test": _URL + "/swda/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @article{stolcke2000dialogue,
                title={Dialogue act modeling for automatic tagging and recognition of conversational speech},
                 author={Stolcke, Andreas and Ries, Klaus and Coccaro, Noah and Shriberg, Elizabeth and
                 Bates, Rebecca and Jurafsky, Daniel and Taylor, Paul and Martin, Rachel and Ess-Dykema,
                 Carol Van and Meteer, Marie},
                 journal={Computational linguistics},
                volume={26},
                number={3},
                pages={339--373},
                year={2000},
                publisher={MIT Press}
                }"""
            ),
            url="https://web.stanford.edu/~jurafsky/ws97/",
        ),
        DAISOConfig(
            name="frames",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["frames"],
            features=[
                "Speaker",
                "Utterance",
                "Dialogue_Act",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/frames/train.csv",
                "test": _URL + "/frames/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @inproceedings{el-asri-etal-2017-frames,
                title = "{F}rames: a corpus for adding memory to goal-oriented dialogue systems",
                author = "El Asri, Layla  and
                  Schulz, Hannes  and
                  Sharma, Shikhar  and
                  Zumer, Jeremie  and
                  Harris, Justin  and
                  Fine, Emery  and
                  Mehrotra, Rahul  and
                  Suleman, Kaheer",
                booktitle = "Proceedings of the 18th Annual {SIG}dial Meeting on Discourse and Dialogue",
                month = aug,
                year = "2017",
                address = {Saarbr{\"u}cken, Germany},
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/W17-5526",
                doi = "10.18653/v1/W17-5526",
                pages = "207--219",
                abstract = "This paper proposes a new dataset, Frames, composed of 1369 human-human dialogues with an average of 15 turns per dialogue. This corpus contains goal-oriented dialogues between users who are given some constraints to book a trip and assistants who search a database to find appropriate trips. The users exhibit complex decision-making behaviour which involve comparing trips, exploring different options, and selecting among the trips that were discussed during the dialogue. To drive research on dialogue systems towards handling such behaviour, we have annotated and released the dataset and we propose in this paper a task called frame tracking. This task consists of keeping track of different semantic frames throughout each dialogue. We propose a rule-based baseline and analyse the frame tracking task through this baseline.",
                }"""
            ),
            url="http://datasets.maluuba.com/Frames",
        ),
        DAISOConfig(
            name="dyda",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["dyda"],
            # {"commissive": {
            #     "base": "Commissive",
            #     "ISO": "commissive"
            # },
            #     "directive": {
            #         "base": "Directive",
            #         "ISO": "directive"
            #     },
            #     "inform": {
            #         "base": "Inform",
            #         "ISO": "inform"
            #     },
            #     "question": {
            #         "base": "Question",
            #         "ISO": None
            #     }
            # },
            features=[
                "Utterance",
                "Dialogue_Act",
                "Emotion",
                "Dialogue_Id",
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
            name="dstc3",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["dstc3"],
            features=[
                "Speaker",
                "Utterance",
                "Dialog_Act",
                "Dialog_Act_ISO"
            ],
            data_url={
                "train": _URL + "/dstc3/train.csv",
                "test": _URL + "/dstc3/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @article{Henderson2014TheTD,
                  title={The third Dialog State Tracking Challenge},
                  author={Matthew Henderson and Blaise Thomson and J. Williams},
                  journal={2014 IEEE Spoken Language Technology Workshop (SLT)},
                  year={2014},
                  pages={324-329},
                  url={https://api.semanticscholar.org/CorpusID:17478615}
                }"""
            ),
            url="http://camdial.org/~mh521/dstc/",
        ),
        DAISOConfig(
            name="dstc8-sgd",
            description=textwrap.dedent(
                """\
                """
            ),
            label_classes=LABELS_MAPPING["dstc8-sgd"],
            features=[
                "Speaker",
                "Utterance",
                "Dialogue_Act",
                "Dialogue_Id",
                "Dialogue_Act_ISO"
            ],
            data_url={
                "train": _URL + "/dstc8-sgd/train.csv",
                "dev": _URL + "/dstc8-sgd/dev.csv",
                "test": _URL + "/dstc8-sgd/test.csv",
            },
            citation=textwrap.dedent(
                """\
                @inproceedings{rastogi2020towards,
                  title={Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset},
                  author={Rastogi, Abhinav and Zang, Xiaoxue and Sunkara, Srinivas and Gupta, Raghav and Khaitan, Pranav},
                  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
                  volume={34},
                  number={05},
                  pages={8689--8696},
                  year={2020}
                }"""
            ),
            url="https://github.com/google-research-datasets/dstc8-schema-guided-dialogue",
        ),

    ]

    DEFAULT_CONFIG_NAME = "dyda"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = {feature: datasets.Value("string") for feature in self.config.features}
        if self.config.label_classes:
            features["Label"] = datasets.features.ClassLabel(names=list(self.config.label_classes.keys()))
            features["Label_ISO"] = datasets.features.ClassLabel(
                names=list(set([map.get("ISO") for map in self.config.label_classes.values()])))
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
