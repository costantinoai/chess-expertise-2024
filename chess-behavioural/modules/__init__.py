# -*- coding: utf-8 -*-
import logging
import seaborn as sns

sns.set(style="darkgrid", palette="muted", font_scale=1.7)

# Setting up the logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('seaborn').setLevel(logging.WARNING)
logging.getLogger('pandas').setLevel(logging.WARNING)

DATASET_CSV_PATH = 'data/dataset.csv'
PALETTE = {'Super-experts': 'darkorange', 'Experts': '#FFD580', 'Non-experts': 'steelblue'}
DNN_RESPONSES_CSV_PATH = '/home/eik-tb/OneDrive_andreaivan.costantino@kuleuven.be/GitHub/chess_expertise/bh_results/misc/correct_responses_human_and_net.csv'

# Subject Lists
EXPERT_SUBJECTS = (
    "03",
    "04",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "16",
    "20",
    "22",
    "23",
    "24",
    "29",
    "30",
    "33",
    "34",
    "36",
)

NONEXPERT_SUBJECTS = (
    "01",
    "02",
    "15",
    "17",
    "18",
    "19",
    "21",
    "25",
    "26",
    "27",
    "28",
    "32",
    "35",
    "37",
    "39",
    "40",
    "41",
    "42",
    "43",
    "44",
)
