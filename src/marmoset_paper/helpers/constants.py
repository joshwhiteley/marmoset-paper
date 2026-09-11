"""Deposited cluster assignments, figure colors, and treatment labels."""

LESS_SEVERE_CLUSTERS = [1, 13, 10, 6, 9, 7, 3, 8]
SEVERE_CLUSTERS = [2, 15, 11, 12, 4, 14, 0, 5]

LESS_SEVERE_COLOR_TUPLE = (0 / 255, 88 / 255, 139 / 255)
SEVERE_COLOR_TUPLE = (230 / 255, 44 / 255, 139 / 255)

PALETTE_16 = [
    (31 / 255, 119 / 255, 180 / 255),  # tab20[0]
    (174 / 255, 199 / 255, 232 / 255),  # tab20[1]
    (255 / 255, 127 / 255, 14 / 255),  # tab20[2]
    (255 / 255, 187 / 255, 120 / 255),  # tab20[3]
    (44 / 255, 160 / 255, 44 / 255),  # tab20[4]
    (152 / 255, 223 / 255, 138 / 255),  # tab20[5]
    (214 / 255, 39 / 255, 40 / 255),  # tab20[6]
    (255 / 255, 152 / 255, 150 / 255),  # tab20[7]
    (148 / 255, 103 / 255, 189 / 255),  # tab20[8]
    (197 / 255, 176 / 255, 213 / 255),  # tab20[9]
    (140 / 255, 86 / 255, 75 / 255),  # tab20[10]
    (196 / 255, 156 / 255, 148 / 255),  # tab20[11]
    (231 / 255, 26 / 255, 128 / 255),  # tab20[12]
    (247 / 255, 182 / 255, 210 / 255),  # tab20[13]
    (127 / 255, 127 / 255, 127 / 255),  # tab20[14]
    (199 / 255, 199 / 255, 199 / 255),  # tab20[15]
]

MARMOSET_ONLY_TUPLE = (
    0.011764705882352955,
    0.5764705882352941,
    0.9764705882352941,
)  # seaborn terrain color 0
MARMOSET_IN_VITRO_TUPLE = (
    0.1450980392156863,
    0.8290196078431373,
    0.42901960784313725,
)  # seaborn terrain color 1


COMPOUND_NAMES = {
    "BDQ+DEL": "BD",
    "BDQ+DEL+QBS": "DBQ",
    "BDQ+LIN": "BL",
    "BDQ+LIN+PRE": "BPaL",
    "BDQ+QBS": "BQ",
    "BDQ+PRE": "BPa",
    "DEL+QBS": "DQ",
    "EMB+INH+PZA+RIF": "HRZE",
    "EMB+MOX+PZA+RIF": "MRZE",
    "INH+PZA": "HZ",
    "LIN+PRE": "PaL",
    "MOX+RIF": "MR",
    "PZA+RIF": "RZ",
}

FXC50_COLS = {
    "equipotent": {
        "FxC50_butyrate_Terminal": "butyrate",
        "FxC50_cholesterol_Terminal": "cholesterol",
        "FxC50_dormancy_Terminal": "dormancy",
        "FxC50_5Nequip": "5N",
        "FxC50_7Nequip": "7N",
        "FxC50_7Hequip": "7H",
    },
    "cas": {
        "FxC50_butyrateCasPK_Terminal": "butyrate",
        "FxC50_cholesterolCasPK_Terminal": "cholesterol",
        "FxC50_dormancyCasPK_Terminal": "dormancy",
        "FxC50_5NcasPK": "5N",
        "FxC50_7NcasPK": "7N",
        "FxC50_7HcasPK": "7H",
    },
    "cell": {
        "FxC50_butyrateCellPK_Terminal": "butyrate",
        "FxC50_cholesterolCellPK_Terminal": "cholesterol",
        "FxC50_dormancyCellPK_Terminal": "dormancy",
        "FxC50_5NcellPK": "5N",
        "FxC50_7NcellPK": "7N",
        "FxC50_7HcellPK": "7H",
    },
}
