import seaborn as sns

# figure 1 classifications
LESS_SEVERE_CLUSTERS = [1, 13, 10, 6, 9, 7, 3, 8]
SEVERE_CLUSTERS = [2, 15, 11, 12, 4, 14, 0, 5]

LESS_SEVERE_COLOR_TUPLE = (  0/255,  88/255, 139/255)
SEVERE_COLOR_TUPLE      = (230/255, 44/255,  139/255)

PALETTE_16 = [
    (31/255, 119/255, 180/255),  # tab20[0]
    (174/255,199/255,232/255),  # tab20[1]
    (255/255,127/255, 14/255),  # tab20[2]
    (255/255,187/255,120/255),  # tab20[3]
    (44/255, 160/255, 44/255),  # tab20[4]
    (152/255,223/255,138/255),  # tab20[5]
    (214/255, 39/255, 40/255),  # tab20[6]
    (255/255,152/255,150/255),  # tab20[7]
    (148/255,103/255,189/255),  # tab20[8]
    (197/255,176/255,213/255),  # tab20[9]
    (140/255, 86/255, 75/255),  # tab20[10]
    (196/255,156/255,148/255),  # tab20[11]
    (231/255, 26/255,128/255),  # tab20[12]
    (247/255,182/255,210/255),  # tab20[13]
    (127/255,127/255,127/255),  # tab20[14]
    (199/255,199/255,199/255),  # tab20[15]
]

# figure 1 clustergram
# (worsening) purple -> green (improvement)
FIGURE_1C_CLUSTERGRAM_PALETTE = sns.diverging_palette(145, 300, s=60, as_cmap=True)

LIDS_DELIMITER = ['AcidicN', 'NeutralN', 'AcidicH', 'NeutralH']
META_DELIMITER = ['Drug', 'Compound', 'Compound', 'MarmID', 'Lesion', 'Consolidate', 'Contour']

FEATURE_CATEGORY_COLORS = {
    "marmoset": "tab:red",
    "simple PK": "tab:orange",
    "LIDS": "tab:green",
    "simple equipotent": "tab:blue",
    "metadata": "tab:grey",
    "unknown": "tab:black"
}

MARMOSET_ONLY_TUPLE = (0.011764705882352955, 0.5764705882352941, 0.9764705882352941) #seaborn terrain color 0
MARMOSET_IN_VITRO_TUPLE = (0.1450980392156863, 0.8290196078431373, 0.42901960784313725) #seaborn terrain color 1
DIAMOND_ONLY_TUPLE = (0.9764705882352941, 0.011764705882352955, 0.5764705882352941) #seaborn terrain color 15

CLUSTER_GROUPS = {
    # less severe lesions, nearest neighbor via dendrogram
    "1a": [1, 13],
    "1b": [10],
    "1c": [6],
    "1d": [7],
    "1e": [3, 8],
    # severe lesions, nearest neighbor via dendrogram
    "2a": [2],
    "2b": [15],
    "2c": [11, 12],
    "2d": [4, 14],
    "2e": [0, 5]
}

COMPOUND_NAMES = {
    "BDQ+DEL": "BD",
    "BDQ+DEL+QBS": "DBO",
    "BDQ+LIN": "BL",
    "BDQ+LIN+PRE": "BPaL",
    "BDQ+QBS": "BO", # no data
    "BDQ+PRE": "BP",
    "DEL+QBS": "DO",
    "EMB+INH+PZA+RIF": "HRZE",
    "EMB+MOX+PZA+RIF": "MRZE",
    "INH+PZA": "HZ",
    "LIN+PRE": "PaL",
    "MOX+RIF": "MR",
    "PZA+RIF": "RZ"
}

FXC50_COLS = {
  'equipotent': {
    'FxC50_butyrate_Terminal': 'butyrate',
    'FxC50_cholesterol_Terminal': 'cholesterol',
    'FxC50_dormancy_Terminal': 'dormancy',
    'FxC50_5Nequip': '5N',
    'FxC50_7Nequip': '7N',
    'FxC50_7Hequip': '7H',
  },
  'cas': {
    'FxC50_butyrateCasPK_Terminal': 'butyrate',
    'FxC50_cholesterolCasPK_Terminal': 'cholesterol',
    'FxC50_dormancyCasPK_Terminal': 'dormancy',
    'FxC50_5NcasPK': '5N',
    'FxC50_7NcasPK': '7N',
    'FxC50_7HcasPK': '7H',
  },
  'cell': {
    'FxC50_butyrateCellPK_Terminal': 'butyrate',
    'FxC50_cholesterolCellPK_Terminal': 'cholesterol',
    'FxC50_dormancyCellPK_Terminal': 'dormancy',
    'FxC50_5NcellPK': '5N',
    'FxC50_7NcellPK': '7N',
    'FxC50_7HcellPK': '7H',
  }
}