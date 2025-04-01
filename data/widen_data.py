import polars as pl

marm_data = pl.read_excel("data/FINALCORRECTED_Merged_CFUandPETCT_20231005.xlsx")

lesion_measurements = ["MeanHU",
                       "MedianHU",
                       "StandDevHU",
                       "MeanSUV",
                       "MedianSUV",
                       "StandDevSUV",
                       "MaxSUV",
                       "HighSUV",
                       "VOLMORE1p5_SUV",
                       "VOLMORE2_SUV",
                       "TLG_SUV",
                       "TLG_HardPlusSoft_SUV",
                       "TLG_HardPlusSoft_SublesionDivMuscleTLG",
                       "SoftVol",
                       "HardVol",
                       "log10CFU",
                       "LesionType",
                       "FinalLesionType",
                       "Consolidate",
                       "Contour"]

# ----------------------------------------------------------------
# FILTERING OUT RESISTANT MARMOSETS / LESIONS
# REMOVING COMPOUNDS WE DO NOT HAVE IN VITRO DATA FOR
# FILTERING OUT REDUNDANT INFORMATION
# ----------------------------------------------------------------

# Resistant marmosets
mask_bk21 = ~marm_data["MarmID"].str.contains("BK21") 
mask_bn20 = ~marm_data["MarmID"].str.contains("BN20")

# Resistant lesions
mask_br01_les05 = ~((marm_data["MarmID"] == "BR01") & (marm_data["Lesion"] == 5))
mask_bj28_les06 = ~((marm_data["MarmID"] == "BJ28") & (marm_data["Lesion"] == 6))
mask_bn22_les09 = ~((marm_data["MarmID"] == "BN22") & (marm_data["Lesion"] == 9))
mask_b011_les11 = ~((marm_data["MarmID"] == "B011") & (marm_data["Lesion"] == 11))
mask_br07_les18 = ~((marm_data["MarmID"] == "BR07") & (marm_data["Lesion"] == 18))
mask_bq47_les04 = ~((marm_data["MarmID"] == "BQ47") & (marm_data["Lesion"] == 4))
mask_bm07_les21 = ~((marm_data["MarmID"] == "BM07") & (marm_data["Lesion"] == 21))
mask_bm07_les22 = ~((marm_data["MarmID"] == "BM07") & (marm_data["Lesion"] == 22))
mask_bm07_les24 = ~((marm_data["MarmID"] == "BM07") & (marm_data["Lesion"] == 24))
mask_bp09_les13 = ~((marm_data["MarmID"] == "BP09") & (marm_data["Lesion"] == 13))
mask_bq42_les20 = ~((marm_data["MarmID"] == "BQ42") & (marm_data["Lesion"] == 20))
mask_bq42_les22 = ~((marm_data["MarmID"] == "BQ42") & (marm_data["Lesion"] == 22))
mask_bq42_les09 = ~((marm_data["MarmID"] == "BQ42") & (marm_data["Lesion"] == 9))
mask_bq42_les13 = ~((marm_data["MarmID"] == "BQ42") & (marm_data["Lesion"] == 13))
mask_bq42_les23 = ~((marm_data["MarmID"] == "BQ42") & (marm_data["Lesion"] == 23))
mask_m289_les08 = ~((marm_data["MarmID"] == "M289") & (marm_data["Lesion"] == 8))
mask_bn18_les09 = ~((marm_data["MarmID"] == "BN18") & (marm_data["Lesion"] == 9))
mask_bn36_les12 = ~((marm_data["MarmID"] == "BN36") & (marm_data["Lesion"] == 12))
mask_bo09_les10 = ~((marm_data["MarmID"] == "BO09") & (marm_data["Lesion"] == 10))
mask_bo09_les12 = ~((marm_data["MarmID"] == "BO09") & (marm_data["Lesion"] == 12))
mask_bn13_les07 = ~((marm_data["MarmID"] == "BN13") & (marm_data["Lesion"] == 7))
mask_bn27_les03 = ~((marm_data["MarmID"] == "BN27") & (marm_data["Lesion"] == 3))
mask_m291_les09 = ~((marm_data["MarmID"] == "M291") & (marm_data["Lesion"] == 9))
mask_bp02_les21 = ~((marm_data["MarmID"] == "BP02") & (marm_data["Lesion"] == 21))
mask_750nb_les06 = ~((marm_data["MarmID"] == "750NB") & (marm_data["Lesion"] == 6))
mask_bm16_les16 = ~((marm_data["MarmID"] == "BM16") & (marm_data["Lesion"] == 16))
mask_bm16_les20 = ~((marm_data["MarmID"] == "BM16") & (marm_data["Lesion"] == 20))
mask_bm40_les18 = ~((marm_data["MarmID"] == "BM40") & (marm_data["Lesion"] == 18))
mask_bm40_les21 = ~((marm_data["MarmID"] == "BM40") & (marm_data["Lesion"] == 21))
mask_bi01_les14 = ~((marm_data["MarmID"] == "BI01") & (marm_data["Lesion"] == 14))
mask_bp14_les11 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 11))
mask_bp14_les14 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 14))
mask_bp14_les27 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 27))
mask_bp14_les28 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 28))
mask_bp14_les29 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 29))
mask_bp14_les30 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 30))
mask_bp14_les44 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 44))
mask_bp14_les47 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 47))
mask_bp14_les50 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 50))
mask_bp14_les51 = ~((marm_data["MarmID"] == "BP14") & (marm_data["Lesion"] == 51))
mask_bp17_les04 = ~((marm_data["MarmID"] == "BP17") & (marm_data["Lesion"] == 4))
mask_bp17_les08 = ~((marm_data["MarmID"] == "BP17") & (marm_data["Lesion"] == 8))
mask_bp17_les18 = ~((marm_data["MarmID"] == "BP17") & (marm_data["Lesion"] == 18))
mask_bp17_les26 = ~((marm_data["MarmID"] == "BP17") & (marm_data["Lesion"] == 26))
mask_bp17_les29 = ~((marm_data["MarmID"] == "BP17") & (marm_data["Lesion"] == 29))
mask_bi07_les04 = ~((marm_data["MarmID"] == "BI07") & (marm_data["Lesion"] == 4))
mask_bi07_les12 = ~((marm_data["MarmID"] == "BI07") & (marm_data["Lesion"] == 12))
mask_bi60_les01 = ~((marm_data["MarmID"] == "BI60") & (marm_data["Lesion"] == 1))
mask_bi76_les01 = ~((marm_data["MarmID"] == "BI76") & (marm_data["Lesion"] == 1))
mask_bi76_les11 = ~((marm_data["MarmID"] == "BI76") & (marm_data["Lesion"] == 11))
mask_bj16_les13 = ~((marm_data["MarmID"] == "BJ16") & (marm_data["Lesion"] == 13))
mask_bj27_les05 = ~((marm_data["MarmID"] == "BJ27") & (marm_data["Lesion"] == 5))
mask_bj27_les12 = ~((marm_data["MarmID"] == "BJ27") & (marm_data["Lesion"] == 12))
mask_bj27_les13 = ~((marm_data["MarmID"] == "BJ27") & (marm_data["Lesion"] == 13))
mask_bj45_les10 = ~((marm_data["MarmID"] == "BJ45") & (marm_data["Lesion"] == 10))
mask_bm01_les03 = ~((marm_data["MarmID"] == "BM01") & (marm_data["Lesion"] == 3))
mask_bm19_les04 = ~((marm_data["MarmID"] == "BM19") & (marm_data["Lesion"] == 4))
mask_bm27_les07 = ~((marm_data["MarmID"] == "BM27") & (marm_data["Lesion"] == 7))
mask_bm27_les13 = ~((marm_data["MarmID"] == "BM27") & (marm_data["Lesion"] == 13))
mask_bm28_les13 = ~((marm_data["MarmID"] == "BM28") & (marm_data["Lesion"] == 13))
mask_bm28_les14 = ~((marm_data["MarmID"] == "BM28") & (marm_data["Lesion"] == 14))
mask_bm34_les09 = ~((marm_data["MarmID"] == "BM34") & (marm_data["Lesion"] == 9))
mask_bm34_les20 = ~((marm_data["MarmID"] == "BM34") & (marm_data["Lesion"] == 20))
mask_bm34_les23 = ~((marm_data["MarmID"] == "BM34") & (marm_data["Lesion"] == 23))
mask_bm40_les09 = ~((marm_data["MarmID"] == "BM40") & (marm_data["Lesion"] == 9))
mask_bn05_les15 = ~((marm_data["MarmID"] == "BN05") & (marm_data["Lesion"] == 15))
mask_bn14_les05 = ~((marm_data["MarmID"] == "BN14") & (marm_data["Lesion"] == 5))
mask_bn14_les09 = ~((marm_data["MarmID"] == "BN14") & (marm_data["Lesion"] == 9))
mask_bn14_les10 = ~((marm_data["MarmID"] == "BN14") & (marm_data["Lesion"] == 10))
mask_bn18_les07 = ~((marm_data["MarmID"] == "BN18") & (marm_data["Lesion"] == 7))
mask_bo11_les23 = ~((marm_data["MarmID"] == "BO11") & (marm_data["Lesion"] == 23))
mask_bo22_les08 = ~((marm_data["MarmID"] == "BO22") & (marm_data["Lesion"] == 8))
mask_bo22_les09 = ~((marm_data["MarmID"] == "BO22") & (marm_data["Lesion"] == 9))
mask_bo23_les06 = ~((marm_data["MarmID"] == "BO23") & (marm_data["Lesion"] == 6))
mask_bo36_les21 = ~((marm_data["MarmID"] == "BO36") & (marm_data["Lesion"] == 21))
mask_bo36_les22 = ~((marm_data["MarmID"] == "BO36") & (marm_data["Lesion"] == 22))
mask_bo37_les11 = ~((marm_data["MarmID"] == "BO37") & (marm_data["Lesion"] == 11))
mask_bo37_les13 = ~((marm_data["MarmID"] == "BO37") & (marm_data["Lesion"] == 13))
mask_bo42_les05 = ~((marm_data["MarmID"] == "BO42") & (marm_data["Lesion"] == 5))
mask_bp17_les12 = ~((marm_data["MarmID"] == "BP17") & (marm_data["Lesion"] == 12))
mask_bp20_les07 = ~((marm_data["MarmID"] == "BP20") & (marm_data["Lesion"] == 7))
mask_bp20_les08 = ~((marm_data["MarmID"] == "BP20") & (marm_data["Lesion"] == 8))
mask_bp20_les12 = ~((marm_data["MarmID"] == "BP20") & (marm_data["Lesion"] == 12))
mask_bp29_les10 = ~((marm_data["MarmID"] == "BP29") & (marm_data["Lesion"] == 10))
mask_bp29_les15 = ~((marm_data["MarmID"] == "BP29") & (marm_data["Lesion"] == 15))
mask_bp35_les04 = ~((marm_data["MarmID"] == "BP35") & (marm_data["Lesion"] == 4))
mask_bp35_les15 = ~((marm_data["MarmID"] == "BP35") & (marm_data["Lesion"] == 15))
mask_bp35_les16 = ~((marm_data["MarmID"] == "BP35") & (marm_data["Lesion"] == 16))
mask_bq02_les23 = ~((marm_data["MarmID"] == "BQ02") & (marm_data["Lesion"] == 23))
mask_bq47_les10 = ~((marm_data["MarmID"] == "BQ47") & (marm_data["Lesion"] == 10))
mask_br07_les12 = ~((marm_data["MarmID"] == "BR07") & (marm_data["Lesion"] == 12))
mask_br07_les18 = ~((marm_data["MarmID"] == "BR07") & (marm_data["Lesion"] == 18))
mask_c508c_les09 = ~((marm_data["MarmID"] == "C508c") & (marm_data["Lesion"] == 9))
mask_c508c_les13 = ~((marm_data["MarmID"] == "C508c") & (marm_data["Lesion"] == 13))
mask_c508c_les16 = ~((marm_data["MarmID"] == "C508c") & (marm_data["Lesion"] == 16))
mask_c508c_les17 = ~((marm_data["MarmID"] == "C508c") & (marm_data["Lesion"] == 17))

# drug that we do not have
mask_bdq20 = ~marm_data["Compound"].str.contains("BDQ20")

resistant_mask = (
    # Full treatment groups & marmosets
    mask_bk21 & mask_bn20 &

    # Individual lesions
    mask_750nb_les06 &
    mask_b011_les11 & 
    mask_bi01_les14 &
    mask_bi07_les04 & mask_bi07_les12 &
    mask_bi60_les01 &
    mask_bi76_les01 & mask_bi76_les11 &
    mask_bj16_les13 &
    mask_bj27_les05 & mask_bj27_les12 & mask_bj27_les13 &
    mask_bj28_les06 &
    mask_bj45_les10 &
    mask_bm01_les03 &
    mask_bm07_les21 & mask_bm07_les22 & mask_bm07_les24 &
    mask_bm16_les16 & mask_bm16_les20 &
    mask_bm19_les04 &
    mask_bm27_les07 & mask_bm27_les13 &
    mask_bm28_les13 & mask_bm28_les14 &
    mask_bm34_les09 & mask_bm34_les20 & mask_bm34_les23 &
    mask_bm40_les09 & mask_bm40_les18 & mask_bm40_les21 &
    mask_bn05_les15 &
    mask_bn13_les07 &
    mask_bn14_les05 & mask_bn14_les09 & mask_bn14_les10 &
    mask_bn18_les07 & mask_bn18_les09 &
    mask_bn22_les09 &
    mask_bn27_les03 &
    mask_bn36_les12 &
    mask_bo09_les10 & mask_bo09_les12 &
    mask_bo11_les23 &
    mask_bo22_les08 & mask_bo22_les09 & mask_bo23_les06 &
    mask_bo36_les21 & mask_bo36_les22 &
    mask_bo37_les11 & mask_bo37_les13 &
    mask_bo42_les05 &
    mask_bp02_les21 &
    mask_bp14_les11 & mask_bp14_les14 & mask_bp14_les27 & mask_bp14_les28 & mask_bp14_les29 & mask_bp14_les30 & mask_bp14_les44 & mask_bp14_les47 & mask_bp14_les50 & mask_bp14_les51 &
    mask_bp17_les04 & mask_bp17_les08 & mask_bp17_les12 & mask_bp17_les18 & mask_bp17_les26 & mask_bp17_les29 &
    mask_bp20_les07 & mask_bp20_les08 & mask_bp20_les12 &
    mask_bp29_les10 & mask_bp29_les15 &
    mask_bp35_les04 & mask_bp35_les15 & mask_bp35_les16 &
    mask_bp09_les13 &
    mask_bq02_les23 &
    mask_bq42_les09 & mask_bq42_les13 & mask_bq42_les20 & mask_bq42_les22 & mask_bq42_les23 &
    mask_bq47_les10 &
    mask_br01_les05 & mask_br07_les12 &
    mask_br07_les18 &
    mask_c508c_les09 & mask_c508c_les13 & mask_c508c_les16 & mask_c508c_les17 &
    mask_m289_les08 &
    mask_m291_les09
)

mask = resistant_mask & mask_bdq20

marm_data = marm_data.filter(mask)

# Remove non-informative features
marm_data = marm_data.drop("Scanner")

# ----------------------------------------------------------------
# PIVOT THE DATA TO A WIDE FORMAT
# UPDATE COLUMN NAMING TO BE:
# TP{timepoint}_{measurement}
# ----------------------------------------------------------------

marm_wide = marm_data.pivot(
    "Timepoint",  # widen on timepoint
    index=["Compound", "MarmID", "Lesion"],
    values=lesion_measurements,
    aggregate_function="first"
)

# Rename columns from {measurement}_{tp} to TP{tp}_{measurement}
new_columns = []
for col in marm_wide.columns:
    if col in ["Compound", "MarmID", "Lesion"]:
        new_columns.append(col)
    else:
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            measurement, tp = parts
            new_columns.append(f"TP{tp}_{measurement}")
        else:
            new_columns.append(col)
marm_wide.columns = new_columns
marm_wide = marm_wide.sort("MarmID")

# ----------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------

marm_wide.write_csv("data/marm_data_wide.csv")

# ----------------------------------------------------------------
# EXPORT OF RESISTANT DATA
# ----------------------------------------------------------------

marm_resistant_raw = pl.read_excel("data/FINALCORRECTED_Merged_CFUandPETCT_20231005.xlsx")
marm_resistant_raw = marm_resistant_raw.filter(~resistant_mask)

marm_resistant_wide = marm_resistant_raw.pivot(
    "Timepoint",
    index=["Compound", "MarmID", "Lesion"],
    values=lesion_measurements,
    aggregate_function="first"
)

# Rename columns from {measurement}_{tp} to TP{tp}_{measurement}
new_columns_resistant = []
for col in marm_resistant_wide.columns:
    if col in ["Compound", "MarmID", "Lesion"]:
        new_columns_resistant.append(col)
    else:
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            measurement, tp = parts
            new_columns_resistant.append(f"TP{tp}_{measurement}")
        else:
            new_columns_resistant.append(col)
marm_resistant_wide.columns = new_columns_resistant
marm_resistant_wide = marm_resistant_wide.sort("MarmID")

marm_resistant_wide.write_csv("data/marm_data_resistant.csv")
