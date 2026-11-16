import pandas as pd


EXCEL_FILE = "data/gst_input.xlsx"


def get_sheet_names():

    xls = pd.ExcelFile(EXCEL_FILE)

    return xls.sheet_names


def load_data(sheet):

    df = pd.read_excel(
        EXCEL_FILE,
        sheet_name=sheet
    )

    df["Month"] = pd.to_datetime(df["Month"])

    return df