import json
import pandas

def output_rows(xlsx_path: str, sh_name: str) -> list[dict]:
    """Wczytuje plik xlsx i zwraca listę słowników (jeden wiersz -> jeden słownik)"""
    xlsx_df = pandas.read_excel(xlsx_path, sheet_name=sh_name)
    xlsx_rows = xlsx_df.to_json(orient='records')
    xlsx_rows = xlsx_rows.lstrip('[').rstrip(']').split('{')
    xlsx_rows = ['{' + i for i in xlsx_rows[1:]]
    xlsx_rows = [i.replace('null', 'None') for i in xlsx_rows]
    xlsx_rows = [eval(i.rstrip(',').replace("true", "'true'").replace("false", "'false'")) for i in xlsx_rows]
    return xlsx_rows
