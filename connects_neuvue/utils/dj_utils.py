import pandas as pd
import datajoint as dj

def df_from_table(table):
    return pd.DataFrame(table.fetch())