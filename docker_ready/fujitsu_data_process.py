import sys, site, pprint
print("EXE:", sys.executable)
print("VERSION:", sys.version)
print("USER_SITE:", site.getusersitepackages())
print("SITE_PKGS:")
pprint.pprint(site.getsitepackages() if hasattr(site, "getsitepackages") else [])
print("FIRST PATHS:", sys.path[:5])

import pandas as pd
import numpy as np
import re
import os
import glob
from datetime import timedelta
from pathlib import Path

data_dir = "raw"
history_pattern = os.path.join(data_dir, "history_fujitsu_*.csv")
histalmevt_pattern = os.path.join(data_dir, "histalmevt_fujitsu_*.csv")

data_path = Path('processed')
data_path.mkdir(parents=True, exist_ok=True)

def find_file_by_pattern(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file matching pattern {pattern} found")
    files.sort()
    print(f"Matched file: {files[-1]}")
    return files[-1]  # latest

# Find matching files
history_file = find_file_by_pattern(history_pattern)
histalmevt_file = find_file_by_pattern(histalmevt_pattern)


def run_data_pipeline():
    print("Starting data pipeline...\n")
    df_dim = process_histcurr()
    df_inst = process_history_only()
    df_alarm = process_alarm_only()
    df_link = create_linktable(df_inst, df_alarm)
    create_date_dimension(df_link)
    print("\n Fujitsu Data Process pipeline completed successfully.")

def process_histcurr():
    
    print(" Processing 'HISTCURR/histcurr_fujitsu.csv'...")
    df = pd.read_csv('HISTCURR/histcurr_fujitsu.csv', low_memory=False)
    df.dropna(inplace=True)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])

    mappings = {
        'S606-KAL': 1001, 'S606-PEL': 1003, 'S606-ALEX': 1002, 'S606-JLNRJH': 1004,
        'S606-THM': 1005, 'S606-GEY': 1006
    }
    for tag, num in mappings.items():
        # print(f"Tag containing '{tag}'➝ RTUNUMBER: {num}")
        df.loc[df["TAGNAME"].str.contains(tag), "RTUNUMBER"] = num

    print("Checking DESCRIPTION-based mappings...")
    df.loc[df["DESCRIPTION"].str.contains('rajah', case=False), "RTUNUMBER"] = 1004
    df.loc[df["DESCRIPTION"].str.contains('marinarwps', case=False), "RTUNUMBER"] = 1007
    df.loc[df["DESCRIPTION"].str.contains('KranjiIPU.*CNF'), "RTUNUMBER"] = 4221

    print("Renaming columns...")
    df.rename(columns={
        "TAGNAME": "Tagname", "IPADDRESS": "IPAddress", "ROW_ID": "RowId",
        "DESCRIPTION": "Description", "RAWTYPE": "RawType", "RTUNUMBER": "RTUNumber",
    }, inplace=True)
    print("columns after renaming... ", df.columns.values)
    print(" Converting datatypes...")
    df = df[['IPAddress', 'RowId', 'Description', 'Tagname', 'RTUNumber', 'RawType']]
    df = df.astype({'RTUNumber': int, 'RawType': int, 'IPAddress': int, 'RowId': int})

    print(" Generating columns: Location, Hkey, Signaltype, Equipment...")
    df['Location'] = df['Description'].apply(lambda a: a.split('-')[0])
    df['Hkey'] = df.IPAddress.astype(str) + df.RowId.astype(str)
    df['Signaltype'] = df['RawType'].apply(lambda x: 'Analog' if x in [1, 5] else 'Digital')
    df['Equipment'] = df.apply(assign_equipment, axis=1)

    df.to_csv('processed/dim.csv', index=False)
    print(" Saved processed dim.csv\n")
    return df

def assign_equipment(row):
    desc = row['Description'].lower()
    if row['Signaltype'] == 'Analog':
        if 'dissolved' in desc: return 'Dissolved Oxygen'
        if 'cond' in desc: return 'Conductivity'
        if 'temp' in desc: return 'Temperature'
        if 'press' in desc: return 'Pressure'
        if 'volt' in desc: return 'Voltage'
        if 'flow' in desc: return 'Flowrate'
        # Analog descriptions that don't match any known keyword fall through
        # to 'Others' below — keep this as-is until the client decides how
        # they want analog miscellany categorised.
        return 'Others'
    else:
        # Digital signals: keep existing specific categories where they match,
        # otherwise label as 'Digital Signal' (a parking bucket) rather than
        # the generic 'Others'. Analysis is currently scoped to analog signals;
        # this label makes it easy for the client to see what's digital and
        # decide on future re-categorisation.
        if 'pump' in desc: return 'Pump'
        if 'valve' in desc: return 'Valve'
        if 'level' in desc: return 'LevelSensor'
        return 'Digital Signal'

def process_history_only():
    print(" Processing 'raw/history_fujitsu.csv'...")
    df = pd.read_csv(history_file, sep=';', low_memory=False, on_bad_lines='skip')
    df.dropna(inplace=True)

    print(" Renaming columns...")
    df.rename(columns={
        "IPADDRESS": "IPAddress", "ROW_ID": "RowId", "DATETIME": "DateTime", "CURRVALUE": "CurrValue"
    }, inplace=True)
    print("columns after renaming... ", df.columns.values)

    print(" Filtering unreasonable CurrValues...")
    original_len = len(df)
    df = df[(df.CurrValue < 1e9) & (df.CurrValue > -1e9)]
    print(f"Filtered out {original_len - len(df)} rows")

    df = df.astype({'IPAddress': int, 'RowId': int})
    df['Hkey'] = df.IPAddress.astype(str) + df.RowId.astype(str)
    df.drop(['IPAddress', 'RowId'], axis=1, inplace=True)

    # Add HkeyDateTime manually (since no resample anymore)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values(['Hkey', 'DateTime'])  # sort for consistency
    df['HkeyDateTime'] = df['Hkey'] + df['DateTime'].dt.strftime('%Y%m%d%H%M%S')

    df[['HkeyDateTime', 'CurrValue']].to_csv('processed/inst.csv', index=False)
    print("Saved processed inst.csv\n")
    return df


def process_alarm_only():
    print(" Processing 'raw/histalmevt_fujitsu.csv'...")
    df = pd.read_csv(histalmevt_file ,sep=';', low_memory=False)
    df.dropna(inplace=True)

    print(" Renaming and cleaning AlarmText...")
    df.rename(columns={"ALARMSETTIME": "AlarmSetTime", "ALARMTEXT": "AlarmText"}, inplace=True)
    print("columns after renaming... ", df.columns.values)
    df['AlarmText'] = df['AlarmText'].str.strip()
    df['Status'] = df.AlarmText.apply(lambda x: x.rsplit(' ', 1)[1])
    df['Alarm'] = df.AlarmText.apply(lambda x: x.rsplit(' ', 1)[0])
    df['Location'] = df['Alarm'].apply(lambda x: x.split('-', 1)[0])
    df['AlarmSetTime'] = pd.to_datetime(df['AlarmSetTime'])

    print("Filtering to last 7 days of alarms...")
    df = df[df.AlarmSetTime > df.AlarmSetTime.max() - timedelta(days=7)]

    print("Adjusting 'Stopped' label...")
    df.loc[df['Status'].str.contains(r'\b(?:Stop)\b'), 'Status'] = 'Stopped'

    df.rename(columns={'AlarmSetTime': 'DateTime'}, inplace=True)
    df['Hkey'] = df['Alarm']
    df['HkeyDateTime'] = df.Hkey.astype(str) + df.DateTime.astype(str)
    df.HkeyDateTime = df.HkeyDateTime.apply(lambda x: re.sub(r'\D', '', x))
    df[['HkeyDateTime', 'Status']].to_csv('processed/alarmevent.csv', index=False)
    print(" Saved processed alarmevent.csv\n")
    return df[['HkeyDateTime', 'DateTime', 'Hkey']]

def create_linktable(df_inst, df_alarm):
    print(" Creating linktable...")
    dfLink = pd.concat([df_alarm[['HkeyDateTime', 'DateTime', 'Hkey']], df_inst[['HkeyDateTime', 'DateTime', 'Hkey']]])
    dfLink.HkeyDateTime = dfLink.HkeyDateTime.apply(lambda x: re.sub(r'\D', '', x))
    dfLink.drop_duplicates(inplace=True)
    dfLink.to_csv('processed/linktable.csv', index=False)
    print(" Saved processed linktable.csv\n")
    return dfLink

def create_date_dimension(df_link):
    print("Creating date dimension...")
    dfDateDim = df_link.drop(['HkeyDateTime', 'Hkey'], axis=1).drop_duplicates()
    dfDateDim['DateTime'] = pd.to_datetime(dfDateDim['DateTime'])
    dfDateDim['Date'] = dfDateDim['DateTime'].dt.date
    dfDateDim['Hour'] = dfDateDim['DateTime'].dt.hour
    dfDateDim['Minute'] = dfDateDim['DateTime'].dt.minute
    dfDateDim[['DateTime', 'Date', 'Hour', 'Minute']].to_csv('processed/dateDim.csv', index=False)
    print(" Saved processed dateDim.csv\n")

if __name__ == '__main__':
    run_data_pipeline()