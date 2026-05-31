import numpy as np

LOCALITIES = {
    "railway_station" : (18.1115, 83.3963),
    "bus_stand"       : (18.1099, 83.3994),
    "fort_park"       : (18.1116, 83.4096),
    "collectorate"    : (18.1176, 83.3870),
    "balaji_nagar"    : (18.1078, 83.3991),
    "mall"            : (18.1103, 83.3976),
    "medical_college" : (18.1426, 83.4024),
    "jntu"            : (18.1511, 83.3757),
}

MY_HOME = (18.1103, 83.3975)
MY_WORK = (18.1115, 83.3963)
MAP_CENTER = MY_HOME

DBSCAN_EPS        = 0.003
DBSCAN_MIN_PTS    = 5
ISO_CONTAMINATION = 0.05

def add_features(df):
    df = df.copy()
    df['lat_rad']  = np.radians(df['lat'])
    df['lng_rad']  = np.radians(df['lng'])
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

FEATURE_COLS = ['lat_rad', 'lng_rad', 'hour_sin', 'hour_cos']