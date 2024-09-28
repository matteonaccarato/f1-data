# https://www.youtube.com/watch?v=29ZQ3TDGgRQ

import pandas as pd

import backend.f1db_utils as f1db_utils

not_a_number_replace = {
    'DNF': f1db_utils.INFINITE_RESULT, 
    'DNS': f1db_utils.INFINITE_RESULT, 
    "DSQ": f1db_utils.INFINITE_RESULT, 
    "DNQ": f1db_utils.INFINITE_RESULT, 
    "NC": f1db_utils.INFINITE_RESULT, 
    "DNPQ": f1db_utils.INFINITE_RESULT, 
    "EX": f1db_utils.INFINITE_RESULT
}

def test(selected_circuit):
    if len(selected_circuit) != 1: f1db_utils.warning_empty_dataframe
    
    df = pd.read_csv(f"{f1db_utils.folder}/{f1db_utils.qualifying_results}")
    df.rename(columns={"positionText":"positionQualifying"}, inplace=True)
    df.drop(columns=df.columns.difference(["raceId","positionQualifying","driverId"]), inplace=True)
    
    df_races_results = pd.read_csv(f"{f1db_utils.folder}/{f1db_utils.races_results}")
    df_races_results.rename(columns={"id": "raceId", "positionText": "positionRace"}, inplace=True)
    df_races_results.drop(columns=df_races_results.columns.difference(["raceId", "positionRace", "driverId"]), inplace=True)
    
    df_drivers_info = pd.read_csv(f"{f1db_utils.folder}/{f1db_utils.drivers_info}")
    df_drivers_info.rename(columns={"id":"driverId", "name":"driverName"}, inplace=True)
    df_drivers_info.drop(columns=df_drivers_info.columns.difference(["driverId", "driverName"]), inplace=True)
    
    df_races = pd.read_csv(f"{f1db_utils.folder}/{f1db_utils.races}")
    df_races.rename(columns={"id":"raceId"}, inplace=True)
    df_races.drop(columns=df_races.columns.difference(["raceId", "circuitId", "officialName"]), inplace=True)
    df_races_circuits = pd.merge(df_races_results, df_races, on="raceId", how="left")
    selected_circuits_mask = df_races_circuits["circuitId"] == selected_circuit
    df_races_circuits = df_races_circuits[selected_circuits_mask]
    
    df = pd.merge(df, df_races_circuits, on=["raceId","driverId"], how="right")
    df = pd.merge(df, df_drivers_info, on="driverId", how="left")

    df['positionRace'] = df['positionRace'].replace(not_a_number_replace) 
    df['positionRace'] = pd.to_numeric(df['positionRace'], errors='coerce')
    df["positionRace"] = df["positionRace"].fillna(f1db_utils.QUALI_FILL_NA)
    df['positionRace'] = df['positionRace'].astype(int)
    df = df[(df["positionRace"] > 0) & (df["positionRace"] < f1db_utils.INFINITE_RESULT - 1)]

    df['positionQualifying'] = df['positionQualifying'].replace(not_a_number_replace) 
    df["positionQualifying"] = df["positionQualifying"].fillna(f1db_utils.INFINITE_RESULT)
    df['positionQualifying'] = df['positionQualifying'].astype(int)
    max_quali_value = df[df["positionQualifying"] != f1db_utils.INFINITE_RESULT]["positionQualifying"].max()
    # Replace qualifying NaN (previously replaced with INFINITE) with max real qualifying result + 1
    df["positionQualifying"] = df["positionQualifying"].replace(f1db_utils.INFINITE_RESULT, max_quali_value + 1)
    df = df[df["positionQualifying"] > 0]

    
    
    
    # ==================================
    # print(df.head())
    df = df.drop(columns=['raceId', 'officialName', 'circuitId', 'driverName'])
    df['driverId_int'] = pd.factorize(df['driverId'])[0]
    map = dict(zip(df['driverId'], df['driverId_int']))
    # print(map)
    df.drop(columns=['driverId', 'driverId_int'], axis=1, inplace=True)
    
    # Data separation X and y
    y = df['positionRace']
    X = df.drop('positionRace', axis=1)
    print(df.head())
    # print(y.head())
    # print(X.head())
    
    # Data splitting (80% train and 20% test)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(X_train)
    # print(X_test)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R² Score: {r2}")
    
    ## Model building using Logistic Regression
    # Training the model
    """from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Applying the model to make a prediction
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)
    
    
    # Evaluate model performance (Compare y_train and y_lr_train_pred)
    from sklearn.metrics import mean_squared_error, r2_score
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)
    
    print('LR MSE (Train): ', lr_train_mse)
    print('LR R2 (Train): ', lr_train_r2)
    print('LR MSE (Test): ', lr_test_mse)
    print('LR R2 (Test): ', lr_test_r2)
    
    # print(lr.predict(df)
    """
    # TODO - rimuovere i piloti , tenere solo postition qualifying
    
    

test('monza')