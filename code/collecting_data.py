
import pandas as pd
import numpy as np
import os


def data_sensor():

    df = pd.read_csv('data\dataset_after_training.csv')

    df.columns

    X = df[['previous_trend_duration', 'previous_percentage', 'harpoon_output_signal', 'harpoon_fuel', 'harpoon_burnout', 'harpoon_h9', 'harpoon_h21', 'harpoon_h14', 'top_gun_output_signal', 'top_gun_top_gun_a', 'top_gun_top_gun_b', 'top_gun_fuel', 'top_gun_burnout', 'maverick_output_signal', 'maverick_fuel', 'maverick_burnout', 'wind_output_signal', 'wind_crossover', 'sonar_output_signal', 'sonar_fuel', 'sonar_burnout', 'Storm', 'Storm_Actual', 'Storm_Impact', 'Storm_MA30', 'Storm_MA7', 'binaryverdict_grouchy', 'BinaryVerdictBernadotte',
            'binaryverdict_Marmont', 'bernadotte_random_forest',
            'Marmont_random_forest', 'Grouchy_random_forest']]
    y = df['classification']
    print(X.shape)
    print(np.unique(y))

    X_np = np.array(X)

    return X_np, y, df


data_sensor()
