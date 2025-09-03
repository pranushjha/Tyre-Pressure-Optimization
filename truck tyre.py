import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = {
    'road_type': ['highway', 'mountain', 'urban', 'offroad', 'highway', 'mountain', 'urban', 'offroad', 'wet', 'snow',
                  'highway', 'mountain', 'urban', 'offroad', 'highway', 'mountain', 'urban', 'offroad', 'wet', 'snow'],
    'load_kg': [20000, 18000, 15000, 12000, 22000, 19000, 16000, 13000, 20000, 17000,
                21000, 17000, 14000, 11000, 23000, 20000, 15000, 12000, 19000, 16000],
    'axles': [3, 3, 2, 2, 4, 4, 2, 3, 3, 2,
              3, 3, 2, 2, 4, 4, 2, 3, 3, 2],
    'climate': ['dry', 'dry', 'dry', 'dry', 'rainy', 'cold', 'dry', 'rainy', 'rainy', 'snow',
                'dry', 'dry', 'dry', 'dry', 'rainy', 'cold', 'dry', 'rainy', 'rainy', 'snow'],
    'steer_tire': ['All-position rib', 'All-position rib', 'All-position rib', 'Off-road', 'All-position rib',
                   'All-position rib', 'All-position rib', 'All-position rib', 'Siped All-position rib', 'Winter tread',
                   'All-position rib', 'All-position rib', 'All-position rib', 'Off-road', 'All-position rib',
                   'All-position rib', 'All-position rib', 'All-position rib', 'Siped All-position rib', 'Winter tread'],
    'drive_tire': ['Deep lug', 'Traction pattern', 'Highway rib', 'Mud terrain', 'Deep lug',
                   'Traction pattern', 'Highway rib', 'All-terrain', 'Wet traction', 'Winter studded',
                   'Deep lug', 'Traction pattern', 'Highway rib', 'Mud terrain', 'Deep lug',
                   'Traction pattern', 'Highway rib', 'All-terrain', 'Wet traction', 'Winter studded'],
    'trailer_tire': ['High mileage', 'Sturdy sidewall', 'Standard', 'Reinforced', 'High mileage',
                     'Sturdy sidewall', 'Standard', 'Reinforced', 'Wet traction', 'Winter',
                     'High mileage', 'Sturdy sidewall', 'Standard', 'Reinforced', 'High mileage',
                     'Sturdy sidewall', 'Standard', 'Reinforced', 'Wet traction', 'Winter'],
    'safety_rating': [9.2, 8.8, 8.5, 8.0, 9.1, 8.7, 8.4, 8.2, 9.0, 9.5,
                      9.1, 8.7, 8.3, 7.9, 9.0, 8.6, 8.3, 8.1, 8.9, 9.4]
}

df = pd.DataFrame(data)


label_encoders = {}
for column in ['road_type', 'climate', 'steer_tire', 'drive_tire', 'trailer_tire']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


X = df[['road_type', 'load_kg', 'axles', 'climate']]
y_steer = df['steer_tire']
y_drive = df['drive_tire']
y_trailer = df['trailer_tire']

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
y_steer_train = y_steer.loc[X_train.index]
y_drive_train = y_drive.loc[X_train.index]
y_trailer_train = y_trailer.loc[X_train.index]


steer_model = RandomForestClassifier(n_estimators=100, random_state=42)
drive_model = RandomForestClassifier(n_estimators=100, random_state=42)
trailer_model = RandomForestClassifier(n_estimators=100, random_state=42)

steer_model.fit(X_train, y_steer_train)
drive_model.fit(X_train, y_drive_train)
trailer_model.fit(X_train, y_trailer_train)


def evaluate_model_on_csv(filepath):
    print("\n=== External CSV Evaluation ===")
    test_df = pd.read_csv(filepath)

    for column in ['road_type', 'climate', 'steer_tire', 'drive_tire', 'trailer_tire']:
        if column in test_df.columns:
            test_df[column] = label_encoders[column].transform(test_df[column])

    X_test_csv = test_df[['road_type', 'load_kg', 'axles', 'climate']]
    y_steer_csv = test_df['steer_tire']
    y_drive_csv = test_df['drive_tire']
    y_trailer_csv = test_df['trailer_tire']

    y_steer_pred = steer_model.predict(X_test_csv)
    y_drive_pred = drive_model.predict(X_test_csv)
    y_trailer_pred = trailer_model.predict(X_test_csv)

    print(f"Steer Tire Accuracy: {accuracy_score(y_steer_csv, y_steer_pred) * 100:.2f}%")
    print(f"Drive Tire Accuracy: {accuracy_score(y_drive_csv, y_drive_pred) * 100:.2f}%")
    print(f"Trailer Tire Accuracy: {accuracy_score(y_trailer_csv, y_trailer_pred) * 100:.2f}%")

def get_user_input():
    print("\n=== Truck Tire Optimization System ===")
    road_type = input("Road type (highway/mountain/urban/offroad/wet/snow): ").lower()
    while road_type not in ['highway', 'mountain', 'urban', 'offroad', 'wet', 'snow']:
        road_type = input("Invalid input. Please enter highway/mountain/urban/offroad/wet/snow: ").lower()
    
    load_kg = int(input("Load in kg (e.g., 15000): "))
    axles = int(input("Number of axles (2-4): "))
    climate = input("Climate (dry/rainy/cold/snow): ").lower()
    while climate not in ['dry', 'rainy', 'cold', 'snow']:
        climate = input("Invalid input. Please enter dry/rainy/cold/snow: ").lower()

    return road_type, load_kg, axles, climate


def predict_tires(road_type, load_kg, axles, climate):
    road_code = label_encoders['road_type'].transform([road_type])[0]
    climate_code = label_encoders['climate'].transform([climate])[0]

    input_data = pd.DataFrame([[road_code, load_kg, axles, climate_code]],
                              columns=['road_type', 'load_kg', 'axles', 'climate'])

    steer_code = steer_model.predict(input_data)[0]
    drive_code = drive_model.predict(input_data)[0]
    trailer_code = trailer_model.predict(input_data)[0]

    steer = label_encoders['steer_tire'].inverse_transform([steer_code])[0]
    drive = label_encoders['drive_tire'].inverse_transform([drive_code])[0]
    trailer = label_encoders['trailer_tire'].inverse_transform([trailer_code])[0]

    return steer, drive, trailer


def display_recommendations(steer, drive, trailer):
    print("\n=== Recommended Tire Configuration ===")
    print(f"\n🛞 Steer Tire: {steer}")
    if 'rib' in steer:
        print("- Reason: Rib design improves straight-line tracking and durability.")
    elif 'Siped' in steer:
        print("- Reason: Siping enhances grip on wet or snowy roads.")
    elif 'Winter' in steer:
        print("- Reason: Designed for cold climates with better low-temp traction.")
    elif 'Off-road' in steer:
        print("- Reason: Thick tread pattern offers better control on rough terrain.")

    print(f"\n🚚 Drive Tire: {drive}")
    if 'Deep' in drive:
        print("- Reason: Deep lug pattern boosts traction under heavy loads.")
    elif 'Traction' in drive:
        print("- Reason: Extra grip for climbing or mountain routes.")
    elif 'Highway' in drive:
        print("- Reason: Ribbed design provides fuel efficiency and longevity.")
    elif 'Mud' in drive:
        print("- Reason: Tread clears mud for off-road performance.")
    elif 'All-terrain' in drive:
        print("- Reason: Balanced for both highway and off-road conditions.")
    elif 'Wet' in drive:
        print("- Reason: Grooves reduce hydroplaning risk.")
    elif 'Winter' in drive:
        print("- Reason: Tread and rubber for snow/ice performance.")

    print(f"\n🚛 Trailer Tire: {trailer}")
    if 'High' in trailer:
        print("- Reason: High mileage compound for long-distance hauling.")
    elif 'Sturdy' in trailer:
        print("- Reason: Strong sidewalls to prevent blowouts on tough terrain.")
    elif 'Standard' in trailer:
        print("- Reason: Economical and versatile for general usage.")
    elif 'Reinforced' in trailer:
        print("- Reason: Additional ply ratings for heavier loads.")
    elif 'Wet' in trailer:
        print("- Reason: Designed for better grip on slippery surfaces.")
    elif 'Winter' in trailer:
        print("- Reason: Rubber compound suitable for snow and freezing temps.")


if __name__ == "__main__":
    try:
        evaluate_model_on_csv("tire_test_data_large.csv")
    except Exception as e:
        print("External CSV Evaluation Skipped:", e)

    road_type, load_kg, axles, climate = get_user_input()
    steer, drive, trailer = predict_tires(road_type, load_kg, axles, climate)
    display_recommendations(steer, drive, trailer)
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = {
    'road_type': ['highway', 'mountain', 'urban', 'offroad', 'highway', 'mountain', 'urban', 'offroad', 'wet', 'snow',
                  'highway', 'mountain', 'urban', 'offroad', 'highway', 'mountain', 'urban', 'offroad', 'wet', 'snow'],
    'load_kg': [20000, 18000, 15000, 12000, 22000, 19000, 16000, 13000, 20000, 17000,
                21000, 17000, 14000, 11000, 23000, 20000, 15000, 12000, 19000, 16000],
    'axles': [3, 3, 2, 2, 4, 4, 2, 3, 3, 2,
              3, 3, 2, 2, 4, 4, 2, 3, 3, 2],
    'climate': ['dry', 'dry', 'dry', 'dry', 'rainy', 'cold', 'dry', 'rainy', 'rainy', 'snow',
                'dry', 'dry', 'dry', 'dry', 'rainy', 'cold', 'dry', 'rainy', 'rainy', 'snow'],
    'steer_tire': ['All-position rib', 'All-position rib', 'All-position rib', 'Off-road', 'All-position rib',
                   'All-position rib', 'All-position rib', 'All-position rib', 'Siped All-position rib', 'Winter tread',
                   'All-position rib', 'All-position rib', 'All-position rib', 'Off-road', 'All-position rib',
                   'All-position rib', 'All-position rib', 'All-position rib', 'Siped All-position rib', 'Winter tread'],
    'drive_tire': ['Deep lug', 'Traction pattern', 'Highway rib', 'Mud terrain', 'Deep lug',
                   'Traction pattern', 'Highway rib', 'All-terrain', 'Wet traction', 'Winter studded',
                   'Deep lug', 'Traction pattern', 'Highway rib', 'Mud terrain', 'Deep lug',
                   'Traction pattern', 'Highway rib', 'All-terrain', 'Wet traction', 'Winter studded'],
    'trailer_tire': ['High mileage', 'Sturdy sidewall', 'Standard', 'Reinforced', 'High mileage',
                     'Sturdy sidewall', 'Standard', 'Reinforced', 'Wet traction', 'Winter',
                     'High mileage', 'Sturdy sidewall', 'Standard', 'Reinforced', 'High mileage',
                     'Sturdy sidewall', 'Standard', 'Reinforced', 'Wet traction', 'Winter'],
    'safety_rating': [9.2, 8.8, 8.5, 8.0, 9.1, 8.7, 8.4, 8.2, 9.0, 9.5,
                      9.1, 8.7, 8.3, 7.9, 9.0, 8.6, 8.3, 8.1, 8.9, 9.4]
}

df = pd.DataFrame(data)


label_encoders = {}
for column in ['road_type', 'climate', 'steer_tire', 'drive_tire', 'trailer_tire']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le


X = df[['road_type', 'load_kg', 'axles', 'climate']]
y_steer = df['steer_tire']
y_drive = df['drive_tire']
y_trailer = df['trailer_tire']

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
y_steer_train = y_steer.loc[X_train.index]
y_drive_train = y_drive.loc[X_train.index]
y_trailer_train = y_trailer.loc[X_train.index]


steer_model = RandomForestClassifier(n_estimators=100, random_state=42)
drive_model = RandomForestClassifier(n_estimators=100, random_state=42)
trailer_model = RandomForestClassifier(n_estimators=100, random_state=42)

steer_model.fit(X_train, y_steer_train)
drive_model.fit(X_train, y_drive_train)
trailer_model.fit(X_train, y_trailer_train)


def evaluate_model_on_csv(filepath):
    print("\n=== External CSV Evaluation ===")
    test_df = pd.read_csv(filepath)

    for column in ['road_type', 'climate', 'steer_tire', 'drive_tire', 'trailer_tire']:
        if column in test_df.columns:
            test_df[column] = label_encoders[column].transform(test_df[column])

    X_test_csv = test_df[['road_type', 'load_kg', 'axles', 'climate']]
    y_steer_csv = test_df['steer_tire']
    y_drive_csv = test_df['drive_tire']
    y_trailer_csv = test_df['trailer_tire']

    y_steer_pred = steer_model.predict(X_test_csv)
    y_drive_pred = drive_model.predict(X_test_csv)
    y_trailer_pred = trailer_model.predict(X_test_csv)

    print(f"Steer Tire Accuracy: {accuracy_score(y_steer_csv, y_steer_pred) * 100:.2f}%")
    print(f"Drive Tire Accuracy: {accuracy_score(y_drive_csv, y_drive_pred) * 100:.2f}%")
    print(f"Trailer Tire Accuracy: {accuracy_score(y_trailer_csv, y_trailer_pred) * 100:.2f}%")


def get_user_input():
    print("\n=== Truck Tire Optimization System ===")
    road_type = input("Road type (highway/mountain/urban/offroad/wet/snow): ").lower()
    while road_type not in ['highway', 'mountain', 'urban', 'offroad', 'wet', 'snow']:
        road_type = input("Invalid input. Please enter highway/mountain/urban/offroad/wet/snow: ").lower()
    
    load_kg = int(input("Load in kg (e.g., 15000): "))
    axles = int(input("Number of axles (2-4): "))
    climate = input("Climate (dry/rainy/cold/snow): ").lower()
    while climate not in ['dry', 'rainy', 'cold', 'snow']:
        climate = input("Invalid input. Please enter dry/rainy/cold/snow: ").lower()

    return road_type, load_kg, axles, climate


def predict_tires(road_type, load_kg, axles, climate):
    road_code = label_encoders['road_type'].transform([road_type])[0]
    climate_code = label_encoders['climate'].transform([climate])[0]

    input_data = pd.DataFrame([[road_code, load_kg, axles, climate_code]],
                              columns=['road_type', 'load_kg', 'axles', 'climate'])

    steer_code = steer_model.predict(input_data)[0]
    drive_code = drive_model.predict(input_data)[0]
    trailer_code = trailer_model.predict(input_data)[0]

    steer = label_encoders['steer_tire'].inverse_transform([steer_code])[0]
    drive = label_encoders['drive_tire'].inverse_transform([drive_code])[0]
    trailer = label_encoders['trailer_tire'].inverse_transform([trailer_code])[0]

    return steer, drive, trailer


def display_recommendations(steer, drive, trailer):
    print("\n=== Recommended Tire Configuration ===")
    print(f"\n🛞 Steer Tire: {steer}")
    if 'rib' in steer:
        print("- Reason: Rib design improves straight-line tracking and durability.")
    elif 'Siped' in steer:
        print("- Reason: Siping enhances grip on wet or snowy roads.")
    elif 'Winter' in steer:
        print("- Reason: Designed for cold climates with better low-temp traction.")
    elif 'Off-road' in steer:
        print("- Reason: Thick tread pattern offers better control on rough terrain.")

    print(f"\n🚚 Drive Tire: {drive}")
    if 'Deep' in drive:
        print("- Reason: Deep lug pattern boosts traction under heavy loads.")
    elif 'Traction' in drive:
        print("- Reason: Extra grip for climbing or mountain routes.")
    elif 'Highway' in drive:
        print("- Reason: Ribbed design provides fuel efficiency and longevity.")
    elif 'Mud' in drive:
        print("- Reason: Tread clears mud for off-road performance.")
    elif 'All-terrain' in drive:
        print("- Reason: Balanced for both highway and off-road conditions.")
    elif 'Wet' in drive:
        print("- Reason: Grooves reduce hydroplaning risk.")
    elif 'Winter' in drive:
        print("- Reason: Tread and rubber for snow/ice performance.")

    print(f"\n🚛 Trailer Tire: {trailer}")
    if 'High' in trailer:
        print("- Reason: High mileage compound for long-distance hauling.")
    elif 'Sturdy' in trailer:
        print("- Reason: Strong sidewalls to prevent blowouts on tough terrain.")
    elif 'Standard' in trailer:
        print("- Reason: Economical and versatile for general usage.")
    elif 'Reinforced' in trailer:
        print("- Reason: Additional ply ratings for heavier loads.")
    elif 'Wet' in trailer:
        print("- Reason: Designed for better grip on slippery surfaces.")
    elif 'Winter' in trailer:
        print("- Reason: Rubber compound suitable for snow and freezing temps.")


if __name__ == "__main__":
    try:
        evaluate_model_on_csv("tire_test_data_large.csv")
    except Exception as e:
        print("External CSV Evaluation Skipped:", e)

    road_type, load_kg, axles, climate = get_user_input()
    steer, drive, trailer = predict_tires(road_type, load_kg, axles, climate)
    display_recommendations(steer, drive, trailer)
