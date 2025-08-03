import json
import pandas as pd
from features.credit_report_feature_creation import extract_all_credit_features

class CreditFeatureExtractor:
    def __init__(self, json_path=None, json_data=None):
        if json_path:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
        elif json_data:
            self.json_data = json_data
        else:
            raise ValueError("Provide either json_path or json_data")

    def flatten_json(self):
        if isinstance(self.json_data, list):
            df_json = pd.DataFrame(self.json_data)
        else:
            raise ValueError("Expected JSON list at top level")

        flattened_rows = []
        for record in df_json.to_dict(orient='records'):
            row = {'application_id': record['application_id']}
            consumer_credit = record.get('data', {}).get('consumerfullcredit', {})
            for key, value in consumer_credit.items():
                row[key] = value
            flattened_rows.append(row)

        return pd.DataFrame(flattened_rows)

    def create_features(self):
        df_flat = self.flatten_json()
        features_df = extract_all_credit_features(df_flat)
        return features_df