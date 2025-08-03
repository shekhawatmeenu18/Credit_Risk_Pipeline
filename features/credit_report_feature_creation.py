import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pandas import json_normalize
from datetime import datetime


# Display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)  # Adjust as needed
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# ## 1. Delinquincy

def extract_delinquency_features(df, col='deliquencyinformation', add_flags=True):
    # Normalize nested data
    records = []
    for _, row in df.iterrows():
        app_id = row['application_id']
        delinq_data = row[col]

        if isinstance(delinq_data, list):
            for item in delinq_data:
                item['application_id'] = app_id
                records.append(item)
        elif isinstance(delinq_data, dict):
            delinq_data['application_id'] = app_id
            records.append(delinq_data)

    if not records:
        return pd.DataFrame(columns=['application_id'])  # Empty case

    delinq_normalized = pd.DataFrame(records)

    # Replace "XXX" with NaN and convert numeric fields
    delinq_normalized.replace("XXX", np.nan, inplace=True)
    delinq_normalized['monthsinarrears'] = pd.to_numeric(delinq_normalized['monthsinarrears'], errors='coerce')
    delinq_normalized['periodnum'] = pd.to_numeric(delinq_normalized['periodnum'], errors='coerce')

    # Aggregate
    df_delinquency_features = delinq_normalized.groupby('application_id').agg(
        total_delinquency_accounts=('accountno', 'count'),
        months_in_arrears=('monthsinarrears', 'max'),
        latest_delinquency_period=('periodnum', 'max'),
        unique_delinquency_subscribers=('subscribername', 'nunique')
    ).reset_index()

    # Add optional flags
    if add_flags:
        df_delinquency_features['has_delinquency_flag'] = (df_delinquency_features['total_delinquency_accounts'] > 0).astype(int)
        df_delinquency_features['severe_delinquency_flag'] = (df_delinquency_features['months_in_arrears'] >= 6).astype(int)

        # Recent delinquency flag: check if latest period within last 6 months
        current_period = int(datetime.today().strftime("%Y%m"))
        six_months_ago = current_period - 6
        df_delinquency_features['recent_delinquency_flag'] = (
            df_delinquency_features['latest_delinquency_period'].fillna(0).astype(int) >= six_months_ago
        ).astype(int)

    # Return with application_id
    return pd.concat([df[['application_id']], df_delinquency_features.drop('application_id', axis=1)], axis=1)



# ## 2. personaldetailssummary

def extract_personal_details_features(df, col='personaldetailssummary'):
    records = []
    for _, row in df.iterrows():
        app_id = row['application_id']
        details = row[col]

        if isinstance(details, dict):
            record = {
                'application_id': app_id,
                'gender': details.get('gender'),
                'birthdate': details.get('birthdate'),
                'dependants': details.get('dependants'),
                'nationality': details.get('nationality'),
                'employerdetail': details.get('employerdetail'),
                'residential_state': details.get('residentialaddress2')  # Extract state from address
            }
            records.append(record)

    if not records:
        return pd.DataFrame(columns=['application_id'])

    # Create DataFrame
    personal_details = pd.DataFrame(records)

    # Convert birthdate to datetime and calculate age
    personal_details['birthdate'] = pd.to_datetime(personal_details['birthdate'], format="%d/%m/%Y", errors='coerce')
    today = pd.Timestamp.today()
    personal_details['age'] = personal_details['birthdate'].apply(lambda x: today.year - x.year if pd.notnull(x) else np.nan)

    # Convert dependants to numeric
    personal_details['dependants'] = pd.to_numeric(personal_details['dependants'], errors='coerce')

    # Drop raw birthdate column (optional)
    personal_details.drop(columns=['birthdate'], inplace=True)

    return personal_details


# ## 3. Guarantor Details
def extract_guarantor_features(df, details_col='guarantordetails', count_col='guarantorcount'):

    # ---- Normalize guarantor details ----
    guarantor_details = []
    for _, row in df.iterrows():
        app_id = row['application_id']
        details = row.get(details_col, {})
        if isinstance(details, dict):
            dob = details.get('guarantordateofbirth')
            if dob == '1900-01-01T00:00:00+01:00':
                dob = None
            guarantor_details.append({
                'application_id': app_id,
                'guarantordateofbirth': dob
            })
        else:
            guarantor_details.append({'application_id': app_id, 'guarantordateofbirth': None})

    guarantor_details_df = pd.DataFrame(guarantor_details)

    # Convert DOB to datetime and calculate age
    guarantor_details_df['guarantordateofbirth'] = pd.to_datetime(
        guarantor_details_df['guarantordateofbirth'], errors='coerce'
    )
    today = pd.to_datetime(datetime.today().date())
    guarantor_details_df['guarantor_age'] = (
        (today - guarantor_details_df['guarantordateofbirth']).dt.days // 365
    )

    # ---- Extract guarantor count ----
    guarantor_count_list = []
    for _, row in df.iterrows():
        app_id = row['application_id']
        count_info = row.get(count_col, {})
        count = None
        if isinstance(count_info, dict):
            count = pd.to_numeric(count_info.get('guarantorssecured'), errors='coerce')
        guarantor_count_list.append({'application_id': app_id, 'guarantor_count': count})

    guarantor_count_df = pd.DataFrame(guarantor_count_list)

    # ---- Merge and create flags ----
    result = guarantor_details_df.merge(guarantor_count_df, on='application_id', how='left')
    result['has_guarantor'] = result['guarantor_count'].apply(
        lambda x: 1 if pd.notna(x) and x > 0 else 0
    )

    return result


# ## 4. Account monthly payment history

def extract_normalized_credit_features(df, history_column='accountmonthlypaymenthistory'):
    dpd_months = {
        "dpd_3m": [f"m{str(i).zfill(2)}" for i in range(22, 25)],
        "dpd_6m": [f"m{str(i).zfill(2)}" for i in range(19, 25)],
        "dpd_12m": [f"m{str(i).zfill(2)}" for i in range(13, 25)],
        "dpd_24m": [f"m{str(i).zfill(2)}" for i in range(1, 25)],
    }

    today = datetime.today()

    def clean_account(account):
        # Clean DPD month fields
        for i in range(1, 25):
            m_key = f"m{str(i).zfill(2)}"
            val = account.get(m_key, None)
            if val in ['XXX', '#']:
                account[m_key] = np.nan
            else:
                try:
                    account[m_key] = int(val)
                except:
                    account[m_key] = np.nan

        # Clean numeric fields
        for amt_field in ['currentbalanceamt', 'openingbalanceamt', 'amountoverdue']:
            val = account.get(amt_field)
            try:
                account[amt_field] = float(str(val).replace(",", ""))
            except:
                account[amt_field] = np.nan

        # Handle invalid date
        if account.get("dateaccountopened") == "1900-01-01":
            account["dateaccountopened"] = None

        return account

    def safe_float(val):
        try:
            return float(str(val).replace(",", ""))
        except:
            return np.nan

    def row_feature_extraction(row):
        # Normalize accounts
        accounts = row.get(history_column, [])
        accounts = [clean_account(acc) for acc in accounts if isinstance(acc, dict)]

        dpd_values = {k: [] for k in dpd_months}
        late_payments_12m = 0
        num_performing = 0
        num_nonperforming = 0
        has_written_off = False
        open_accounts = 0
        new_accounts_6m = 0
        util_ratios = []
        total_amount_overdue = 0
        account_ages = []

        for acc in accounts:
            perf_status = acc.get("performancestatus", "")
            acc_status = acc.get("accountstatus", "")
            date_opened_str = acc.get("dateaccountopened", None)

            if perf_status == "Performing":
                num_performing += 1
            else:
                num_nonperforming += 1

            if acc_status == "WrittenOff":
                has_written_off = True

            if acc_status == "Open":
                open_accounts += 1

            if date_opened_str:
                try:
                    date_opened = datetime.strptime(date_opened_str, "%d/%m/%Y")
                    acc_age_months = (today.year - date_opened.year) * 12 + (today.month - date_opened.month)
                    account_ages.append(acc_age_months)
                    if acc_age_months <= 6:
                        new_accounts_6m += 1
                except:
                    pass

            curr_bal = safe_float(acc.get("currentbalanceamt"))
            open_bal = safe_float(acc.get("openingbalanceamt"))
            if curr_bal is not None and open_bal is not None and open_bal > 0:
                util_ratios.append(curr_bal / open_bal)

            overdue = safe_float(acc.get("amountoverdue"))
            if overdue:
                total_amount_overdue += overdue

            for k, months in dpd_months.items():
                for m in months:
                    val = acc.get(m, '#')
                    try:
                        dpd_val = int(val)
                        dpd_values[k].append(dpd_val)
                        if k == 'dpd_12m' and dpd_val > 0:
                            late_payments_12m += 1
                    except (ValueError, TypeError):
                        continue

        return pd.Series({
            "dpd_3m": max(dpd_values["dpd_3m"], default=0),
            "dpd_6m": max(dpd_values["dpd_6m"], default=0),
            "dpd_12m": max(dpd_values["dpd_12m"], default=0),
            "dpd_24m": max(dpd_values["dpd_24m"], default=0),
            "late_payments_12m_count": late_payments_12m,
            "num_performing_accounts": num_performing,
            "num_nonperforming_accounts": num_nonperforming,
            "has_written_off_account": int(has_written_off),
            "num_open_accounts": open_accounts,
            "num_new_accounts_6m": new_accounts_6m,
            "avg_utilization_ratio": np.nanmean(util_ratios) if util_ratios else np.nan,
            "total_amount_overdue": total_amount_overdue,
            "avg_account_age_months": np.mean(account_ages) if account_ages else np.nan,
            "max_account_age_months": max(account_ages) if account_ages else np.nan,
        })

    # Apply feature extraction
    features_df = df.apply(row_feature_extraction, axis=1)

    # Combine with application_id
    result = pd.concat([df[["application_id"]].reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    return result

# ## 5. Account Ratings
def extract_account_rating_features(df, col='accountrating'):
    product_mapping = {
        'creditcard': 'credit_cards',
        'personalloan': 'personal_loans',
        'autoloan': 'auto_loans',
        'homeloan': 'home_loans',
        'retailaccounts': 'retail_accounts',
        'telecomaccounts': 'telecom_accounts',
        'studyloan': 'study_loans',
        'jointloan': 'joint_loans',
        'otheraccounts': 'other_accounts'
    }

    def process_row(d):
        result = {}

        if not isinstance(d, dict):
            # Return all zeros
            base = {f'num_good_{v}': 0 for v in product_mapping.values()}
            base.update({f'num_bad_{v}': 0 for v in product_mapping.values()})
            base.update({
                'total_good_accounts': 0,
                'total_bad_accounts': 0,
                'total_accounts': 0,
                'share_bad_accounts': 0
            })
            return base

        total_good = 0
        total_bad = 0

        for key, val in d.items():
            try:
                count = int(val)
            except (ValueError, TypeError):
                count = 0

            for prefix, readable in product_mapping.items():
                if key.startswith(f'noof{prefix}'):
                    if key.endswith('good'):
                        result[f'num_good_{readable}'] = result.get(f'num_good_{readable}', 0) + count
                        total_good += count
                    elif key.endswith('bad'):
                        result[f'num_bad_{readable}'] = result.get(f'num_bad_{readable}', 0) + count
                        total_bad += count

        # Fill missing product vars with 0
        for v in product_mapping.values():
            result.setdefault(f'num_good_{v}', 0)
            result.setdefault(f'num_bad_{v}', 0)

        result['total_good_accounts'] = total_good
        result['total_bad_accounts'] = total_bad
        result['total_accounts'] = total_good + total_bad
        result['share_bad_accounts'] = round(total_bad / (total_good + total_bad), 3) if (total_good + total_bad) > 0 else 0

        return result

    features = df[col].apply(process_row).apply(pd.Series)
    return pd.concat([df['application_id'], features], axis=1)


# ## 6. credit account summary
def extract_credit_account_summary_features(df, col='creditaccountsummary'):
    def parse_amount(val):
        try:
            return float(str(val).replace(',', '').replace(' ', '').replace('-', '0'))
        except:
            return 0.0

    def process_row(d):
        result = {}

        if not isinstance(d, dict):
            return {
                'credit_rating': 0,
                'amount_in_arrear': 0.0,
                'total_accounts': 0,
                'total_accounts_in_arrear': 0,
                'total_outstanding_debt': 0.0,
                'monthly_instalment_total': 0.0,
                'judgement_amount_total': 0.0,
                'num_judgements': 0,
                'num_dishonoured': 0,
                'num_good_condition_accounts': 0,
                'arrear_per_account': 0.0,
                'debt_to_instalment_ratio': 0.0,
                'arrear_rate': 0.0,
                'has_judgement': 0,
                'has_dishonoured': 0
            }

        # Raw values
        result['credit_rating'] = int(d.get('rating', 0))
        result['amount_in_arrear'] = parse_amount(d.get('amountarrear', 0))
        result['total_accounts'] = int(d.get('totalaccounts', 0))
        result['total_accounts_in_arrear'] = int(d.get('totalaccountarrear', 0))
        result['total_outstanding_debt'] = parse_amount(d.get('totaloutstandingdebt', 0))
        result['monthly_instalment_total'] = parse_amount(d.get('totalmonthlyinstalment', 0))
        result['judgement_amount_total'] = parse_amount(d.get('totaljudgementamount', 0))
        result['num_judgements'] = int(d.get('totalnumberofjudgement', 0))
        result['num_dishonoured'] = int(d.get('totalnumberofdishonoured', 0))
        result['num_good_condition_accounts'] = int(d.get('totalaccountingodcondition', 0))

        # Derived
        result['arrear_per_account'] = round(result['amount_in_arrear'] / result['total_accounts'], 2) if result['total_accounts'] else 0.0
        result['debt_to_instalment_ratio'] = round(result['total_outstanding_debt'] / result['monthly_instalment_total'], 2) if result['monthly_instalment_total'] else 0.0
        result['arrear_rate'] = round(result['total_accounts_in_arrear'] / result['total_accounts'], 2) if result['total_accounts'] else 0.0
        result['has_judgement'] = int(result['num_judgements'] > 0)
        result['has_dishonoured'] = int(result['num_dishonoured'] > 0)

        return result

    features = df[col].apply(process_row).apply(pd.Series)
    return pd.concat([df['application_id'], features], axis=1)


# ## 7. Credit Agreementy summmary

def safe_float(val):
    try:
        return float(str(val).replace(",", "").strip())
    except:
        return np.nan

def extract_credit_agreement_summary_features(df, col='creditagreementsummary'):
    today = datetime.today()

    def process_credit_agreement(agreements):
        if not isinstance(agreements, list):
            return pd.Series({})

        num_open_accounts = 0
        num_written_off_accounts = 0
        num_new_accounts_6m = 0
        num_personal_loans = 0
        num_overdraft_accounts = 0
        num_installment_accounts = 0

        total_current_balance = 0
        total_opening_balance = 0
        total_amount_overdue = 0
        max_amount_overdue = 0
        utilisation_ratios = []
        instalment_amounts = []
        account_ages = []

        high_utilization_flag = 0
        high_overdue_flag = 0

        for acc in agreements:
            status = str(acc.get('accountstatus', '') or '')
            perf_status = str(acc.get('performancestatus', '') or '')
            indicator = str(acc.get('indicatordescription', '') or '')

            # Status-based features
            if status == 'Open':
                num_open_accounts += 1
            if status == 'WrittenOff':
                num_written_off_accounts += 1

            # Loan type classification
            if 'Personal' in indicator:
                num_personal_loans += 1
            if 'Overdraft' in indicator:
                num_overdraft_accounts += 1
            if 'Installment' in indicator:
                num_installment_accounts += 1

            # Balances & ratios
            current_balance = safe_float(acc.get('currentbalanceamt'))
            opening_balance = safe_float(acc.get('openingbalanceamt'))
            overdue = safe_float(acc.get('amountoverdue'))
            instalment_amount = safe_float(acc.get('instalmentamount'))

            total_current_balance += current_balance
            total_opening_balance += opening_balance
            total_amount_overdue += overdue
            max_amount_overdue = max(max_amount_overdue, overdue)

            if opening_balance > 0:
                util = current_balance / opening_balance
                utilisation_ratios.append(util)
                if util > 0.9:
                    high_utilization_flag = 1

            if overdue > 50000:
                high_overdue_flag = 1

            if instalment_amount > 0:
                instalment_amounts.append(instalment_amount)

            # Account age calculation
            date_opened_str = acc.get('dateaccountopened')
            if date_opened_str:
                try:
                    date_opened = datetime.strptime(date_opened_str, "%d/%m/%Y")
                    age_months = (today.year - date_opened.year) * 12 + (today.month - date_opened.month)
                    account_ages.append(age_months)
                    if age_months <= 6:
                        num_new_accounts_6m += 1
                except:
                    pass

        # Derived metrics
        avg_utilization_ratio = np.nanmean(utilisation_ratios) if utilisation_ratios else np.nan
        avg_instalment_amount = np.nanmean(instalment_amounts) if instalment_amounts else np.nan
        avg_account_age_months = np.nanmean(account_ages) if account_ages else np.nan
        max_account_age_months = max(account_ages) if account_ages else np.nan
        overdue_to_balance_ratio = total_amount_overdue / total_current_balance if total_current_balance > 0 else 0

        # Flags
        has_written_off = 1 if num_written_off_accounts > 0 else 0
        multiple_written_off_flag = 1 if num_written_off_accounts > 1 else 0
        recent_opening_spike_flag = 1 if num_new_accounts_6m > 2 else 0

        return pd.Series({
            "num_open_accounts": num_open_accounts,
            "num_written_off_accounts": num_written_off_accounts,
            "num_new_accounts_6m": num_new_accounts_6m,
            "total_current_balance": total_current_balance,
            "total_opening_balance": total_opening_balance,
            "total_amount_overdue": total_amount_overdue,
            "max_amount_overdue": max_amount_overdue,
            "avg_utilization_ratio": avg_utilization_ratio,
            "avg_account_age_months": avg_account_age_months,
            "max_account_age_months": max_account_age_months,
            "num_personal_loans": num_personal_loans,
            "num_overdraft_accounts": num_overdraft_accounts,
            "num_installment_accounts": num_installment_accounts,
            "avg_instalment_amount": avg_instalment_amount,
            "overdue_to_balance_ratio": overdue_to_balance_ratio,
            "has_written_off": has_written_off,
            "high_utilization_flag": high_utilization_flag,
            "high_overdue_flag": high_overdue_flag,
            "multiple_written_off_flag": multiple_written_off_flag,
            "recent_opening_spike_flag": recent_opening_spike_flag
        })

    features_df = df.apply(lambda row: process_credit_agreement(row[col]), axis=1)
    return pd.concat([df[['application_id']], features_df], axis=1)

# ## 8.Employment history

def extract_employment_history_features(df, col='employmenthistory'):
    today = datetime.today()
    
    def process_employment(history):
        if not isinstance(history, list):
            return pd.Series({})

        num_employers = len(history)
        unique_occupations = set()
        num_military_or_defense_jobs = 0
        last_update_date = None
        current_occupation = None

        for entry in history:
            occupation = str(entry.get('occupation', '') or '').strip().upper()
            employer = str(entry.get('employerdetail', '') or '').strip().upper()
            update_date_str = entry.get('updatedate') or entry.get('updateondate')

            if occupation:
                unique_occupations.add(occupation)
            if 'ARMY' in occupation or 'DEFENCE' in occupation or 'MILITARY' in occupation or 'NAVY' in employer:
                num_military_or_defense_jobs += 1

            if update_date_str:
                try:
                    update_date = datetime.strptime(update_date_str, "%d/%m/%Y")
                    if (last_update_date is None) or (update_date > last_update_date):
                        last_update_date = update_date
                except:
                    pass

        # Current occupation → last record in the list
        if history and isinstance(history[-1], dict):
            current_occupation = str(history[-1].get('occupation', '') or '').upper()

        years_since_last_update = (today - last_update_date).days / 365 if last_update_date else np.nan
        is_government_job = 1 if 'PUBLIC SERVANTS' in current_occupation else 0
        is_high_stability_job = 1 if is_government_job or num_military_or_defense_jobs > 0 else 0

        return pd.Series({
            # "num_employers": num_employers,
            "num_unique_occupations": len(unique_occupations),
            "num_military_or_defense_jobs": num_military_or_defense_jobs,
            "years_since_last_update": years_since_last_update,
            "current_occupation": current_occupation,
            "is_government_job": is_government_job,
            "is_high_stability_job": is_high_stability_job
        })

    features_df = df.apply(lambda row: process_employment(row[col]), axis=1)
    return pd.concat([df[['application_id']], features_df], axis=1)


# ## 9. Enquiry details
def extract_enquiry_features(df, enquiry_details_col='enquirydetails', enquiry_history_col='enquiryhistorytop'):
    today = datetime.today()

    def process_enquiry_features(row):
        enquiry_details = row.get(enquiry_details_col, {})
        enquiry_history = row.get(enquiry_history_col, [])

        # ---- From enquirydetails ----
        latest_matching_rate = float(enquiry_details.get('matchingrate', 0)) if enquiry_details else 0
        latest_product_id = enquiry_details.get('productid', None)
        has_recent_enquiry = 1 if enquiry_details else 0

        # ---- From enquiryhistorytop ----
        if not enquiry_history:
            return pd.Series({
                "latest_matching_rate": latest_matching_rate,
                "latest_product_id": latest_product_id,
                "has_recent_enquiry": has_recent_enquiry,
                "num_total_enquiries": 0,
                "num_enquiries_last_30d": 0,
                "num_enquiries_last_90d": 0,
                "days_since_last_enquiry": np.nan,
                "num_unique_subscribers": 0,
                "recent_enquiry_flag": 0,
                "high_enquiry_risk_bucket": "Low"
            })

        dates = []
        subscribers = []
        for e in enquiry_history:
            date_str = e.get('daterequested')
            if date_str:
                try:
                    date_obj = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
                    dates.append(date_obj)
                except:
                    pass
            subscribers.append(e.get('subscribername'))

        num_total = len(enquiry_history)
        unique_subs = len(set(subscribers))

        if dates:
            last_date = max(dates)
            days_since_last = (today - last_date).days
        else:
            days_since_last = np.nan

        num_last_30d = sum((today - d).days <= 30 for d in dates)
        num_last_90d = sum((today - d).days <= 90 for d in dates)

        # Flags & Risk bucket
        recent_flag = 1 if num_last_30d > 0 else 0
        if num_last_90d <= 3:
            bucket = "Low"
        elif num_last_90d <= 6:
            bucket = "Medium"
        else:
            bucket = "High"

        return pd.Series({
            "latest_matching_rate": latest_matching_rate,
            "latest_product_id": latest_product_id,
            "has_recent_enquiry": has_recent_enquiry,
            "num_total_enquiries": num_total,
            "num_enquiries_last_30d": num_last_30d,
            "num_enquiries_last_90d": num_last_90d,
            "days_since_last_enquiry": days_since_last,
            "num_unique_subscribers": unique_subs,
            "recent_enquiry_flag": recent_flag,
            "high_enquiry_risk_bucket": bucket
        })

    features_df = df.apply(process_enquiry_features, axis=1)
    return pd.concat([df[['application_id']], features_df], axis=1)

# ## 10. Telephone History
def extract_telephone_history_features(df, col='telephonehistory'):
    
    today = datetime.today()
    
    features = []
    
    for idx, row in df.iterrows():
        tel_history = row.get(col, [])
        
        home_numbers = set()
        mobile_numbers = set()
        update_dates = []
        
        for record in tel_history:
            home = record.get('hometelephonenumber', None)
            mobile = record.get('mobiletelephonenumber', None)
            update_date_str = record.get('homenoupdatedondate', None)
            
            if home and home != 'XXX':
                home_numbers.add(home)
            if mobile and mobile != 'XXX':
                mobile_numbers.add(mobile)
            
            if update_date_str and update_date_str != 'XXX':
                try:
                    update_date = datetime.strptime(update_date_str, '%d/%m/%Y')
                    update_dates.append(update_date)
                except:
                    pass
        
        # Compute numeric features
        num_unique_home_phones = len(home_numbers)
        num_unique_mobile_phones = len(mobile_numbers)
        num_phone_updates = len(update_dates)
        
        # Date-based features
        days_since_last_phone_update = None
        days_between_first_and_last_update = None
        if update_dates:
            last_update = max(update_dates)
            first_update = min(update_dates)
            days_since_last_phone_update = (today - last_update).days
            days_between_first_and_last_update = (last_update - first_update).days
        
        # Flags
        has_multiple_mobile_numbers = 1 if num_unique_mobile_phones > 1 else 0
        has_recent_update_flag = 1 if days_since_last_phone_update is not None and days_since_last_phone_update <= 90 else 0
        
        features.append({
            'num_unique_home_phones': num_unique_home_phones,
            'num_unique_mobile_phones': num_unique_mobile_phones,
            'num_phone_updates': num_phone_updates,
            'days_since_last_phone_update': days_since_last_phone_update,
            'days_between_first_and_last_update': days_between_first_and_last_update,
            'has_multiple_mobile_numbers': has_multiple_mobile_numbers,
            'has_recent_update_flag': has_recent_update_flag
        })
    
    features_df = pd.DataFrame(features)
    return pd.concat([df[['application_id']], features_df], axis=1)


# ## Merge All Variables

def extract_all_credit_features(df):
    """
    Combine all feature extraction functions into a single function.
    Input: df (DataFrame with all raw JSON fields)
    Output: DataFrame with application_id and all engineered features.
    """

    # 1. Account Monthly Payment History
    df_payment_history_features = extract_normalized_credit_features(df)

    # 2. Credit Agreement Summary
    credit_agreement_features = extract_credit_agreement_summary_features(df, col='creditagreementsummary')

    # 3. Account Rating
    account_rating_features = extract_account_rating_features(df, col='accountrating')

    # 4. Credit Account Summary
    credit_account_summary_features = extract_credit_account_summary_features(df, col='creditaccountsummary')

    # 5. Employment History
    employment_history_features = extract_employment_history_features(df, col='employmenthistory')

    # 6. Enquiry Details & History
    enquiry_features = extract_enquiry_features(df) #details_col='enquirydetails', history_col='enquiryhistorytop')

    # 7. Telephone History
    telephone_history_features = extract_telephone_history_features(df, col='telephonehistory')

    # 8. Delinquency Information
    delinquency_features = extract_delinquency_features(df, col='deliquencyinformation')

    # 9. Personal Details Summary
    personal_details_features = extract_personal_details_features(df, col='personaldetailssummary')

    # 10. Guarantor Details
    guarantor_features = extract_guarantor_features(df)

    # Combine all
    all_features = [
        df_payment_history_features,
        credit_agreement_features,
        account_rating_features,
        credit_account_summary_features,
        employment_history_features,
        enquiry_features,
        telephone_history_features,
        delinquency_features,
        personal_details_features,
        guarantor_features
    ]

    # Merge all on application_id
    final_features = all_features[0]
    for f in all_features[1:]:
        final_features = final_features.merge(f, on='application_id', how='left')

    return final_features