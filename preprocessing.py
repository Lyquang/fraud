"""
Data Preprocessing Module for Fraud Detection
==============================================
Module chuẩn hóa cho tất cả các thí nghiệm
Bao gồm: Feature Engineering, Imputation, Scaling, Encoding
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Feature Engineering transformer với fit/transform pattern
    Tránh data leakage bằng cách học statistics từ training data
    """
    def __init__(self,
                 freq_cols=('P_emaildomain','R_emaildomain','card1','card2','card3','card4',
                            'addr1','ProductCD','M1','M2','M3','M4','M5','M6','M7','M8','M9'),
                 agg_keys_card=('card1',),
                 agg_keys_uid=('card1','addr1'),
                 agg_keys_email=('P_emaildomain',),
                 iqr_clip=True):
        self.freq_cols = freq_cols
        self.agg_keys_card = agg_keys_card
        self.agg_keys_uid = agg_keys_uid
        self.agg_keys_email = agg_keys_email
        self.iqr_clip = iqr_clip
        
        # Learned from training data
        self.freq_maps_ = {}
        self.card_amt_stats_ = None
        self.uid_amt_stats_ = None
        self.email_amt_stats_ = None
        self.train_amt_median_ = None

    @staticmethod
    def _safe_div(a, b):
        """Safe division avoiding divide by zero"""
        return a / np.where(b == 0, 1, b)

    @staticmethod
    def _top_free_email(domain: str) -> int:
        """Check if email is from free provider"""
        if pd.isna(domain):
            return 0
        domain = str(domain).lower()
        free = {'gmail.com','googlemail.com','yahoo.com','yahoo.co.jp','hotmail.com',
                'outlook.com','live.com','aol.com','icloud.com','protonmail.com',
                'msn.com','hotmail.co.uk'}
        return int(domain in free)

    @staticmethod
    def _get_tld(domain):
        """Extract top-level domain"""
        if pd.isna(domain):
            return 'unknown'
        parts = str(domain).lower().split('.')
        return parts[-1] if len(parts) >= 2 else 'unknown'

    @staticmethod
    def _get_root_domain(domain):
        """Extract root domain"""
        if pd.isna(domain):
            return 'unknown'
        parts = str(domain).lower().split('.')
        return '.'.join(parts[-2:]) if len(parts) >= 2 else parts[0]

    @staticmethod
    def _get_device_brand(dev):
        """Extract device brand from DeviceInfo"""
        if pd.isna(dev):
            return 'unknown'
        s = str(dev).lower()
        if 'samsung' in s: return 'samsung'
        if 'huawei' in s: return 'huawei'
        if 'xiaomi' in s or 'mi ' in s: return 'xiaomi'
        if 'lg' in s: return 'lg'
        if 'motorola' in s or 'moto' in s: return 'motorola'
        if 'oneplus' in s: return 'oneplus'
        if 'apple' in s or 'mac' in s or 'ios' in s or 'iphone' in s or 'ipad' in s: return 'apple'
        if 'windows' in s: return 'windows'
        if 'android' in s: return 'android'
        return 'other'

    @staticmethod
    def _parse_os(id30):
        """Parse OS from id_30"""
        if pd.isna(id30):
            return 'unknown'
        s = str(id30).lower()
        if 'windows' in s: return 'windows'
        if 'mac' in s or 'ios' in s: return 'apple'
        if 'android' in s: return 'android'
        if 'linux' in s: return 'linux'
        return 'other'

    @staticmethod
    def _parse_browser(id31):
        """Parse browser from id_31"""
        if pd.isna(id31):
            return 'unknown'
        s = str(id31).lower()
        if 'chrome' in s: return 'chrome'
        if 'safari' in s and 'mobile' in s: return 'mobile_safari'
        if 'safari' in s: return 'safari'
        if 'edge' in s: return 'edge'
        if 'firefox' in s: return 'firefox'
        if 'ie' in s or 'internet explorer' in s: return 'ie'
        if 'opera' in s: return 'opera'
        return 'other'

    def fit(self, X, y=None):
        """Learn statistics from training data"""
        X = X.copy()

        # Median TransactionAmt for fallback
        if 'TransactionAmt' in X.columns:
            self.train_amt_median_ = float(np.nanmedian(X['TransactionAmt'].values))
        else:
            self.train_amt_median_ = 0.0

        # Frequency encoding maps
        for col in self.freq_cols:
            if col in X.columns:
                vc = X[col].astype(str).fillna('nan').value_counts(dropna=False)
                self.freq_maps_[col] = vc.to_dict()

        # Aggregate statistics
        if set(self.agg_keys_card).issubset(X.columns) and 'TransactionAmt' in X.columns:
            g = X.groupby(list(self.agg_keys_card))['TransactionAmt']
            self.card_amt_stats_ = pd.DataFrame({
                'mean': g.mean(),
                'std': g.std().fillna(0),
                'median': g.median(),
                'count': g.size()
            })

        if set(self.agg_keys_uid).issubset(X.columns) and 'TransactionAmt' in X.columns:
            g = X.groupby(list(self.agg_keys_uid))['TransactionAmt']
            self.uid_amt_stats_ = pd.DataFrame({
                'mean': g.mean(),
                'std': g.std().fillna(0),
                'median': g.median(),
                'count': g.size()
            })

        if set(self.agg_keys_email).issubset(X.columns) and 'TransactionAmt' in X.columns:
            g = X.groupby(list(self.agg_keys_email))['TransactionAmt']
            self.email_amt_stats_ = pd.DataFrame({
                'mean': g.mean(),
                'std': g.std().fillna(0),
                'median': g.median(),
                'count': g.size()
            })

        return self

    def transform(self, X):
        """Apply feature engineering transformations"""
        X = X.copy()

        # 1. Transaction Amount Features
        if 'TransactionAmt' in X.columns:
            amt = X['TransactionAmt'].astype(float)
            X['TransactionAmt_log'] = np.log1p(np.clip(amt, 0, None))
            X['TransactionAmt_sqrt'] = np.sqrt(np.clip(amt, 0, None))
            X['amt_round_0'] = amt.round(0)
            X['amt_round_1'] = amt.round(1)
            X['amt_decimal'] = (amt - amt.astype(np.int64)).abs()
            X['amt_is_whole'] = (X['amt_decimal'] == 0).astype(int)
            
            X['TransactionAmt_bin'] = pd.cut(
                amt, bins=[-np.inf, 10, 50, 100, 500, 2000, np.inf],
                labels=['vlow','low','med','high','vhigh','extreme']
            ).astype(str)

            if self.iqr_clip:
                q1, q3 = np.nanpercentile(amt, [25, 75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
                X['TransactionAmt_clip'] = np.clip(amt, lo, hi)
            else:
                X['TransactionAmt_clip'] = amt

        # 2. Email Domain Features
        for col in ['P_emaildomain','R_emaildomain']:
            if col in X.columns:
                s = X[col]
                X[f'{col}_root'] = s.apply(self._get_root_domain)
                X[f'{col}_tld'] = s.apply(self._get_tld)
                X[f'{col}_is_free'] = s.apply(self._top_free_email).astype(int)

        if 'P_emaildomain' in X.columns:
            def get_email_type(domain):
                if pd.isna(domain): return 'unknown'
                d = str(domain).lower()
                commons = {'gmail.com','yahoo.com','hotmail.com','outlook.com','icloud.com'}
                if d in commons: return 'common'
                if 'edu' in d: return 'edu'
                if 'gov' in d: return 'gov'
                if 'biz' in d or 'corp' in d or 'company' in d or 'business' in d:
                    return 'business'
                return 'other'
            X['P_emaildomain_type'] = X['P_emaildomain'].apply(get_email_type).astype(str)

        # 3. Card Features
        if 'card1' in X.columns and 'card2' in X.columns:
            X['card1_card2'] = X['card1'].astype(str) + '_' + X['card2'].astype(str)
        if 'card4' in X.columns:
            X['is_visa'] = (X['card4'] == 'visa').astype(int)
            X['is_mastercard'] = (X['card4'] == 'mastercard').astype(int)

        # 4. Address Features
        addr_cols = [c for c in X.columns if 'addr' in c.lower()]
        if len(addr_cols) >= 2:
            X['addr_match'] = (X[addr_cols[0]] == X[addr_cols[1]]).astype(int)

        # 5. Device Features
        if 'DeviceInfo' in X.columns:
            def get_device_type(device):
                if pd.isna(device): return 'unknown'
                d = str(device).lower()
                if 'windows' in d: return 'windows'
                if 'mac' in d or 'ios' in d: return 'apple'
                if 'android' in d: return 'android'
                return 'other'
            X['device_type'] = X['DeviceInfo'].apply(get_device_type).astype(str)
            X['device_brand'] = X['DeviceInfo'].apply(self._get_device_brand).astype(str)

        if 'id_30' in X.columns:
            X['device_os'] = X['id_30'].apply(self._parse_os).astype(str)
        if 'id_31' in X.columns:
            X['device_browser'] = X['id_31'].apply(self._parse_browser).astype(str)

        # 6. Time Features
        if 'TransactionDT' in X.columns:
            hours = (X['TransactionDT'] / 3600) % 24
            days = (X['TransactionDT'] // (3600*24)).astype(np.int64)

            X['hour'] = hours
            X['day_index'] = days
            X['dayofweek'] = (days % 7).astype(np.int8)
            X['is_weekend'] = X['dayofweek'].isin([5,6]).astype(int)
            X['is_night'] = ((hours >= 22) | (hours <= 6)).astype(int)
            X['is_business_hours'] = ((hours >= 9) & (hours <= 18)).astype(int)

            X['hour_sin'] = np.sin(2*np.pi*hours/24)
            X['hour_cos'] = np.cos(2*np.pi*hours/24)
            X['dow_sin'] = np.sin(2*np.pi*X['dayofweek']/7)
            X['dow_cos'] = np.cos(2*np.pi*X['dayofweek']/7)

            X['week'] = (X['TransactionDT'] // (3600*24*7)).astype(np.int32)
            X['month_rel'] = (X['TransactionDT'] // (3600*24*30)).astype(np.int32)

        # 7. V/C/D/M Features Statistics
        for prefix in ['V','C','D','M']:
            cols = [c for c in X.columns if c.startswith(prefix)]
            if cols:
                num_cols_prefix = [c for c in cols if np.issubdtype(X[c].dtype, np.number)]
                if num_cols_prefix:
                    X[f'{prefix}_mean'] = X[num_cols_prefix].mean(axis=1)
                    X[f'{prefix}_std'] = X[num_cols_prefix].std(axis=1)
                X[f'{prefix}_nulls'] = X[cols].isnull().sum(axis=1)

        # 8. Frequency Encoding
        for col, mapping in self.freq_maps_.items():
            if col in X.columns:
                X[f'{col}_freq'] = X[col].astype(str).fillna('nan').map(mapping).fillna(0).astype(np.int64)

        # 9. Group Aggregates
        def _merge_group_stats(df, key_cols, stats_df, name):
            if isinstance(stats_df, pd.DataFrame) and not stats_df.empty:
                stats = stats_df.reset_index().copy()
                stats.columns = list(key_cols) + [f'{name}_mean', f'{name}_std', f'{name}_median', f'{name}_count']
                return df.merge(stats, how='left', on=list(key_cols))
            return df

        if isinstance(self.card_amt_stats_, pd.DataFrame) and not self.card_amt_stats_.empty and set(self.agg_keys_card).issubset(X.columns):
            X = _merge_group_stats(X, self.agg_keys_card, self.card_amt_stats_, 'card_amt')
        
        if isinstance(self.uid_amt_stats_, pd.DataFrame) and not self.uid_amt_stats_.empty and set(self.agg_keys_uid).issubset(X.columns):
            X = _merge_group_stats(X, self.agg_keys_uid, self.uid_amt_stats_, 'uid_amt')
        
        if isinstance(self.email_amt_stats_, pd.DataFrame) and not self.email_amt_stats_.empty and set(self.agg_keys_email).issubset(X.columns):
            X = _merge_group_stats(X, self.agg_keys_email, self.email_amt_stats_, 'email_amt')

        # Ratios vs group means
        if 'TransactionAmt' in X.columns:
            if 'card_amt_mean' in X.columns:
                X['amt_vs_card_mean'] = self._safe_div(X['TransactionAmt'], X['card_amt_mean'])
            if 'uid_amt_mean' in X.columns:
                X['amt_vs_uid_mean'] = self._safe_div(X['TransactionAmt'], X['uid_amt_mean'])
            if 'email_amt_mean' in X.columns:
                X['amt_vs_email_mean'] = self._safe_div(X['TransactionAmt'], X['email_amt_mean'])
            
            # Fallback
            if 'amt_vs_card_mean' not in X.columns:
                X['amt_vs_card_mean'] = self._safe_div(X['TransactionAmt'], (self.train_amt_median_ if self.train_amt_median_ else 1))
            if 'amt_vs_uid_mean' not in X.columns:
                X['amt_vs_uid_mean'] = self._safe_div(X['TransactionAmt'], (self.train_amt_median_ if self.train_amt_median_ else 1))

        # 10. Interactions
        if {'TransactionAmt','card1'}.issubset(X.columns) and 'card_amt_mean' in X.columns:
            X['amt_minus_card_mean'] = X['TransactionAmt'] - X['card_amt_mean']
        if {'TransactionAmt','addr1'}.issubset(X.columns) and 'uid_amt_mean' in X.columns and 'card1' in X.columns:
            X['amt_minus_uid_mean'] = X['TransactionAmt'] - X['uid_amt_mean']

        # 11. Ensure string type for categorical features
        cat_features = [
            'TransactionAmt_bin','P_emaildomain_type','card1_card2','device_type',
            'device_brand','device_os','device_browser','P_emaildomain_root','P_emaildomain_tld',
            'R_emaildomain_root','R_emaildomain_tld'
        ]
        for feat in cat_features:
            if feat in X.columns:
                X[feat] = X[feat].astype(str)

        return X


class CustomImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputation strategy:
    - Fill -1 for specific columns (addr, card, D9)
    - Fill median for numerical columns
    - Fill 'unknown' for categorical columns
    """
    def __init__(self, fill_minus_1_cols, fill_median_cols, cat_cols):
        self.fill_minus_1_cols = fill_minus_1_cols
        self.fill_median_cols = fill_median_cols
        self.cat_cols = cat_cols
        self.median_values = {}
        
    def fit(self, X, y=None):
        for col in self.fill_median_cols:
            if col in X.columns:
                self.median_values[col] = X[col].median()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        for col in self.fill_minus_1_cols:
            if col in X.columns:
                X[col] = X[col].fillna(-1)
        
        for col in self.fill_median_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.median_values[col])
        
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna('unknown')
        
        return X


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers using percentile bounds
    """
    def __init__(self, columns, lower_percentile=1, upper_percentile=99):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.bounds = {}
    
    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                lower_bound = np.percentile(X[col], self.lower_percentile)
                upper_bound = np.percentile(X[col], self.upper_percentile)
                self.bounds[col] = (lower_bound, upper_bound)
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns and col in self.bounds:
                lower_bound, upper_bound = self.bounds[col]
                X[col] = np.clip(X[col], lower_bound, upper_bound)
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log transformation to reduce skewness
    """
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = np.log1p(X[col] - X[col].min() + 1)
        return X


def get_column_groups(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target if exists
    if 'isFraud' in num_cols:
        num_cols.remove('isFraud')
    
    # Define fill strategies
    fill_minus_1_cols = []
    if 'D9' in df.columns:
        fill_minus_1_cols.append('D9')
    
    # Add addr columns
    addr_cols = [col for col in df.columns if 'addr' in col and df[col].dtype in ['float64', 'int64']]
    fill_minus_1_cols.extend(addr_cols)
    
    # Add card columns
    card_cols = [col for col in df.columns if 'card' in col and df[col].dtype in ['float64', 'int64']]
    fill_minus_1_cols.extend(card_cols)
    
    # Remove duplicates
    fill_minus_1_cols = list(set(fill_minus_1_cols))
    fill_median_cols = [col for col in num_cols if col not in fill_minus_1_cols]
    
    return {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'fill_minus_1_cols': fill_minus_1_cols,
        'fill_median_cols': fill_median_cols
    }


def create_preprocessing_pipeline(col_groups, include_fe=True, include_smote=False, 
                                  feature_selector=None, smote_strategy=0.3):
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    
    steps = []
    
    # Feature Engineering
    if include_fe:
        steps.append(('feature_engineering', FeatureEngineering()))
    
    # Imputation
    steps.append(('custom_imputer', CustomImputer(
        col_groups['fill_minus_1_cols'],
        col_groups['fill_median_cols'],
        col_groups['cat_cols']
    )))
    
    # Outlier Clipping
    steps.append(('outlier_clipper', OutlierClipper(col_groups['num_cols'])))
    
    # Log Transformation for skewed features
    # Note: Skewed columns should be calculated after FE if include_fe=True
    steps.append(('log_transformer', LogTransformer([])))  # Empty by default
    
    # Scaling and Encoding
    steps.append(('column_transformer', ColumnTransformer([
        ('num_scaler', RobustScaler(), 
         col_groups['fill_median_cols'] + col_groups['fill_minus_1_cols']),
        ('cat_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
         col_groups['cat_cols'])
    ], remainder='passthrough')))
    
    # Feature Selection
    if feature_selector is not None:
        steps.append(('feature_selector', feature_selector))
    
    # SMOTE
    if include_smote:
        steps.append(('smote', SMOTE(random_state=42, sampling_strategy=smote_strategy)))
        return ImbPipeline(steps)
    
    return Pipeline(steps)


def update_column_groups_after_fe(X_sample, fe_transformer, original_col_groups):
    """
    Update column groups sau khi apply Feature Engineering
    
    Args:
        X_sample: Sample data để test FE
        fe_transformer: FeatureEngineering transformer đã fit
        original_col_groups: Column groups gốc
        
    Returns:
        dict: Updated column groups
    """
    # Apply FE to sample
    X_fe = fe_transformer.transform(X_sample)
    
    # Get new column groups
    num_cols_fe = X_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols_fe = X_fe.select_dtypes(include=['object']).columns.tolist()
    
    # Keep original fill strategies but update with new columns
    fill_minus_1_cols_fe = []
    if 'D9' in X_fe.columns:
        fill_minus_1_cols_fe.append('D9')
    
    addr_cols = [col for col in X_fe.columns if 'addr' in col and X_fe[col].dtype in ['float64', 'int64']]
    fill_minus_1_cols_fe.extend(addr_cols)
    
    card_cols = [col for col in X_fe.columns if 'card' in col and X_fe[col].dtype in ['float64', 'int64']]
    fill_minus_1_cols_fe.extend(card_cols)
    
    fill_minus_1_cols_fe = list(set(fill_minus_1_cols_fe))
    fill_median_cols_fe = [col for col in num_cols_fe if col not in fill_minus_1_cols_fe]
    
    return {
        'num_cols': num_cols_fe,
        'cat_cols': cat_cols_fe,
        'fill_minus_1_cols': fill_minus_1_cols_fe,
        'fill_median_cols': fill_median_cols_fe
    }


def get_skewed_columns(X, threshold=0.75):
    """
    Identify skewed numerical columns
    
    Args:
        X: DataFrame
        threshold: Skewness threshold
        
    Returns:
        list: List of skewed column names
    """
    skewed_cols = []
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    for col in num_cols:
        if col in X.columns:
            skewness = X[col].skew()
            if abs(skewness) > threshold:
                skewed_cols.append(col)
    
    return skewed_cols
