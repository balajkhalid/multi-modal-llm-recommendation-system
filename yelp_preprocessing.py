import dask.dataframe as dd
import dask.bag as db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
import ast
warnings.filterwarnings('ignore')

class AttributeExtractor:
    """
    Modular attribute extraction utility for Yelp business attributes.
    Handles nested JSON parsing with various extraction strategies.
    """
    
    @staticmethod
    def _parse_to_dict(attr_value):
        """
        Helper method to convert attribute value to dictionary.
        Handles string representations of dicts and already-parsed dicts.
        
        Args:
            attr_value: Raw attribute value (can be string, dict, or None)
            
        Returns:
            dict or None
        """
        if pd.isna(attr_value):
            return None
        
        if isinstance(attr_value, str):
            try:
                return ast.literal_eval(attr_value)
            except:
                return None
        
        if isinstance(attr_value, dict):
            return attr_value
        
        return None
    
    @staticmethod
    def _parse_nested_dict(value_str):
        """
        Parse a nested dictionary string (e.g., parking, meal times).
        Handles unicode markers and various string formats.
        
        Args:
            value_str: String representation of nested dict or dict object
            
        Returns:
            dict or None
        """
        if not value_str or value_str == 'None':
            return None
        
        if isinstance(value_str, dict):
            return value_str
        
        if isinstance(value_str, str):
            try:
                # Clean unicode markers
                cleaned = value_str.replace("u'", "'").replace('u"', '"')
                return ast.literal_eval(cleaned)
            except:
                return None
        
        return None
    
    @staticmethod
    def _clean_string_value(value):
        """
        Clean string values by removing quotes and unicode markers.
        
        Args:
            value: String value to clean
            
        Returns:
            Cleaned string or None
        """
        if not value or value == 'None':
            return None
        
        if isinstance(value, str):
            return value.strip("'\"u ")
        
        return value
    
    @classmethod
    def extract_simple_attribute(cls, attr_value, attr_key):
        """
        Extract a simple attribute (string, boolean, or number).
        
        Args:
            attr_value: The entire attributes dict
            attr_key: The specific attribute key to extract
            
        Returns:
            Extracted value or None
        """
        attr_dict = cls._parse_to_dict(attr_value)
        if not attr_dict:
            return None
        
        value = attr_dict.get(attr_key)
        
        # Handle boolean values
        if isinstance(value, bool):
            return value
        
        # Handle string values
        if isinstance(value, str):
            return cls._clean_string_value(value)
        
        # Handle numeric values
        if isinstance(value, (int, float)):
            return value
        
        return None
    
    @classmethod
    def extract_nested_true_values(cls, attr_value, attr_key):
        """
        Extract nested dictionary and return only keys with True values.
        Used for: parking options, meal times, ambience types, etc.
        
        Args:
            attr_value: The entire attributes dict
            attr_key: The specific attribute key containing nested dict
            
        Returns:
            Comma-separated string of True values or None
        """
        attr_dict = cls._parse_to_dict(attr_value)
        if not attr_dict:
            return None
        
        nested_value = attr_dict.get(attr_key)
        nested_dict = cls._parse_nested_dict(nested_value)
        
        if not isinstance(nested_dict, dict):
            return None
        
        # Extract only True values
        true_values = [k for k, v in nested_dict.items() if v is True]
        return ', '.join(true_values) if true_values else None
    
    @classmethod
    def extract_nested_all_values(cls, attr_value, attr_key):
        """
        Extract nested dictionary and return all key-value pairs.
        Used when you want to see all options, not just True ones.
        
        Args:
            attr_value: The entire attributes dict
            attr_key: The specific attribute key containing nested dict
            
        Returns:
            Dictionary of all values or None
        """
        attr_dict = cls._parse_to_dict(attr_value)
        if not attr_dict:
            return None
        
        nested_value = attr_dict.get(attr_key)
        return cls._parse_nested_dict(nested_value)
    
    @classmethod
    def check_attribute_exists(cls, attr_value, attr_keys):
        """
        Check if any of the specified attributes exist.
        Used for checking business types (e.g., is restaurant).
        
        Args:
            attr_value: The entire attributes dict
            attr_keys: List of attribute keys to check for
            
        Returns:
            Boolean indicating if any key exists
        """
        attr_dict = cls._parse_to_dict(attr_value)
        if not attr_dict:
            return False
        
        return any(key in attr_dict for key in attr_keys)


class YelpPreprocessor:
    """
    Data cleaning and EDA pipeline for multi-modal Yelp dataset.
    Designed to support cold-start recommendation systems with LLM-based explanations.
    """
    
    def __init__(self, data_dir, photo_dir=None):
        """
        Initialize the preprocessor with data directories.
        
        Args:
            data_dir: Path to directory containing JSON files
            photo_dir: Path to directory containing photos (optional)
        """
        self.data_dir = Path(data_dir)
        self.photo_dir = Path(photo_dir) if photo_dir else None
        self.cold_start_threshold = 5  # Reviews threshold for cold-start classification
        
        # Initialize attribute extractor
        self.attr_extractor = AttributeExtractor()
        
        # Data containers
        self.business_df = None
        self.checkin_df = None
        self.review_df = None
        self.tip_df = None
        self.user_df = None
        self.photo_df = None
        
        # Processed data
        self.user_interaction_counts = None
        self.business_interaction_counts = None
    
    def _parse_all_attributes(self, df):
        """
        Parse and extract all relevant business attributes using modular extraction.
        This replaces all the individual extraction methods with a cleaner approach.
        
        Args:
            df: Business dataframe with 'attributes' column
            
        Returns:
            DataFrame with extracted attribute columns
        """
        print("Parsing business attributes...")
        
        # Define attribute extraction configurations
        # Format: (column_name, extraction_method, attribute_key, [additional_params])
        
        # Simple string/boolean attributes
        simple_attrs = [
            ('accepts_credit_cards', 'simple', 'BusinessAcceptsCreditCards'),
            ('bike_parking', 'simple', 'BikeParking'),
            ('wheelchair_accessible', 'simple', 'WheelchairAccessible'),
            ('outdoor_seating', 'simple', 'OutdoorSeating'),
            ('has_tv', 'simple', 'HasTV'),
            ('dogs_allowed', 'simple', 'DogsAllowed'),
            ('good_for_kids', 'simple', 'GoodForKids'),
            ('takes_reservations', 'simple', 'RestaurantsReservations'),
            ('delivery', 'simple', 'RestaurantsDelivery'),
            ('takeout', 'simple', 'RestaurantsTakeOut'),
            ('price_range', 'simple', 'RestaurantsPriceRange2'),
            ('alcohol', 'simple', 'Alcohol'),
            ('wifi', 'simple', 'WiFi'),
            ('noise_level', 'simple', 'NoiseLevel'),
            ('attire', 'simple', 'RestaurantsAttire'),
            ('smoking', 'simple', 'Smoking'),
            ('byob', 'simple', 'BYOB'),
            ('corkage', 'simple', 'Corkage'),
            ('good_for_groups', 'simple', 'RestaurantsGoodForGroups'),
            ('table_service', 'simple', 'RestaurantsTableService'),
            ('waiter_service', 'simple', 'WaiterService'),
            ('drive_thru', 'simple', 'DriveThru'),
            ('caters', 'simple', 'Caters'),
        ]
        
        # Nested attributes - extract only True values
        nested_true_attrs = [
            ('parking_options', 'nested_true', 'BusinessParking'),
            ('good_for_meal', 'nested_true', 'GoodForMeal'),
            ('ambience', 'nested_true', 'Ambience'),
            ('best_nights', 'nested_true', 'BestNights'),
            ('music_types', 'nested_true', 'Music'),
        ]
        
        # Special case: check if business is a restaurant
        restaurant_attrs = [
            'RestaurantsPriceRange2', 'RestaurantsPriceRange',
            'RestaurantsTakeOut', 'RestaurantsDelivery',
            'RestaurantsReservations', 'RestaurantsGoodForGroups',
            'RestaurantsTableService', 'RestaurantsAttire'
        ]
        
        # Extract simple attributes
        for col_name, method, attr_key in simple_attrs:
            df[col_name] = df['attributes'].apply(
                lambda x: self.attr_extractor.extract_simple_attribute(x, attr_key)
            )
        
        # Extract nested attributes with True values
        for col_name, method, attr_key in nested_true_attrs:
            df[col_name] = df['attributes'].apply(
                lambda x: self.attr_extractor.extract_nested_true_values(x, attr_key)
            )
        
        # Check if restaurant
        df['is_restaurant_attr'] = df['attributes'].apply(
            lambda x: self.attr_extractor.check_attribute_exists(x, restaurant_attrs)
        )
        
        print(f"Extracted {len(simple_attrs) + len(nested_true_attrs) + 1} attribute features")
        
        return df
    
    def load_data(self, use_sample=False, sample_frac=0.1):
        """
        Load all Yelp dataset files using Dask for efficient processing.
        
        Args:
            use_sample: Whether to use a sample of the data for faster prototyping
            sample_frac: Fraction of data to sample if use_sample=True
        """
        print("Loading Yelp dataset files...")
        
        # Load business data
        print("Loading business data...")
        business_path = self.data_dir / "yelp_academic_dataset_business.json"
        self.business_df = dd.read_json(business_path, lines=True, blocksize='64MB')
        if use_sample:
            self.business_df = self.business_df.sample(frac=sample_frac, random_state=42)
        self.business_df = self.business_df.compute()
        print(f"Loaded {len(self.business_df)} businesses")
        
        # Load review data (largest file - keep as Dask initially)
        print("Loading review data...")
        review_path = self.data_dir / "yelp_academic_dataset_review.json"
        self.review_df = dd.read_json(review_path, lines=True, blocksize='64MB')
        if use_sample:
            self.review_df = self.review_df.sample(frac=sample_frac, random_state=42)
        print(f"Review data loaded (Dask DataFrame with {self.review_df.npartitions} partitions)")
        
        # Load user data
        print("Loading user data...")
        user_path = self.data_dir / "yelp_academic_dataset_user.json"
        self.user_df = dd.read_json(user_path, lines=True, blocksize='64MB')
        if use_sample:
            self.user_df = self.user_df.sample(frac=sample_frac, random_state=42)
        self.user_df = self.user_df.compute()
        print(f"Loaded {len(self.user_df)} users")
        
        # Load tips
        print("Loading tips data...")
        tip_path = self.data_dir / "yelp_academic_dataset_tip.json"
        self.tip_df = dd.read_json(tip_path, lines=True, blocksize='64MB').compute()
        if use_sample and len(self.tip_df) > 0:
            self.tip_df = self.tip_df.sample(frac=sample_frac, random_state=42)
        print(f"Loaded {len(self.tip_df)} tips")
        
        # Load check-ins
        print("Loading check-in data...")
        checkin_path = self.data_dir / "yelp_academic_dataset_checkin.json"
        self.checkin_df = dd.read_json(checkin_path, lines=True, blocksize='64MB').compute()
        print(f"Loaded {len(self.checkin_df)} check-in records")
        
        # Load photos (if directory provided)
        if self.photo_dir and (self.data_dir / "photos.json").exists():
            print("Loading photo metadata...")
            photo_path = self.data_dir / "photos.json"
            self.photo_df = dd.read_json(photo_path, lines=True, blocksize='64MB').compute()
            if use_sample and len(self.photo_df) > 0:
                self.photo_df = self.photo_df.sample(frac=sample_frac, random_state=42)
            print(f"Loaded {len(self.photo_df)} photo records")
        
        print("\nData loading complete!\n")
    
    def clean_business_data(self):
        """Clean and preprocess business information."""
        print("Cleaning business data...")
        
        df = self.business_df.copy()
        
        # Handle missing values
        df['stars'] = df['stars'].fillna(0)
        df['review_count'] = df['review_count'].fillna(0).astype(int)
        df['is_open'] = df['is_open'].fillna(0).astype(int)
        
        # Extract and clean location data
        df['state'] = df['state'].fillna('Unknown')
        df['city'] = df['city'].fillna('Unknown')
        df['postal_code'] = df['postal_code'].fillna('Unknown')
        
        # Handle coordinates
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Parse attributes (nested JSON) - extract ALL attributes dynamically
        if 'attributes' in df.columns:
            df = self._parse_all_attributes(df)
        
        # Parse categories (comma-separated string to list)
        if 'categories' in df.columns:
            df['category_list'] = df['categories'].fillna('').str.split(', ')
            df['num_categories'] = df['category_list'].apply(len)
            df['is_restaurant'] = df['categories'].fillna('').str.contains('Restaurant', case=False)
        
        self.business_df = df
        print(f"Business data cleaned: {len(df)} records\n")
    
    def clean_review_data(self):
        """Clean and preprocess review text data."""
        print("Cleaning review data...")
        
        # Work with Dask DataFrame
        df = self.review_df
        
        # Convert date to datetime
        df['date'] = dd.to_datetime(df['date'], errors='coerce')
        
        # Text length features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Handle missing text
        df['text'] = df['text'].fillna('')
        
        # Engagement metrics
        df['useful'] = df['useful'].fillna(0).astype(int)
        df['funny'] = df['funny'].fillna(0).astype(int)
        df['cool'] = df['cool'].fillna(0).astype(int)
        df['total_engagement'] = df['useful'] + df['funny'] + df['cool']
        
        # Compute to pandas for interaction counting
        print("Computing review statistics...")
        self.review_df = df.compute()
        print(f"Review data cleaned: {len(self.review_df)} records\n")
    
    def clean_user_data(self):
        """Clean and preprocess user metadata."""
        print("Cleaning user data...")
        
        df = self.user_df.copy()
        
        # Handle temporal features
        df['yelping_since'] = pd.to_datetime(df['yelping_since'], errors='coerce')
        df['account_age_days'] = (datetime.now() - df['yelping_since']).dt.days
        
        # Clean engagement metrics
        numeric_cols = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Parse friends list
        if 'friends' in df.columns:
            df['friends_list'] = df['friends'].fillna('').apply(
                lambda x: x.split(', ') if x != 'None' and x != '' else []
            )
            df['num_friends'] = df['friends_list'].apply(len)
        
        # Parse elite years
        if 'elite' in df.columns:
            df['elite_years_list'] = df['elite'].fillna('').apply(
                lambda x: x.split(',') if x != 'None' and x != '' else []
            )
            df['num_elite_years'] = df['elite_years_list'].apply(len)
            df['is_elite'] = df['num_elite_years'] > 0
        
        # Aggregate compliment scores
        compliment_cols = [col for col in df.columns if 'compliment' in col]
        if compliment_cols:
            df['total_compliments'] = df[compliment_cols].sum(axis=1)
        
        self.user_df = df
        print(f"User data cleaned: {len(df)} records\n")
    
    def clean_tip_data(self):
        """Clean and preprocess tip data."""
        print("Cleaning tip data...")
        
        df = self.tip_df.copy()
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Text features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        # Handle missing values
        df['text'] = df['text'].fillna('')
        df['compliment_count'] = df['compliment_count'].fillna(0).astype(int)
        
        self.tip_df = df
        print(f"Tip data cleaned: {len(df)} records\n")
    
    def clean_photo_data(self):
        """Clean and preprocess photo metadata."""
        if self.photo_df is None:
            print("No photo data loaded. Skipping photo cleaning.\n")
            return
        
        print("Cleaning photo data...")
        
        df = self.photo_df.copy()
        
        # Handle missing captions
        df['caption'] = df['caption'].fillna('')
        df['has_caption'] = df['caption'] != ''
        
        # Count photos per business
        photos_per_business = df.groupby('business_id').size().reset_index(name='photo_count')
        
        self.photo_df = df
        print(f"Photo data cleaned: {len(df)} records\n")
    
    def identify_cold_start_entities(self):
        """
        Identify cold-start users and businesses based on interaction counts.
        Cold-start defined as entities with <= threshold reviews.
        """
        print("Identifying cold-start entities...")
        
        # Count user interactions
        self.user_interaction_counts = self.review_df.groupby('user_id').size().reset_index(name='review_count')
        cold_start_users = self.user_interaction_counts[
            self.user_interaction_counts['review_count'] <= self.cold_start_threshold
        ]
        
        # Count business interactions
        self.business_interaction_counts = self.review_df.groupby('business_id').size().reset_index(name='review_count')
        cold_start_businesses = self.business_interaction_counts[
            self.business_interaction_counts['review_count'] <= self.cold_start_threshold
        ]
        
        print(f"Cold-start users (≤{self.cold_start_threshold} reviews): {len(cold_start_users)}")
        print(f"Cold-start businesses (≤{self.cold_start_threshold} reviews): {len(cold_start_businesses)}\n")
        
        return cold_start_users, cold_start_businesses
    
    def run_full_pipeline(self, use_sample=False, sample_frac=0.1):
        """
        Execute the complete data cleaning pipeline.
        
        Args:
            use_sample: Whether to use a sample for faster execution
            sample_frac: Fraction of data to sample
        """
        print("="*60)
        print("YELP MULTI-MODAL DATA PREPROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        self.load_data(use_sample=use_sample, sample_frac=sample_frac)
        
        # Clean individual datasets
        self.clean_business_data()
        self.clean_review_data()
        self.clean_user_data()
        self.clean_tip_data()
        self.clean_photo_data()
        
        # # Identify cold-start entities
        self.identify_cold_start_entities()
        
        print("="*60)
        print("PIPELINE COMPLETE")
        print("="*60)