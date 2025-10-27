"""
Exploratory Data Analysis (EDA) Module for Yelp Multi-Modal Cold-Start Project

This module provides comprehensive analysis and visualization for:
1. Cold-start distribution patterns
2. Attribute coverage across businesses
3. Multi-modal feature availability
4. Key insights for model development

Author: Research Project - Cold-Start Recommendation with Multi-Modal LLMs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class YelpEDA:
    """
    Comprehensive EDA toolkit for analyzing cold-start patterns in Yelp dataset.
    """
    
    def __init__(self, preprocessor, output_dir='eda_outputs'):
        """
        Initialize EDA module with preprocessed data.
        
        Args:
            preprocessor: YelpMultiModalPreprocessor instance with cleaned data
            output_dir: Directory to save plots and analysis results
        """
        self.preprocessor = preprocessor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data references
        self.business_df = preprocessor.business_df
        self.review_df = preprocessor.review_df
        self.user_df = preprocessor.user_df
        self.tip_df = preprocessor.tip_df
        self.photo_df = preprocessor.photo_df
        
        # Analysis results storage
        self.cold_start_stats = {}
        self.attribute_coverage = {}
        self.multimodal_availability = {}
        
    def analyze_cold_start_distributions(self, thresholds=[1, 3, 5, 10, 20]):
        """
        Analyze and visualize cold-start patterns for users and businesses.
        
        Args:
            thresholds: List of review count thresholds to analyze
        """
        print("=" * 80)
        print("COLD-START DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # User interaction distribution
        user_reviews = self.review_df.groupby('user_id').size().reset_index(name='review_count')
        business_reviews = self.review_df.groupby('business_id').size().reset_index(name='review_count')
        
        # Calculate cold-start percentages at different thresholds
        print("\n📊 COLD-START THRESHOLDS ANALYSIS")
        print("-" * 80)
        
        for threshold in thresholds:
            user_coldstart_pct = (user_reviews['review_count'] <= threshold).mean() * 100
            biz_coldstart_pct = (business_reviews['review_count'] <= threshold).mean() * 100
            
            self.cold_start_stats[f'threshold_{threshold}'] = {
                'user_percentage': user_coldstart_pct,
                'business_percentage': biz_coldstart_pct,
                'user_count': (user_reviews['review_count'] <= threshold).sum(),
                'business_count': (business_reviews['review_count'] <= threshold).sum()
            }
            
            print(f"\nThreshold ≤ {threshold} reviews:")
            print(f"  Users:      {user_coldstart_pct:.2f}% ({self.cold_start_stats[f'threshold_{threshold}']['user_count']:,} users)")
            print(f"  Businesses: {biz_coldstart_pct:.2f}% ({self.cold_start_stats[f'threshold_{threshold}']['business_count']:,} businesses)")
        
        # Create comprehensive visualization
        self._plot_cold_start_distributions(user_reviews, business_reviews, thresholds)
        
        return self.cold_start_stats
    
    def _plot_cold_start_distributions(self, user_reviews, business_reviews, thresholds):
        """Create multi-panel visualization of cold-start distributions."""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. User review count distribution (log scale)
        ax1 = fig.add_subplot(gs[0, 0])
        user_reviews['review_count'].plot(kind='hist', bins=50, ax=ax1, 
                                          color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Number of Reviews per User', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('User Review Distribution', fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. Business review count distribution (log scale)
        ax2 = fig.add_subplot(gs[0, 1])
        business_reviews['review_count'].plot(kind='hist', bins=50, ax=ax2,
                                              color='coral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Number of Reviews per Business', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Business Review Distribution', fontsize=13, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        ax3 = fig.add_subplot(gs[0, 2])
        user_sorted = np.sort(user_reviews['review_count'].values)
        biz_sorted = np.sort(business_reviews['review_count'].values)
        user_cdf = np.arange(1, len(user_sorted) + 1) / len(user_sorted) * 100
        biz_cdf = np.arange(1, len(biz_sorted) + 1) / len(biz_sorted) * 100
        
        ax3.plot(user_sorted, user_cdf, label='Users', linewidth=2, color='steelblue')
        ax3.plot(biz_sorted, biz_cdf, label='Businesses', linewidth=2, color='coral')
        ax3.set_xlabel('Number of Reviews', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
        ax3.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
        ax3.set_xscale('log')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Cold-start threshold comparison
        ax4 = fig.add_subplot(gs[1, :])
        threshold_data = []
        for threshold in thresholds:
            stats = self.cold_start_stats[f'threshold_{threshold}']
            threshold_data.append({
                'Threshold': f'≤{threshold}',
                'Users': stats['user_percentage'],
                'Businesses': stats['business_percentage']
            })
        
        threshold_df = pd.DataFrame(threshold_data)
        x = np.arange(len(threshold_df))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, threshold_df['Users'], width, label='Users', 
                       color='steelblue', edgecolor='black', alpha=0.8)
        bars2 = ax4.bar(x + width/2, threshold_df['Businesses'], width, label='Businesses',
                       color='coral', edgecolor='black', alpha=0.8)
        
        ax4.set_xlabel('Review Count Threshold', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cold-Start Percentage (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Cold-Start Severity Across Thresholds', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(threshold_df['Threshold'])
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 5. Box plot comparison
        ax5 = fig.add_subplot(gs[2, 0])
        box_data = [
            user_reviews['review_count'].values,
            business_reviews['review_count'].values
        ]
        bp = ax5.boxplot(box_data, labels=['Users', 'Businesses'], patch_artist=True,
                        showfliers=False)
        colors = ['steelblue', 'coral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_ylabel('Number of Reviews (log scale)', fontsize=11, fontweight='bold')
        ax5.set_title('Review Count Distribution', fontsize=13, fontweight='bold')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Statistical summary table
        ax6 = fig.add_subplot(gs[2, 1:])
        ax6.axis('tight')
        ax6.axis('off')
        
        stats_data = [
            ['Metric', 'Users', 'Businesses'],
            ['Total Count', f"{len(user_reviews):,}", f"{len(business_reviews):,}"],
            ['Mean Reviews', f"{user_reviews['review_count'].mean():.2f}", 
             f"{business_reviews['review_count'].mean():.2f}"],
            ['Median Reviews', f"{user_reviews['review_count'].median():.0f}",
             f"{business_reviews['review_count'].median():.0f}"],
            ['Std Dev', f"{user_reviews['review_count'].std():.2f}",
             f"{business_reviews['review_count'].std():.2f}"],
            ['Min Reviews', f"{user_reviews['review_count'].min():.0f}",
             f"{business_reviews['review_count'].min():.0f}"],
            ['Max Reviews', f"{user_reviews['review_count'].max():.0f}",
             f"{business_reviews['review_count'].max():.0f}"],
            ['Gini Coefficient', f"{self._calculate_gini(user_reviews['review_count']):.3f}",
             f"{self._calculate_gini(business_reviews['review_count']):.3f}"]
        ]
        
        table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.35, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(stats_data)):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(3):
                table[(i, j)].set_facecolor(color)
        
        plt.suptitle('Cold-Start Problem: Comprehensive Distribution Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = self.output_dir / 'cold_start_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved: {output_path}")
        plt.close()
    
    def analyze_attribute_coverage(self):
        """
        Analyze coverage and completeness of business attributes.
        Critical for understanding feature availability in cold-start scenarios.
        """
        print("\n" + "=" * 80)
        print("ATTRIBUTE COVERAGE ANALYSIS")
        print("=" * 80)
        
        # Identify attribute columns (from refactored extraction)
        attribute_cols = [
            'accepts_credit_cards', 'bike_parking', 'wheelchair_accessible', 
            'outdoor_seating', 'has_tv', 'dogs_allowed', 'good_for_kids',
            'takes_reservations', 'delivery', 'takeout', 'price_range',
            'alcohol', 'wifi', 'noise_level', 'attire', 'smoking',
            'parking_options', 'good_for_meal', 'ambience', 'good_for_groups'
        ]
        
        # Filter to columns that exist
        existing_attrs = [col for col in attribute_cols if col in self.business_df.columns]
        
        # Calculate coverage
        coverage_data = []
        for attr in existing_attrs:
            non_null_count = self.business_df[attr].notna().sum()
            coverage_pct = (non_null_count / len(self.business_df)) * 100
            
            # For boolean columns
            if self.business_df[attr].dtype == bool or self.business_df[attr].dropna().isin([True, False]).all():
                true_count = (self.business_df[attr] == True).sum()
                true_pct = (true_count / non_null_count * 100) if non_null_count > 0 else 0
                value_dist = f"{true_pct:.1f}% True"
            else:
                # For categorical/string columns
                top_value = self.business_df[attr].mode()[0] if len(self.business_df[attr].mode()) > 0 else 'N/A'
                value_dist = f"Top: {top_value}"
            
            coverage_data.append({
                'Attribute': attr,
                'Coverage_Pct': coverage_pct,
                'Non_Null_Count': non_null_count,
                'Value_Distribution': value_dist
            })
        
        coverage_df = pd.DataFrame(coverage_data).sort_values('Coverage_Pct', ascending=False)
        self.attribute_coverage = coverage_df
        
        print("\n📋 ATTRIBUTE COVERAGE SUMMARY")
        print("-" * 80)
        print(coverage_df.to_string(index=False))
        
        # Visualize coverage
        self._plot_attribute_coverage(coverage_df)
        
        # Analyze cold-start businesses specifically
        self._analyze_coldstart_attribute_coverage()
        
        return coverage_df
    
    def _plot_attribute_coverage(self, coverage_df):
        """Visualize attribute coverage patterns."""
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Business Attribute Coverage Analysis', fontsize=16, fontweight='bold')
        
        # 1. Coverage bar chart
        ax1 = axes[0, 0]
        colors = ['#2ecc71' if x >= 50 else '#e74c3c' if x < 20 else '#f39c12' 
                  for x in coverage_df['Coverage_Pct']]
        bars = ax1.barh(coverage_df['Attribute'], coverage_df['Coverage_Pct'], color=colors, alpha=0.8)
        ax1.set_xlabel('Coverage Percentage (%)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Attribute', fontsize=11, fontweight='bold')
        ax1.set_title('Attribute Coverage Across All Businesses', fontsize=12, fontweight='bold')
        ax1.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='50% threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, val) in enumerate(zip(bars, coverage_df['Coverage_Pct'])):
            ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', fontsize=8)
        
        # 2. Coverage distribution
        ax2 = axes[0, 1]
        ax2.hist(coverage_df['Coverage_Pct'], bins=20, color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Coverage Percentage (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Number of Attributes', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of Coverage Rates', fontsize=12, fontweight='bold')
        ax2.axvline(coverage_df['Coverage_Pct'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {coverage_df['Coverage_Pct'].mean():.1f}%")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Coverage tiers
        ax3 = axes[1, 0]
        tiers = {
            'High (≥50%)': (coverage_df['Coverage_Pct'] >= 50).sum(),
            'Medium (20-50%)': ((coverage_df['Coverage_Pct'] >= 20) & 
                               (coverage_df['Coverage_Pct'] < 50)).sum(),
            'Low (<20%)': (coverage_df['Coverage_Pct'] < 20).sum()
        }
        colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']
        wedges, texts, autotexts = ax3.pie(tiers.values(), labels=tiers.keys(), 
                                           autopct='%1.1f%%', colors=colors_pie,
                                           startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Attribute Coverage Tiers', fontsize=12, fontweight='bold')
        
        # 4. Top 10 attributes by coverage
        ax4 = axes[1, 1]
        top_10 = coverage_df.head(10)
        bars = ax4.bar(range(len(top_10)), top_10['Coverage_Pct'], 
                      color='teal', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Attribute Rank', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Coverage Percentage (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Top 10 Most Complete Attributes', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(top_10)))
        ax4.set_xticklabels(range(1, len(top_10) + 1))
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add attribute names as text
        for i, (bar, attr) in enumerate(zip(bars, top_10['Attribute'])):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    attr, ha='center', va='bottom', fontsize=7, rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / 'attribute_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
    
    def _analyze_coldstart_attribute_coverage(self):
        """Compare attribute availability in cold-start vs warm-start businesses."""
        
        print("\n📊 COLD-START VS WARM-START ATTRIBUTE COMPARISON")
        print("-" * 80)
        
        # Get review counts per business
        business_reviews = self.review_df.groupby('business_id').size().reset_index(name='actual_review_count')
        
        # Merge avoiding column conflicts - only take business_id from business_df first
        business_with_reviews = self.business_df.copy()
        business_with_reviews = business_with_reviews.merge(
            business_reviews, 
            on='business_id', 
            how='left'
        )
        business_with_reviews['actual_review_count'] = business_with_reviews['actual_review_count'].fillna(0)
        
        # Define cold-start threshold
        threshold = self.preprocessor.cold_start_threshold
        coldstart_mask = business_with_reviews['actual_review_count'] <= threshold
        
        coldstart_biz = business_with_reviews[coldstart_mask]
        warmstart_biz = business_with_reviews[~coldstart_mask]
        
        print(f"\nCold-start businesses (≤{threshold} reviews): {len(coldstart_biz):,}")
        print(f"Warm-start businesses (>{threshold} reviews): {len(warmstart_biz):,}")
        
        # Compare attribute coverage
        attribute_cols = [col for col in self.attribute_coverage['Attribute'].values 
                         if col in business_with_reviews.columns]
        
        comparison_data = []
        for attr in attribute_cols:
            cold_coverage = coldstart_biz[attr].notna().mean() * 100
            warm_coverage = warmstart_biz[attr].notna().mean() * 100
            difference = warm_coverage - cold_coverage
            
            comparison_data.append({
                'Attribute': attr,
                'Cold_Start_Coverage': cold_coverage,
                'Warm_Start_Coverage': warm_coverage,
                'Difference': difference
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Difference', ascending=False)
        print("\nTop 10 attributes with biggest coverage gap (favoring warm-start):")
        print(comparison_df.head(10).to_string(index=False))
        
        # Visualize comparison
        self._plot_coldstart_warmstart_comparison(comparison_df, len(coldstart_biz), len(warmstart_biz))
    
    def _plot_coldstart_warmstart_comparison(self, comparison_df, cold_count, warm_count):
        """Visualize cold-start vs warm-start attribute coverage."""
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Side-by-side comparison
        ax1 = axes[0]
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax1.barh(x - width/2, comparison_df['Cold_Start_Coverage'], width,
                        label=f'Cold-Start (n={cold_count:,})', color='coral', alpha=0.8)
        bars2 = ax1.barh(x + width/2, comparison_df['Warm_Start_Coverage'], width,
                        label=f'Warm-Start (n={warm_count:,})', color='steelblue', alpha=0.8)
        
        ax1.set_yticks(x)
        ax1.set_yticklabels(comparison_df['Attribute'], fontsize=9)
        ax1.set_xlabel('Coverage Percentage (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Attribute Coverage: Cold-Start vs Warm-Start', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Coverage gap visualization
        ax2 = axes[1]
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in comparison_df['Difference']]
        bars = ax2.barh(comparison_df['Attribute'], comparison_df['Difference'], 
                       color=colors, alpha=0.8)
        ax2.set_xlabel('Coverage Difference (Warm - Cold) %', fontsize=11, fontweight='bold')
        ax2.set_title('Attribute Coverage Gap', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add annotations for significant gaps
        for bar, diff in zip(bars, comparison_df['Difference']):
            if abs(diff) > 5:  # Only annotate significant differences
                ax2.text(diff + (0.5 if diff > 0 else -0.5), bar.get_y() + bar.get_height()/2,
                        f'{diff:+.1f}%', va='center', fontsize=8,
                        ha='left' if diff > 0 else 'right')
        
        plt.tight_layout()
        output_path = self.output_dir / 'coldstart_vs_warmstart_coverage.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
    
    def analyze_multimodal_availability(self):
        """
        Analyze availability of multi-modal features: text, images, metadata.
        Critical for understanding which modalities can be leveraged for cold-start.
        """
        print("\n" + "=" * 80)
        print("MULTI-MODAL FEATURE AVAILABILITY ANALYSIS")
        print("=" * 80)
        
        # Get business review counts
        business_reviews = self.review_df.groupby('business_id').size().reset_index(name='review_count')
        business_tips = self.tip_df.groupby('business_id').size().reset_index(name='tip_count') if self.tip_df is not None else pd.DataFrame()
        business_photos = self.photo_df.groupby('business_id').size().reset_index(name='photo_count') if self.photo_df is not None else pd.DataFrame()
        
        # Merge all modalities
        multimodal_df = self.business_df[['business_id']].copy()
        multimodal_df = multimodal_df.merge(business_reviews, on='business_id', how='left')
        if not business_tips.empty:
            multimodal_df = multimodal_df.merge(business_tips, on='business_id', how='left')
        if not business_photos.empty:
            multimodal_df = multimodal_df.merge(business_photos, on='business_id', how='left')
        
        # Fill NaN with 0
        multimodal_df = multimodal_df.fillna(0)
        
        # Define modality flags
        multimodal_df['has_reviews'] = multimodal_df['review_count'] > 0
        
        # Safely create has_tips flag
        if 'tip_count' in multimodal_df.columns:
            multimodal_df['has_tips'] = multimodal_df['tip_count'] > 0
        else:
            multimodal_df['has_tips'] = False
        
        # Safely create has_photos flag
        if 'photo_count' in multimodal_df.columns:
            multimodal_df['has_photos'] = multimodal_df['photo_count'] > 0
        else:
            multimodal_df['has_photos'] = False
        
        # Count attribute coverage
        attribute_cols = [col for col in self.business_df.columns 
                         if col in ['price_range', 'wifi', 'parking_options', 'good_for_meal', 
                                   'outdoor_seating', 'takes_reservations']]
        
        # Safely create has_attributes flag - ensure it's boolean
        if attribute_cols:
            multimodal_df['has_attributes'] = self.business_df[attribute_cols].notna().any(axis=1).astype(bool)
        else:
            # If no attribute columns exist, all businesses have no attributes
            multimodal_df['has_attributes'] = False
        
        # Analyze modality combinations
        print("\n📊 MODALITY AVAILABILITY")
        print("-" * 80)
        print(f"Businesses with reviews: {multimodal_df['has_reviews'].sum():,} ({multimodal_df['has_reviews'].mean()*100:.2f}%)")
        if 'has_tips' in multimodal_df.columns:
            print(f"Businesses with tips: {multimodal_df['has_tips'].sum():,} ({multimodal_df['has_tips'].mean()*100:.2f}%)")
        if 'has_photos' in multimodal_df.columns:
            print(f"Businesses with photos: {multimodal_df['has_photos'].sum():,} ({multimodal_df['has_photos'].mean()*100:.2f}%)")
        print(f"Businesses with attributes: {multimodal_df['has_attributes'].sum():,} ({multimodal_df['has_attributes'].mean()*100:.2f}%)")
        
        # Analyze cold-start businesses specifically
        threshold = self.preprocessor.cold_start_threshold
        coldstart_mask = multimodal_df['review_count'] <= threshold
        coldstart_multimodal = multimodal_df[coldstart_mask]
        
        print(f"\n📊 COLD-START BUSINESSES (≤{threshold} reviews): {len(coldstart_multimodal):,}")
        print("-" * 80)
        if 'has_tips' in coldstart_multimodal.columns:
            print(f"With tips: {coldstart_multimodal['has_tips'].sum():,} ({coldstart_multimodal['has_tips'].mean()*100:.2f}%)")
        if 'has_photos' in coldstart_multimodal.columns:
            print(f"With photos: {coldstart_multimodal['has_photos'].sum():,} ({coldstart_multimodal['has_photos'].mean()*100:.2f}%)")
        print(f"With attributes: {coldstart_multimodal['has_attributes'].sum():,} ({coldstart_multimodal['has_attributes'].mean()*100:.2f}%)")
        
        # Store results
        self.multimodal_availability = {
            'all_businesses': multimodal_df,
            'coldstart_businesses': coldstart_multimodal
        }
        
        # Visualize
        self._plot_multimodal_availability(multimodal_df, coldstart_multimodal, threshold)
        
        return multimodal_df, coldstart_multimodal
    
    def _plot_multimodal_availability(self, all_data, coldstart_data, threshold):
        """Visualize multi-modal feature availability."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Modal Feature Availability Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall modality availability
        ax1 = axes[0, 0]
        modalities = ['Reviews', 'Tips', 'Photos', 'Attributes']
        availability = [
            all_data['has_reviews'].mean() * 100,
            all_data['has_tips'].mean() * 100 if 'has_tips' in all_data.columns else 0,
            all_data['has_photos'].mean() * 100 if 'has_photos' in all_data.columns else 0,
            all_data['has_attributes'].mean() * 100
        ]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax1.bar(modalities, availability, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Availability (%)', fontsize=11, fontweight='bold')
        ax1.set_title('Overall Modality Availability', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, availability):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 2,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        # 2. Cold-start vs All comparison
        ax2 = axes[0, 1]
        
        # Safely get coldstart availability for each modality
        coldstart_availability = [
            coldstart_data['has_reviews'].mean() * 100,
            coldstart_data['has_tips'].mean() * 100 if 'has_tips' in coldstart_data.columns else 0,
            coldstart_data['has_photos'].mean() * 100 if 'has_photos' in coldstart_data.columns else 0,
            coldstart_data['has_attributes'].mean() * 100
        ]
        
        x = np.arange(len(modalities))
        width = 0.35
        bars1 = ax2.bar(x - width/2, availability, width, label='All Businesses', 
                       color='steelblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, coldstart_availability, width, 
                       label=f'Cold-Start (≤{threshold} reviews)', color='coral', alpha=0.8)
        
        ax2.set_ylabel('Availability (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Modality Availability: All vs Cold-Start', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modalities)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Modality combinations (Venn-like analysis)
        ax3 = axes[1, 0]
        
        # Helper function to safely get boolean column
        def get_bool_col(df, col_name):
            """Get boolean column or return Series of False if column doesn't exist."""
            if col_name in df.columns:
                return df[col_name].astype(bool)
            else:
                return pd.Series([False] * len(df), index=df.index, dtype=bool)
        
        # Get boolean columns safely
        has_tips_col = get_bool_col(all_data, 'has_tips')
        has_photos_col = get_bool_col(all_data, 'has_photos')
        has_attributes_col = get_bool_col(all_data, 'has_attributes')
        has_reviews_col = get_bool_col(all_data, 'has_reviews')
        
        # Count businesses with different modality combinations
        combo_counts = {
            'All 4': ((has_reviews_col) & 
                     (has_tips_col) & 
                     (has_photos_col) & 
                     (has_attributes_col)).sum(),
            'Reviews +\n2 others': ((has_reviews_col) & 
                                   ((has_tips_col & has_photos_col) |
                                    (has_tips_col & has_attributes_col) |
                                    (has_photos_col & has_attributes_col))).sum(),
            'Reviews +\n1 other': ((has_reviews_col) & 
                                  ((has_tips_col) | 
                                   (has_photos_col) | 
                                   (has_attributes_col))).sum(),
            'Reviews\nonly': ((has_reviews_col) & 
                            ~has_tips_col & 
                            ~has_photos_col & 
                            ~has_attributes_col).sum(),
            'No\nreviews': (~has_reviews_col).sum()
        }
        
        labels = list(combo_counts.keys())
        sizes = list(combo_counts.values())
        colors_pie = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90,
                                           textprops={'fontsize': 9})
        ax3.set_title('Modality Combination Distribution', fontsize=12, fontweight='bold')
        
        # 4. Feature counts distribution for cold-start
        ax4 = axes[1, 1]
        
        if 'review_count' in coldstart_data.columns:
            review_dist = coldstart_data['review_count'].value_counts().sort_index()
            ax4.bar(review_dist.index, review_dist.values, color='coral', 
                   alpha=0.8, edgecolor='black')
            ax4.set_xlabel('Number of Reviews', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Number of Businesses', fontsize=11, fontweight='bold')
            ax4.set_title(f'Review Distribution (Cold-Start Businesses)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'multimodal_availability.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_path}")
        plt.close()
    
    # def identify_key_features(self, top_n=15):
    #     """
    #     Identify and rank the most important features for cold-start modeling.
    #     Based on coverage, variance, and relevance.
    #     """
    #     print("\n" + "=" * 80)
    #     print("KEY FEATURE IDENTIFICATION FOR COLD-START MODELING")
    #     print("=" * 80)
        
    #     feature_importance = []
        
    #     # 1. Attribute features
    #     if hasattr(self, 'attribute_coverage') and not self.attribute_coverage.empty:
    #         for _, row in self.attribute_coverage.iterrows():
    #             attr = row['Attribute']
    #             if attr in self.business_df.columns:
    #                 coverage = row['Coverage_Pct']
                    
    #                 # Calculate variance (for boolean/categorical)
    #                 if self.business_df[attr].dtype == bool:
    #                     variance = self.business_df[attr].var()
    #                 else:
    #                     # For categorical, use entropy as variance proxy
    #                     value_counts = self.business_df[attr].value_counts(normalize=True)
    #                     variance = -(value_counts * np.log2(value_counts + 1e-10)).sum()
                    
    #                 # Importance score: coverage × variance
    #                 importance = coverage * variance * 10  # Scale factor
                    
    #                 feature_importance.append({
    #                     'Feature': attr,
    #                     'Type': 'Attribute',
    #                     'Coverage': coverage,
    #                     'Variance': variance,
    #                     'Importance_Score': importance
    #                 })
        
    #     # 2. Metadata features (always available)
    #     metadata_features = ['stars', 'review_count', 'latitude', 'longitude', 'is_open']
    #     for feat in metadata_features:
    #         if feat in self.business_df.columns:
    #             coverage = self.business_df[feat].notna().mean() * 100
    #             variance = self.business_df[feat].var()
    #             importance = coverage * variance * 10
                
    #             feature_importance.append({
    #                 'Feature': feat,
    #                 'Type': 'Metadata',
    #                 'Coverage': coverage,
    #                 'Variance': variance if not pd.isna(variance) else 0,
    #                 'Importance_Score': importance if not pd.isna(importance) else 0
    #             })
        
    #     # 3. Categorical features
    #     if 'num_categories' in self.business_df.columns:
    #         coverage = self.business_df['num_categories'].notna().mean() * 100
    #         variance = self.business_df['num_categories'].var()
    #         importance = coverage * variance * 10
            
    #         feature_importance.append({
    #             'Feature': 'num_categories',
    #             'Type': 'Categorical',
    #             'Coverage': coverage,
    #             'Variance': variance,
    #             'Importance_Score': importance
    #         })
        
    #     # Sort by importance
    #     feature_df = pd.DataFrame(feature_importance).sort_values('Importance_Score', ascending=False)
        
    #     print(f"\n🔑 TOP {top_n} KEY FEATURES FOR COLD-START MODELING")
    #     print("-" * 80)
    #     print(feature_df.head(top_n).to_string(index=False))
        
    #     # Visualize
    #     self._plot_feature_importance(feature_df, top_n)
        
    #     return feature_df
    
    # def _plot_feature_importance(self, feature_df, top_n):
    #     """Visualize feature importance ranking."""
        
    #     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    #     fig.suptitle('Feature Importance for Cold-Start Modeling', fontsize=16, fontweight='bold')
        
    #     top_features = feature_df.head(top_n)
        
    #     # 1. Importance scores
    #     ax1 = axes[0]
    #     colors = ['#2ecc71' if t == 'Attribute' else '#3498db' if t == 'Metadata' else '#f39c12' 
    #               for t in top_features['Type']]
    #     bars = ax1.barh(range(len(top_features)), top_features['Importance_Score'], 
    #                    color=colors, alpha=0.8, edgecolor='black')
    #     ax1.set_yticks(range(len(top_features)))
    #     ax1.set_yticklabels(top_features['Feature'])
    #     ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    #     ax1.set_title(f'Top {top_n} Features by Importance', fontsize=12, fontweight='bold')
    #     ax1.grid(True, alpha=0.3, axis='x')
    #     ax1.invert_yaxis()
        
    #     # 2. Coverage vs Variance scatter
    #     ax2 = axes[1]
    #     types = top_features['Type'].unique()
    #     type_colors = {'Attribute': '#2ecc71', 'Metadata': '#3498db', 'Categorical': '#f39c12'}
        
    #     for t in types:
    #         mask = top_features['Type'] == t
    #         ax2.scatter(top_features[mask]['Coverage'], 
    #                    top_features[mask]['Variance'],
    #                    label=t, s=top_features[mask]['Importance_Score']*2,
    #                    alpha=0.6, color=type_colors.get(t, 'gray'),
    #                    edgecolors='black', linewidth=1)
        
    #     ax2.set_xlabel('Coverage (%)', fontsize=11, fontweight='bold')
    #     ax2.set_ylabel('Variance', fontsize=11, fontweight='bold')
    #     ax2.set_title('Feature Quality: Coverage vs Variance', fontsize=12, fontweight='bold')
    #     ax2.legend()
    #     ax2.grid(True, alpha=0.3)
        
    #     # Annotate top 5
    #     for _, row in top_features.head(5).iterrows():
    #         ax2.annotate(row['Feature'], 
    #                     (row['Coverage'], row['Variance']),
    #                     fontsize=8, alpha=0.7)
        
    #     plt.tight_layout()
    #     output_path = self.output_dir / 'feature_importance.png'
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"💾 Saved: {output_path}")
    #     plt.close()
    
    def generate_eda_report(self):
        """Generate comprehensive EDA summary report."""
        
        print("\n" + "=" * 80)
        print("GENERATING COMPREHENSIVE EDA REPORT")
        print("=" * 80)
        
        report_lines = []
        report_lines.append("# Yelp Cold-Start Recommendation System - EDA Report\n")
        report_lines.append(f"Generated: {pd.Timestamp.now()}\n\n")
        
        # Dataset overview
        report_lines.append("## Dataset Overview\n")
        report_lines.append(f"- **Total Businesses**: {len(self.business_df):,}\n")
        report_lines.append(f"- **Total Reviews**: {len(self.review_df):,}\n")
        report_lines.append(f"- **Total Users**: {len(self.user_df):,}\n")
        if self.tip_df is not None:
            report_lines.append(f"- **Total Tips**: {len(self.tip_df):,}\n")
        if self.photo_df is not None:
            report_lines.append(f"- **Total Photos**: {len(self.photo_df):,}\n")
        report_lines.append("\n")
        
        # Cold-start statistics
        if self.cold_start_stats:
            report_lines.append("## Cold-Start Analysis\n")
            threshold = self.preprocessor.cold_start_threshold
            stats = self.cold_start_stats[f'threshold_{threshold}']
            report_lines.append(f"Using threshold: ≤{threshold} reviews\n\n")
            report_lines.append(f"- **Cold-Start Users**: {stats['user_count']:,} ({stats['user_percentage']:.2f}%)\n")
            report_lines.append(f"- **Cold-Start Businesses**: {stats['business_count']:,} ({stats['business_percentage']:.2f}%)\n")
            report_lines.append("\n")
        
        # Attribute coverage summary
        if hasattr(self, 'attribute_coverage') and not self.attribute_coverage.empty:
            report_lines.append("## Attribute Coverage Summary\n")
            report_lines.append(f"- **Total Attributes Analyzed**: {len(self.attribute_coverage)}\n")
            report_lines.append(f"- **Mean Coverage**: {self.attribute_coverage['Coverage_Pct'].mean():.2f}%\n")
            report_lines.append(f"- **Median Coverage**: {self.attribute_coverage['Coverage_Pct'].median():.2f}%\n")
            high_coverage = (self.attribute_coverage['Coverage_Pct'] >= 50).sum()
            report_lines.append(f"- **High Coverage Attributes (≥50%)**: {high_coverage}\n")
            report_lines.append("\n")
        
        # Multi-modal availability
        if self.multimodal_availability:
            all_data = self.multimodal_availability['all_businesses']
            coldstart_data = self.multimodal_availability['coldstart_businesses']
            
            report_lines.append("## Multi-Modal Feature Availability\n")
            report_lines.append("### All Businesses:\n")
            report_lines.append(f"- Reviews: {all_data['has_reviews'].mean()*100:.2f}%\n")
            if 'has_tips' in all_data.columns:
                report_lines.append(f"- Tips: {all_data['has_tips'].mean()*100:.2f}%\n")
            if 'has_photos' in all_data.columns:
                report_lines.append(f"- Photos: {all_data['has_photos'].mean()*100:.2f}%\n")
            report_lines.append(f"- Attributes: {all_data['has_attributes'].mean()*100:.2f}%\n\n")
            
            report_lines.append("### Cold-Start Businesses:\n")
            if 'has_tips' in coldstart_data.columns:
                report_lines.append(f"- Tips: {coldstart_data['has_tips'].mean()*100:.2f}%\n")
            if 'has_photos' in coldstart_data.columns:
                report_lines.append(f"- Photos: {coldstart_data['has_photos'].mean()*100:.2f}%\n")
            report_lines.append(f"- Attributes: {coldstart_data['has_attributes'].mean()*100:.2f}%\n\n")
        
        # Key insights
        report_lines.append("## Key Insights for Model Development\n\n")
        report_lines.append("### Recommendations:\n")
        report_lines.append("1. **Cold-Start Severity**: Significant portion of data falls into cold-start category\n")
        report_lines.append("2. **Multi-Modal Opportunity**: Multiple modalities available for feature extraction\n")
        report_lines.append("3. **Attribute Heterogeneity**: Coverage varies significantly across attributes\n")
        report_lines.append("4. **Metadata Availability**: Location and rating data universally available\n\n")
        
        report_lines.append("### Next Steps:\n")
        report_lines.append("1. Feature engineering from high-coverage attributes\n")
        report_lines.append("2. Text embedding generation from reviews and tips\n")
        report_lines.append("3. Image embedding extraction from business photos\n")
        report_lines.append("4. Multi-modal fusion strategy design\n")
        
        # Save report
        report_path = self.output_dir / 'eda_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        print(f"\n💾 EDA Report saved: {report_path}")
        print("\n" + "=" * 80)
        print("EDA COMPLETE - All visualizations and reports generated")
        print("=" * 80)
    
    @staticmethod
    def _calculate_gini(values):
        """Calculate Gini coefficient for inequality measurement."""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((n - np.arange(1, n + 1) + 1) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    def run_full_eda(self, cold_start_thresholds=[1, 3, 5, 10, 20], top_features=15):
        """
        Execute complete EDA pipeline.
        
        Args:
            cold_start_thresholds: List of review thresholds to analyze
            top_features: Number of top features to identify
        """
        print("\n" + "=" * 80)
        print("YELP COLD-START EDA - COMPREHENSIVE ANALYSIS PIPELINE")
        print("=" * 80 + "\n")
        
        # 1. Cold-start distributions
        self.analyze_cold_start_distributions(thresholds=cold_start_thresholds)
        
        # 2. Attribute coverage
        self.analyze_attribute_coverage()
        
        # 3. Multi-modal availability
        self.analyze_multimodal_availability()
        
        # 4. Key feature identification
        # self.identify_key_features(top_n=top_features)
        
        # 5. Generate final report
        self.generate_eda_report()
        
        print(f"\n✅ All EDA outputs saved to: {self.output_dir.absolute()}")


# Example usage
if __name__ == "__main__":
    from yelp_preprocessor_refactored import YelpMultiModalPreprocessor
    
    # Initialize preprocessor and load data
    preprocessor = YelpMultiModalPreprocessor(
        data_dir='path/to/yelp_dataset',
        photo_dir='path/to/photos'
    )
    
    # Run preprocessing
    preprocessor.run_full_pipeline(use_sample=True, sample_frac=0.1)
    
    # Initialize EDA module
    eda = YelpColdStartEDA(preprocessor, output_dir='eda_outputs')
    
    # Run full EDA
    eda.run_full_eda()