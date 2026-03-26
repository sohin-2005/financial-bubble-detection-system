"""
SENTIMENT-PRICE DIVERGENCE VISUALIZATION
=========================================
Plots Nifty 50 price with:
  - Bubble periods (from drawdown labeling)
  - Sentiment-price divergence zones
  - Price Z-score levels
  - Sentiment trend

This helps visualize when price was high but sentiment was deteriorating
(classic bubble fragility signal).
"""

from bubble_detection_fixed import (
    load_data,
    label_bubbles_by_drawdown,
    compute_zscore_features,
    integrate_sentiment_features
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Import functions from the fixed pipeline
import sys
sys.path.append('.')


def create_divergence_visualization():
    """
    Create comprehensive visualization of sentiment-price divergence.
    """
    print("\n" + "="*70)
    print("SENTIMENT-PRICE DIVERGENCE VISUALIZATION")
    print("="*70)

    # Load and prepare data
    print("\nLoading data...")
    df = load_data()
    df = label_bubbles_by_drawdown(df)
    df = compute_zscore_features(df)
    df = integrate_sentiment_features(df)

    # Filter out NaN values
    df = df.dropna(
        subset=['price_zscore', 'sentiment_raw', 'sentiment_price_divergence'])

    print(
        f"Plotting {len(df)} days of data from {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('Nifty 50: Sentiment-Price Divergence Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # Convert dates for plotting
    dates = pd.to_datetime(df['Date'])

    # ─────────────────────────────────────────────────────────────────────
    # SUBPLOT 1: Price with Bubble Periods
    # ─────────────────────────────────────────────────────────────────────
    ax1 = axes[0]

    # Plot price
    ax1.plot(dates, df['Close'], color='#2E86AB',
             linewidth=1.5, label='Nifty 50 Price')

    # Shade bubble periods
    bubble_mask = df['bubble'] == 1
    if bubble_mask.any():
        bubble_dates = dates[bubble_mask]
        bubble_prices = df.loc[bubble_mask, 'Close']
        ax1.scatter(bubble_dates, bubble_prices, color='red', s=20,
                    alpha=0.6, zorder=5, label='Bubble Period')

        # Shade background for bubble regions
        for idx in df[bubble_mask].index:
            ax1.axvspan(dates[idx], dates[idx], alpha=0.1, color='red')

    # Shade sentiment-price divergence periods
    divergence_mask = df['sentiment_price_divergence'] == 1
    if divergence_mask.any():
        div_dates = dates[divergence_mask]
        div_prices = df.loc[divergence_mask, 'Close']
        ax1.scatter(div_dates, div_prices, color='orange', s=30,
                    marker='v', alpha=0.8, zorder=6,
                    label='Sentiment-Price Divergence')

    ax1.set_ylabel('Price (INR)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Chart with Bubble Periods & Divergence Signals',
                  fontsize=12, pad=10)

    # ─────────────────────────────────────────────────────────────────────
    # SUBPLOT 2: Price Z-Score
    # ─────────────────────────────────────────────────────────────────────
    ax2 = axes[1]

    # Plot Z-score
    ax2.plot(dates, df['price_zscore'], color='#A23B72', linewidth=1.5,
             label='Price Z-Score (252d)')

    # Add threshold lines
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1,
                alpha=0.7, label='Z = +2 (High)')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=-2.0, color='green', linestyle='--', linewidth=1,
                alpha=0.7, label='Z = -2 (Low)')

    # Shade divergence periods
    if divergence_mask.any():
        for idx in df[divergence_mask].index:
            ax2.axvspan(dates[idx], dates[idx], alpha=0.15, color='orange')

    ax2.set_ylabel('Z-Score', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Price Z-Score (Rolling 252-day)', fontsize=12, pad=10)

    # ─────────────────────────────────────────────────────────────────────
    # SUBPLOT 3: Sentiment Trend
    # ─────────────────────────────────────────────────────────────────────
    ax3 = axes[2]

    # Plot sentiment
    ax3.plot(dates, df['sentiment_raw'], color='#F18F01', linewidth=1,
             alpha=0.6, label='Raw Sentiment')
    ax3.plot(dates, df['sentiment_7d'], color='#C73E1D', linewidth=1.5,
             label='7-day MA')
    ax3.plot(dates, df['sentiment_30d'], color='#6A994E', linewidth=1.5,
             label='30-day MA')

    # Shade divergence periods
    if divergence_mask.any():
        for idx in df[divergence_mask].index:
            ax3.axvspan(dates[idx], dates[idx], alpha=0.15, color='orange')

    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax3.set_ylabel('Sentiment Score', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Sentiment Trend (VIX-based)', fontsize=12, pad=10)

    # ─────────────────────────────────────────────────────────────────────
    # SUBPLOT 4: Sentiment Momentum
    # ─────────────────────────────────────────────────────────────────────
    ax4 = axes[3]

    # Plot sentiment momentum
    colors = ['red' if x < 0 else 'green' for x in df['sentiment_momentum_custom']]
    ax4.bar(dates, df['sentiment_momentum_custom'], color=colors,
            alpha=0.6, width=1, label='Sentiment Momentum (7d - 30d)')

    # Mark divergence periods
    if divergence_mask.any():
        div_momentum = df.loc[divergence_mask, 'sentiment_momentum_custom']
        ax4.scatter(div_dates, div_momentum, color='orange', s=50,
                    marker='v', alpha=0.9, zorder=5, edgecolors='black',
                    linewidth=1, label='Divergence (Price↑ + Sentiment↓)')

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    ax4.set_ylabel('Momentum', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper left', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Sentiment Momentum (Negative = Deteriorating)',
                  fontsize=12, pad=10)

    # ─────────────────────────────────────────────────────────────────────
    # Format x-axis
    # ─────────────────────────────────────────────────────────────────────
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save figure
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/sentiment_price_divergence.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")

    # Show figure
    plt.show()

    # ─────────────────────────────────────────────────────────────────────
    # Print Summary Statistics
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    bubble_periods = df['bubble'].sum()
    bubble_pct = bubble_periods / len(df) * 100

    divergence_periods = df['sentiment_price_divergence'].sum()
    divergence_pct = divergence_periods / len(df) * 100

    # Count periods where divergence coincides with bubbles
    overlap = ((df['bubble'] == 1) & (
        df['sentiment_price_divergence'] == 1)).sum()
    overlap_pct = overlap / bubble_periods * 100 if bubble_periods > 0 else 0

    print(f"\nTotal trading days: {len(df)}")
    print(f"\nBubble periods (drawdown-based):")
    print(f"  Count: {bubble_periods} days ({bubble_pct:.1f}%)")

    print(f"\nSentiment-Price Divergence periods:")
    print(f"  Count: {divergence_periods} days ({divergence_pct:.1f}%)")
    print(f"  Definition: Price Z-score > 1.0 AND Sentiment momentum < 0")

    print(f"\nOverlap (Divergence during Bubbles):")
    print(f"  Count: {overlap} days ({overlap_pct:.1f}% of bubble periods)")

    if divergence_periods > 0:
        print(f"\nDivergence periods detected:")
        divergence_dates = df[df['sentiment_price_divergence'] == 1]['Date']

        # Group consecutive dates
        divergence_groups = []
        current_group = []

        for i, date in enumerate(divergence_dates):
            if not current_group:
                current_group = [date]
            else:
                # Check if consecutive (within 5 days)
                if (date - current_group[-1]).days <= 5:
                    current_group.append(date)
                else:
                    divergence_groups.append(current_group)
                    current_group = [date]

        if current_group:
            divergence_groups.append(current_group)

        print(f"\n  {len(divergence_groups)} distinct divergence episodes:")
        for i, group in enumerate(divergence_groups[:10], 1):  # Show first 10
            start = group[0]
            end = group[-1]
            duration = len(group)
            print(f"    {i}. {start.date()} to {end.date()} ({duration} days)")

    # Analyze divergence effectiveness
    print("\n" + "="*70)
    print("DIVERGENCE AS BUBBLE PREDICTOR")
    print("="*70)

    if divergence_periods > 0 and bubble_periods > 0:
        # Check if divergence precedes bubbles
        df['future_bubble'] = df['bubble'].shift(-30)  # Check 30 days ahead

        divergence_followed_by_bubble = ((df['sentiment_price_divergence'] == 1) &
                                         (df['future_bubble'] == 1)).sum()

        precision = divergence_followed_by_bubble / \
            divergence_periods * 100 if divergence_periods > 0 else 0

        print(f"\nDivergence followed by bubble (within 30 days):")
        print(
            f"  {divergence_followed_by_bubble} / {divergence_periods} = {precision:.1f}%")

        if precision > 50:
            print("  ✅ Divergence is a useful early warning signal")
        elif precision > 30:
            print("  ~ Divergence has moderate predictive value")
        else:
            print("  ⚠️ Divergence has low predictive value")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
Sentiment-Price Divergence indicates:
  • Price is elevated (Z-score > 1.0)
  • BUT sentiment is deteriorating (7d trend < 30d trend)
  • Classic "losing confidence while price still high" pattern
  • Can signal bubble fragility

Orange triangles on the chart show these periods.
Red dots show confirmed bubble periods (60 days before 30% crash).

If divergence periods cluster before crashes, it's a useful signal.
If they're scattered randomly, sentiment data may be noisy.
    """)

    return df


if __name__ == "__main__":
    df = create_divergence_visualization()
