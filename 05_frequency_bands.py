import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define Periods
pre_start = '2019-12-01'
pre_end = '2020-02-29'
post_start = '2020-03-01'
post_end = '2020-05-31'

# Bands to look for in the raw files
# Based on typical file content: 4.0-14.0 is primary. 
# Others might be 0.1-1.0 (microseism), 1.0-20.0, 4.0-20.0
target_bands = ['0.1-1.0', '1.0-20.0', '4.0-14.0', '4.0-20.0']

# Storage
# List of dicts: {Station, Band, Period, Median_Value}
data_list = []

files = glob.glob('PaperZero_RMS/*.csv')
print(f"Found {len(files)} files in PaperZero_RMS/. Processing...")

count = 0
for f in files:
    try:
        # Read CSV
        # Index is usually datetime string
        df = pd.read_csv(f, index_col=0)
        
        # Parse Dates safely
        try:
             df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        except Exception as e:
             # Fallback for naive dates
             df.index = pd.to_datetime(df.index)
        
        # Filter for our total range first to speed up
        mask = (df.index >= pre_start) & (df.index <= post_end)
        df = df[mask]
        
        if df.empty: 
            continue
            
        station = os.path.basename(f).split('.')[1]
        
        # Check available bands in this file
        for band in target_bands:
            if band in df.columns:
                # Pre Data
                pre_data = df.loc[(df.index >= pre_start) & (df.index <= pre_end), band]
                # Post Data
                post_data = df.loc[(df.index >= post_start) & (df.index <= post_end), band]
                
                # We calculate the MEDIAN for this station/band/period
                if not pre_data.empty:
                    val = pre_data.median()
                    if not np.isnan(val):
                        data_list.append({'Station': station, 'Band': band, 'Period': 'Pré-Lockdown', 'Value': val})
                        
                if not post_data.empty:
                    val = post_data.median()
                    if not np.isnan(val):
                        data_list.append({'Station': station, 'Band': band, 'Period': 'Lockdown', 'Value': val})
        
        count += 1
        if count % 20 == 0:
            print(f"Processed {count} files...")
            
    except Exception as e:
        print(f"Error processing {f}: {e}")

# Create DataFrame
results_df = pd.DataFrame(data_list)
print(f"\nExtraction complete. Total records: {len(results_df)}")
print(results_df.head())

if not results_df.empty:
    # Aggregate: Global Median per Band and Period
    # This gives us the "typical" noise level for that band globally
    global_stats = results_df.groupby(['Band', 'Period'])['Value'].median().reset_index()
    
    print("\nGlobal Statistics (Median Displacement):")
    print(global_stats)
    
    # Visualization
    plt.figure(figsize=(10, 7))
    
    # Sort bands logically (numeric sort by start freq)
    # 0.1-1.0, 1.0-20.0, 4.0-14.0, 4.0-20.0
    # Custom sort order
    band_order = ['0.1-1.0', '1.0-20.0', '4.0-14.0', '4.0-20.0']
    
    # Check which bands actually exist in data
    existing_bands = [b for b in band_order if b in global_stats['Band'].unique()]
    
    # Bar Chart
    # Use Log Scale because 0.1-1.0 (Ocean) is usually much stronger than 4.0-14.0 (Anthropogenic)
    sns.barplot(data=global_stats, x='Band', y='Value', hue='Period', 
                order=existing_bands,
                palette={'Pré-Lockdown': 'gray', 'Lockdown': 'tab:red'})
    
    plt.yscale('log')
    plt.title('Energia Sísmica Global por Banda de Frequência (Mediana)', fontsize=14)
    plt.ylabel('Deslocamento RMS (Escala Log)', fontsize=12)
    plt.xlabel('Faixa de Frequência (Hz)', fontsize=12)
    plt.grid(True, which="major", axis='y', ls="-", alpha=0.5)
    plt.grid(True, which="minor", axis='y', ls=":", alpha=0.3)
    
    plt.savefig('frequency_band_comparison.png', dpi=300)
    print("Saved plot to 'frequency_band_comparison.png'")
    
else:
    print("No valid data found for frequency analysis.")
