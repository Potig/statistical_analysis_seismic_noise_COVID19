
import pandas as pd
import os
import glob
from datetime import datetime

def load_and_process_data():
    """
    Loads station metadata and RMS data, merges them, and creates a consolidated dataset.
    """
    base_dir = "/Users/leonardodesa/Geoestatistica/ThomasLecocq-2020_Science_GlobalQuieting-584a221"
    metadata_file = os.path.join(base_dir, "Supplementary Files/Table S1 - Lockdown stations.csv")
    rms_dir = os.path.join(base_dir, "PaperZero_RMS")
    mobility_file = os.path.join(base_dir, "Global_Mobility_Report.csv")
    output_file = os.path.join(base_dir, "processed_data.pkl")

    print(f"Loading mobility data from {mobility_file}...")
    try:
        # Optimizing load by reading only necessary columns
        mobility_cols = [
            'country_region', 'sub_region_1', 'sub_region_2', 'metro_area', 'date', 
            'retail_and_recreation_percent_change_from_baseline',
            'grocery_and_pharmacy_percent_change_from_baseline',
            'parks_percent_change_from_baseline',
            'transit_stations_percent_change_from_baseline',
            'workplaces_percent_change_from_baseline',
            'residential_percent_change_from_baseline'
        ]
        mobility_df = pd.read_csv(mobility_file, usecols=mobility_cols)
        
        # Filter for strictly Country-level data (all sub-regions and metro areas must be NaN)
        mobility_df = mobility_df[
            mobility_df['sub_region_1'].isna() & 
            mobility_df['sub_region_2'].isna() & 
            mobility_df['metro_area'].isna()
        ]
        
        # Drop potential duplicates
        mobility_df = mobility_df.drop_duplicates(subset=['country_region', 'date'])
        
        mobility_df['date'] = pd.to_datetime(mobility_df['date'])
        # Rename country column to match our convention if needed, or keeping as is for merge
        print(f"Loaded {len(mobility_df)} mobility records (Strict Country-level).")
    except Exception as e:
        print(f"Error loading mobility data: {e}")
        return

    print(f"Loading metadata from {metadata_file}...")
    try:
        metadata = pd.read_csv(metadata_file, encoding='latin1')
    except Exception as e:
        print(f"Error loading metadata with latin1, trying default: {e}")
        try:
             metadata = pd.read_csv(metadata_file)
        except Exception as e2:
             print(f"Error loading metadata: {e2}")
             return

    # Clean metadata column names if necessary
    metadata.columns = [c.strip() for c in metadata.columns]
    
    # We need to map Station_Code to filenames. 
    # Filenames format: Network.Station.Location.Channel.csv (e.g., AM.R091F.00.EHZ.csv)
    # Metadata Station_Code: Network.Station (e.g., AM.R091F)
    
    processed_dfs = []
    
    print(f"Found {len(metadata)} stations in metadata.")
    
    for index, row in metadata.iterrows():
        station_code = row['Station_Code']
        # Construct expected filename pattern
        # The file system has filenames like AM.R091F.00.EHZ.csv
        # We need to find the file that starts with the station code.
        
        search_pattern = os.path.join(rms_dir, f"{station_code}.*.csv")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            # print(f"Warning: No file found for station {station_code}")
            continue
        
        # Take the first matching file (usually there's only one per station/channel used)
        file_path = matching_files[0]
        
        try:
            # Load RMS data
            # The CSVs have no header or variable header (based on previous head command, it has a header)
            # Header: ,0.1-1.0,2.0-10.0,4.0-20.0,10.0-20.0
            # The first column is unnamed and contains datetime.
            
            df = pd.read_csv(file_path)
            
            # Renaissance of the first column
            df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # --- DATE FILTERING ---
            # User requested analysis starting from Dec 2019
            df = df[df['timestamp'] >= '2019-12-01']
            
            if df.empty:
                # print(f"No data after 2019-12-01 for {station_code}")
                continue
            
            # Filter for high frequency noise (4.0-20.0 Hz)
            if '4.0-20.0' in df.columns:
                df['noise_level'] = df['4.0-20.0']
            else:
                # Fallback if specific column name varies, though expected to be consistent
                print(f"Column '4.0-20.0' not found in {os.path.basename(file_path)}. Skipping.")
                continue
                
            # Add metadata to the dataframe
            for col in ['Country name', 'City', 'pop_den_30s', 'latitude', 'longitude', 'Date_LD1']:
                if col in row:
                    df[col] = row[col]
            
            df['Station_Code'] = station_code
            
            # Determine Lockdown Status
            # Date_LD1 is in format DD/MM/YYYY
            if pd.notna(row['Date_LD1']):
                try:
                    ld_date = pd.to_datetime(row['Date_LD1'], dayfirst=True)
                    # Label: 'Lockdown' if timestamp >= ld_date, else 'Pre-Lockdown'
                    # Note: This is a simplification. The study might define specific windows.
                    # For this analysis, we'll use a binary split.
                    df['condition'] = df['timestamp'].apply(lambda x: 'Lockdown' if x >= ld_date else 'Pre-Lockdown')
                except Exception as e:
                    # print(f"Error parsing date {row['Date_LD1']} for {station_code}: {e}")
                    df['condition'] = 'Unknown'
            else:
                df['condition'] = 'No Lockdown Info'

            # Resample to Daily Median as requested
            # Group by Date (ignoring time)
            df['date'] = df['timestamp'].dt.date
            daily_median = df.groupby('date')['noise_level'].median().reset_index()
            daily_median['timestamp'] = pd.to_datetime(daily_median['date'])
            daily_median['Station_Code'] = station_code
            
            # Determine Lockdown Status for the DAY
            if pd.notna(row['Date_LD1']):
                try:
                    ld_date = pd.to_datetime(row['Date_LD1'], dayfirst=True)
                    # Label: 'Lockdown' if date >= ld_date, else 'Pre-Lockdown'
                    daily_median['condition'] = daily_median['timestamp'].apply(lambda x: 'Lockdown' if x >= ld_date else 'Pre-Lockdown')
                    
                    # --- ARTICLE METHODOLOGY NORMALIZATION ---
                    # Calculate Q15 and Q85 of the PRE-LOCKDOWN Daily Medians
                    pre_data = daily_median[daily_median['condition'] == 'Pre-Lockdown']
                    
                    if not pre_data.empty:
                        q15 = pre_data['noise_level'].quantile(0.15)
                        q85 = pre_data['noise_level'].quantile(0.85)
                        iqr_range = q85 - q15
                        
                        if iqr_range > 0:
                            # Normalize: (Value - Q15) / (Q85 - Q15)
                            daily_median['normalized_level'] = (daily_median['noise_level'] - q15) / iqr_range
                            
                            # Article Metric: Centered percentage (assuming median is ~0.5 of range)
                            # Formula from Figure 2 notebook: ((100 * normalized) - 50)
                            daily_median['noise_change_pct'] = (100 * daily_median['normalized_level']) - 50
                        else:
                             # Fallback if range is 0 (unlikely)
                             daily_median['normalized_level'] = np.nan
                             daily_median['noise_change_pct'] = np.nan
                    else:
                        daily_median['normalized_level'] = np.nan
                        daily_median['noise_change_pct'] = np.nan
                        
                except Exception as e:
                    print(f"Error calculating baseline for {station_code}: {e}")
                    daily_median['condition'] = 'Unknown'
                    daily_median['noise_change_pct'] = np.nan
            else:
                daily_median['condition'] = 'No Lockdown Info'
                daily_median['noise_change_pct'] = np.nan

            # Add metadata
            for col in ['Country name', 'City', 'pop_den_30s', 'latitude', 'longitude']:
                if col in row:
                    daily_median[col] = row[col]
            
            processed_dfs.append(daily_median)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    if processed_dfs:
        all_data = pd.concat(processed_dfs, ignore_index=True)
        
        print("Merging with Google Mobility Data...")
        # Merge on Date and Country
        # Ensure date columns match types
        all_data['date'] = pd.to_datetime(all_data['date'])
        
        # Mobility data has 'country_region', our data has 'Country name'
        # We need to ensure we merge correctly.
        merged_data = pd.merge(
            all_data, 
            mobility_df, 
            how='left', 
            left_on=['date', 'Country name'], 
            right_on=['date', 'country_region']
        )
        
        print(f"Saving consolidated DAILY MEDIAN data with {len(merged_data)} rows to {output_file}...")
        merged_data.to_pickle(output_file)
        print("Done.")

    else:
        print("No data processed.")

if __name__ == "__main__":
    load_and_process_data()
