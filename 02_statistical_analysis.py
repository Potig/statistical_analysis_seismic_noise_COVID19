
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_analysis():
    base_dir = "/Users/leonardodesa/Geoestatistica/ThomasLecocq-2020_Science_GlobalQuieting-584a221"
    data_file = os.path.join(base_dir, "processed_data.pkl")
    
    if not os.path.exists(data_file):
        print("Data file not found. Run 01_data_processing.py first.")
        return

    print("Loading DAILY MEDIAN data...")
    df = pd.read_pickle(data_file)
    
    # Check if we have noise_change_pct
    if 'noise_change_pct' not in df.columns:
        print("Warning: noise_change_pct not found. Re-run data processing.")
        return

    # 1. Normality Test (Shapiro-Wilk) - SPLIT POPULATIONS
    print("\n--- Section: Probability (Normality Test by Population) ---")
    
    # Split Data
    pre_data = df[df['condition'] == 'Pre-Lockdown']['noise_change_pct'].dropna()
    lockdown_data = df[df['condition'] == 'Lockdown']['noise_change_pct'].dropna()
    
    # Report Sizes
    print(f"Pre-Lockdown Samples: {len(pre_data)}")
    print(f"Lockdown Samples: {len(lockdown_data)}")

    def test_normality(data, name):
        # Sample if too large for Shapiro
        if len(data) > 4000:
            sample = data.sample(4000, random_state=42)
        else:
            sample = data
        
        stat, p = stats.shapiro(sample)
        print(f"[{name}] Shapiro-Wilk (N={len(sample)}): Stat={stat:.4f}, p={p:.4e}")
        
        # KS Test (Standardized)
        if len(data) > 0:
             z_score = (data - data.mean()) / data.std()
             ks_stat, ks_p = stats.kstest(z_score, 'norm')
             print(f"[{name}] KS Test: D={ks_stat:.4f}, p={ks_p:.4e}")
        
        if p > 0.05:
            print(f"-> {name}: Distribution is Normal (fail to reject H0)")
        else:
            print(f"-> {name}: Distribution is NOT Normal (reject H0)")

    test_normality(pre_data, "Pre-Lockdown")
    test_normality(lockdown_data, "Lockdown")

    # 2. Descriptive & Comparative Stats (T-test & F-Test)
    print("\n--- Section: Comparative Statistics (3-Month Window) ---")
    
    # Define Periods
    # Pre: 2019-12-01 to 2020-02-29
    # Post: 2020-03-01 to 2020-05-31
    pre_start = pd.to_datetime('2019-12-01')
    pre_end = pd.to_datetime('2020-02-29')
    post_start = pd.to_datetime('2020-03-01')
    post_end = pd.to_datetime('2020-05-31')
    
    print(f"Pre-Pandemic Period: {pre_start} to {pre_end}")
    print(f"Post-Pandemic Period: {post_start} to {post_end}")
    
    # Filter Data
    df_pre = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)]['noise_change_pct'].dropna()
    df_post = df[(df['date'] >= post_start) & (df['date'] <= post_end)]['noise_change_pct'].dropna()
    
    print(f"Samples: Pre={len(df_pre)}, Post={len(df_post)}")
    
    if len(df_pre) > 2 and len(df_post) > 2:
        # --- T-Test (Means) ---
        mean_pre = df_pre.mean()
        mean_post = df_post.mean()
        print(f"Mean Noise Change: Pre={mean_pre:.2f}%, Post={mean_post:.2f}%")
        
        t_stat, t_p = stats.ttest_ind(df_pre, df_post, equal_var=False) # Welch's t-test
        print(f"T-Test (Difference of Means): T={t_stat:.4f}, p={t_p:.4e}")
        
        # --- F-Test (Variances) ---
        # H0: Var_pre = Var_post
        # H1: Var_pre != Var_post
        var_pre = df_pre.var()
        var_post = df_post.var()
        print(f"Variance: Pre={var_pre:.2f}, Post={var_post:.2f}")
        
        # F statistic is the ratio of variances. Usually larger/smaller for 1-tailed, 
        # but for 2-tailed we check the ratio 
        if var_pre > var_post:
            f_stat = var_pre / var_post
            df1 = len(df_pre) - 1
            df2 = len(df_post) - 1
        else:
            f_stat = var_post / var_pre
            df1 = len(df_post) - 1
            df2 = len(df_pre) - 1
            
        # P-value for F-test
        p_f = 1 - stats.f.cdf(f_stat, df1, df2)
        p_f = p_f * 2 # Two-tailed
        
        print(f"F-Test (Equality of Variances): F={f_stat:.4f}, p={p_f:.4e}")
        if p_f < 0.05:
            print("-> Significant difference in Variances (Reject H0)")
        else:
            print("-> No significant difference in Variances (Fail to Reject H0)")
            
        # Visualization: Comparative Density Plot (KDE) - REMOVED per user request
        # The user decided this "theoretical density" plot was not useful.
        # Keeping F-test text output but skipping the plot generation.
        pass

    else:
        print("Insufficient data for 3-month comparison.")

    # 2. Descriptive & Comparative Stats (T-test)
    print("\n--- Section: Comparative Statistics ---")
    alpha = 0.05 # Define alpha for all tests
    
    # Visualization: Comparative Histogram
    plt.figure(figsize=(12, 7))
    
    # Filter for plotting (exclude extreme outliers visually)
    range_mask = (df['noise_change_pct'] > -150) & (df['noise_change_pct'] < 150)
    plot_df = df[range_mask]
    
    # Use KDE Plot for cleaner curves (as requested by user preference in other plot)
    # Explicitly defining legend
    sns.kdeplot(data=plot_df, x='noise_change_pct', hue='condition', fill=True, 
                common_norm=False, palette={'Pre-Lockdown': 'gray', 'Lockdown': 'tab:red'},
                alpha=0.3, linewidth=2)
    
    plt.title('Comparação de Distribuições: Pré-Lockdown vs Lockdown')
    plt.xlabel('Variação % da Baseline')
    plt.ylabel('Densidade')
    plt.axvline(0, color='k', linestyle='--', alpha=0.5, label='Baseline (0%)')
    
    # Custom Legend to be absolutely sure
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', lw=2),
                    Line2D([0], [0], color='tab:red', lw=2),
                    Line2D([0], [0], color='k', linestyle='--', lw=1)]
    plt.legend(custom_lines, ['Pré-Lockdown (Cinza)', 'Lockdown (Vermelho)', 'Baseline (0)'], loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.savefig('distribution_comparison.png', dpi=300)
    plt.close()
    print("Saved comparative KDE to 'distribution_comparison.png'")
    
    # Old plotting block removal (handled by this new block)
    # The original code had a second descriptive stats block start here logic flows naturally to "comparative stats" now in next lines



    # 3. Correlation / Regression (Population Density vs Mean % Drop)
    # Regression Analysis: Population Density vs Median % Drop (Pre vs Post)
    print("\n--- Section: Regression Analysis (Pre vs Post) ---")
    
    # 1. Post-Lockdown Stats (Already calculated)
    lockdown_df = df[df['condition'] == 'Lockdown']
    post_stats = lockdown_df.groupby('Station_Code')['noise_change_pct'].median().reset_index()
    post_stats['Period'] = 'Pós-Lockdown (Vermelho)'

    # 2. Pre-Lockdown Stats (New)
    pre_df = df[df['condition'] == 'Pre-Lockdown']
    pre_stats = pre_df.groupby('Station_Code')['noise_change_pct'].median().reset_index()
    pre_stats['Period'] = 'Pré-Lockdown (Azul)'
    
    # Combined Data
    combined_stats = pd.concat([post_stats, pre_stats])
    
    # Add Metadata
    meta = df.groupby('Station_Code')[['pop_den_30s', 'latitude', 'longitude']].first().reset_index()
    analysis_df = pd.merge(combined_stats, meta, on='Station_Code')
    
    # Filter valid data
    reg_df = analysis_df.dropna(subset=['noise_change_pct', 'pop_den_30s'])
    
    if len(reg_df) > 10:
        plt.figure(figsize=(10, 6))
        
        # Prepare Log Scale for X
        reg_df['pop_den_log'] = reg_df['pop_den_30s'] + 1
        
        # Plot Scatter with Hue
        sns.scatterplot(data=reg_df, x='pop_den_log', y='noise_change_pct', hue='Period', 
                        palette={'Pré-Lockdown (Azul)': 'tab:blue', 'Pós-Lockdown (Vermelho)': 'tab:red'},
                        alpha=0.6, style='Period')
        
        # Add Regression Line ONLY for Post-Lockdown (to show the trend there)
        post_reg_df = reg_df[reg_df['Period'] == 'Pós-Lockdown (Vermelho)']
        sns.regplot(data=post_reg_df, x='pop_den_log', y='noise_change_pct', scatter=False, color='tab:red', line_kws={'linestyle': '--'})
        
        # Stats for Post-Lockdown
        slope, intercept, r_value, p_value, std_err = stats.linregress(post_reg_df['pop_den_log'], post_reg_df['noise_change_pct'])
        print(f"Linear Regression (Post): y = {slope:.4f}x + {intercept:.4f}, R2={r_value**2:.4f}")
        
        plt.xscale('log')
        plt.title('Densidade Populacional vs Variação do Ruído (Pré vs Pós)')
        plt.xlabel('Densidade Populacional (Escala Log, x+1)')
        plt.ylabel('Variação % Mediana do Ruído')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5) # Baseline
        
        plt.legend()
        plt.savefig('regression_pop_vs_drop.png')
        plt.close()
        
        # Save stats for report
        analysis_df.to_csv('station_stats_refined.csv', index=False)
        print("Regression plot upgraded with Pre-Lockdown comparison.")
    else:
        print("Not enough data for meaningful regression analysis.")



    # --- NEW SECTION: Mobility Correlation & T-Tests ---
    print("\n--- Section: Google Mobility Analysis (Refined) ---")
    mobility_cols = [
        'retail_and_recreation_percent_change_from_baseline',
        'grocery_and_pharmacy_percent_change_from_baseline',
        'parks_percent_change_from_baseline',
        'transit_stations_percent_change_from_baseline',
        'workplaces_percent_change_from_baseline',
        'residential_percent_change_from_baseline'
    ]
    
    # Check if mobility columns exist and have data
    available_mob_cols = [c for c in mobility_cols if c in df.columns]
    
    if available_mob_cols:
        print(f"Found {len(available_mob_cols)} mobility columns.")
        
        # METHOD 1: Global Temporal Correlation (Smoothed)
        # We want to compare the GLOBAL signal of quieting vs the GLOBAL signal of mobility
        # Applying 7-Day Rolling Average to capture the TREND (removing weekly noise)
        
        print("\n--- Method 1: Global Temporal Correlation (7-Day Smoothed) ---")
        daily_global = df.groupby('date')[['noise_change_pct'] + available_mob_cols].median().reset_index()
        
        # Apply Smoothing
        numeric_cols = ['noise_change_pct'] + available_mob_cols
        daily_global[numeric_cols] = daily_global[numeric_cols].rolling(window=7, center=True).mean()
        
        daily_global = daily_global.dropna()
        
        corr_results = []
        
        if not daily_global.empty:
            print(f"Calculating Global Smoothed correlations using {len(daily_global)} days.")
            for mob_col in available_mob_cols:
                r, p = stats.pearsonr(daily_global['noise_change_pct'], daily_global[mob_col])
                
                # Calculate T-Statistic for Correlation
                n = len(daily_global)
                if abs(r) < 1.0:
                    t_stat_corr = r * np.sqrt((n - 2) / (1 - r**2))
                else:
                    t_stat_corr = np.inf # Should not happen with real data usually
                
                # Hypothesis Test Details
                dof_corr = n - 2
                t_crit_corr = stats.t.ppf(1 - alpha/2, dof_corr)
                is_sig = p < alpha
                
                # Validation: Spearman Rank Correlation (Non-Parametric)
                rho, p_rho = stats.spearmanr(daily_global['noise_change_pct'], daily_global[mob_col])
                
                cat_name = mob_col.replace('_percent_change_from_baseline', '')
                corr_results.append({
                    'Mobility_Category': cat_name, 
                    'Global_Temporal_R': r, 
                    'Global_Temporal_T': t_stat_corr,
                    'DoF_Corr': dof_corr,
                    'T_Crit_Corr': t_crit_corr,
                    'Global_Temporal_P': p,
                    'Significant_Corr': 'Sim' if is_sig else 'Não',
                    'Spearman_Rho': rho
                })
                print(f"Global Smoothed Correlation (Noise vs {cat_name}): Pearson r={r:.4f}, Spearman rho={rho:.4f}")
        
        # METHOD 2: Per-Station Correlation Median
        # Calculate correlation for each station individually, then take the median of those correlations.
        print("\n--- Method 2: Per-Station Correlation Distribution ---")
        station_corrs = {cat: [] for cat in available_mob_cols}
        
        for station, group in df.groupby('Station_Code'):
            # clean NaNs for this station
            clean_group = group.dropna(subset=['noise_change_pct'] + available_mob_cols)
            if len(clean_group) < 10: # Need minimum points for reliable correlation
                continue
                
            for mob_col in available_mob_cols:
                # Need variance to correlate
                if clean_group[mob_col].nunique() > 1 and clean_group['noise_change_pct'].nunique() > 1:
                    r, _ = stats.pearsonr(clean_group['noise_change_pct'], clean_group[mob_col])
                    if not np.isnan(r):
                        station_corrs[mob_col].append(r)
        
        # Add per-station medians to results
        final_results = []
        for res in corr_results:
            cat = res['Mobility_Category']
            # Find the original column name to look up in station_corrs
            orig_col = [c for c in available_mob_cols if c.replace('_percent_change_from_baseline', '') == cat][0]
            
            if station_corrs[orig_col]:
                res['Median_Station_R'] = np.median(station_corrs[orig_col])
                print(f"Median Station Correlation (Noise vs {cat}): r={res['Median_Station_R']:.4f}")
            else:
                res['Median_Station_R'] = np.nan
            final_results.append(res)
        
        # Rename columns for Portuguese output
        if final_results: # Use final_results here, as it's the list being built
            corr_df = pd.DataFrame(final_results)
            # Translate categories
            cat_map = {
                'retail_and_recreation': 'Varejo e Lazer',
                'grocery_and_pharmacy': 'Mercado e Farmácia',
                'parks': 'Parques',
                'transit_stations': 'Estações de Transporte',
                'workplaces': 'Locais de Trabalho',
                'residential': 'Residencial'
            }
            corr_df['Mobility_Category'] = corr_df['Mobility_Category'].map(cat_map).fillna(corr_df['Mobility_Category'])
            
            # Rename columns
            corr_df = corr_df.rename(columns={
                'Mobility_Category': 'Categoria de Mobilidade',
                'Global_Temporal_R': 'R (Pearson)',
                'Global_Temporal_T': 'T-Calculado (Corr)',
                'DoF_Corr': 'Graus de Liberdade',
                'T_Crit_Corr': 'T-Crítico',
                'Global_Temporal_P': 'P-Valor (Global)',
                'Significant_Corr': 'Significativo?',
                'Spearman_Rho': 'Rho (Spearman)',
                'Median_Station_R': 'R (Mediana Estações)'
            })
            corr_df.to_csv('mobility_correlations.csv', index=False)
            
        # 2. T-Tests for Mobility Changes during Lockdown (Test against 0)
        # Using daily global medians to avoid inflating N
        print("\n--- Mobility T-Tests (Lockdown Days vs Baseline=0) ---")
        # Filter global days that are considered 'Lockdown' (approximate by date or aggregated condition)
        # Re-using the raw data for T-test distribution is fine, but let's use global daily means for robustness?
        # Actually sticking to raw distribution for T-test is acceptable to show "days are lower", 
        # but let's stick to the previous method for T-test as it answers "was mobility lower on average?"
        
        lockdown_mob = df[df['condition'] == 'Lockdown'].dropna(subset=available_mob_cols)
        
        ttest_results = []
        alpha = 0.05
        
        print(f"Calculating Detailed T-Tests for {len(available_mob_cols)} categories...")
        
        for mob_col in available_mob_cols:
            data = lockdown_mob[mob_col]
            n = len(data)
            if n > 1:
                t_stat, p_val = stats.ttest_1samp(data, 0)
                mean_val = data.mean()
                cat_name = mob_col.replace('_percent_change_from_baseline', '')
                
                # Detailed Stats
                dof = n - 1
                t_crit = stats.t.ppf(1 - alpha/2, dof)
                
                # Validation: Wilcoxon Signed-Rank Test (Non-Parametric)
                # Tests if median difference from 0 is significant
                # Wilcoxon requires differences to be non-zero mostly, handles standard ranking
                try:
                    w_stat, p_wilc = stats.wilcoxon(data - 0)
                except Exception as e:
                    p_wilc = np.nan
                    print(f"Wilcoxon failed for {cat_name}: {e}")
                
                ttest_results.append({
                    'Mobility_Category': cat_name, 
                    'Mean_Change': mean_val, 
                    'T_Stat': t_stat,
                    'T_Critical': t_crit,
                    'DoF': dof,
                    'P_Value': p_val,
                    'Significant': 'Sim' if p_val < alpha else 'Não',
                    'P_Wilcoxon': p_wilc
                })
        
        ttest_df = pd.DataFrame(ttest_results)
        
        # Translate categories
        cat_map = {
            'retail_and_recreation': 'Varejo e Lazer',
            'grocery_and_pharmacy': 'Mercado e Farmácia',
            'parks': 'Parques',
            'transit_stations': 'Estações de Transporte',
            'workplaces': 'Locais de Trabalho',
            'residential': 'Residencial'
        }
        ttest_df['Mobility_Category'] = ttest_df['Mobility_Category'].map(cat_map).fillna(ttest_df['Mobility_Category'])
        
        # Rename columns
        ttest_df = ttest_df.rename(columns={
            'Mobility_Category': 'Categoria de Mobilidade',
            'Mean_Change': 'Variação Média (%)',
            'T_Stat': 'T-Calculado',
            'T_Critical': 'T-Crítico',
            'DoF': 'Graus de Liberdade',
            'P_Value': 'P-Valor (T-Test)',
            'P_Wilcoxon': 'P-Valor (Wilcoxon)',
            'Significant': 'Significativo?'
        })
        
        ttest_df.to_csv('mobility_ttests.csv', index=False)

    else:
        print("No mobility columns found in dataset.")

    # 4. Spatial Map of % Reduction
    print("\n--- Section: Spatial Analysis ---")
    try:
        import geopandas as gpd
        # Use locally downloaded shapefile
        world = gpd.read_file("natural_earth_data/ne_110m_admin_0_countries.shp")
        use_geopandas = True
    except Exception as e:
        print(f"Could not use geopandas for map: {e}")
        use_geopandas = False

    # --- Standard Change Map (Single - Lockdown Only) ---
    plt.figure(figsize=(15, 8))
    # Filter for Lockdown period only to show the "quieting"
    # Using the Period column we created earlier or checking 'condition' if merged
    # analysis_df has 'Period' from the regression block merge
    map_plot_data = analysis_df[analysis_df['Period'] == 'Pós-Lockdown (Vermelho)']
    
    if use_geopandas:
        ax = plt.gca()
        world.plot(ax=ax, color='lightgrey', edgecolor='white')
        sc = plt.scatter(map_plot_data['longitude'], map_plot_data['latitude'], 
                         c=map_plot_data['noise_change_pct'], cmap='RdBu_r', 
                         vmin=-50, vmax=50, s=60, alpha=0.9, edgecolor='k', zorder=5)
    else:
        sc = plt.scatter(map_plot_data['longitude'], map_plot_data['latitude'], 
                        c=map_plot_data['noise_change_pct'], cmap='RdBu_r', 
                        vmin=-50, vmax=50, s=50, alpha=0.8, edgecolor='k')
    
    plt.colorbar(sc, label='Variação do Ruído (%)', shrink=0.6)
    plt.title('Mapa Global de Variação do Ruído Sísmico (Lockdown)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig('spatial_map_change.png')
    plt.close()
    
    # --- NEW: Side-by-Side Absolute Level Comparison ---
    print("\n--- Section: Spatial Comparison Map (Pre vs Post) ---")
    
    # Calculate Median Absolute Noise Level per station for each period
    spatial_stats = df.groupby(['Station_Code', 'condition'])['noise_level'].median().unstack()
    # Ensure we have both columns
    if 'Pre-Lockdown' in spatial_stats.columns and 'Lockdown' in spatial_stats.columns:
        # Add coords
        coords = df.groupby('Station_Code')[['longitude', 'latitude']].first()
        map_data = spatial_stats.join(coords)
        
        # Define common log-scale for absolute noise (it varies wildly)
        # Using simple Min/Max for color scaling
        vmin = map_data[['Pre-Lockdown', 'Lockdown']].min().min()
        vmax = map_data[['Pre-Lockdown', 'Lockdown']].max().max()
        # Log scale for color
        import matplotlib.colors as colors
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
        
        labels = ['Pre-Lockdown', 'Lockdown']
        
        for i, ax in enumerate(axes):
            period = labels[i]
            if use_geopandas:
                world.plot(ax=ax, color='lightgrey', edgecolor='white')
            
            # Plot
            data = map_data[period]
            sc2 = ax.scatter(map_data['longitude'], map_data['latitude'], 
                             c=data, cmap='inferno', # 'inferno' goes Black->Red->Yellow (Good for intensity)
                             norm=norm, s=80, alpha=0.9, edgecolor='k', zorder=5)
            
            ax.set_title(f'Nível de Ruído Absoluto ({period})', fontsize=14)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # Remove axes for clean look
            ax.set_xticks([])
            ax.set_yticks([])

        # Shared Colorbar
        cbar = fig.colorbar(sc2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
        cbar.set_label('Deslocamento RMS (Escala Log)', fontsize=12)
        
        plt.suptitle('Comparação Global: "Apagão" do Ruído Sísmico', fontsize=16)
        plt.savefig('spatial_comparison_pre_post.png', dpi=300)
        plt.close()
        print("Comparison map saved to 'spatial_comparison_pre_post.png'")
    else:
        print("Skipping spatial comparison: Missing Pre/Post data columns.")







    # 5. Time Series Plot (Global Evolution)
    print("\n--- Section: Time Series Visualization ---")
    # Group by date to get global signal
    global_ts = df.groupby('date')['noise_change_pct'].median().reset_index()
    global_ts['date'] = pd.to_datetime(global_ts['date'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(global_ts['date'], global_ts['noise_change_pct'], color='tab:blue', linewidth=2, label='Variação Mediana Global')
    
    # Add a smoothed rolling average
    rolling = global_ts.set_index('date')['noise_change_pct'].rolling('7D', center=True).mean()
    plt.plot(rolling.index, rolling, color='tab:red', linewidth=3, linestyle='-', label='Média Móvel de 7 Dias')
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Baseline (0%)')
    
    # Highlight general lockdown period (approx March 2020)
    plt.axvline(pd.to_datetime('2020-03-11'), color='gray', linestyle=':', label='Declaração de Pandemia da OMS')
    
    plt.title('Quieting Global do Ruído Sísmico ao Longo do Tempo (2020)')
    plt.ylabel('Variação do Ruído relativa à Baseline (%)')
    plt.xlabel('Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('global_time_series.png')
    plt.close()

    # 6. All Stations Spaghetti Plot
    print("\n--- Section: All Stations Visualization ---")
    # All Stations Heatmap
    print("\n--- Section: All Stations Visualization (Heatmap) ---")
    import matplotlib.dates as mdates # Ensure import
    plt.figure(figsize=(15, 10))
    
    # Create pivot_df (missing from previous step)
    pivot_df = df.pivot(index='date', columns='Station_Code', values='noise_change_pct')
    pivot_df.index = pd.to_datetime(pivot_df.index)
    
    # pivot_df has Index=Date, Columns=Station_Code
    # We need to sort columns (Stations) by Latitude
    
    # Get metadata for sorting
    station_meta = df.groupby('Station_Code')['latitude'].first()
    
    # Get available stations in pivot
    available_stations = pivot_df.columns
    
    # Filter metadata
    station_meta = station_meta.reindex(available_stations)
    
    # Sort stations by Latitude (North to South usually, or Vice Versa)
    sorted_stations = station_meta.sort_values(ascending=False).index
    
    # Reorder pivot_df
    sorted_pivot = pivot_df[sorted_stations]
    
    # Create Meshgrid for pcolormesh
    # We need numeric dates for plotting or just use indices
    dates = mdates.date2num(sorted_pivot.index)
    # Stations on Y axis (0 to N)
    y_vals = np.arange(len(sorted_stations) + 1)
    x_vals = dates
    
    # Prepare data matrix (Transpose so Stations are Rows)
    data_matrix = sorted_pivot.T.values # Shape: (Stations, Days)
    
    # Plot Pcolormesh
    # RdBu_r: Red (High/Increase), Blue (Low/Decrease) - Reversed so Blue is Quieting
    # Center at 0 using vmin/vmax symmetric or divergent norm
    
    # Using pcolormesh
    # Note: x_vals and y_vals define the EDGES of the grid squares
    # We need edges for dates which can be tricky.
    # Simpler: Use imshow with aspect='auto' and manual extent
    
    ax = plt.gca()
    
    # Define bounds for color map to center 0 at white/neutral
    # Common range for this study is +/- 50% or +/- 100%
    # We use vmin=-60, vmax=60 to highlight the drop better
    
    im = ax.pcolormesh(sorted_pivot.index, np.arange(len(sorted_stations)), data_matrix, 
                       cmap='RdBu_r', vmin=-70, vmax=70, shading='auto')
    
    # Formatting
    plt.colorbar(im, label='Variação do Ruído (%)', pad=0.01)
    
    plt.title(f'Variação do Ruído Sísmico por Estação (Ordenado por Latitude)\nTotal: {len(sorted_stations)} Estações', fontsize=14)
    plt.ylabel('Estações (Norte -> Sul)', fontsize=12)
    plt.xlabel('Data', fontsize=12)
    
    # Format X-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Remove Y ticks (too many station names)
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('all_stations_heatmap.png', dpi=300)
    plt.close()
    print("Heatmap saved to 'all_stations_heatmap.png'")

    # All Stations Record Section (Wiggle Plot)
    print("\n--- Section: All Stations visualization (Record Section) ---")
    plt.figure(figsize=(12, 16)) # Taller figure for many stations
    
    # Use sorted_pivot (Stations as columns, Date as index)
    # We will plot each station as a line offset by an integer
    
    # Scaling factor: How much 100% change occupies on Y-axis
    # If standard spacing is 1.0, and we want +/- 50% to take up 0.5 units
    scale_factor = 0.5 / 50 # 50% change = 0.5 units visual
    
    # Create offset for each station
    # We iterate and plot
    stations = sorted_pivot.columns
    n_stations = len(stations)
    
    # Define Y-values for stations (0 to N-1)
    # We plot from top to bottom (Latitude N -> S matches top -> bottom?)
    # Usually standard plots are Y=0 at bottom.
    # Heatmap had N->S as top->bottom logic if mapped to matrix rows 0..N.
    # Let's keep Index 0 = Top Latitude (High Y) or Bottom Y?
    # Let's simple plot: Index 0 (North) at Top (Y=N). Index N (South) at Bottom (Y=0).
    
    ax = plt.gca()
    
    # Clip data for visualization to avoid massive overlap from outliers
    # Clip at +/- 150%
    clipped_data = sorted_pivot.clip(-150, 150)
    
    print("Plotting Record Section lines...")
    for i, station in enumerate(stations):
        # Position: N-1-i so first station (North) is at top
        y_center = n_stations - 1 - i 
        
        # Data
        series = clipped_data[station]
        
        # Handle NaNs (break lines)
        # Plot: X=Date, Y= (Value * scale) + y_center
        y_values = (series * scale_factor) + y_center
        
        # Plot distinctive 'Quieting' in Blue, 'Noise' in Red? 
        # Or just black lines like ObsPy section. User said "like obspy".
        # ObsPy section is usually black lines.
        plt.plot(series.index, y_values, color='black', linewidth=0.6)
        
        # Optional: Fill negative parts (Quieting) with blue?
        # This makes the "silence" pop out.
        # Let's fill below y_center for negative values
        # We need to handle NaNs for fill_between
        valid_indices = series.dropna().index
        if len(valid_indices) > 0:
             # Re-extract to ensure alignment
             v_series = series.loc[valid_indices]
             v_y_values = y_values.loc[valid_indices]
             
             # Fill logic: where series < 0, fill between y_value and y_center
             # ax.fill_between(valid_indices, v_y_values, y_center, 
             #                where=(v_series < 0), color='blue', alpha=0.3)
             pass # Skipping fill to keep it clean like typical section for now

    # Y-Axis Labels
    # Label every 5th station to avoid clutter
    step = 5
    yticks = np.arange(n_stations - 1, -1, -step) # corresponding to indices 0, 5, 10
    ytick_labels = [stations[n_stations - 1 - y] for y in yticks] # map back to station names
    
    # Correction: The y_center logic was N-1-i.
    # If i=0, y=N-1. Station=stations[0].
    # So yticks should properly map to stations.
    # Let's just set ticks at integer locations
    plt.yticks(np.arange(n_stations), ["" for _ in range(n_stations)]) # Clear all first
    
    # Set specific ticks
    # We want labels for stations[0], stations[5]...
    # stations[0] is at y = N-1
    # stations[5] is at y = N-6
    n_step = 5
    label_indices = range(0, n_stations, n_step)
    label_y_pos = [n_stations - 1 - i for i in label_indices]
    label_text = [stations[i] for i in label_indices]
    
    plt.yticks(label_y_pos, label_text, fontsize=8)
    
    plt.title(f'Record Section (Todas as Estações)\nOrdenado por Latitude (Topo = Norte)', fontsize=14)
    plt.xlabel('Data')
    plt.ylabel('Estações')
    
    # X-Axis format
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.grid(False) # Sections usually don't have grid, or minimal vertical
    plt.axvline(pd.to_datetime('2020-03-11'), color='red', linestyle='--', alpha=0.7, label='Pandemia')
    
    plt.tight_layout()
    plt.savefig('all_stations_section_plot.png', dpi=300)
    plt.close()
    print("Section plot saved to 'all_stations_section_plot.png'")

    # 7. Noise vs Mobility Comparison Plot
    print("\n--- Section: Noise vs Mobility Visualization ---")
    if not daily_global.empty:
        plt.figure(figsize=(14, 7))
        
        # Plot Seismic Noise (Primary Axis)
        # daily_global is already 7-day smoothed from Method 1 calculations
        plt.plot(daily_global['date'], daily_global['noise_change_pct'], 
                 color='black', linewidth=3, label='Ruído Sísmico (Mediana Global)')
        
        # Plot Mobility Columns
        # We can pick key ones: Retail, Transit, Residential
        mob_styles = {
            'retail_and_recreation_percent_change_from_baseline': {'color': 'tab:blue', 'label': 'Varejo e Lazer', 'style': '--'},
            'transit_stations_percent_change_from_baseline': {'color': 'tab:orange', 'label': 'Estações de Transporte', 'style': '--'},
            'residential_percent_change_from_baseline': {'color': 'tab:green', 'label': 'Residencial', 'style': ':'}
        }
        
        for col, style in mob_styles.items():
            if col in daily_global.columns:
                plt.plot(daily_global['date'], daily_global[col], 
                         color=style['color'], linestyle=style['style'], linewidth=2, label=style['label'])
        
        plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
        plt.title('Comparação: Ruído Sísmico Global vs Tendências de Mobilidade Google (Suavizado 7 Dias)')
        plt.ylabel('Variação % da Baseline')
        plt.xlabel('Data')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format Date Axis
        import matplotlib.dates as mdates
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        
        plt.savefig('noise_mobility_timeseries.png')
        plt.close()
        print("Noise vs Mobility plot generated.")

    # 8. Scatter Plot: Noise vs Mobility (Regression)
    print("\n--- Section: Scatter Plot Visualization ---")
    if not daily_global.empty:
        plt.figure(figsize=(10, 8))
        
        mob_styles = {
            'retail_and_recreation_percent_change_from_baseline': {'color': 'tab:blue', 'label': 'Varejo e Lazer'},
            'transit_stations_percent_change_from_baseline': {'color': 'tab:orange', 'label': 'Estações de Transporte'},
            'residential_percent_change_from_baseline': {'color': 'tab:green', 'label': 'Residencial'}
        }
        
        for col, style in mob_styles.items():
            if col in daily_global.columns:
                # Calculate regression for legend
                slope, intercept, r_value, p_value, std_err = stats.linregress(daily_global[col], daily_global['noise_change_pct'])
                
                # Plot Scatter
                plt.scatter(daily_global[col], daily_global['noise_change_pct'], 
                            color=style['color'], alpha=0.5, s=30, label=f"{style['label']} (R²={r_value**2:.2f})")
                
                # Plot Regression Line
                x_vals = np.array([daily_global[col].min(), daily_global[col].max()])
                plt.plot(x_vals, slope*x_vals + intercept, color=style['color'], linewidth=2)

        plt.title('Correlação: Mobilidade vs Variação do Ruído Sísmico (Suavizado 7 Dias)')
        plt.xlabel('Variação da Mobilidade Google (%)')
        plt.ylabel('Variação do Ruído Sísmico (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        
        plt.savefig('scatter_noise_vs_mobility.png')
        plt.close()
        print("Scatter plot generated.")
    
    
    # 9. Relative Frequency Table
    print("\n--- Section: Relative Frequency Distribution ---")
    if not daily_global.empty:
        # Define 10% bins from -100 to +60
        bins = range(-100, 70, 10)
        labels = [f"{i} a {i+10}%" for i in bins[:-1]]
        
        freq_df = pd.DataFrame({'Intervalo': labels})
        freq_df.set_index('Intervalo', inplace=True)
        
        # Columns to analyze
        cols_to_analyze = {
            'Ruído Global': 'noise_change_pct',
        }
        # Add mobility cols if they exist
        mob_map = {
            'retail_and_recreation_percent_change_from_baseline': 'Varejo',
            'transit_stations_percent_change_from_baseline': 'Transporte',
            'residential_percent_change_from_baseline': 'Residencial'
        }
        for mob_col, nice_name in mob_map.items():
            if mob_col in daily_global.columns:
                cols_to_analyze[nice_name] = mob_col
        
        for name, col_name in cols_to_analyze.items():
            # Cut into bins
            cats = pd.cut(daily_global[col_name], bins=bins, labels=labels, right=False)
            # Calculate relative frequency
            counts = cats.value_counts(normalize=True).sort_index()
            freq_df[name] = counts
            
        # Fill NaNs with 0
        freq_df = freq_df.fillna(0)
        
        # Add Total Row to verify validation (should be ~1.0)
        # We process the sum before formatting percentages if we were doing formatting, 
        # but here values are floats 0-1.
        total_row = freq_df.sum()
        total_row.name = 'Total'
        freq_df = pd.concat([freq_df, total_row.to_frame().T])
        
        freq_df.reset_index().to_csv('frequency_distribution.csv', index=False)
        print("Frequency distribution table generated.")

    print("Analysis Complete. Figures generated.")

if __name__ == "__main__":
    perform_analysis()
