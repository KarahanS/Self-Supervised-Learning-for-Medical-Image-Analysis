import pandas as pd
import numpy as np
from scipy import stats
import os

# Configurable parameters
input_csv_path = 'paper_downstream_results.csv'  # Path to the input CSV file
output_csv_path = 'output_with_confidence_intervalss.csv'  # Path to the output CSV file
confidence_level = 0.95  # Confidence level for intervals

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate the confidence interval for a given dataset.
    
    Parameters:
        data (list or np.array): Array-like dataset to calculate the interval for.
        confidence (float): Confidence level for the interval.
    
    Returns:
        tuple: Mean of data, lower bound, and upper bound of the confidence interval.
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard Error of the Mean
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_of_error = t_value * sem
    return mean, mean - margin_of_error, mean + margin_of_error

def main():
    # Read the input CSV file
    print(f"Reading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # Check if all groups have the same number of seeds
    print("Checking the consistency of the number of seeds across all runs...")
    seed_counts = df.groupby(['model_name', 'downstream_classifier_name']).size()
    if seed_counts.nunique() != 1:
        inconsistent_models = seed_counts[seed_counts != seed_counts.mode()[0]]
        print("Error: Not all runs have the same number of seeds for the following model-classifier combinations:")
        for idx in inconsistent_models.index:
            print(f"Model: {idx[0]}, Classifier: {idx[1]}, Seed Count: {inconsistent_models[idx]}")
        return
    
    # Group by model_name and downstream_classifier_name to calculate confidence intervals
    results = []

    print("Calculating confidence intervals...")
    grouped = df.groupby(['model_name', 'downstream_classifier_name'])
    for (model_name, classifier_name), group in grouped:
        accuracies = group['test_acc'].values
        balanced_accuracies = group['balanced_acc'].values
        aurocs = group['test_auroc'].values
        
        if len(accuracies) > 1:
            mean_acc, lower_bound_acc, upper_bound_acc = calculate_confidence_interval(accuracies, confidence=confidence_level)
            mean_bal_acc, lower_bound_bal_acc, upper_bound_bal_acc = calculate_confidence_interval(balanced_accuracies, confidence=confidence_level)
            mean_auroc, lower_bound_auroc, upper_bound_auroc = calculate_confidence_interval(aurocs, confidence=confidence_level)
        else:
            mean_acc = accuracies[0]
            lower_bound_acc = upper_bound_acc = mean_acc
            mean_bal_acc = balanced_accuracies[0]
            lower_bound_bal_acc = upper_bound_bal_acc = mean_bal_acc
            mean_auroc = aurocs[0]
            lower_bound_auroc = upper_bound_auroc = mean_auroc
        
        # Add results to the list, keeping all original columns
        result_dict = group.iloc[0].to_dict()  # Copy original row information
        result_dict.update({
            'mean_test_acc': mean_acc,
            'ci_lower_test_acc': lower_bound_acc,
            'ci_upper_test_acc': upper_bound_acc,
            'mean_balanced_acc': mean_bal_acc,
            'ci_lower_balanced_acc': lower_bound_bal_acc,
            'ci_upper_balanced_acc': upper_bound_bal_acc,
            'mean_test_auroc': mean_auroc,
            'ci_lower_test_auroc': lower_bound_auroc,
            'ci_upper_test_auroc': upper_bound_auroc,
            'number_of_seeds': len(accuracies)
        })
        results.append(result_dict)
    
    # Create a DataFrame from results
    result_df = pd.DataFrame(results)

    # Drop original columns for accuracy and AUROC
    if 'test_acc' in result_df.columns:
        result_df = result_df.drop(columns=['test_acc'])
    if 'balanced_acc' in result_df.columns:
        result_df = result_df.drop(columns=['balanced_acc'])
    if 'test_auroc' in result_df.columns:
        result_df = result_df.drop(columns=['test_auroc'])
    if 'seed' in result_df.columns:
        result_df = result_df.drop(columns=['seed'])
    
    # Load existing output CSV if it exists
    rows_overwritten = 0
    rows_added = 0
    if os.path.exists(output_csv_path) and not os.path.getsize(output_csv_path) == 0:
        print(f"Reading existing data from {output_csv_path}...")

        existing_df = pd.read_csv(output_csv_path)

        if existing_df.empty:
            print("The output file is empty. All results will be added.")
            merged_df = result_df
        else:
            # Merge results with the existing data, updating where necessary
            merged_df = pd.merge(existing_df, result_df, on=['model_name', 'downstream_classifier_name'], how='outer', suffixes=('', '_new'))

        # Identify overwritten rows and new rows
        for index, row in merged_df.iterrows():
            if pd.notna(row['mean_test_acc_new']):  # Check if this row has updated data
                if pd.notna(row['mean_test_acc']):  # It was already present, so it's an overwrite
                    rows_overwritten += 1
                else:  # It was not present, so it's an addition
                    rows_added += 1

        # Keep the new data where it exists
        combined_df = merged_df.drop(columns=['mean_test_acc', 'ci_lower_test_acc', 'ci_upper_test_acc', 
                                              'mean_balanced_acc', 'ci_lower_balanced_acc', 'ci_upper_balanced_acc',
                                              'mean_test_auroc', 'ci_lower_test_auroc', 'ci_upper_test_auroc',
                                              'number_of_seeds'])
    else:
        combined_df = result_df
        rows_added = len(result_df)

    # Write the results to the output CSV file
    print(f"Writing results to {output_csv_path}...")
    combined_df.to_csv(output_csv_path, index=False)
    
    # Total rows processed (excluding the header)
    total_rows = len(combined_df)
    print(f"{rows_overwritten} rows overwritten, {rows_added} rows added for a total of {total_rows} rows (excluding the header).")

if __name__ == '__main__':
    main()
