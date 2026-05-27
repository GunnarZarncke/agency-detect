#!/usr/bin/env python
"""
Discrete-Appropriate CMI Estimators for Agent Detection

Analysis of CMI estimation methods suitable for discrete data with:
- Limited cardinality (3-10 values per variable)
- Reasonable sample size (15K samples)
- Low-dimensional conditioning (2-4D)
"""

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score
from collections import Counter
import itertools
from math import log2, log
import warnings
warnings.filterwarnings('ignore')


def plugin_cmi_estimator(X, Y, Z, base=2):
    """
    Method 1: Plug-in estimator using direct frequency counting.
    
    Best for: Pure discrete data with reasonable sample sizes
    Pros: Theoretically exact for discrete data, fast
    Cons: Suffers from curse of dimensionality, needs smoothing
    """
    # Convert to tuples for hashing
    n = len(X)
    
    # Convert Z to tuples for hashing
    if Z.ndim > 1:
        Z_tuples = [tuple(row) for row in Z]
    else:
        Z_tuples = Z
    
    # Count joint frequencies
    xyz_counts = Counter(zip(X, Y, Z_tuples))
    xz_counts = Counter(zip(X, Z_tuples))
    yz_counts = Counter(zip(Y, Z_tuples))
    z_counts = Counter(Z_tuples)
    
    # Get all possible combinations
    x_values = set(X)
    y_values = set(Y)
    z_values = set(Z_tuples)
    
    # Calculate CMI: I(X;Y|Z) = sum over x,y,z of P(x,y,z) * log(P(x,y,z)*P(z) / (P(x,z)*P(y,z)))
    cmi = 0.0
    
    for x in x_values:
        for y in y_values:
            for z in z_values:
                p_xyz = xyz_counts.get((x, y, z), 0) / n
                p_xz = xz_counts.get((x, z), 0) / n  
                p_yz = yz_counts.get((y, z), 0) / n
                p_z = z_counts.get(z, 0) / n
                
                if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                    cmi += p_xyz * log(p_xyz * p_z / (p_xz * p_yz), base)
    
    return cmi


def smoothed_plugin_cmi(X, Y, Z, alpha=0.5, base=2):
    """
    Method 2: Plug-in estimator with Laplace smoothing.
    
    Best for: Discrete data with potential zero counts
    Pros: Handles zero frequencies, more robust
    Cons: Introduces bias, needs tuning of alpha
    """
    n = len(X)
    
    # Get cardinalities
    card_x = len(set(X))
    card_y = len(set(Y))
    card_z = len(set(tuple(row) for row in Z)) if Z.ndim > 1 else len(set(Z))
    
    # Convert Z to tuples for hashing
    if Z.ndim > 1:
        Z_tuples = [tuple(row) for row in Z]
    else:
        Z_tuples = Z
    
    # Count with smoothing
    xyz_counts = Counter(zip(X, Y, Z_tuples))
    xz_counts = Counter(zip(X, Z_tuples))
    yz_counts = Counter(zip(Y, Z_tuples))
    z_counts = Counter(Z_tuples)
    
    # All possible combinations
    x_values = sorted(set(X))
    y_values = sorted(set(Y))
    z_values = sorted(set(Z_tuples))
    
    cmi = 0.0
    
    for x in x_values:
        for y in y_values:
            for z in z_values:
                # Smoothed counts
                n_xyz = xyz_counts.get((x, y, z), 0) + alpha
                n_xz = xz_counts.get((x, z), 0) + alpha * card_y
                n_yz = yz_counts.get((y, z), 0) + alpha * card_x  
                n_z = z_counts.get(z, 0) + alpha * card_x * card_y
                
                # Smoothed probabilities
                p_xyz = n_xyz / (n + alpha * card_x * card_y * card_z)
                p_xz = n_xz / (n + alpha * card_y * card_x * card_z) 
                p_yz = n_yz / (n + alpha * card_x * card_y * card_z)
                p_z = n_z / (n + alpha * card_x * card_y * card_z)
                
                cmi += p_xyz * log(p_xyz * p_z / (p_xz * p_yz), base)
    
    return cmi


def bootstrap_cmi_estimator(X, Y, Z, n_bootstrap=100, base=2):
    """
    Method 3: Bootstrap CMI estimator for uncertainty quantification.
    
    Best for: Getting confidence intervals on CMI estimates
    Pros: Provides uncertainty estimates, robust to outliers
    Cons: Computationally expensive, still needs base estimator
    """
    n = len(X)
    cmi_estimates = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n, size=n, replace=True)
        X_boot = X[indices]
        Y_boot = Y[indices]
        Z_boot = Z[indices]
        
        # Estimate CMI on bootstrap sample
        try:
            cmi_boot = plugin_cmi_estimator(X_boot, Y_boot, Z_boot, base=base)
            cmi_estimates.append(cmi_boot)
        except:
            continue
    
    if not cmi_estimates:
        return 0.0, 0.0, 0.0
    
    cmi_estimates = np.array(cmi_estimates)
    return np.mean(cmi_estimates), np.std(cmi_estimates), np.percentile(cmi_estimates, [5, 95])


def adaptive_binning_cmi(X, Y, Z, max_bins=None, base=2):
    """
    Method 4: Adaptive binning for mixed discrete/continuous data.
    
    Best for: When some variables might be continuous-like discrete
    Pros: Handles mixed data types, reduces dimensionality
    Cons: Loses information through binning, needs tuning
    """
    if max_bins is None:
        max_bins = int(np.sqrt(len(X)))
    
    # Adaptively bin each variable if it has too many unique values
    def adaptive_bin(var):
        unique_vals = len(np.unique(var))
        if unique_vals <= max_bins:
            return var  # Keep as-is if few unique values
        else:
            # Bin into max_bins categories
            return np.digitize(var, np.linspace(var.min(), var.max(), max_bins+1)[1:-1])
    
    X_binned = adaptive_bin(X)
    Y_binned = adaptive_bin(Y)
    
    if Z.ndim > 1:
        Z_binned = np.column_stack([adaptive_bin(Z[:, i]) for i in range(Z.shape[1])])
    else:
        Z_binned = adaptive_bin(Z)
    
    return plugin_cmi_estimator(X_binned, Y_binned, Z_binned, base=base)


def ksg_discrete_cmi(X, Y, Z, k=3, discrete_features=None, base=2):
    """
    Method 5: Modified KSG estimator for mixed discrete-continuous data.
    
    Best for: When you want k-NN benefits but handle discrete properly
    Pros: Principled approach, handles mixed data
    Cons: Complex implementation, needs careful tuning
    """
    # For pure discrete data, fall back to plugin estimator
    if discrete_features is None or all(discrete_features):
        return plugin_cmi_estimator(X, Y, Z, base=base)
    
    # This would require complex implementation for mixed data
    # Placeholder for the concept
    return plugin_cmi_estimator(X, Y, Z, base=base)


def chi_square_independence_test(X, Y, Z, alpha=0.05):
    """
    Method 6: Chi-square test of conditional independence.
    
    Best for: Binary hypothesis testing rather than CMI estimation
    Pros: Well-established, gives p-values
    Cons: Doesn't give CMI value, just independence test
    """
    # Create contingency table for each value of Z
    if Z.ndim > 1:
        z_values = [tuple(row) for row in Z]
    else:
        z_values = Z
    
    p_values = []
    
    for z_val in set(z_values):
        mask = np.array(z_values) == z_val if Z.ndim == 1 else np.array([tuple(row) == z_val for row in Z])
        
        if np.sum(mask) < 5:  # Skip if too few samples
            continue
            
        X_cond = X[mask]
        Y_cond = Y[mask]
        
        # Create contingency table
        unique_x = sorted(set(X_cond))
        unique_y = sorted(set(Y_cond))
        
        contingency = np.zeros((len(unique_x), len(unique_y)))
        for i, x in enumerate(unique_x):
            for j, y in enumerate(unique_y):
                contingency[i, j] = np.sum((X_cond == x) & (Y_cond == y))
        
        # Chi-square test
        if contingency.sum() > 0 and contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_val, _, _ = chi2_contingency(contingency)
            p_values.append(p_val)
    
    if not p_values:
        return True, 1.0  # Cannot reject independence
    
    # Use minimum p-value (most significant test)
    min_p = min(p_values)
    is_independent = min_p > alpha
    
    return is_independent, min_p


def compare_estimators_on_data():
    """
    Compare different estimators on simulated data matching your characteristics.
    """
    print("=== Comparing Discrete CMI Estimators ===")
    
    np.random.seed(42)
    n_samples = 1000  # Smaller for demonstration
    
    # Test Case 1: Independent variables (should give CMI ≈ 0)
    print("\n1. Testing Independent Variables:")
    X = np.random.randint(0, 8, n_samples)
    Y = np.random.randint(0, 8, n_samples)  
    Z = np.random.randint(0, 6, (n_samples, 2))
    
    methods = [
        ("Plug-in", lambda: plugin_cmi_estimator(X, Y, Z)),
        ("Smoothed", lambda: smoothed_plugin_cmi(X, Y, Z, alpha=0.1)),
        ("Adaptive", lambda: adaptive_binning_cmi(X, Y, Z, max_bins=8)),
        ("Chi-square", lambda: chi_square_independence_test(X, Y, Z)),
    ]
    
    for name, method in methods:
        try:
            if name == "Chi-square":
                is_indep, p_val = method()
                print(f"  {name}: Independent={is_indep}, p-value={p_val:.4f}")
            else:
                cmi = method()
                print(f"  {name}: CMI = {cmi:.4f} nats")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    
    # Test Case 2: Correlated variables (should give CMI > 0)
    print("\n2. Testing Correlated Variables:")
    X = np.random.randint(0, 8, n_samples)
    noise = np.random.randint(0, 3, n_samples)
    Y = (X + noise) % 8  # Y correlated with X
    Z = np.random.randint(0, 6, (n_samples, 2))
    
    for name, method in methods[:-1]:  # Skip chi-square for this test
        try:
            if name == "Plug-in":
                method = lambda: plugin_cmi_estimator(X, Y, Z)
            elif name == "Smoothed":
                method = lambda: smoothed_plugin_cmi(X, Y, Z, alpha=0.1)
            elif name == "Adaptive":
                method = lambda: adaptive_binning_cmi(X, Y, Z, max_bins=8)
            
            cmi = method()
            print(f"  {name}: CMI = {cmi:.4f} nats")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")


def main():
    """
    Main analysis of discrete CMI estimators.
    """
    print("📊 DISCRETE-APPROPRIATE CMI ESTIMATORS")
    print("=" * 60)
    print("For discrete data with 3-10 cardinality, 15K samples, 2-4D conditioning")
    print("=" * 60)
    
    estimators = [
        {
            "name": "1. Plug-in Estimator",
            "best_for": "Pure discrete data, exact calculation",
            "pros": ["Theoretically exact", "Fast computation", "No parameters"],
            "cons": ["Curse of dimensionality", "Zero count problems", "Needs large samples"],
            "sample_requirement": "~1000 per conditioning value",
            "recommended": "YES - Primary choice for your data"
        },
        {
            "name": "2. Smoothed Plug-in (Laplace)",
            "best_for": "Discrete data with potential sparse counts",
            "pros": ["Handles zero counts", "Robust", "Theoretically grounded"],
            "cons": ["Introduces bias", "Needs tuning α", "Still suffers curse of dimensionality"],
            "sample_requirement": "~500 per conditioning value",
            "recommended": "YES - Good fallback option"
        },
        {
            "name": "3. Bootstrap CMI",
            "best_for": "Uncertainty quantification",
            "pros": ["Confidence intervals", "Robust to outliers", "No distributional assumptions"],
            "cons": ["Computationally expensive", "Still needs base estimator", "Slower"],
            "sample_requirement": "Same as base estimator",
            "recommended": "MAYBE - For uncertainty analysis"
        },
        {
            "name": "4. Adaptive Binning",
            "best_for": "Mixed discrete/continuous-like data",
            "pros": ["Handles high cardinality", "Reduces dimensionality", "Flexible"],
            "cons": ["Loses information", "Parameter tuning", "Not exact for discrete"],
            "sample_requirement": "~200 per bin",
            "recommended": "NO - Your data is already discrete"
        },
        {
            "name": "5. Modified KSG",
            "best_for": "Mixed discrete-continuous data",
            "pros": ["Principled approach", "Handles mixed types", "Research active"],
            "cons": ["Complex implementation", "Limited software", "Not widely tested"],
            "sample_requirement": "Variable",
            "recommended": "NO - Overkill for pure discrete"
        },
        {
            "name": "6. Chi-square Independence Test",
            "best_for": "Binary hypothesis testing",
            "pros": ["Well-established", "P-values", "Handles categorical data"],
            "cons": ["Binary output only", "No CMI estimate", "Multiple testing issues"],
            "sample_requirement": "~30 per cell",
            "recommended": "MAYBE - As validation check"
        },
    ]
    
    for est in estimators:
        print(f"\n{est['name']}")
        print(f"  Best for: {est['best_for']}")
        print(f"  Pros: {', '.join(est['pros'])}")
        print(f"  Cons: {', '.join(est['cons'])}")
        print(f"  Sample requirement: {est['sample_requirement']}")
        print(f"  Recommended: {est['recommended']}")
    
    print(f"\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS FOR YOUR SIMULATION")
    print("=" * 60)
    
    print("Given your characteristics:")
    print("  • 15K samples")
    print("  • 3-10 cardinality per variable")
    print("  • 2-4D conditioning")
    print("  • Pure discrete data")
    print()
    
    print("RECOMMENDED APPROACH:")
    print("  1. PRIMARY: Smoothed Plug-in Estimator (α=0.1-1.0)")
    print("     - Handles your sample size well")
    print("     - Robust to zero counts")
    print("     - Theoretically grounded for discrete data")
    print()
    print("  2. VALIDATION: Chi-square Independence Test")
    print("     - Binary validation of conditional independence")
    print("     - Complements CMI with hypothesis testing")
    print()
    print("  3. FALLBACK: Standard Plug-in (if enough samples per combination)")
    print("     - Most exact when sample size permits")
    print()
    
    print("IMPLEMENTATION STRATEGY:")
    print("  1. Replace k-NN CMI with smoothed plug-in")
    print("  2. Tune α parameter (0.1-1.0) based on validation performance")
    print("  3. Set threshold based on empirical calibration")
    print("  4. Use chi-square test as additional validation")
    
    # Run comparison
    compare_estimators_on_data()


if __name__ == '__main__':
    main() 