import pandas as pd

def run_scenarios(X_test, X_test_scaled, model, scaler):
    
    # Baseline
    baseline_pred = model.predict(X_test_scaled)
    results = {"Baseline": baseline_pred.mean()}

    # Scenario 1: Increase TV by 10%
    X_tv = X_test.copy()
    X_tv["TV"] *= 1.10
    results["+10% TV"] = model.predict(scaler.transform(X_tv)).mean()

    # Scenario 2: Reduce Newspaper by 20%
    X_news = X_test.copy()
    X_news["Newspaper"] *= 0.80
    results["-20% Newspaper"] = model.predict(scaler.transform(X_news)).mean()

    # Scenario 3: Increase Radio by 10%
    X_radio = X_test.copy()
    X_radio["Radio"] *= 1.10
    results["+10% Radio"] = model.predict(scaler.transform(X_radio)).mean()

    # Scenario 4: Increase TV by 10% & Radio by 15%
    X_tv_radio = X_test.copy()
    X_tv_radio["TV"] *= 1.10
    X_tv_radio["Radio"] *= 1.15
    results["+10% TV & +15% Radio"] = model.predict(scaler.transform(X_tv_radio)).mean()

    # Scenario 5: Reduce all spends by 10%
    X_cut = X_test.copy() * 0.90
    results["-10% All Spends"] = model.predict(scaler.transform(X_cut)).mean()

    # Impacts (difference from baseline)
    impacts = {scenario: avg - results["Baseline"]
               for scenario, avg in results.items() if scenario != "Baseline"}

    return results, impacts
