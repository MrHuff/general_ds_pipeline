import shap
shap.initjs()

# model = load_model
#load test_data
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

#Single pred
shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_display.iloc[0,:])

#many preds
shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], X_display.iloc[:1000,:])

#summary plot
shap.summary_plot(shap_values, X)


#dependenct plot
for name in X_train.columns:
    shap.dependence_plot(name, shap_values[1], X, display_features=X_display)