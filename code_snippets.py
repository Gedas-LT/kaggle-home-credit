threshold = [i * 0.001 for i in range(1, 10)]

for i in threshold:

    log_pipe = Pipeline(
        steps=[
            ("custom_preprocessor_1", get_organization_type),
            ("custom_preprocessor_2", get_credit_card_dpd),
            ("custom_preprocessor_3", get_flag_insurance),
            ("custom_preprocessor_4", get_age_binning),
            ("custom_preprocessor_5", get_bureau_credit_type_counter),
            ("custom_preprocessor_6", get_prev_credit_type_counter),
            ("custom_preprocessor_7", get_prev_flag_insurance),
            ("custom_preprocessor_8", get_annuity_income_ratio),
            ("custom_preprocessor_9", get_prev_annuity_income_ratio),
            ("custom_preprocessor_10", get_installments_version),
            ("custom_preprocessor_11", get_drop_id),
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(estimator=log_reg, threshold=i)),
            ("model", log_reg)
        ]
    )

    log_pipe.fit(X_train, y_train)
    predictions = log_pipe.predict_proba(X_test)

    output = pd.DataFrame({"SK_ID_CURR": SK_ID_CURR, "TARGET": predictions[:, 1]})
    output.to_csv(f"submissions/temp/log_{i}_mean.csv", index=False)