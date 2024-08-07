import logging

from tasks import (evaluate_predictions, fit_models, load_input_data, predict,
                   preprocess_input_data, save_results, setup_pipeline_run,
                   tune_models)

CONFIG_DIR = "numerical_illustration/config/demo_config.yaml"

# Set up a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Setup the numerical illustration
    config, rng, output_path = setup_pipeline_run(config_path=CONFIG_DIR)
    logger.info("Setup complete")

    # Load data
    raw_input_data = load_input_data(config=config, rng=rng)
    logger.info("Data loaded")

    # Preprocess data
    train_data, test_data = preprocess_input_data(
        config=config, data=raw_input_data, rng=rng
    )
    logger.info("Data preprocessed")

    # Tune models
    tuning_results, n_estimators = tune_models(
        config=config,
        train_data=train_data,
        rng=rng,
    )
    logger.info("Models tuned")

    # Fit models
    models = fit_models(
        config=config, train_data=train_data, rng=rng, n_estimators=n_estimators
    )
    logger.info("Models fitted")

    # Predict
    train_data = predict(models=models, data=train_data)
    test_data = predict(models=models, data=test_data)
    logger.info("Predictions made")

    # Evaluate
    metrics = evaluate_predictions(
        train_data=train_data, test_data=test_data, config=config
    )
    logger.info("Predictions evaluated")

    # Save results
    save_results(
        train_data=train_data,
        test_data=test_data,
        loss_results=tuning_results,
        metrics=metrics,
        models=models,
        output_path=output_path,
    )
    logger.info("Results saved")

    logger.info((metrics.astype(float) / 1e2).round(2))


if __name__ == "__main__":
    main()
